"""
Context Manager - Tracks BCI conversation state and interaction history.

State machine:
  IDLE → DETECTING → INTENT_LOCKED → AWAITING_LLM
       → PRESENTING_CANDIDATES → SELECTING → COMPLETED → IDLE

The context manager ensures:
  1. Clean state transitions (no jumping from COMPLETED to AWAITING_LLM)
  2. Timeout handling (auto-select best candidate if user doesn't respond)
  3. Conversation context window (last N turns for LLM prompt)
  4. Thread-safe state access for the visual feedback server
"""

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
import time
import threading


class BCIState(Enum):
    """BCI interaction state machine states."""
    IDLE = "idle"
    DETECTING = "detecting"
    INTENT_LOCKED = "intent_locked"
    AWAITING_LLM = "awaiting_llm"
    PRESENTING_CANDIDATES = "presenting_candidates"
    SELECTING = "selecting"
    COMPLETED = "completed"


@dataclass
class InteractionTurn:
    """One complete BCI-LLM interaction turn."""
    turn_id: int
    intent_mode: str
    intent_confidence: float
    intent_label: str = ""
    llm_candidates: list[str] = field(default_factory=list)
    selected_index: int = -1
    final_response: str = ""
    timestamp: float = field(default_factory=time.time)
    duration: float = 0.0

    def to_dict(self) -> dict:
        return {
            "turn_id": self.turn_id,
            "intent_mode": self.intent_mode,
            "intent_label": self.intent_label,
            "intent_confidence": round(self.intent_confidence, 3),
            "n_candidates": len(self.llm_candidates),
            "selected_index": self.selected_index,
            "final_response": self.final_response[:200],  # Truncate for logging
            "timestamp": self.timestamp,
            "duration": round(self.duration, 2),
        }


class ContextManager:
    """
    Manages BCI conversation context and state machine.

    Thread-safe: uses a lock for state transitions, so the visual feedback
    server can read state from a different thread.
    """

    MAX_CONTEXT_TURNS = 3  # Keep last 3 turns for LLM context window

    def __init__(self, selection_timeout: float = 15.0):
        """
        Args:
            selection_timeout: Seconds to wait for user selection before auto-selecting.
        """
        self._lock = threading.Lock()
        self._state = BCIState.IDLE
        self._turns: list[InteractionTurn] = []
        self._current_turn: Optional[InteractionTurn] = None
        self._turn_counter = 0
        self._selection_timeout = selection_timeout
        self._last_state_change = time.time()
        self._turn_start_time = 0.0

    @property
    def state(self) -> BCIState:
        with self._lock:
            return self._state

    @property
    def current_turn(self) -> Optional[InteractionTurn]:
        with self._lock:
            return self._current_turn

    @property
    def turns(self) -> list[InteractionTurn]:
        with self._lock:
            return list(self._turns)

    def transition_to(self, new_state: BCIState):
        """Transition to a new state (thread-safe)."""
        with self._lock:
            self._state = new_state
            self._last_state_change = time.time()

    def start_turn(self, intent_mode: str, confidence: float, label: str = "") -> InteractionTurn:
        """
        Start a new interaction turn.

        Args:
            intent_mode: Cognitive mode string (query/reason/create/review).
            confidence: Decoder confidence for this intent.
            label: Human-readable intent label.

        Returns:
            The new InteractionTurn.
        """
        with self._lock:
            self._turn_counter += 1
            self._turn_start_time = time.time()
            self._current_turn = InteractionTurn(
                turn_id=self._turn_counter,
                intent_mode=intent_mode,
                intent_confidence=confidence,
                intent_label=label,
            )
            self._state = BCIState.INTENT_LOCKED
            self._last_state_change = time.time()
            return self._current_turn

    def set_awaiting_llm(self):
        """Mark that we're waiting for LLM response."""
        with self._lock:
            self._state = BCIState.AWAITING_LLM
            self._last_state_change = time.time()

    def set_candidates(self, candidates: list[str]):
        """Set LLM-generated candidates and transition to presenting."""
        with self._lock:
            if self._current_turn:
                self._current_turn.llm_candidates = candidates
            self._state = BCIState.PRESENTING_CANDIDATES
            self._last_state_change = time.time()

    def select_candidate(self, index: int) -> Optional[str]:
        """
        User selects a candidate (via second BCI round or auto-timeout).

        Args:
            index: Index of selected candidate.

        Returns:
            The selected response text, or None if invalid.
        """
        with self._lock:
            if not self._current_turn or not self._current_turn.llm_candidates:
                return None
            if index < 0 or index >= len(self._current_turn.llm_candidates):
                return None

            self._current_turn.selected_index = index
            self._current_turn.final_response = self._current_turn.llm_candidates[index]
            self._current_turn.duration = time.time() - self._turn_start_time

            # Archive turn
            self._turns.append(self._current_turn)
            if len(self._turns) > self.MAX_CONTEXT_TURNS:
                self._turns = self._turns[-self.MAX_CONTEXT_TURNS:]

            self._state = BCIState.COMPLETED
            self._last_state_change = time.time()
            return self._current_turn.final_response

    def auto_select_best(self) -> Optional[str]:
        """Auto-select the first candidate (highest ranked by LLM) on timeout."""
        return self.select_candidate(0)

    def get_context_for_llm(self) -> list[dict]:
        """
        Get conversation context formatted for LLM prompt.

        Returns:
            List of message dicts (role/content) for the LLM.
        """
        with self._lock:
            context = []
            for turn in self._turns:
                context.append({
                    "role": "user",
                    "content": f"[{turn.intent_mode}] (confidence: {turn.intent_confidence:.2f})",
                })
                context.append({
                    "role": "assistant",
                    "content": turn.final_response,
                })
            return context

    def check_timeout(self) -> bool:
        """
        Check if current state has timed out.

        Returns:
            True if timed out and auto-selection should trigger.
        """
        with self._lock:
            if self._state == BCIState.PRESENTING_CANDIDATES:
                elapsed = time.time() - self._last_state_change
                return elapsed > self._selection_timeout
            return False

    def reset_to_idle(self):
        """Reset to idle state, ready for next interaction."""
        with self._lock:
            self._state = BCIState.IDLE
            self._current_turn = None
            self._last_state_change = time.time()

    def get_state_dict(self) -> dict:
        """Get current state as dict (for visual feedback)."""
        with self._lock:
            return {
                "state": self._state.value,
                "turn_id": self._turn_counter,
                "current_intent": (
                    self._current_turn.intent_label if self._current_turn else None
                ),
                "current_mode": (
                    self._current_turn.intent_mode if self._current_turn else None
                ),
                "current_confidence": (
                    round(self._current_turn.intent_confidence, 3)
                    if self._current_turn else None
                ),
                "candidates": (
                    self._current_turn.llm_candidates if self._current_turn else []
                ),
                "n_past_turns": len(self._turns),
            }
