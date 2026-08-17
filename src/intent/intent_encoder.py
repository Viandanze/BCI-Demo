"""
Intent Encoder - Maps BCI decoder output to structured cognitive intents.

Design: Collaborative Reasoning Mode (Approach D)
  Instead of treating BCI as a keyboard (one label = one character),
  we map motor imagery classes to high-level cognitive modes.
  The LLM does the heavy lifting (knowledge generation, reasoning),
  while the human brain does what it's best at: rapid intuitive selection.

Mapping (PhysioNet 4-class motor imagery):
  Left Hand  → QUERY   (search for knowledge / factual lookup)
  Right Hand → REASON  (logical deduction / calculation / analysis)
  Feet       → CREATE  (generate solutions / creative ideas)
  Tongue     → REVIEW  (summarize / synthesize current context)
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Optional
import time


class CognitiveMode(Enum):
    """High-level cognitive modes mapped from BCI motor imagery classes."""
    QUERY = "query"     # Left hand → search knowledge
    REASON = "reason"   # Right hand → logical reasoning / calculation
    CREATE = "create"   # Feet → generate solutions / creative ideas
    REVIEW = "review"   # Tongue → summarize / synthesize context

    @property
    def description(self) -> str:
        descriptions = {
            CognitiveMode.QUERY: "Search for knowledge or factual information",
            CognitiveMode.REASON: "Logical reasoning, calculation, or step-by-step analysis",
            CognitiveMode.CREATE: "Generate creative solutions, ideas, or novel approaches",
            CognitiveMode.REVIEW: "Summarize and synthesize the current conversation context",
        }
        return descriptions.get(self, "General assistance")


class DecoderLabel(Enum):
    """Motor imagery class labels (PhysioNet standard)."""
    LEFT_HAND = 0
    RIGHT_HAND = 1
    FEET = 2
    TONGUE = 3
    REST = 4  # Idle / no clear intent


@dataclass
class Intent:
    """Structured intent output from the encoder."""
    mode: CognitiveMode
    confidence: float
    raw_label: int
    raw_probabilities: list[float]
    timestamp: float = field(default_factory=time.time)

    @property
    def is_confident(self) -> bool:
        """Whether the intent confidence exceeds the activation threshold."""
        return self.confidence >= 0.6

    @property
    def mode_label(self) -> str:
        """Human-readable mode label for display."""
        labels = {
            CognitiveMode.QUERY: "🔍 Query",
            CognitiveMode.REASON: "🧠 Reason",
            CognitiveMode.CREATE: "✨ Create",
            CognitiveMode.REVIEW: "📋 Review",
        }
        return labels.get(self.mode, self.mode.value)

    def to_dict(self) -> dict:
        """Serialize to dict for JSON transmission."""
        return {
            "mode": self.mode.value,
            "mode_label": self.mode_label,
            "description": self.mode.description,
            "confidence": round(self.confidence, 3),
            "raw_label": self.raw_label,
            "timestamp": self.timestamp,
        }


class IntentEncoder:
    """
    Maps raw decoder output to structured cognitive intents.

    Features:
      - Debounce: requires N consecutive identical predictions to lock intent
      - Confidence threshold: ignores low-confidence predictions
      - Rest state detection: idle when REST label dominates
    """

    # PhysioNet label → CognitiveMode mapping
    LABEL_TO_MODE = {
        DecoderLabel.LEFT_HAND: CognitiveMode.QUERY,
        DecoderLabel.RIGHT_HAND: CognitiveMode.REASON,
        DecoderLabel.FEET: CognitiveMode.CREATE,
        DecoderLabel.TONGUE: CognitiveMode.REVIEW,
    }

    def __init__(
        self,
        confidence_threshold: float = 0.6,
        debounce_frames: int = 3,
        rest_label: int = 4,
    ):
        """
        Args:
            confidence_threshold: Minimum softmax probability to accept a prediction.
            debounce_frames: Number of consecutive identical predictions required to lock.
            rest_label: Label index for idle/rest state.
        """
        self.confidence_threshold = confidence_threshold
        self.debounce_frames = debounce_frames
        self.rest_label = rest_label
        self._label_history: list[int] = []
        self._last_locked: Optional[Intent] = None

    def encode(
        self,
        label: int,
        probabilities: list[float],
        timestamp: Optional[float] = None,
    ) -> Optional[Intent]:
        """
        Encode a single decoder output frame to structured intent.

        Args:
            label: Predicted class index from decoder (0-4).
            probabilities: Softmax probability vector from decoder.
            timestamp: Optional timestamp; defaults to current time.

        Returns:
            Intent object if intent is locked after debounce, None otherwise.
        """
        if timestamp is None:
            timestamp = time.time()

        # Update debounce history
        self._label_history.append(label)
        if len(self._label_history) > self.debounce_frames:
            self._label_history.pop(0)

        # Need enough frames to debounce
        if len(self._label_history) < self.debounce_frames:
            return None

        # All frames must agree
        if not all(l == label for l in self._label_history):
            return None

        # Rest state → no intent
        if label == self.rest_label:
            return None

        # Confidence check
        confidence = probabilities[label] if probabilities else 0.0
        if confidence < self.confidence_threshold:
            return None

        # Map to cognitive mode
        try:
            decoder_label = DecoderLabel(label)
        except ValueError:
            return None

        mode = self.LABEL_TO_MODE.get(decoder_label)
        if mode is None:
            return None

        intent = Intent(
            mode=mode,
            confidence=confidence,
            raw_label=label,
            raw_probabilities=probabilities,
            timestamp=timestamp,
        )
        self._last_locked = intent
        return intent

    def encode_selection(
        self,
        label: int,
        probabilities: list[float],
        n_candidates: int,
        timestamp: Optional[float] = None,
    ) -> Optional[int]:
        """
        Encode a BCI selection from candidate list.

        For Phase 1, we use a simple mapping:
          Left Hand  → candidate 0 (first)
          Right Hand → candidate 1 (second)
          Feet       → candidate 2 (third)
          Tongue     → candidate 3 (fourth, if exists)

        Args:
            label: Predicted class index.
            probabilities: Softmax probabilities.
            n_candidates: Number of available candidates.
            timestamp: Optional timestamp.

        Returns:
            Selected candidate index, or None if invalid.
        """
        if timestamp is None:
            timestamp = time.time()

        # Debounce check
        self._label_history.append(label)
        if len(self._label_history) > self.debounce_frames:
            self._label_history.pop(0)

        if len(self._label_history) < self.debounce_frames:
            return None
        if not all(l == label for l in self._label_history):
            return None

        # Map label to candidate index
        if label >= n_candidates:
            return None

        confidence = probabilities[label] if probabilities else 0.0
        if confidence < self.confidence_threshold:
            return None

        return label

    def reset(self):
        """Reset debounce history (call after each completed interaction)."""
        self._label_history.clear()
        self._last_locked = None
