"""
Tests for ContextManager module.

Covers:
  - State machine transitions (IDLE → DETECTING → ... → COMPLETED → IDLE)
  - Turn management (start, set candidates, select, archive)
  - Context window (last N turns for LLM prompt)
  - Timeout detection
  - Auto-select best candidate
  - Thread safety (concurrent access)
  - State serialization for visual feedback
"""

import pytest
import sys
import time
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.intent.context_manager import (
    ContextManager,
    BCIState,
    InteractionTurn,
)


class TestBCIState:
    """Test BCIState enum."""

    def test_state_values(self):
        assert BCIState.IDLE.value == "idle"
        assert BCIState.DETECTING.value == "detecting"
        assert BCIState.INTENT_LOCKED.value == "intent_locked"
        assert BCIState.AWAITING_LLM.value == "awaiting_llm"
        assert BCIState.PRESENTING_CANDIDATES.value == "presenting_candidates"
        assert BCIState.SELECTING.value == "selecting"
        assert BCIState.COMPLETED.value == "completed"


class TestContextManagerBasics:
    """Test basic ContextManager functionality."""

    def test_initial_state_is_idle(self):
        cm = ContextManager()
        assert cm.state == BCIState.IDLE

    def test_no_current_turn_initially(self):
        cm = ContextManager()
        assert cm.current_turn is None

    def test_no_past_turns_initially(self):
        cm = ContextManager()
        assert len(cm.turns) == 0

    def test_transition_to(self):
        cm = ContextManager()
        cm.transition_to(BCIState.DETECTING)
        assert cm.state == BCIState.DETECTING


class TestTurnManagement:
    """Test interaction turn lifecycle."""

    def test_start_turn(self):
        cm = ContextManager()
        turn = cm.start_turn("query", 0.85, "Query")

        assert turn.turn_id == 1
        assert turn.intent_mode == "query"
        assert turn.intent_confidence == 0.85
        assert turn.intent_label == "Query"
        assert cm.state == BCIState.INTENT_LOCKED
        assert cm.current_turn is turn

    def test_start_multiple_turns_increments_id(self):
        cm = ContextManager()

        cm.start_turn("query", 0.8, "Query")
        cm.set_awaiting_llm()
        cm.set_candidates(["a", "b", "c"])
        cm.select_candidate(0)
        cm.reset_to_idle()

        cm.start_turn("create", 0.7, "Create")
        assert cm.current_turn.turn_id == 2

    def test_set_candidates(self):
        cm = ContextManager()
        cm.start_turn("reason", 0.9, "Reason")
        cm.set_awaiting_llm()
        cm.set_candidates(["Option A", "Option B", "Option C"])

        assert cm.state == BCIState.PRESENTING_CANDIDATES
        assert len(cm.current_turn.llm_candidates) == 3

    def test_select_candidate_valid(self):
        cm = ContextManager()
        cm.start_turn("create", 0.75, "Create")
        cm.set_awaiting_llm()
        cm.set_candidates(["Idea A", "Idea B", "Idea C"])

        result = cm.select_candidate(1)
        assert result == "Idea B"
        assert cm.current_turn.selected_index == 1
        assert cm.current_turn.final_response == "Idea B"
        assert cm.state == BCIState.COMPLETED

    def test_select_candidate_invalid_index(self):
        cm = ContextManager()
        cm.start_turn("create", 0.75, "Create")
        cm.set_candidates(["A", "B"])

        result = cm.select_candidate(5)  # Out of range
        assert result is None

    def test_select_candidate_negative_index(self):
        cm = ContextManager()
        cm.start_turn("create", 0.75, "Create")
        cm.set_candidates(["A", "B"])

        result = cm.select_candidate(-1)
        assert result is None

    def test_select_without_candidates(self):
        cm = ContextManager()
        cm.start_turn("create", 0.75, "Create")
        # No candidates set
        result = cm.select_candidate(0)
        assert result is None

    def test_auto_select_best(self):
        cm = ContextManager()
        cm.start_turn("query", 0.8, "Query")
        cm.set_candidates(["Best", "Second", "Third"])

        result = cm.auto_select_best()
        assert result == "Best"  # First candidate


class TestContextWindow:
    """Test conversation context window management."""

    def test_turns_archived_after_completion(self):
        cm = ContextManager()
        cm.start_turn("query", 0.8, "Query")
        cm.set_candidates(["Response A", "Response B"])
        cm.select_candidate(0)

        assert len(cm.turns) == 1
        assert cm.turns[0].final_response == "Response A"

    def test_max_context_turns(self):
        cm = ContextManager(selection_timeout=100.0)

        # Create 5 turns
        for i in range(5):
            cm.reset_to_idle()
            cm.start_turn("query", 0.8, "Query")
            cm.set_candidates([f"Response {i}"])
            cm.select_candidate(0)

        # Should only keep last 3
        assert len(cm.turns) == ContextManager.MAX_CONTEXT_TURNS
        assert cm.turns[-1].final_response == "Response 4"

    def test_get_context_for_llm(self):
        cm = ContextManager(selection_timeout=100.0)
        cm.start_turn("query", 0.8, "Query")
        cm.set_candidates(["Answer 1"])
        cm.select_candidate(0)

        context = cm.get_context_for_llm()
        assert len(context) == 2  # user + assistant message
        assert context[0]["role"] == "user"
        assert context[1]["role"] == "assistant"
        assert context[1]["content"] == "Answer 1"


class TestTimeout:
    """Test timeout detection and auto-selection."""

    def test_no_timeout_in_idle(self):
        cm = ContextManager(selection_timeout=0.1)
        assert cm.check_timeout() is False

    def test_timeout_in_presenting_candidates(self):
        cm = ContextManager(selection_timeout=0.1)
        cm.start_turn("query", 0.8, "Query")
        cm.set_candidates(["A", "B", "C"])

        time.sleep(0.15)  # Wait past timeout
        assert cm.check_timeout() is True

    def test_no_timeout_before_expiry(self):
        cm = ContextManager(selection_timeout=5.0)
        cm.start_turn("query", 0.8, "Query")
        cm.set_candidates(["A", "B", "C"])

        assert cm.check_timeout() is False


class TestResetToIdle:
    """Test reset behavior."""

    def test_reset_to_idle(self):
        cm = ContextManager()
        cm.start_turn("query", 0.8, "Query")
        cm.set_candidates(["A", "B"])
        cm.select_candidate(0)

        cm.reset_to_idle()
        assert cm.state == BCIState.IDLE
        assert cm.current_turn is None

    def test_reset_preserves_past_turns(self):
        cm = ContextManager()
        cm.start_turn("query", 0.8, "Query")
        cm.set_candidates(["A"])
        cm.select_candidate(0)

        cm.reset_to_idle()
        assert len(cm.turns) == 1  # Archived turn preserved


class TestStateDict:
    """Test state serialization for visual feedback."""

    def test_state_dict_idle(self):
        cm = ContextManager()
        d = cm.get_state_dict()
        assert d["state"] == "idle"
        assert d["turn_id"] == 0
        assert d["current_intent"] is None
        assert d["candidates"] == []

    def test_state_dict_with_active_turn(self):
        cm = ContextManager()
        cm.start_turn("create", 0.85, "Create")
        cm.set_candidates(["Idea 1", "Idea 2"])

        d = cm.get_state_dict()
        assert d["state"] == "presenting_candidates"
        assert d["current_mode"] == "create"
        assert d["current_confidence"] == 0.85
        assert len(d["candidates"]) == 2


class TestThreadSafety:
    """Test thread-safe access."""

    def test_concurrent_transitions(self):
        cm = ContextManager()
        errors = []

        def worker():
            try:
                for _ in range(100):
                    cm.transition_to(BCIState.DETECTING)
                    cm.transition_to(BCIState.IDLE)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert cm.state in (BCIState.IDLE, BCIState.DETECTING)

    def test_concurrent_read_write(self):
        cm = ContextManager()
        cm.start_turn("query", 0.8, "Query")
        cm.set_candidates(["A", "B", "C"])

        read_results = []
        errors = []

        def reader():
            try:
                for _ in range(50):
                    d = cm.get_state_dict()
                    read_results.append(d)
            except Exception as e:
                errors.append(e)

        def writer():
            try:
                for _ in range(50):
                    cm.transition_to(BCIState.SELECTING)
                    cm.transition_to(BCIState.PRESENTING_CANDIDATES)
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=reader)
        t2 = threading.Thread(target=writer)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert len(errors) == 0
        assert len(read_results) == 50


class TestEnhancedContext:
    """Session summary + EEG stats + enhanced context."""

    @staticmethod
    def _run_turn(cm, mode="query", conf=0.8, response="answer"):
        cm.start_turn(mode, conf)
        cm.set_candidates([response, "other"])
        return cm.select_candidate(0)

    def test_summary_empty_session(self):
        cm = ContextManager()
        assert cm.get_session_summary() == "First turn of the session."

    def test_summary_counts_all_turns(self):
        cm = ContextManager()
        self._run_turn(cm, mode="query", conf=0.8)
        self._run_turn(cm, mode="reason", conf=0.6)
        summary = cm.get_session_summary()
        assert "2 completed turn(s)" in summary
        assert "queryx1" in summary and "reasonx1" in summary
        assert "average decoder confidence 0.70" in summary

    def test_summary_survives_turn_trimming(self):
        cm = ContextManager()
        for i in range(5):
            self._run_turn(cm, mode="query", conf=0.5)
        assert len(cm.turns) == cm.MAX_CONTEXT_TURNS  # list trimmed
        assert "5 completed turn(s)" in cm.get_session_summary()  # stats kept

    def test_record_eeg_stats_appears_in_context(self):
        cm = ContextManager()
        cm.record_eeg_stats(
            mean_confidence=0.86,
            intent_distribution={"left_hand": 0.6, "rest": 0.2},
        )
        ctx = cm.get_enhanced_context()
        text = ctx[0]["content"]
        assert "mean confidence 0.86" in text
        assert "left_hand" in text

    def test_enhanced_context_alternates_roles(self):
        cm = ContextManager()
        self._run_turn(cm, mode="query", conf=0.9, response="first")
        self._run_turn(cm, mode="reason", conf=0.7, response="second")
        ctx = cm.get_enhanced_context()
        roles = [m["role"] for m in ctx]
        # Strictly alternating user/assistant starting with user.
        assert roles == ["user", "assistant"] * (len(roles) // 2)

    def test_enhanced_context_includes_recent_turns(self):
        cm = ContextManager()
        self._run_turn(cm, mode="query", conf=0.9, response="first")
        self._run_turn(cm, mode="reason", conf=0.7, response="second")
        ctx = cm.get_enhanced_context()
        contents = " | ".join(m["content"] for m in ctx)
        assert "first" in contents and "second" in contents

    def test_enhanced_context_max_turns_limit(self):
        cm = ContextManager()
        for i in range(4):
            self._run_turn(cm, mode="query", conf=0.5, response=f"r{i}")
        ctx = cm.get_enhanced_context(max_turns=2)
        contents = " | ".join(m["content"] for m in ctx)
        assert "r3" in contents and "r2" in contents
        assert "r0" not in contents and "r1" not in contents
        # 1 summary pair + 2 turn pairs = 6 messages.
        assert len(ctx) == 6

    def test_enhanced_context_empty_session(self):
        cm = ContextManager()
        ctx = cm.get_enhanced_context()
        assert len(ctx) == 2  # only the summary pair
        assert "First turn" in ctx[0]["content"]
