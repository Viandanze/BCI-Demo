"""
Tests for AudioFeedback (Phase 2).

Covers:
  - Event pattern registry completeness
  - Unknown event rejection
  - Enabled / disabled behaviour
  - Synchronous (block=True) playback via an injected engine
  - Missing platform engine degrades silently
"""

import pytest
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.feedback.audio_feedback import AudioFeedback, EVENT_PATTERNS


class FakeEngine:
    """Records beep calls instead of making sound."""

    def __init__(self, fail: bool = False):
        self.calls: list[tuple[int, int]] = []
        self.fail = fail

    def beep(self, frequency_hz: int, duration_ms: int) -> None:
        if self.fail:
            raise RuntimeError("no audio device")
        self.calls.append((frequency_hz, duration_ms))


class TestEventPatterns:
    def test_all_events_have_patterns(self):
        for event, pattern in EVENT_PATTERNS.items():
            assert isinstance(pattern, list) and pattern, event
            for freq, dur in pattern:
                assert 20 <= freq <= 20000, (event, freq)
                assert 0.01 <= dur <= 1.0, (event, dur)

    def test_expected_event_set(self):
        assert set(EVENT_PATTERNS) == {
            "intent_locked",
            "candidates_ready",
            "candidate_selected",
            "turn_completed",
            "error",
        }


class TestPlayBehaviour:
    def test_unknown_event_raises(self):
        audio = AudioFeedback(block=True, engine=FakeEngine())
        with pytest.raises(ValueError, match="Unknown audio event"):
            audio.play("nope")

    def test_disabled_silences_everything(self):
        engine = FakeEngine()
        audio = AudioFeedback(enabled=False, block=True, engine=engine)
        assert audio.play("error") is False
        assert engine.calls == []

    def test_no_engine_degrades_silently(self, monkeypatch):
        # Force the "no platform backend" path even on Windows (winsound) or CI runners.
        monkeypatch.setattr("src.feedback.audio_feedback._HAS_WINSOUND", False)
        audio = AudioFeedback(block=True, engine=None)
        assert audio.play("error") is False  # must not raise

    def test_set_enabled_at_runtime(self):
        engine = FakeEngine()
        audio = AudioFeedback(enabled=False, block=True, engine=engine)
        audio.set_enabled(True)
        assert audio.enabled is True
        assert audio.play("intent_locked") is True
        assert len(engine.calls) == 1

    def test_blocking_playback_matches_pattern(self):
        engine = FakeEngine()
        audio = AudioFeedback(block=True, engine=engine)
        assert audio.play("candidates_ready") is True
        expected = [
            (660, 70), (880, 70), (1100, 90),
        ]
        assert engine.calls == expected

    def test_async_playback_returns_immediately(self):
        engine = FakeEngine()
        audio = AudioFeedback(block=False, engine=engine)
        start = time.time()
        assert audio.play("turn_completed") is True
        assert time.time() - start < 0.2  # did not block on 3 beeps
        # Daemon thread fires shortly after.
        deadline = time.time() + 2.0
        while len(engine.calls) < 3 and time.time() < deadline:
            time.sleep(0.01)
        assert len(engine.calls) == 3

    def test_engine_failure_never_propagates(self):
        audio = AudioFeedback(block=True, engine=FakeEngine(fail=True))
        assert audio.play("error") is True  # dispatched, then swallowed
