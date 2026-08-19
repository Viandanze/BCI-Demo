"""
Audio feedback for BCI interaction events (Phase 2).

Zero third-party dependencies:
  - Windows: winsound.Beep (system speaker / default audio device).
  - Other platforms: silent no-op fallback (beeps are optional UX sugar,
    never on the critical path).

All playback happens on a daemon thread by default, so the BCI main loop
is never blocked by audio latency.
"""

import logging
import threading
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import winsound  # Windows only
    _HAS_WINSOUND = True
except ImportError:
    winsound = None
    _HAS_WINSOUND = False

# Event name -> list of (frequency_hz, duration_s) steps.
# Distinct rhythms let the user hear which stage the loop is in without
# looking at the screen.
EVENT_PATTERNS: dict[str, list[tuple[int, float]]] = {
    "intent_locked": [(740, 0.10)],
    "candidates_ready": [(660, 0.07), (880, 0.07), (1100, 0.09)],
    "candidate_selected": [(990, 0.12)],
    "turn_completed": [(523, 0.09), (659, 0.09), (784, 0.16)],
    "error": [(220, 0.30)],
}


class AudioFeedback:
    """Non-blocking audio cues for BCI interaction events.

    Args:
        enabled: Master switch (False silences everything).
        block: If True, play synchronously on the calling thread (used by
            tests and headless scripts); default False spawns a daemon
            thread so the real-time loop never waits.
        engine: Optional beep backend injected by tests. Must expose
            `beep(frequency_hz, duration_ms)`.
    """

    def __init__(
        self,
        enabled: bool = True,
        block: bool = False,
        engine: Optional[object] = None,
    ):
        self._enabled = enabled
        self._block = block
        self._engine = engine if engine is not None else self._default_engine()

    # ------------------------------------------------------------------
    # Engine plumbing
    # ------------------------------------------------------------------

    @staticmethod
    def _default_engine():
        """Return a beep engine for the current platform (or None)."""

        class _WinsoundEngine:
            @staticmethod
            def beep(frequency_hz: int, duration_ms: int) -> None:
                winsound.Beep(frequency_hz, duration_ms)

        if _HAS_WINSOUND:
            return _WinsoundEngine()
        logger.debug(
            "AudioFeedback: no platform beep backend (non-Windows); "
            "audio cues are silently disabled."
        )
        return None

    @property
    def enabled(self) -> bool:
        return self._enabled

    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable audio at runtime."""
        self._enabled = enabled

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def play(self, event: str) -> bool:
        """Play the audio pattern for a BCI event.

        Args:
            event: One of EVENT_PATTERNS keys (e.g. "intent_locked").

        Returns:
            True if a pattern was dispatched, False if muted / unknown
            event / no engine available.

        Raises:
            ValueError: If the event name is unknown.
        """
        if event not in EVENT_PATTERNS:
            raise ValueError(
                f"Unknown audio event: {event!r}. "
                f"Known events: {sorted(EVENT_PATTERNS)}"
            )
        if not self._enabled:
            return False
        if self._engine is None:
            return False

        pattern = EVENT_PATTERNS[event]
        if self._block:
            self._beep_pattern(pattern)
        else:
            threading.Thread(
                target=self._beep_pattern, args=(pattern,), daemon=True
            ).start()
        return True

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _beep_pattern(self, pattern: list[tuple[int, float]]) -> None:
        import time

        for frequency_hz, duration_s in pattern:
            try:
                self._engine.beep(frequency_hz, int(duration_s * 1000))
            except Exception as exc:  # never let audio break the loop
                logger.debug("AudioFeedback beep failed: %s", exc)
                return
            time.sleep(0.02)
