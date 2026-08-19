"""
DeepSeek cost-aware scheduling (Phase 2).

DeepSeek roughly doubles its prices during peak hours
(09:00-12:00 and 14:00-18:00, Beijing time). This module lets the demo
automatically pick the cheap/fast model during peak hours and the strong
model off-peak, so long demo sessions do not burn tokens at 2x price.

Pure scheduling logic (no network calls here) - fully unit-testable via
`now_fn` injection. The actual HTTP call is inherited from APIClient.
"""

import os
from datetime import datetime
from typing import Callable, Optional

from .llm_client import APIClient

logger = __import__("logging").getLogger(__name__)

# Peak pricing windows as (start_hour, end_hour), end-exclusive, local time.
# DeepSeek peak windows: 09:00-12:00 and 14:00-18:00 (Beijing time).
PEAK_WINDOWS: tuple[tuple[int, int], ...] = ((9, 12), (14, 18))

DEFAULT_API_URL = "https://api.deepseek.com/v1"
DEFAULT_FLASH_MODEL = "deepseek-chat"      # cheap & fast tier
DEFAULT_PRO_MODEL = "deepseek-reasoner"    # strong tier


def is_peak_hour(now: Optional[datetime] = None) -> bool:
    """Return True if `now` falls inside a peak pricing window.

    Args:
        now: Optional datetime to test (defaults to current local time).

    Returns:
        True during peak hours, False off-peak.
    """
    hour = now.hour if now is not None else datetime.now().hour
    return any(start <= hour < end for start, end in PEAK_WINDOWS)


class DeepSeekClient(APIClient):
    """OpenAI-compatible client that switches models by pricing window.

    Behavior:
      - Off-peak  -> strong model (`pro_model`, default deepseek-reasoner)
      - Peak      -> cheap model (`flash_model`, default deepseek-chat),
                     configurable via `peak_uses_flash=False` to keep the
                     strong model at all times.
      - API key resolution: `api_key` arg > DEEPSEEK_API_KEY > LLM_API_KEY.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
        flash_model: str = DEFAULT_FLASH_MODEL,
        pro_model: str = DEFAULT_PRO_MODEL,
        peak_uses_flash: bool = True,
        now_fn: Optional[Callable[[], datetime]] = None,
        temperature: float = 0.8,
        timeout: int = 30,
    ):
        """
        Args:
            api_key: DeepSeek API key (env: DEEPSEEK_API_KEY or LLM_API_KEY).
            api_url: Endpoint (env: DEEPSEEK_API_URL or LLM_API_URL; defaults
                to https://api.deepseek.com/v1).
            flash_model: Model used during peak hours (cheap tier).
            pro_model: Model used off-peak (strong tier).
            peak_uses_flash: If False, always use the pro model.
            now_fn: Injectable clock for tests.
            temperature: Sampling temperature.
            timeout: Request timeout in seconds.
        """
        resolved_key = (
            api_key
            or os.getenv("DEEPSEEK_API_KEY", "")
            or os.getenv("LLM_API_KEY", "")
        )
        resolved_url = (
            api_url
            or os.getenv("DEEPSEEK_API_URL", "")
            or os.getenv("LLM_API_URL", "")
            or DEFAULT_API_URL
        )
        super().__init__(
            api_url=resolved_url,
            api_key=resolved_key,
            model=pro_model,
            temperature=temperature,
            timeout=timeout,
        )
        self.flash_model = flash_model
        self.pro_model = pro_model
        self.peak_uses_flash = peak_uses_flash
        self._now_fn = now_fn or datetime.now

    def _now(self) -> datetime:
        return self._now_fn()

    def select_model(self, now: Optional[datetime] = None) -> str:
        """Pick the model for the current pricing window.

        Args:
            now: Optional datetime (defaults to the injected clock).

        Returns:
            The model name to use.
        """
        moment = now if now is not None else self._now()
        if self.peak_uses_flash and is_peak_hour(moment):
            return self.flash_model
        return self.pro_model

    def schedule_info(self, now: Optional[datetime] = None) -> dict:
        """Describe the current scheduling decision (for logging / UI)."""
        moment = now if now is not None else self._now()
        model = self.select_model(moment)
        return {
            "peak": is_peak_hour(moment),
            "current_model": model,
            "flash_model": self.flash_model,
            "pro_model": self.pro_model,
            "peak_windows": [list(w) for w in PEAK_WINDOWS],
        }

    def generate_candidates(
        self,
        intent_mode: str,
        context: list[dict],
        n_candidates: int = 3,
        topic_hint: str = "",
    ) -> list[str]:
        """Generate candidates using the cost-aware model selection."""
        self.model = self.select_model()
        return super().generate_candidates(
            intent_mode, context, n_candidates, topic_hint
        )

    def expand_response(self, selected_response: str, intent_mode: str,
                        context: list[dict]) -> str:
        """Expand a response using the cost-aware model selection."""
        self.model = self.select_model()
        return super().expand_response(selected_response, intent_mode, context)
