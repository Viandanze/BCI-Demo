"""
Tests for DeepSeek cost-aware scheduling.

Covers:
  - is_peak_hour boundary conditions (09-12 / 14-18, end-exclusive)
  - DeepSeekClient env resolution (DEEPSEEK_API_KEY > LLM_API_KEY)
  - Model switching by pricing window (flash on peak, pro off-peak)
  - schedule_info payload
  - generate_candidates sends the scheduled model to the API
"""

import pytest
import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm_bridge.deepseek_scheduler import (
    DeepSeekClient,
    is_peak_hour,
    PEAK_WINDOWS,
    DEFAULT_API_URL,
)


def _at(hour: int, minute: int = 0) -> datetime:
    return datetime(2026, 8, 18, hour, minute)


class TestIsPeakHour:
    """Peak window boundaries (end-exclusive)."""

    @pytest.mark.parametrize("hour,expected", [
        (0, False), (8, False),          # before morning peak
        (9, True), (10, True), (11, True),  # morning peak 09-12
        (12, False), (13, False),        # lunch gap
        (14, True), (15, True), (17, True),  # afternoon peak 14-18
        (18, False), (23, False),        # evening off-peak
    ])
    def test_hour_boundaries(self, hour, expected):
        assert is_peak_hour(_at(hour)) is expected

    def test_peak_windows_shape(self):
        assert PEAK_WINDOWS == ((9, 12), (14, 18))


class TestDeepSeekClientEnv:
    """Credential / endpoint resolution."""

    @pytest.fixture(autouse=True)
    def _clean_env(self, monkeypatch):
        for key in ("DEEPSEEK_API_KEY", "LLM_API_KEY",
                    "DEEPSEEK_API_URL", "LLM_API_URL"):
            monkeypatch.delenv(key, raising=False)

    def test_explicit_key_wins(self):
        client = DeepSeekClient(api_key="explicit")
        assert client.api_key == "explicit"

    def test_deepseek_env_preferred_over_generic(self, monkeypatch):
        monkeypatch.setenv("LLM_API_KEY", "generic")
        monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek")
        client = DeepSeekClient()
        assert client.api_key == "deepseek"

    def test_generic_env_fallback(self, monkeypatch):
        monkeypatch.setenv("LLM_API_KEY", "generic")
        client = DeepSeekClient()
        assert client.api_key == "generic"

    def test_default_url(self):
        client = DeepSeekClient(api_key="k")
        assert client.api_url == DEFAULT_API_URL

    def test_custom_url(self, monkeypatch):
        monkeypatch.setenv("LLM_API_URL", "https://custom.example/v1/")
        client = DeepSeekClient(api_key="k")
        assert client.api_url == "https://custom.example/v1"


class TestModelSelection:
    """Flash on peak, pro off-peak."""

    def test_peak_uses_flash(self):
        client = DeepSeekClient(api_key="k", now_fn=lambda: _at(10))
        assert client.select_model() == client.flash_model

    def test_offpeak_uses_pro(self):
        client = DeepSeekClient(api_key="k", now_fn=lambda: _at(20))
        assert client.select_model() == client.pro_model

    def test_peak_disabled_keeps_pro(self):
        client = DeepSeekClient(
            api_key="k", peak_uses_flash=False, now_fn=lambda: _at(10)
        )
        assert client.select_model() == client.pro_model

    def test_custom_model_names(self):
        client = DeepSeekClient(
            api_key="k", flash_model="m-flash", pro_model="m-pro",
            now_fn=lambda: _at(10),
        )
        assert client.select_model() == "m-flash"

    def test_schedule_info(self):
        client = DeepSeekClient(api_key="k", now_fn=lambda: _at(10))
        info = client.schedule_info()
        assert info["peak"] is True
        assert info["current_model"] == client.flash_model
        assert info["peak_windows"] == [[9, 12], [14, 18]]


class TestGenerateSwitchesModel:
    """The scheduled model must reach the HTTP payload."""

    class _Resp:
        @staticmethod
        def raise_for_status():
            return None

        @staticmethod
        def json():
            return {"choices": [{"message": {"content": '["a", "b", "c"]'}}]}

    def test_generate_uses_scheduled_model(self):
        client = DeepSeekClient(api_key="k", now_fn=lambda: _at(10))
        with patch("requests.post", return_value=self._Resp()) as mock_post:
            result = client.generate_candidates("query", [])
        assert result == ["a", "b", "c"]
        sent_model = mock_post.call_args.kwargs["json"]["model"]
        assert sent_model == client.flash_model

        # Off-peak: the same client switches to the pro model.
        client._now_fn = lambda: _at(20)
        with patch("requests.post", return_value=self._Resp()) as mock_post:
            client.generate_candidates("query", [])
        sent_model = mock_post.call_args.kwargs["json"]["model"]
        assert sent_model == client.pro_model
