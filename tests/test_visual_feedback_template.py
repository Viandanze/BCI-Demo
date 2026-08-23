"""
Tests for VisualFeedback template and performance changes.

Covers:
  - Template externalization: explicit path, bundled file, inline fallback
  - EEG downsampling before SSE push
  - Pre-serialized JSON payloads in the EEG queue
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import flask  # noqa: F401
except ImportError:
    flask = None

pytestmark = pytest.mark.skipif(
    flask is None, reason="flask not installed in this environment"
)

from src.feedback.visual_feedback import VisualFeedback


def _make_batch(n_samples: int, n_channels: int = 8) -> list:
    return [[0.01 * ch for ch in range(n_channels)] for _ in range(n_samples)]


class TestTemplateLoading:
    def test_explicit_path_wins(self, tmp_path):
        custom = tmp_path / "custom.html"
        custom.write_text("<html>CUSTOM-MARKER</html>", encoding="utf-8")
        vf = VisualFeedback(template_path=str(custom))
        assert "CUSTOM-MARKER" in vf._template

    def test_bundled_template_used_by_default(self):
        vf = VisualFeedback()
        assert "<html" in vf._template
        assert len(vf._template) > 20000  # full dashboard, not a stub

    def test_bad_path_falls_back_to_inline(self, tmp_path):
        missing = tmp_path / "does_not_exist.html"
        vf = VisualFeedback(template_path=str(missing))
        assert vf._template == VisualFeedback.HTML_TEMPLATE

    def test_non_html_file_rejected(self, tmp_path):
        junk = tmp_path / "junk.html"
        junk.write_text("just text, no markup", encoding="utf-8")
        vf = VisualFeedback(template_path=str(junk))
        assert vf._template == VisualFeedback.HTML_TEMPLATE

    def test_bundled_file_matches_inline(self):
        bundled = Path(VisualFeedback.DEFAULT_TEMPLATE_PATH)
        assert bundled.exists()
        assert bundled.read_text(encoding="utf-8") == VisualFeedback.HTML_TEMPLATE


class TestEEGDownsample:
    def test_downsample_halves_payload(self):
        vf = VisualFeedback(eeg_downsample=2)
        assert vf.update_eeg(_make_batch(400)) is True
        payload = vf._eeg_queue.get_nowait()
        # Queue now holds pre-serialized JSON strings.
        assert isinstance(payload, str)
        event = json.loads(payload)
        assert event["type"] == "eeg_batch"
        assert len(event["data"]) == 200

    def test_no_downsample_keeps_all_samples(self):
        vf = VisualFeedback(eeg_downsample=1)
        vf.update_eeg(_make_batch(100))
        event = json.loads(vf._eeg_queue.get_nowait())
        assert len(event["data"]) == 100

    def test_short_batches_survive_downsample(self):
        vf = VisualFeedback(eeg_downsample=4)
        vf.update_eeg(_make_batch(3))  # shorter than the factor
        event = json.loads(vf._eeg_queue.get_nowait())
        assert len(event["data"]) == 3  # shorter than factor: batch kept intact

    def test_channel_count_preserved(self):
        vf = VisualFeedback(eeg_downsample=2)
        vf.update_eeg(_make_batch(100, n_channels=4))
        event = json.loads(vf._eeg_queue.get_nowait())
        assert len(event["data"][0]) == 4
