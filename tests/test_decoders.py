"""
Tests for decoder modules (MockDecoder and RealDecoder).

Covers:
  - MockDecoder: predict() return format, probability distribution, call count
  - MockDecoder: different sample_rate values, multi-channel input
  - RealDecoder: model not found raises FileNotFoundError
  - RealDecoder: fallback when model path is invalid (mock file scenario)
  - RealDecoder: _preprocess() with mock torch/scipy dependencies
  - RealDecoder: _infer_architecture() from mock state_dict
"""

import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import numpy as np

from src.decoders.mock_decoder import MockDecoder


# ---------------------------------------------------------------------------
# Helper: create a fake torch module so RealDecoder.__init__ can import it
# ---------------------------------------------------------------------------
def _make_mock_torch():
    """Return a MagicMock that quacks like the torch module."""
    mock_torch = MagicMock()
    mock_torch.device.return_value = 'cpu'
    mock_torch.cuda.is_available.return_value = False
    # torch.load returns a checkpoint dict
    mock_torch.load.return_value = {
        'model_state_dict': {},
        'best_accuracy': 0.75,
    }
    # torch.FloatTensor, torch.no_grad, torch.softmax for predict
    mock_torch.FloatTensor = lambda x: MagicMock()
    mock_torch.no_grad = MagicMock()
    mock_torch.softmax = lambda x, dim=1: MagicMock()
    return mock_torch


def _make_mock_models():
    """Return dict of mock modules for src.models.eegnet and src.models."""
    mock_model_cls = MagicMock()
    mock_model = MagicMock()
    mock_model.load_state_dict = MagicMock()  # success by default
    mock_model_cls.return_value = mock_model
    mock_model_cls.side_effect = lambda **kw: mock_model

    mock_eegnet_mod = MagicMock()
    mock_eegnet_mod.EEGNetClassifier = mock_model_cls
    mock_models_mod = MagicMock()

    return {
        'src.models': mock_models_mod,
        'src.models.eegnet': mock_eegnet_mod,
    }


# ---------------------------------------------------------------------------
# MockDecoder tests
# ---------------------------------------------------------------------------
class TestMockDecoderInit:
    """Test MockDecoder initialization."""

    def test_default_init(self):
        dec = MockDecoder()
        assert dec.n_classes == 4
        assert dec.sample_rate == 250
        assert dec._call_count == 0

    def test_custom_init(self):
        dec = MockDecoder(n_classes=3, sample_rate=128)
        assert dec.n_classes == 3
        assert dec.sample_rate == 128

    def test_call_count_starts_zero(self):
        dec = MockDecoder()
        assert dec._call_count == 0


class TestMockDecoderPredict:
    """Test MockDecoder.predict() method."""

    @pytest.fixture
    def eeg_data(self):
        """Generate realistic EEG-like data: 8 channels, 250 samples."""
        rng = np.random.RandomState(42)
        return rng.randn(8, 250)

    def test_returns_tuple(self, eeg_data):
        dec = MockDecoder()
        result = dec.predict(eeg_data)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_label_is_int(self, eeg_data):
        dec = MockDecoder()
        label, probs = dec.predict(eeg_data)
        assert isinstance(label, int)

    def test_label_in_valid_range(self, eeg_data):
        dec = MockDecoder()
        label, probs = dec.predict(eeg_data)
        assert 0 <= label < 4

    def test_probabilities_is_list(self, eeg_data):
        dec = MockDecoder()
        label, probs = dec.predict(eeg_data)
        assert isinstance(probs, list)

    def test_probabilities_length_matches_classes(self, eeg_data):
        dec = MockDecoder()
        label, probs = dec.predict(eeg_data)
        assert len(probs) == 4

    def test_probabilities_sum_to_one(self, eeg_data):
        dec = MockDecoder()
        label, probs = dec.predict(eeg_data)
        assert abs(sum(probs) - 1.0) < 1e-6

    def test_probabilities_non_negative(self, eeg_data):
        dec = MockDecoder()
        label, probs = dec.predict(eeg_data)
        assert all(p >= 0.0 for p in probs)

    def test_call_count_increments(self, eeg_data):
        dec = MockDecoder()
        assert dec._call_count == 0
        dec.predict(eeg_data)
        assert dec._call_count == 1
        dec.predict(eeg_data)
        assert dec._call_count == 2

    def test_different_sample_rate(self):
        """Predict with sample_rate=128 should still produce valid output."""
        rng = np.random.RandomState(99)
        data = rng.randn(8, 128)
        dec = MockDecoder(sample_rate=128)
        label, probs = dec.predict(data)
        assert 0 <= label < 4
        assert len(probs) == 4
        assert abs(sum(probs) - 1.0) < 1e-6

    def test_single_channel_input(self):
        """Decoder should handle single-channel data."""
        rng = np.random.RandomState(7)
        data = rng.randn(1, 250)
        dec = MockDecoder()
        label, probs = dec.predict(data)
        assert 0 <= label < 4
        assert len(probs) == 4

    def test_label_matches_argmax(self, eeg_data):
        """The returned label should be the argmax of probabilities."""
        dec = MockDecoder()
        label, probs = dec.predict(eeg_data)
        assert label == int(np.argmax(probs))


# ---------------------------------------------------------------------------
# RealDecoder tests (all mocked — no real torch / model required)
# ---------------------------------------------------------------------------
class TestRealDecoderModelNotFound:
    """Test RealDecoder when no trained model exists."""

    def test_raises_filenotfound_when_no_model(self):
        """Should raise FileNotFoundError when no model path and no glob match."""
        from src.decoders.real_decoder import RealDecoder

        mock_torch = _make_mock_torch()
        with patch.dict(sys.modules, {'torch': mock_torch}), \
             patch('glob.glob', return_value=[]):
            with pytest.raises(FileNotFoundError, match="No trained model found"):
                RealDecoder(model_path=None)

    def test_raises_filenotfound_for_nonexistent_path(self):
        """When an explicit model path is given but torch.load fails, error propagates."""
        from src.decoders.real_decoder import RealDecoder

        mock_torch = _make_mock_torch()
        mock_torch.load.side_effect = FileNotFoundError("File not found: /fake/model.pt")
        with patch.dict(sys.modules, {'torch': mock_torch}):
            with pytest.raises(FileNotFoundError):
                RealDecoder(model_path='/nonexistent/model.pt')

    def test_auto_detect_picks_last_candidate(self):
        """When model_path is None and glob returns candidates, uses last one."""
        from src.decoders.real_decoder import RealDecoder

        mock_torch = _make_mock_torch()
        mock_modules = _make_mock_models()
        mock_modules['torch'] = mock_torch

        candidates = [
            'outputs/eegnet_s1/best_model.pt',
            'outputs/eegnet_s2/best_model.pt',
        ]
        with patch.dict(sys.modules, mock_modules), \
             patch('glob.glob', return_value=candidates):
            decoder = RealDecoder(model_path=None)
            # torch.load should have been called with the last candidate
            mock_torch.load.assert_called_once()
            called_path = mock_torch.load.call_args[0][0]
            assert called_path == 'outputs/eegnet_s2/best_model.pt'


class TestRealDecoderInferArchitecture:
    """Test _infer_architecture() with a mock state_dict."""

    def test_infer_default_values_when_no_matching_keys(self):
        """When no keys match, should fall back to defaults (64ch, 640 samples)."""
        from src.decoders.real_decoder import RealDecoder

        decoder = RealDecoder.__new__(RealDecoder)
        decoder.N_CLASSES = 4

        state_dict = {'unrelated.weight': np.array([1.0])}

        mock_models = _make_mock_models()
        with patch.dict(sys.modules, mock_models):
            n_ch, n_times = decoder._infer_architecture(state_dict)

        assert n_ch == 64
        assert isinstance(n_times, int)

    def test_infer_n_channels_from_spatial_weight(self):
        """Should infer n_channels from a 'spatial' weight key."""
        from src.decoders.real_decoder import RealDecoder

        decoder = RealDecoder.__new__(RealDecoder)
        decoder.N_CLASSES = 4

        # shape[2] should be the channel count
        spatial_weight = np.zeros((2, 1, 32, 1))  # n_channels=32
        state_dict = {
            'spatial.weight': spatial_weight,
            'unrelated.bias': np.zeros(4),
        }

        mock_models = _make_mock_models()
        with patch.dict(sys.modules, mock_models):
            n_ch, n_times = decoder._infer_architecture(state_dict)

        assert n_ch == 32


class TestRealDecoderPreprocess:
    """Test _preprocess() method with mocked dependencies."""

    def _make_decoder(self):
        """Create a RealDecoder instance bypassing __init__."""
        from src.decoders.real_decoder import RealDecoder

        decoder = RealDecoder.__new__(RealDecoder)
        decoder.MODEL_SAMPLE_RATE = 128.0
        decoder.BANDPASS_LOW = 4.0
        decoder.BANDPASS_HIGH = 38.0
        decoder.N_CLASSES = 4
        decoder.source_sample_rate = 250.0
        decoder.model_channels = 64
        decoder.model_times = 640
        decoder._sos = np.array([[[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]]])
        return decoder

    def test_preprocess_resamples_to_model_rate(self):
        """250Hz data should be resampled to 128Hz."""
        decoder = self._make_decoder()
        eeg = np.random.randn(8, 1000)  # 4s at 250Hz

        with patch('scipy.signal.resample') as mock_resample, \
             patch('scipy.signal.sosfiltfilt') as mock_filt:
            mock_resample.side_effect = lambda x, n, axis=1: x[:, :n]
            mock_filt.side_effect = lambda sos, x, axis=1: x
            result = decoder._preprocess(eeg, source_rate=250.0)

        # Should have been called to resample
        mock_resample.assert_called_once()

    def test_preprocess_crops_to_model_times(self):
        """When data is longer than model_times, it gets cropped."""
        decoder = self._make_decoder()
        # Simulate 128Hz data already at correct rate → no resample
        eeg = np.random.randn(64, 1000)  # longer than 640

        with patch('scipy.signal.resample') as mock_resample, \
             patch('scipy.signal.sosfiltfilt') as mock_filt:
            mock_filt.side_effect = lambda sos, x, axis=1: x
            result = decoder._preprocess(eeg, source_rate=128.0)

        assert result.shape[1] == 640  # cropped to model_times
        assert result.shape[0] == 64   # 64 channels (no adaptation needed)

    def test_preprocess_pads_short_data(self):
        """When data is shorter than model_times, it gets zero-padded."""
        decoder = self._make_decoder()
        eeg = np.random.randn(64, 400)  # shorter than 640

        with patch('scipy.signal.sosfiltfilt') as mock_filt:
            mock_filt.side_effect = lambda sos, x, axis=1: x
            result = decoder._preprocess(eeg, source_rate=128.0)

        assert result.shape[1] == 640  # padded to model_times

    def test_preprocess_channel_adaptation_repeat(self):
        """When fewer channels than model, channels are repeated."""
        decoder = self._make_decoder()
        eeg = np.random.randn(8, 640)  # 8 channels, model wants 64

        with patch('scipy.signal.sosfiltfilt') as mock_filt:
            mock_filt.side_effect = lambda sos, x, axis=1: x
            result = decoder._preprocess(eeg, source_rate=128.0)

        assert result.shape[0] == 64  # repeated to 64 channels

    def test_preprocess_channel_adaptation_truncate(self):
        """When more channels than model, channels are truncated."""
        decoder = self._make_decoder()
        eeg = np.random.randn(100, 640)  # 100 channels, model wants 64

        with patch('scipy.signal.sosfiltfilt') as mock_filt:
            mock_filt.side_effect = lambda sos, x, axis=1: x
            result = decoder._preprocess(eeg, source_rate=128.0)

        assert result.shape[0] == 64  # truncated to 64 channels

    def test_preprocess_output_is_float32(self):
        """Preprocessed data should be float32."""
        decoder = self._make_decoder()
        eeg = np.random.randn(64, 640).astype(np.float64)

        with patch('scipy.signal.sosfiltfilt') as mock_filt:
            mock_filt.side_effect = lambda sos, x, axis=1: x
            result = decoder._preprocess(eeg, source_rate=128.0)

        assert result.dtype == np.float32
