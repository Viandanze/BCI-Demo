"""
Tests for EEGStreamManager module.

Covers:
  - EEGStreamConfig dataclass: default values, custom values
  - EEGStreamConfig.window_samples property
  - EEGStreamManager initialization (default and custom config)
  - update() interval gating (no push before interval)
  - update() batch push when interval elapsed
  - Batch format (transposed to [n_samples, n_channels])
  - display_channels limiting
  - None / empty data handling
  - Multiple update calls and last_push_time tracking
"""

import os
import sys
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import numpy as np

from src.acquisition.eeg_stream_manager import EEGStreamConfig, EEGStreamManager


class TestEEGStreamConfig:
    """Test EEGStreamConfig dataclass."""

    def test_default_values(self):
        cfg = EEGStreamConfig()
        assert cfg.window_seconds == 2.0
        assert cfg.push_interval == 0.2
        assert cfg.display_channels == 8
        assert cfg.sample_rate == 250

    def test_custom_values(self):
        cfg = EEGStreamConfig(
            window_seconds=5.0,
            push_interval=0.5,
            display_channels=16,
            sample_rate=500,
        )
        assert cfg.window_seconds == 5.0
        assert cfg.push_interval == 0.5
        assert cfg.display_channels == 16
        assert cfg.sample_rate == 500

    def test_window_samples_default(self):
        cfg = EEGStreamConfig()
        assert cfg.window_samples == 500  # 2.0 * 250

    def test_window_samples_custom(self):
        cfg = EEGStreamConfig(window_seconds=3.0, sample_rate=500)
        assert cfg.window_samples == 1500  # 3.0 * 500

    def test_window_samples_fractional(self):
        """window_samples truncates to int."""
        cfg = EEGStreamConfig(window_seconds=1.5, sample_rate=250)
        assert cfg.window_samples == 375  # 1.5 * 250 = 375.0 -> int

    def test_window_samples_tiny_window(self):
        cfg = EEGStreamConfig(window_seconds=0.004, sample_rate=250)
        assert cfg.window_samples == 1  # 0.004 * 250 = 1.0 -> int


class TestEEGStreamManagerInit:
    """Test EEGStreamManager initialization."""

    def test_init_with_default_config(self):
        acq = MagicMock()
        mgr = EEGStreamManager(acq)
        assert mgr.config.window_seconds == 2.0
        assert mgr.config.push_interval == 0.2
        assert mgr._last_push_time == 0.0

    def test_init_with_custom_config(self):
        acq = MagicMock()
        cfg = EEGStreamConfig(window_seconds=5.0, push_interval=1.0)
        mgr = EEGStreamManager(acq, config=cfg)
        assert mgr.config.window_seconds == 5.0
        assert mgr.config.push_interval == 1.0

    def test_init_stores_acquisition(self):
        acq = MagicMock()
        mgr = EEGStreamManager(acq)
        assert mgr._acquisition is acq


class TestUpdateIntervalGating:
    """Test that update() respects the push interval."""

    def test_no_push_before_interval(self):
        acq = MagicMock()
        feedback = MagicMock()
        mgr = EEGStreamManager(acq)

        # current_time (0.1) - last_push_time (0.0) = 0.1 < 0.2 interval
        mgr.update(0.1, feedback)
        acq.get_recent_data.assert_not_called()
        feedback.update_eeg.assert_not_called()

    def test_push_at_exact_interval(self):
        acq = MagicMock()
        acq.get_recent_data.return_value = np.zeros((8, 500))
        feedback = MagicMock()
        mgr = EEGStreamManager(acq)

        # 0.2 - 0.0 = 0.2, not < 0.2 → should push
        mgr.update(0.2, feedback)
        acq.get_recent_data.assert_called_once_with(500)
        feedback.update_eeg.assert_called_once()

    def test_push_after_interval(self):
        acq = MagicMock()
        acq.get_recent_data.return_value = np.zeros((8, 500))
        feedback = MagicMock()
        mgr = EEGStreamManager(acq)

        mgr.update(0.5, feedback)
        acq.get_recent_data.assert_called_once()

    def test_second_push_respects_new_interval(self):
        acq = MagicMock()
        acq.get_recent_data.return_value = np.zeros((8, 500))
        feedback = MagicMock()
        mgr = EEGStreamManager(acq)

        mgr.update(0.2, feedback)   # first push at 0.2
        assert feedback.update_eeg.call_count == 1

        mgr.update(0.3, feedback)   # 0.3 - 0.2 = 0.1 < 0.2 → no push
        assert feedback.update_eeg.call_count == 1

        mgr.update(0.5, feedback)   # 0.5 - 0.2 = 0.3 >= 0.2 → push
        assert feedback.update_eeg.call_count == 2


class TestUpdateBatchFormat:
    """Test the batch data format pushed to feedback."""

    def test_batch_is_transposed(self):
        """Data (n_channels, n_samples) → batch (n_samples, n_channels)."""
        acq = MagicMock()
        data = np.random.randn(8, 500)
        acq.get_recent_data.return_value = data
        feedback = MagicMock()
        mgr = EEGStreamManager(acq)

        mgr.update(0.2, feedback)
        batch = feedback.update_eeg.call_args[0][0]
        assert len(batch) == 500          # n_samples rows
        assert len(batch[0]) == 8         # n_channels cols

    def test_batch_is_list_of_lists(self):
        acq = MagicMock()
        acq.get_recent_data.return_value = np.zeros((4, 100))
        feedback = MagicMock()
        mgr = EEGStreamManager(acq)

        mgr.update(0.2, feedback)
        batch = feedback.update_eeg.call_args[0][0]
        assert isinstance(batch, list)
        assert isinstance(batch[0], list)

    def test_display_channels_limits_output(self):
        """When data has more channels than display_channels, only use first N."""
        acq = MagicMock()
        acq.get_recent_data.return_value = np.zeros((16, 500))
        feedback = MagicMock()
        cfg = EEGStreamConfig(display_channels=4)
        mgr = EEGStreamManager(acq, config=cfg)

        mgr.update(0.2, feedback)
        batch = feedback.update_eeg.call_args[0][0]
        assert len(batch[0]) == 4  # only 4 channels

    def test_display_channels_fewer_than_data(self):
        """When data has fewer channels than display_channels, use all."""
        acq = MagicMock()
        acq.get_recent_data.return_value = np.zeros((3, 500))
        feedback = MagicMock()
        cfg = EEGStreamConfig(display_channels=8)
        mgr = EEGStreamManager(acq, config=cfg)

        mgr.update(0.2, feedback)
        batch = feedback.update_eeg.call_args[0][0]
        assert len(batch[0]) == 3  # only 3 channels available


class TestUpdateEdgeCases:
    """Test edge cases for update()."""

    def test_none_data_skips_push(self):
        acq = MagicMock()
        acq.get_recent_data.return_value = None
        feedback = MagicMock()
        mgr = EEGStreamManager(acq)

        mgr.update(0.2, feedback)
        feedback.update_eeg.assert_not_called()

    def test_empty_data_skips_push(self):
        acq = MagicMock()
        acq.get_recent_data.return_value = np.zeros((8, 0))
        feedback = MagicMock()
        mgr = EEGStreamManager(acq)

        mgr.update(0.2, feedback)
        feedback.update_eeg.assert_not_called()

    def test_window_size_matches_config(self):
        """get_recent_data is called with the correct window size."""
        acq = MagicMock()
        acq.get_recent_data.return_value = np.zeros((8, 750))
        feedback = MagicMock()
        cfg = EEGStreamConfig(window_seconds=3.0, sample_rate=250)
        mgr = EEGStreamManager(acq, config=cfg)

        mgr.update(0.2, feedback)
        acq.get_recent_data.assert_called_once_with(750)  # 3.0 * 250
