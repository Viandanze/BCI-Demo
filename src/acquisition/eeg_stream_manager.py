"""Rolling-window EEG stream manager for real-time visualization."""

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EEGStreamConfig:
    """Configuration for rolling-window EEG stream display.

    Instead of pushing individual samples one at a time, the stream manager
    retrieves a complete rolling window of recent EEG data at fixed intervals.
    This ensures the visualization buffer is always full from the first push.

    Attributes:
        window_seconds: Duration of the rolling window in seconds.
        push_interval: Interval between pushes in seconds (controls refresh rate).
        display_channels: Number of EEG channels to include in each push.
        sample_rate: Sampling rate of the data source in Hz.
    """

    window_seconds: float = 2.0
    push_interval: float = 0.2
    display_channels: int = 8
    sample_rate: int = 250

    @property
    def window_samples(self) -> int:
        """Number of samples in one rolling window."""
        return int(self.window_seconds * self.sample_rate)


class EEGStreamManager:
    """Rolling-window EEG stream manager for real-time visualization.

    This is the standard approach used by professional EEG visualization tools
    (BrainFlow examples, EDF Viewer, OpenViBE). Instead of streaming individual
    samples and slowly filling a display buffer, each push delivers a complete
    window of recent data that directly replaces the visualization buffer.

    The manager decouples data acquisition timing from display refresh timing,
    allowing independent control of acquisition rate and visualization FPS.
    """

    def __init__(self, acquisition, config: EEGStreamConfig = None):
        """Initialize the EEG stream manager.

        Args:
            acquisition: Data acquisition object implementing get_recent_data().
            config: Stream configuration. Uses defaults if None.
        """
        self._acquisition = acquisition
        self.config = config or EEGStreamConfig()
        self._last_push_time: float = 0.0

    def update(self, current_time: float, feedback) -> None:
        """Push a rolling window of EEG data to the feedback interface.

        Called from the main processing loop. If the configured push interval
        has elapsed, retrieves a complete window of recent EEG data and forwards
        it as a batch. The feedback interface replaces its entire buffer with
        each batch, ensuring the display is always full.

        Args:
            current_time: Current timestamp in seconds.
            feedback: VisualFeedback instance with update_eeg() method.
        """
        if current_time - self._last_push_time < self.config.push_interval:
            return

        self._last_push_time = current_time
        window_size = self.config.window_samples

        eeg_data = self._acquisition.get_recent_data(window_size)
        if eeg_data is None or eeg_data.shape[1] == 0:
            return

        n_channels = min(self.config.display_channels, eeg_data.shape[0])
        batch = eeg_data[:n_channels, :].T.tolist()

        feedback.update_eeg(batch)
