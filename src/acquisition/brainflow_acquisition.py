"""BrainFlow synthetic board data acquisition."""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class BrainFlowAcquisition:
    """BrainFlow synthetic board data acquisition.

    Wraps the BrainFlow BoardShim API to provide a simple interface for
    acquiring EEG data from a synthetic (simulated) board.

    Attributes:
        board_id: BrainFlow board identifier.
        sampling_rate: Sampling rate in Hz.
        n_eeg_channels: Number of EEG channels.
        channel_names: List of EEG channel names.
    """

    def __init__(self, board_id: int = -1):
        try:
            import brainflow
            from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
            self.BoardShim = BoardShim

            # Use synthetic board
            self.board_id = BoardIds.SYNTHETIC_BOARD if hasattr(BoardIds, 'SYNTHETIC_BOARD') else board_id
            self.params = BrainFlowInputParams()
            self.board = BoardShim(self.board_id, self.params)
            self.channel_names = BoardShim.get_eeg_names(self.board_id)
            self.sampling_rate = BoardShim.get_sampling_rate(self.board_id)
            self.n_eeg_channels = len(self.channel_names)
            self._available = True
            logger.info(f"BrainFlow ready. Board ID: {self.board_id}, "
                        f"Rate: {self.sampling_rate}Hz, "
                        f"EEG Channels: {self.n_eeg_channels} ({self.channel_names})")
        except ImportError:
            logger.warning("BrainFlow not installed. Install with: pip install brainflow")
            self._available = False

    @property
    def available(self) -> bool:
        return self._available

    def start(self):
        if not self._available:
            return
        self.board.prepare_session()
        self.board.start_stream(450000)
        logger.info("BrainFlow stream started")

    def stop(self):
        if not self._available:
            return
        self.board.stop_stream()
        self.board.release_session()
        logger.info("BrainFlow stream stopped")

    def get_recent_data(self, n_samples: int) -> Optional[np.ndarray]:
        """Get most recent n_samples of EEG data.

        Returns:
            numpy array of shape (n_eeg_channels, n_samples) or None.
        """
        if not self._available:
            return None
        data = self.board.get_current_board_data(n_samples)
        eeg_data = data[:self.n_eeg_channels, :]
        return eeg_data
