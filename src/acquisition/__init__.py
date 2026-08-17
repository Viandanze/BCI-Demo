"""Data acquisition and EEG streaming modules."""

from .brainflow_acquisition import BrainFlowAcquisition
from .eeg_stream_manager import EEGStreamConfig, EEGStreamManager

__all__ = ['BrainFlowAcquisition', 'EEGStreamConfig', 'EEGStreamManager']
