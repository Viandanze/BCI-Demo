"""EEG decoder modules (Mock and Real EEGNet)."""

from .mock_decoder import MockDecoder
from .real_decoder import RealDecoder

__all__ = ['MockDecoder', 'RealDecoder']
