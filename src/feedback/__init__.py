"""Feedback module - Visual and audio feedback for BCI-LLM interaction."""

from .visual_feedback import VisualFeedback
from .audio_feedback import AudioFeedback

__all__ = ["VisualFeedback", "AudioFeedback"]
