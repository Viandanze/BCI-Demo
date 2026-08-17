"""Intent module - Maps BCI decoder output to structured cognitive intents."""

from .intent_encoder import IntentEncoder, Intent, CognitiveMode, DecoderLabel
from .context_manager import ContextManager, BCIState, InteractionTurn

__all__ = [
    "IntentEncoder", "Intent", "CognitiveMode", "DecoderLabel",
    "ContextManager", "BCIState", "InteractionTurn",
]
