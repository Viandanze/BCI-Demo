"""LLM Bridge module - Abstract interface for LLM backends.

Also provides AsyncLLMBridge, a non-blocking wrapper that runs synchronous
LLM calls on a background worker thread so real-time loops never stall.
"""

from .llm_client import (
    LLMClient, OllamaClient, APIClient, MockLLMClient,
    CachedLLMClient, create_llm_client,
)
from .async_bridge import AsyncLLMBridge, LLMResult
from .deepseek_scheduler import DeepSeekClient, is_peak_hour

__all__ = [
    "LLMClient", "OllamaClient", "APIClient", "MockLLMClient",
    "CachedLLMClient", "create_llm_client", "AsyncLLMBridge", "LLMResult",
    "DeepSeekClient", "is_peak_hour",
]
