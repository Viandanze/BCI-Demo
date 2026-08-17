"""
LLM Bridge - Abstract interface for LLM backends.

Supports:
  - Ollama (local, private, zero-cost) — default for Phase 1
  - OpenAI-compatible API (cloud, higher quality) — Phase 2

Design:
  The LLM acts as the "knowledge engine" in the collaborative reasoning loop.
  It receives a cognitive mode (query/reason/create/review) + conversation context,
  then generates multiple candidate responses from different perspectives.
  The user selects the best candidate via a second BCI round.

Prompt strategy:
  - System prompt establishes the BCI-collaboration context
  - Mode-specific instructions guide the response style
  - Candidates are generated with different "personalities" (conservative/bold/balanced)
  - Responses are kept short (≤3 sentences) for rapid BCI selection
"""

from abc import ABC, abstractmethod
from typing import Optional
import json
import time
import logging

logger = logging.getLogger(__name__)


class LLMClient(ABC):
    """Abstract LLM client interface."""

    @abstractmethod
    def generate_candidates(
        self,
        intent_mode: str,
        context: list[dict],
        n_candidates: int = 3,
        topic_hint: str = "",
    ) -> list[str]:
        """
        Generate candidate responses based on intent and context.

        Args:
            intent_mode: Cognitive mode (query/reason/create/review).
            context: Previous conversation context (list of message dicts).
            n_candidates: Number of candidates to generate.
            topic_hint: Optional hint about the topic.

        Returns:
            List of candidate response strings.
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the LLM backend is available."""
        pass

    def expand_response(self, selected_response: str, intent_mode: str,
                        context: list[dict]) -> str:
        """
        Expand the selected short candidate into a fuller response.

        Called after the user selects a candidate via BCI.
        The LLM elaborates on the chosen direction.

        Args:
            selected_response: The short candidate the user selected.
            intent_mode: Original cognitive mode.
            context: Conversation context.

        Returns:
            Expanded response string.
        """
        return selected_response  # Default: no expansion


class OllamaClient(LLMClient):
    """Ollama local LLM client (default backend for Phase 1)."""

    MODE_DESCRIPTIONS = {
        "query": (
            "The user wants to search for knowledge or factual information. "
            "Provide accurate, informative answers with specific details."
        ),
        "reason": (
            "The user wants logical reasoning or calculation. "
            "Show clear step-by-step reasoning. Be rigorous and precise."
        ),
        "create": (
            "The user wants creative solutions or ideas. "
            "Be innovative, propose novel approaches, think outside the box."
        ),
        "review": (
            "The user wants a summary or review of the current context. "
            "Synthesize key points, distill insights, identify patterns."
        ),
    }

    CANDIDATE_PERSONAS = [
        "conservative (safe, well-established, widely accepted)",
        "bold (novel, unconventional, thought-provoking)",
        "balanced (practical, nuanced, considers trade-offs)",
    ]

    def __init__(
        self,
        model: str = "qwen2.5:7b",
        host: str = "http://localhost:11434",
        temperature: float = 0.8,
        timeout: int = 120,
    ):
        """
        Args:
            model: Ollama model name (e.g., qwen2.5:7b, llama3.1:8b).
            host: Ollama server URL.
            temperature: Sampling temperature (higher = more creative).
            timeout: Request timeout in seconds (120s for first-time model loading).
        """
        self.model = model
        self.host = host.rstrip("/")
        self.temperature = temperature
        self.timeout = timeout

    def is_available(self) -> bool:
        """Check if Ollama server is running and model is available."""
        try:
            import requests
            resp = requests.get(f"{self.host}/api/tags", timeout=3)
            if resp.status_code != 200:
                return False
            models = resp.json().get("models", [])
            model_names = [m.get("name", "") for m in models]
            return any(self.model in name for name in model_names)
        except Exception as e:
            logger.debug(f"Ollama not available: {e}")
            return False

    def _build_system_prompt(self, intent_mode: str, n_candidates: int) -> str:
        mode_desc = self.MODE_DESCRIPTIONS.get(
            intent_mode, "Assist the user with their query."
        )

        personas = self.CANDIDATE_PERSONAS[:n_candidates]
        persona_list = "\n".join(
            f"  {i+1}. {p}" for i, p in enumerate(personas)
        )

        return (
            "You are an AI assistant collaborating with a human through a "
            "Brain-Computer Interface (BCI). The user's intent is decoded from "
            "EEG signals and may be imprecise. Your role is to generate "
            "candidate responses that the user will select from using their brain signals.\n\n"
            f"Current cognitive mode: {intent_mode}\n"
            f"Mode description: {mode_desc}\n\n"
            f"Generate exactly {n_candidates} candidate responses, each from a "
            f"different perspective:\n{persona_list}\n\n"
            "Rules:\n"
            "- Keep each response under 3 sentences (for rapid BCI selection)\n"
            "- Make responses genuinely different in approach, not just rephrasings\n"
            "- Be specific and actionable, not vague\n"
            "- Return as a JSON array of strings, e.g.: "
            '["response 1", "response 2", "response 3"]\n'
            "- Do not include any text outside the JSON array"
        )

    def generate_candidates(
        self,
        intent_mode: str,
        context: list[dict],
        n_candidates: int = 3,
        topic_hint: str = "",
    ) -> list[str]:
        """Generate candidate responses via Ollama."""
        import requests

        system_prompt = self._build_system_prompt(intent_mode, n_candidates)

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(context[-6:])  # Last 3 turns = 6 messages max

        user_content = "Generate candidate responses."
        if topic_hint:
            user_content += f" Topic hint: {topic_hint}"
        messages.append({"role": "user", "content": user_content})

        try:
            resp = requests.post(
                f"{self.host}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": False,
                    "options": {"temperature": self.temperature},
                    "format": "json",
                },
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            content = data.get("message", {}).get("content", "[]")

            # Parse JSON array
            try:
                candidates = json.loads(content)
                if isinstance(candidates, list) and candidates:
                    return [str(c) for c in candidates[:n_candidates]]
            except json.JSONDecodeError:
                # Fallback: try to extract JSON array from text
                import re
                match = re.search(r'\[.*\]', content, re.DOTALL)
                if match:
                    try:
                        candidates = json.loads(match.group())
                        if isinstance(candidates, list):
                            return [str(c) for c in candidates[:n_candidates]]
                    except json.JSONDecodeError:
                        pass
                # Last resort: return as single candidate
                logger.warning(
                    "Failed to parse JSON from LLM, returning raw content"
                )
                return [content]

            return [content]

        except requests.exceptions.ConnectionError:
            logger.error("Cannot connect to Ollama. Is it running?")
            return ["[Error: Ollama not connected. Run 'ollama serve' first.]"]
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return [f"[LLM Error: {e}]"]

    def expand_response(self, selected_response: str, intent_mode: str,
                        context: list[dict]) -> str:
        """Expand the selected candidate into a fuller response."""
        import requests

        system_prompt = (
            "You are an AI assistant collaborating with a human through a BCI. "
            "The user has selected a response direction. Now elaborate on it "
            "with more detail and depth. Keep it under 200 words."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Selected direction: {selected_response}"},
        ]
        messages.extend(context[-4:])

        try:
            resp = requests.post(
                f"{self.host}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": False,
                    "options": {"temperature": 0.6},
                },
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("message", {}).get("content", selected_response)
        except Exception as e:
            logger.error(f"Response expansion failed: {e}")
            return selected_response


class APIClient(LLMClient):
    """
    OpenAI-compatible API client (for Phase 2).

    Works with any OpenAI-compatible endpoint:
      - DeepSeek API
      - OpenAI API
      - Local vLLM server
      - Any OpenAI-compatible service
    """

    def __init__(
        self,
        api_url: str,
        api_key: str,
        model: str,
        temperature: float = 0.8,
        timeout: int = 30,
    ):
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.timeout = timeout

    def is_available(self) -> bool:
        return bool(self.api_key and self.api_url)

    def generate_candidates(
        self,
        intent_mode: str,
        context: list[dict],
        n_candidates: int = 3,
        topic_hint: str = "",
    ) -> list[str]:
        """Generate candidates via OpenAI-compatible API."""
        import requests

        # Reuse OllamaClient's prompt logic
        ollama_prompt = OllamaClient.MODE_DESCRIPTIONS.get(intent_mode, "")
        system_prompt = (
            f"You are a BCI-collaborative AI. Mode: {intent_mode}. {ollama_prompt}\n"
            f"Generate {n_candidates} distinct candidate responses "
            f"(conservative, bold, balanced). "
            f"Each under 3 sentences. Return as JSON array of strings."
        )

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(context[-6:])
        if topic_hint:
            messages.append({"role": "user", "content": f"Topic: {topic_hint}"})

        try:
            resp = requests.post(
                f"{self.api_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": messages,
                    "temperature": self.temperature,
                    "response_format": {"type": "json_object"},
                },
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]

            try:
                candidates = json.loads(content)
                if isinstance(candidates, list):
                    return [str(c) for c in candidates[:n_candidates]]
                if isinstance(candidates, dict) and "candidates" in candidates:
                    return [str(c) for c in candidates["candidates"][:n_candidates]]
            except json.JSONDecodeError:
                pass
            return [content]
        except Exception as e:
            logger.error(f"API LLM failed: {e}")
            return [f"[API Error: {e}]"]


class MockLLMClient(LLMClient):
    """
    Mock LLM client for testing without Ollama or API.

    Returns canned responses based on cognitive mode.
    Useful for development and demos where Ollama is not installed.
    """

    MOCK_RESPONSES = {
        "query": [
            "Based on current research, BCI systems typically achieve 70-85% accuracy in motor imagery classification using EEGNet architectures.",
            "The latest neural interfaces use high-density electrode arrays with 256+ channels, enabling more granular intent decoding than ever before.",
            "Studies show that combining BCI with LLMs can reduce user cognitive load by up to 40% compared to traditional input methods.",
        ],
        "reason": [
            "If we assume 250Hz sampling rate and 2s windows, that's 500 samples per channel. With 8 channels, the EEGNet input tensor is 8×500 — well within real-time processing budget.",
            "The bottleneck isn't decoding speed (EEGNet inference < 10ms) but rather the LLM generation latency. We can pipeline: start LLM while still acquiring confirmation EEG.",
            "By quantizing the EEGNet model to INT8, we reduce inference latency by 3x with <1% accuracy drop, making edge deployment feasible on Raspberry Pi.",
        ],
        "create": [
            "What if we used a continuous BCI signal (not just discrete classes) to control a slider that adjusts LLM creativity temperature in real-time?",
            "Imagine a 'neural thermostat' — the BCI reads your cognitive load and automatically adjusts the LLM's response complexity to match your mental state.",
            "We could train a personalized EEG-to-intent model that learns each user's unique neural patterns, creating a truly individualized AI collaboration interface.",
        ],
        "review": [
            "So far we've discussed BCI decoding accuracy, LLM integration architecture, and the collaborative reasoning paradigm. The key insight is leveraging human intuition for selection.",
            "The conversation has covered: (1) hardware simulation via BrainFlow, (2) intent encoding from motor imagery, and (3) the candidate selection loop. Next step is real-time integration.",
            "Summary: We're building a system where BCI provides coarse intent, LLM generates detailed candidates, and human brain makes the final selection — combining AI knowledge with human intuition.",
        ],
    }

    def is_available(self) -> bool:
        return True

    def generate_candidates(
        self,
        intent_mode: str,
        context: list[dict],
        n_candidates: int = 3,
        topic_hint: str = "",
    ) -> list[str]:
        """Return mock candidates for testing."""
        time.sleep(0.5)  # Simulate LLM latency
        responses = self.MOCK_RESPONSES.get(intent_mode, self.MOCK_RESPONSES["query"])
        return responses[:n_candidates]

    def expand_response(self, selected_response: str, intent_mode: str,
                        context: list[dict]) -> str:
        return selected_response + "\n\n[Mock expansion: This would be elaborated by the LLM with more detail and context.]"


def create_llm_client(backend: str = "ollama", **kwargs) -> LLMClient:
    """Factory function to create LLM client.

    Args:
        backend: 'ollama', 'api', or 'mock'.
        **kwargs: Backend-specific arguments.

    Returns:
        LLMClient instance.
    """
    if backend == "ollama":
        return OllamaClient(**kwargs)
    elif backend == "api":
        return APIClient(**kwargs)
    elif backend == "mock":
        return MockLLMClient(**kwargs)
    else:
        raise ValueError(f"Unknown backend: {backend}. Use 'ollama', 'api', or 'mock'.")
