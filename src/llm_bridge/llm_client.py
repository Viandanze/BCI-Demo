"""
LLM Bridge - Abstract interface for LLM backends.

Supports:
  - Ollama (local, private, zero-cost) — default backend
  - OpenAI-compatible API (cloud, higher quality)
  - Coze agent (deployed Coze Coding agent relay, async task API)

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
import os
import time
import logging
import threading
from collections import OrderedDict

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
    """Ollama local LLM client (default backend)."""

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
    OpenAI-compatible API client.

    Works with any OpenAI-compatible endpoint:
      - DeepSeek API
      - OpenAI API
      - Local vLLM server
      - Any OpenAI-compatible service

    Configuration can come from constructor args or from the environment
    variables LLM_API_URL / LLM_API_KEY / LLM_API_MODEL (or a .env file,
    which the demo loads automatically).
    """

    def __init__(
        self,
        api_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.8,
        timeout: int = 30,
    ):
        """
        Args:
            api_url: OpenAI-compatible endpoint, e.g. https://api.deepseek.com/v1
                (env: LLM_API_URL).
            api_key: API key (env: LLM_API_KEY).
            model: Model name, e.g. deepseek-chat (env: LLM_API_MODEL).
            temperature: Sampling temperature.
            timeout: Request timeout in seconds.
        """
        self.api_url = (api_url or os.getenv("LLM_API_URL", "")).rstrip("/")
        self.api_key = api_key or os.getenv("LLM_API_KEY", "")
        self.model = model or os.getenv("LLM_API_MODEL", "")
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


class CozeClient(LLMClient):
    """
    Coze agent backend: calls a deployed Coze Coding agent (pure LLM relay).

    API flow (Coze agent service, asynchronous mode):
      1. POST {domain}/async_run  -> {"task_id": "...", "status": "pending"}
      2. GET  {domain}/task/{task_id}  (poll until a terminal status)
      3. Extract result.messages -> last message with type == "ai" -> content

    The deployed relay agent forwards the query verbatim to the underlying
    model, so the system prompt and the user request are composed into a
    single query text.

    Credentials are read from environment variables (never hardcode them):
      COZE_AGENT_DOMAIN  e.g. "https://xxxx.coze.site"
      COZE_PROJECT_ID    numeric project id, e.g. "7600000000000000000"
      COZE_API_TOKEN     personal access token ("pat_...") or project API token

    Note: the agent sandbox may be reclaimed after ~1 hour of inactivity;
    the first request after an idle period can take noticeably longer
    (cold start). Total wall time is bounded by `poll_timeout`.
    """

    def __init__(
        self,
        domain: Optional[str] = None,
        project_id=None,
        api_token: Optional[str] = None,
        session_id: str = "neurodecode",
        poll_interval: float = 1.0,
        poll_timeout: float = 120.0,
        request_timeout: float = 15.0,
    ):
        """
        Args:
            domain: Coze agent service domain (env: COZE_AGENT_DOMAIN).
            project_id: Numeric project id (env: COZE_PROJECT_ID).
            api_token: Bearer token (env: COZE_API_TOKEN).
            session_id: Server-side conversation id for multi-turn context.
            poll_interval: Seconds between task status polls.
            poll_timeout: Max seconds to wait for task completion.
            request_timeout: Per-HTTP-request timeout in seconds.
        """
        self.domain = (domain or os.getenv("COZE_AGENT_DOMAIN", "")).rstrip("/")
        raw_pid = str(project_id) if project_id is not None else os.getenv("COZE_PROJECT_ID", "")
        try:
            self.project_id = int(raw_pid) if str(raw_pid).strip() else None
        except (TypeError, ValueError):
            self.project_id = None
        self.api_token = api_token or os.getenv("COZE_API_TOKEN", "")
        self.session_id = session_id
        self.poll_interval = poll_interval
        self.poll_timeout = poll_timeout
        self.request_timeout = request_timeout

    def is_available(self) -> bool:
        """All three credentials must be configured (no network probe)."""
        return bool(self.domain and self.project_id and self.api_token)

    def _auth_headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json",
        }

    def _build_system_prompt(self, intent_mode: str, n_candidates: int) -> str:
        """Same prompt contract as OllamaClient (candidates as JSON array)."""
        mode_desc = OllamaClient.MODE_DESCRIPTIONS.get(
            intent_mode, "Assist the user with their query."
        )
        personas = OllamaClient.CANDIDATE_PERSONAS[:n_candidates]
        persona_list = "\n".join(
            f"  {i + 1}. {p}" for i, p in enumerate(personas)
        )
        return (
            "You are an AI assistant collaborating with a human through a "
            "Brain-Computer Interface (BCI). The user's intent is decoded from "
            "EEG signals and may be imprecise. Your role is to generate "
            "candidate responses that the user will select from using their "
            "brain signals.\n\n"
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

    def _submit_task(self, query_text: str) -> str:
        """Submit an async_run task and return its task_id."""
        import requests

        resp = requests.post(
            f"{self.domain}/async_run",
            headers=self._auth_headers(),
            json={
                "content": {
                    "query": {
                        "prompt": [
                            {"content": {"text": query_text}, "type": "text"}
                        ]
                    }
                },
                "type": "query",
                "session_id": self.session_id,
                "project_id": self.project_id,
            },
            timeout=self.request_timeout,
        )
        resp.raise_for_status()
        task_id = resp.json().get("task_id")
        if not task_id:
            raise RuntimeError("Coze async_run response did not contain a task_id")
        return task_id

    def _wait_for_result(self, task_id: str) -> str:
        """Poll task status until terminal, return the raw answer text."""
        import requests

        deadline = time.time() + self.poll_timeout
        while time.time() < deadline:
            resp = requests.get(
                f"{self.domain}/task/{task_id}",
                headers=self._auth_headers(),
                timeout=self.request_timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            status = data.get("status")
            if status == "completed":
                return self._extract_answer(data.get("result") or {})
            if status in ("failed", "timeout"):
                raise RuntimeError(
                    f"Coze task {task_id} ended with status={status}: "
                    f"{data.get('error')}"
                )
            time.sleep(self.poll_interval)
        raise TimeoutError(
            f"Coze task {task_id} did not complete within {self.poll_timeout:.0f}s"
        )

    @staticmethod
    def _extract_answer(result: dict) -> str:
        """Pull the final assistant text out of a completed task result."""
        messages = result.get("messages") or []
        for msg in reversed(messages):
            if msg.get("type") != "ai":
                continue
            content = msg.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):  # multimodal content blocks
                return "".join(
                    block.get("text", "")
                    for block in content
                    if isinstance(block, dict)
                )
        # Fallback: last message, whatever its type.
        if messages:
            content = messages[-1].get("content")
            if isinstance(content, str):
                return content
        return ""

    def _query(self, query_text: str) -> str:
        """Submit one async task and return its final answer text."""
        task_id = self._submit_task(query_text)
        return self._wait_for_result(task_id)

    def _parse_candidates(self, content: str, n_candidates: int) -> list:
        """Parse a JSON array of candidates, with resilient fallbacks."""
        try:
            candidates = json.loads(content)
            if isinstance(candidates, list) and candidates:
                return [str(c) for c in candidates[:n_candidates]]
        except json.JSONDecodeError:
            import re
            match = re.search(r"\[.*\]", content, re.DOTALL)
            if match:
                try:
                    candidates = json.loads(match.group())
                    if isinstance(candidates, list) and candidates:
                        return [str(c) for c in candidates[:n_candidates]]
                except json.JSONDecodeError:
                    pass
        return [content] if content else []

    def generate_candidates(
        self,
        intent_mode: str,
        context: list[dict],
        n_candidates: int = 3,
        topic_hint: str = "",
    ) -> list[str]:
        """Generate candidate responses via the Coze agent relay."""
        if not self.is_available():
            missing = [
                name
                for name, value in (
                    ("COZE_AGENT_DOMAIN", self.domain),
                    ("COZE_PROJECT_ID", self.project_id),
                    ("COZE_API_TOKEN", self.api_token),
                )
                if not value
            ]
            return [
                f"[Coze Error: missing configuration ({', '.join(missing)}). "
                "Set them via constructor or environment variables.]"
            ]

        query_text = self._build_system_prompt(intent_mode, n_candidates)
        context_tail = "\n".join(
            f"{m.get('role', 'user')}: {m.get('content', '')}"
            for m in context[-6:]
        )
        if context_tail:
            query_text += f"\n\nRecent conversation:\n{context_tail}"
        user_line = "Generate candidate responses."
        if topic_hint:
            user_line += f" Topic hint: {topic_hint}"
        query_text += f"\n\nUser request: {user_line}"

        try:
            content = self._query(query_text)
            candidates = self._parse_candidates(content, n_candidates)
            if candidates:
                return candidates
            return ["[Coze Error: agent returned an empty response.]"]
        except Exception as e:
            logger.error(f"Coze generation failed: {e}")
            return [f"[Coze Error: {e}]"]

    def expand_response(self, selected_response: str, intent_mode: str,
                        context: list[dict]) -> str:
        """Expand the selected candidate into a fuller response."""
        if not self.is_available():
            return selected_response

        query_text = (
            "You are an AI assistant collaborating with a human through a BCI. "
            "The user has selected a response direction. Elaborate on it with "
            "more detail and depth. Keep it under 200 words. "
            "Do not add any preamble; reply with the elaboration only.\n\n"
            f"Selected direction: {selected_response}"
        )
        try:
            expanded = self._query(query_text)
            return expanded if expanded else selected_response
        except Exception as e:
            logger.error(f"Coze expansion failed: {e}")
            return selected_response


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


class CachedLLMClient(LLMClient):
    """Caching decorator around any LLMClient.

    Repeated identical requests (same intent mode, context, candidate
    count and topic hint) within the TTL return a cached copy without
    hitting the backend. Handy for demos where the same intent fires
    again and again.

    Error placeholder responses (e.g. "[API Error: ...]") are never
    cached. Thread-safe: guarded by a lock, safe to share with the
    AsyncLLMBridge worker thread.
    """

    def __init__(self, inner: LLMClient, maxsize: int = 64, ttl: float = 300.0,
                 clock=None):
        """
        Args:
            inner: The wrapped LLMClient.
            maxsize: Maximum number of cached entries (LRU eviction).
            ttl: Seconds until a cached entry expires.
            clock: Injectable time function for tests.
        """
        self.inner = inner
        self._maxsize = max(1, int(maxsize))
        self._ttl = float(ttl)
        self._clock = clock or time.time
        self._lock = threading.Lock()
        self._cache: "OrderedDict[tuple, tuple[float, object]]" = OrderedDict()
        self._hits = 0
        self._misses = 0

    @staticmethod
    def _freeze(messages: list[dict]) -> tuple:
        """Make a hashable fingerprint of a message list."""
        return tuple(
            (m.get("role", ""), m.get("content", "")) for m in messages
        )

    @staticmethod
    def _cacheable(candidates: list[str]) -> bool:
        """Error placeholder responses must not poison the cache."""
        if not candidates:
            return False
        bad = ("[API Error", "[LLM Error", "[Error:", "LLM offline")
        return not any(any(marker in c for marker in bad) for c in candidates)

    def _lookup(self, key: tuple):
        now = self._clock()
        with self._lock:
            item = self._cache.get(key)
            if item is not None:
                expires_at, value = item
                if now < expires_at:
                    self._hits += 1
                    return list(value) if isinstance(value, list) else value
                del self._cache[key]
            self._misses += 1
            return None

    def _store(self, key: tuple, value) -> None:
        with self._lock:
            self._cache[key] = (self._clock() + self._ttl, value)
            while len(self._cache) > self._maxsize:
                self._cache.popitem(last=False)

    def generate_candidates(
        self,
        intent_mode: str,
        context: list[dict],
        n_candidates: int = 3,
        topic_hint: str = "",
    ) -> list[str]:
        """Cached variant of inner.generate_candidates()."""
        key = ("gen", intent_mode, n_candidates, topic_hint,
               self._freeze(context))
        cached = self._lookup(key)
        if cached is not None:
            logger.debug("LLM cache hit for mode=%s", intent_mode)
            return cached
        result = self.inner.generate_candidates(
            intent_mode, context, n_candidates, topic_hint
        )
        if self._cacheable(result):
            self._store(key, list(result))
        return result

    def is_available(self) -> bool:
        return self.inner.is_available()

    def expand_response(self, selected_response: str, intent_mode: str,
                        context: list[dict]) -> str:
        """Cached variant of inner.expand_response()."""
        key = ("exp", selected_response, intent_mode, self._freeze(context))
        cached = self._lookup(key)
        if cached is not None:
            return cached
        result = self.inner.expand_response(selected_response, intent_mode, context)
        if result and "[API Error" not in result and "[LLM Error" not in result:
            self._store(key, result)
        return result

    def cache_stats(self) -> dict:
        """Return cache counters (hits/misses/size) for logging."""
        with self._lock:
            return {
                "hits": self._hits,
                "misses": self._misses,
                "size": len(self._cache),
                "maxsize": self._maxsize,
                "ttl": self._ttl,
            }

    def clear_cache(self) -> None:
        """Drop all cached entries."""
        with self._lock:
            self._cache.clear()


def create_llm_client(backend: str = "ollama", cache: bool = False,
                      **kwargs) -> LLMClient:
    """Factory function to create LLM client.

    Args:
        backend: 'ollama', 'api', 'coze' or 'mock'.
        cache: Wrap the client in CachedLLMClient (skipped for 'mock').
        **kwargs: Backend-specific arguments.

    Returns:
        LLMClient instance.
    """
    if backend == "ollama":
        client = OllamaClient(**kwargs)
    elif backend == "api":
        client = APIClient(**kwargs)
    elif backend == "coze":
        client = CozeClient(**kwargs)
    elif backend == "mock":
        client = MockLLMClient(**kwargs)
    else:
        raise ValueError(
            f"Unknown backend: {backend}. Use 'ollama', 'api', 'coze', or 'mock'."
        )
    if cache and backend != "mock":
        return CachedLLMClient(client)
    return client

