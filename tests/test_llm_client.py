"""
Tests for LLM Bridge module.

Covers:
  - MockLLMClient: is_available, generate_candidates, expand_response
  - OllamaClient: is_available when Ollama not running
  - APIClient: initialization and availability
  - create_llm_client factory: all backends + error handling
  - LLMClient abstract interface compliance
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm_bridge.llm_client import (
    LLMClient,
    MockLLMClient,
    OllamaClient,
    APIClient,
    CozeClient,
    CachedLLMClient,
    create_llm_client,
)


class TestMockLLMClient:
    """Test MockLLMClient (no external dependencies needed)."""

    @pytest.fixture
    def client(self):
        return MockLLMClient()

    def test_is_available(self, client):
        assert client.is_available() is True

    def test_generate_candidates_query(self, client):
        candidates = client.generate_candidates("query", [], n_candidates=3)
        assert len(candidates) == 3
        assert all(isinstance(c, str) for c in candidates)
        assert all(len(c) > 0 for c in candidates)

    def test_generate_candidates_reason(self, client):
        candidates = client.generate_candidates("reason", [], n_candidates=2)
        assert len(candidates) == 2

    def test_generate_candidates_create(self, client):
        candidates = client.generate_candidates("create", [], n_candidates=3)
        assert len(candidates) == 3

    def test_generate_candidates_review(self, client):
        candidates = client.generate_candidates("review", [], n_candidates=3)
        assert len(candidates) == 3

    def test_generate_candidates_unknown_mode(self, client):
        """Unknown mode should fall back to query responses."""
        candidates = client.generate_candidates("unknown_mode", [], n_candidates=3)
        assert len(candidates) == 3

    def test_generate_candidates_respects_n(self, client):
        for n in [1, 2, 3]:
            candidates = client.generate_candidates("query", [], n_candidates=n)
            assert len(candidates) == n

    def test_expand_response(self, client):
        original = "Test response"
        expanded = client.expand_response(original, "query", [])
        assert original in expanded
        assert len(expanded) > len(original)

    def test_implements_llm_client_interface(self, client):
        assert isinstance(client, LLMClient)


class TestOllamaClient:
    """Test OllamaClient (without actual Ollama running)."""

    def test_is_available_false_when_not_running(self):
        client = OllamaClient(host="http://localhost:99999")
        # Ollama not running on port 99999
        assert client.is_available() is False

    def test_implements_llm_client_interface(self):
        client = OllamaClient()
        assert isinstance(client, LLMClient)


class TestAPIClient:
    """Test APIClient initialization."""

    def test_init_with_valid_params(self):
        client = APIClient(
            api_url="https://api.deepseek.com/v1",
            api_key="test-key",
            model="deepseek-chat",
        )
        assert isinstance(client, LLMClient)

    def test_is_available_with_key_and_url(self):
        """is_available returns True when api_key and api_url are set."""
        client = APIClient(
            api_url="https://api.deepseek.com/v1",
            api_key="test-key",
            model="deepseek-chat",
        )
        assert client.is_available() is True

    def test_env_variable_fallback(self, monkeypatch):
        """Config falls back to LLM_API_URL / LLM_API_KEY / LLM_API_MODEL."""
        monkeypatch.setenv("LLM_API_URL", "https://api.example.com/v1/")
        monkeypatch.setenv("LLM_API_KEY", "sk-test")
        monkeypatch.setenv("LLM_API_MODEL", "test-model")
        client = APIClient()
        assert client.api_url == "https://api.example.com/v1"
        assert client.api_key == "sk-test"
        assert client.model == "test-model"
        assert client.is_available() is True

    def test_explicit_args_override_env(self, monkeypatch):
        """Constructor arguments take precedence over environment variables."""
        monkeypatch.setenv("LLM_API_URL", "https://env.example.com/v1")
        monkeypatch.setenv("LLM_API_KEY", "sk-env")
        client = APIClient(
            api_url="https://arg.example.com/v1",
            api_key="sk-arg",
            model="arg-model",
        )
        assert client.api_url == "https://arg.example.com/v1"
        assert client.api_key == "sk-arg"
        assert client.model == "arg-model"


class TestCreateLLMClient:
    """Test factory function."""

    def test_create_mock(self):
        client = create_llm_client("mock")
        assert isinstance(client, MockLLMClient)
        assert isinstance(client, LLMClient)

    def test_create_ollama(self):
        client = create_llm_client("ollama")
        assert isinstance(client, OllamaClient)

    def test_create_api(self):
        client = create_llm_client(
            "api",
            api_url="https://api.deepseek.com/v1",
            api_key="test-key",
            model="deepseek-chat",
        )
        assert isinstance(client, APIClient)

    def test_create_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            create_llm_client("unknown_backend")

    def test_default_backend_is_ollama(self):
        """Default backend should be ollama."""
        client = create_llm_client()
        assert isinstance(client, OllamaClient)


class TestLLMClientAbstract:
    """Test LLMClient abstract interface."""

    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            LLMClient()

    def test_all_concrete_clients_have_required_methods(self):
        """All LLM client implementations must have required methods."""
        clients = [
            MockLLMClient(),
            OllamaClient(),
        ]

        for client in clients:
            assert hasattr(client, "generate_candidates")
            assert hasattr(client, "is_available")
            assert hasattr(client, "expand_response")
            assert callable(client.generate_candidates)
            assert callable(client.is_available)
            assert callable(client.expand_response)


class TestCozeClient:
    """Test CozeClient (all HTTP traffic mocked, no real API calls)."""

    @pytest.fixture(autouse=True)
    def _clean_env(self, monkeypatch):
        """Keep tests independent from the developer's COZE_* env vars."""
        for key in ("COZE_AGENT_DOMAIN", "COZE_PROJECT_ID", "COZE_API_TOKEN"):
            monkeypatch.delenv(key, raising=False)

    def _make_client(self, **overrides):
        params = dict(
            domain="https://fake.coze.site",
            project_id=123,
            api_token="pat_test",
            poll_interval=0.01,
            poll_timeout=1.0,
            request_timeout=5.0,
        )
        params.update(overrides)
        return CozeClient(**params)

    @staticmethod
    def _completed(messages):
        return {"task_id": "t", "status": "completed", "result": {"messages": messages}}

    def test_is_available_with_credentials(self):
        assert self._make_client().is_available()

    def test_is_available_false_without_credentials(self):
        client = CozeClient(domain="", project_id="", api_token="")
        assert not client.is_available()

    def test_env_variable_fallback(self, monkeypatch):
        monkeypatch.setenv("COZE_AGENT_DOMAIN", "https://env.coze.site/")
        monkeypatch.setenv("COZE_PROJECT_ID", "42")
        monkeypatch.setenv("COZE_API_TOKEN", "pat_env")
        client = CozeClient()
        assert client.domain == "https://env.coze.site"
        assert client.project_id == 42
        assert client.api_token == "pat_env"
        assert client.is_available()

    def test_invalid_project_id_becomes_none(self):
        client = CozeClient(domain="https://x.coze.site", project_id="not-a-number",
                            api_token="pat")
        assert client.project_id is None
        assert not client.is_available()

    def test_extract_answer_prefers_last_ai_message(self):
        result = {
            "messages": [
                {"type": "human", "content": [{"text": "q", "type": "text"}]},
                {"type": "ai", "content": "first answer"},
                {"type": "ai", "content": "final answer"},
            ]
        }
        assert CozeClient._extract_answer(result) == "final answer"

    def test_extract_answer_multimodal_blocks(self):
        result = {"messages": [{"type": "ai", "content": [{"text": "a"}, {"text": "b"}]}]}
        assert CozeClient._extract_answer(result) == "ab"

    def test_extract_answer_empty(self):
        assert CozeClient._extract_answer({}) == ""

    @patch("requests.post")
    @patch("requests.get")
    def test_generate_candidates_success(self, mock_get, mock_post):
        client = self._make_client()
        mock_post.return_value.json.return_value = {"task_id": "t1", "status": "pending"}
        mock_get.return_value.json.return_value = self._completed([
            {"type": "human", "content": [{"text": "input", "type": "text"}]},
            {"type": "ai", "content": '["first", "second", "third"]'},
        ])

        result = client.generate_candidates("query", [], n_candidates=3)

        assert result == ["first", "second", "third"]
        # Verify the async_run request body contract.
        _, kwargs = mock_post.call_args
        body = kwargs["json"]
        assert body["type"] == "query"
        assert body["project_id"] == 123
        assert body["session_id"] == "neurodecode"
        query_text = body["content"]["query"]["prompt"][0]["content"]["text"]
        assert "candidate responses" in query_text
        assert kwargs["headers"]["Authorization"] == "Bearer pat_test"

    @patch("requests.post")
    @patch("requests.get")
    def test_generate_candidates_non_json_reply(self, mock_get, mock_post):
        client = self._make_client()
        mock_post.return_value.json.return_value = {"task_id": "t2"}
        mock_get.return_value.json.return_value = self._completed([
            {"type": "ai", "content": "plain text answer"},
        ])

        result = client.generate_candidates("query", [])

        assert result == ["plain text answer"]

    @patch("requests.post")
    def test_generate_candidates_missing_config_returns_error(self, mock_post):
        client = CozeClient(domain="", project_id="", api_token="")

        result = client.generate_candidates("query", [])

        assert len(result) == 1
        assert "Coze Error" in result[0]
        assert "missing configuration" in result[0]
        mock_post.assert_not_called()

    @patch("requests.post")
    @patch("requests.get")
    def test_generate_candidates_task_failed(self, mock_get, mock_post):
        client = self._make_client()
        mock_post.return_value.json.return_value = {"task_id": "t3"}
        mock_get.return_value.json.return_value = {
            "task_id": "t3", "status": "failed", "error": "boom",
        }

        result = client.generate_candidates("query", [])

        assert len(result) == 1
        assert "Coze Error" in result[0]
        assert "failed" in result[0]

    @patch("requests.post")
    @patch("requests.get")
    def test_generate_candidates_poll_timeout(self, mock_post, mock_get):
        client = self._make_client(poll_timeout=0.05)
        mock_post.return_value.json.return_value = {"task_id": "t4"}
        mock_get.return_value.json.return_value = {
            "task_id": "t4", "status": "running",
        }

        result = client.generate_candidates("query", [])

        assert len(result) == 1
        assert "Coze Error" in result[0]

    @patch("requests.post")
    @patch("requests.get")
    def test_generate_candidates_context_and_topic_hint(self, mock_get, mock_post):
        client = self._make_client()
        mock_post.return_value.json.return_value = {"task_id": "t5"}
        mock_get.return_value.json.return_value = self._completed([
            {"type": "ai", "content": '["only one"]'},
        ])

        client.generate_candidates(
            "reason",
            [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}],
            topic_hint="EEG decoding",
        )

        _, kwargs = mock_post.call_args
        query_text = kwargs["json"]["content"]["query"]["prompt"][0]["content"]["text"]
        assert "hello" in query_text          # context is folded into the query
        assert "EEG decoding" in query_text   # topic hint is folded in too

    @patch("requests.post")
    @patch("requests.get")
    def test_expand_response_success(self, mock_get, mock_post):
        client = self._make_client()
        mock_post.return_value.json.return_value = {"task_id": "t6"}
        mock_get.return_value.json.return_value = self._completed([
            {"type": "ai", "content": "elaborated answer"},
        ])

        assert client.expand_response("direction", "query", []) == "elaborated answer"

    @patch("requests.post")
    def test_expand_response_unavailable_returns_original(self, mock_post):
        client = CozeClient(domain="", project_id="", api_token="")

        assert client.expand_response("direction", "query", []) == "direction"
        mock_post.assert_not_called()

    def test_implements_llm_client_interface(self):
        client = self._make_client()
        assert isinstance(client, LLMClient)
        assert callable(client.generate_candidates)
        assert callable(client.is_available)
        assert callable(client.expand_response)


class TestCreateLLMClientCoze:
    """Factory tests for the coze backend."""

    @pytest.fixture(autouse=True)
    def _clean_env(self, monkeypatch):
        for key in ("COZE_AGENT_DOMAIN", "COZE_PROJECT_ID", "COZE_API_TOKEN"):
            monkeypatch.delenv(key, raising=False)

    def test_create_coze(self):
        client = create_llm_client(
            "coze",
            domain="https://x.coze.site",
            project_id="7",
            api_token="pat_x",
        )
        assert isinstance(client, CozeClient)
        assert client.project_id == 7
        assert client.is_available()

    def test_create_coze_empty_needs_env(self):
        client = create_llm_client("coze")
        assert isinstance(client, CozeClient)
        assert not client.is_available()


class CountingClient(LLMClient):
    """Deterministic inner client that counts backend calls."""

    def __init__(self, candidates=None):
        self.calls = 0
        self._candidates = candidates or ["alpha", "beta", "gamma"]

    def generate_candidates(self, intent_mode, context, n_candidates=3,
                            topic_hint=""):
        self.calls += 1
        return list(self._candidates)

    def is_available(self):
        return True

    def expand_response(self, selected_response, intent_mode, context):
        self.calls += 1
        return f"expanded::{selected_response}"


class TestCachedLLMClient:
    """Response cache."""

    def test_identical_request_hits_cache(self):
        inner = CountingClient()
        cached = CachedLLMClient(inner)
        ctx = [{"role": "user", "content": "hi"}]
        r1 = cached.generate_candidates("query", ctx)
        r2 = cached.generate_candidates("query", ctx)
        assert inner.calls == 1
        assert r1 == r2

    def test_different_args_miss_cache(self):
        inner = CountingClient()
        cached = CachedLLMClient(inner)
        cached.generate_candidates("query", [])
        cached.generate_candidates("reason", [])       # different mode
        cached.generate_candidates("query", [], n_candidates=5)  # different n
        assert inner.calls == 3

    def test_returned_list_is_a_copy(self):
        inner = CountingClient()
        cached = CachedLLMClient(inner)
        r1 = cached.generate_candidates("query", [])
        r1.append("polluted")
        r2 = cached.generate_candidates("query", [])
        assert r2 == ["alpha", "beta", "gamma"]

    def test_ttl_expiry_forces_backend_call(self):
        fake_now = [1000.0]
        inner = CountingClient()
        cached = CachedLLMClient(inner, ttl=10.0, clock=lambda: fake_now[0])
        cached.generate_candidates("query", [])
        fake_now[0] += 11.0  # past the TTL
        cached.generate_candidates("query", [])
        assert inner.calls == 2

    def test_error_placeholders_not_cached(self):
        inner = CountingClient(candidates=["[API Error: boom]"])
        cached = CachedLLMClient(inner)
        cached.generate_candidates("query", [])
        cached.generate_candidates("query", [])
        assert inner.calls == 2

    def test_ollama_offline_placeholder_not_cached(self):
        inner = CountingClient(
            candidates=["[Error: Ollama not connected. Run 'ollama serve' first.]"]
        )
        cached = CachedLLMClient(inner)
        cached.generate_candidates("query", [])
        cached.generate_candidates("query", [])
        assert inner.calls == 2

    def test_expand_response_cached(self):
        inner = CountingClient()
        cached = CachedLLMClient(inner)
        ctx = [{"role": "user", "content": "ctx"}]
        e1 = cached.expand_response("pick me", "query", ctx)
        e2 = cached.expand_response("pick me", "query", ctx)
        assert e1 == "expanded::pick me"
        assert inner.calls == 1
        assert e1 == e2

    def test_is_available_delegates(self):
        cached = CachedLLMClient(CountingClient())
        assert cached.is_available() is True

    def test_cache_stats_and_clear(self):
        inner = CountingClient()
        cached = CachedLLMClient(inner)
        cached.generate_candidates("query", [])   # miss
        cached.generate_candidates("query", [])   # hit
        stats = cached.cache_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["size"] == 1
        cached.clear_cache()
        assert cached.cache_stats()["size"] == 0

    def test_lru_eviction_respects_maxsize(self):
        inner = CountingClient()
        cached = CachedLLMClient(inner, maxsize=2)
        for mode in ("a", "b", "c"):
            cached.generate_candidates(mode, [])
        assert cached.cache_stats()["size"] == 2
        cached.generate_candidates("a", [])  # evicted -> backend again
        assert inner.calls == 4


class TestFactoryCacheFlag:
    """create_llm_client(cache=...) wiring."""

    def test_mock_is_never_wrapped(self):
        client = create_llm_client("mock", cache=True)
        assert isinstance(client, MockLLMClient)

    def test_api_wrapped_when_cache_true(self):
        client = create_llm_client(
            "api", cache=True,
            api_url="https://api.example.com/v1", api_key="k",
        )
        assert isinstance(client, CachedLLMClient)
        assert isinstance(client.inner, APIClient)

    def test_api_not_wrapped_by_default(self):
        client = create_llm_client(
            "api",
            api_url="https://api.example.com/v1", api_key="k",
        )
        assert isinstance(client, APIClient)

