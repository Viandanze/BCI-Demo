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

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm_bridge.llm_client import (
    LLMClient,
    MockLLMClient,
    OllamaClient,
    APIClient,
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
