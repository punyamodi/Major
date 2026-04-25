"""Unit tests for LLM Provider abstraction layer."""

import asyncio
import json
import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from medaide_plus.utils.llm_provider import (
    BaseLLMProvider,
    LLMResponse,
    ProviderType,
    OpenAIProvider,
    OllamaProvider,
    AnthropicProvider,
    AzureOpenAIProvider,
    GoogleGeminiProvider,
    HuggingFaceProvider,
    OpenAICompatibleProvider,
    create_provider,
    create_provider_from_yaml,
)


# ---------------------------------------------------------------------------
# LLMResponse & ProviderType
# ---------------------------------------------------------------------------


class TestLLMResponse:
    """Tests for the LLMResponse dataclass."""

    def test_basic_fields(self):
        resp = LLMResponse(text="hello", model="gpt-4o", provider="openai")
        assert resp.text == "hello"
        assert resp.model == "gpt-4o"
        assert resp.provider == "openai"
        assert resp.usage == {}
        assert resp.latency_ms == 0.0
        assert resp.metadata == {}

    def test_with_usage(self):
        resp = LLMResponse(
            text="answer",
            model="llama3",
            provider="ollama",
            usage={"prompt_tokens": 10, "completion_tokens": 20},
            latency_ms=150.5,
        )
        assert resp.usage["prompt_tokens"] == 10
        assert resp.usage["completion_tokens"] == 20
        assert resp.latency_ms == 150.5

    def test_metadata(self):
        resp = LLMResponse(
            text="test",
            model="m",
            provider="p",
            metadata={"finish_reason": "stop", "extra": 42},
        )
        assert resp.metadata["finish_reason"] == "stop"
        assert resp.metadata["extra"] == 42


class TestProviderType:
    """Tests for the ProviderType enum."""

    def test_all_providers_present(self):
        expected = {
            "openai", "ollama", "anthropic", "azure_openai",
            "google_gemini", "huggingface", "openai_compatible",
        }
        actual = {p.value for p in ProviderType}
        assert actual == expected

    def test_enum_lookup(self):
        assert ProviderType("ollama") == ProviderType.OLLAMA
        assert ProviderType("openai") == ProviderType.OPENAI


# ---------------------------------------------------------------------------
# env resolver
# ---------------------------------------------------------------------------


class TestEnvResolver:
    """Test the _resolve_env helper on BaseLLMProvider subclasses."""

    def test_resolves_env_var(self):
        os.environ["_TEST_MEDAIDE_KEY"] = "secret123"
        provider = OllamaProvider()
        assert provider._resolve_env("${_TEST_MEDAIDE_KEY}") == "secret123"
        del os.environ["_TEST_MEDAIDE_KEY"]

    def test_missing_env_var_returns_none(self):
        provider = OllamaProvider()
        assert provider._resolve_env("${_NONEXISTENT_KEY_12345}") is None

    def test_plain_string_returned_as_is(self):
        provider = OllamaProvider()
        assert provider._resolve_env("plain-key") == "plain-key"

    def test_none_input(self):
        provider = OllamaProvider()
        assert provider._resolve_env(None) is None


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------


class TestCreateProvider:
    """Tests for the create_provider factory function."""

    def test_creates_openai(self):
        p = create_provider({"provider": "openai", "api_key": "sk-test", "model": "gpt-4o"})
        assert isinstance(p, OpenAIProvider)
        assert p.model == "gpt-4o"

    def test_creates_ollama(self):
        p = create_provider({"provider": "ollama", "model": "llama3"})
        assert isinstance(p, OllamaProvider)
        assert p.model == "llama3"
        assert p.base_url == "http://localhost:11434"

    def test_creates_anthropic(self):
        p = create_provider({"provider": "anthropic", "api_key": "test-key", "model": "claude-3-sonnet"})
        assert isinstance(p, AnthropicProvider)

    def test_creates_azure(self):
        p = create_provider({
            "provider": "azure_openai",
            "api_key": "az-key",
            "endpoint": "https://example.openai.azure.com",
            "deployment": "my-gpt4",
        })
        assert isinstance(p, AzureOpenAIProvider)

    def test_creates_gemini(self):
        p = create_provider({"provider": "google_gemini", "api_key": "gm-key"})
        assert isinstance(p, GoogleGeminiProvider)

    def test_creates_huggingface(self):
        p = create_provider({"provider": "huggingface", "api_key": "hf-key"})
        assert isinstance(p, HuggingFaceProvider)

    def test_creates_openai_compatible(self):
        p = create_provider({
            "provider": "openai_compatible",
            "api_key": "oc-key",
            "base_url": "http://localhost:8080/v1",
            "model": "local-model",
        })
        assert isinstance(p, OpenAICompatibleProvider)

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            create_provider({"provider": "doesnt_exist"})

    def test_default_provider_is_openai(self):
        p = create_provider({"api_key": "test"})
        assert isinstance(p, OpenAIProvider)


class TestCreateProviderFromYaml:
    """Tests for the create_provider_from_yaml factory function."""

    def test_ollama_from_yaml(self):
        config = {
            "llm": {
                "provider": "ollama",
                "ollama": {
                    "base_url": "http://myhost:11434",
                    "model": "mistral",
                },
            }
        }
        p = create_provider_from_yaml(config)
        assert isinstance(p, OllamaProvider)
        assert p.model == "mistral"
        assert p.base_url == "http://myhost:11434"

    def test_openai_from_yaml(self):
        config = {
            "llm": {
                "provider": "openai",
                "openai": {"api_key": "sk-test", "model": "gpt-4o"},
            }
        }
        p = create_provider_from_yaml(config)
        assert isinstance(p, OpenAIProvider)

    def test_missing_llm_section_defaults_openai(self):
        p = create_provider_from_yaml({})
        assert isinstance(p, OpenAIProvider)

    def test_anthropic_from_yaml(self):
        config = {
            "llm": {
                "provider": "anthropic",
                "anthropic": {"api_key": "ant-key", "model": "claude-3-opus"},
            }
        }
        p = create_provider_from_yaml(config)
        assert isinstance(p, AnthropicProvider)


# ---------------------------------------------------------------------------
# Ollama Provider (mocked HTTP)
# ---------------------------------------------------------------------------


class TestOllamaProvider:
    """Test OllamaProvider with mocked httpx calls."""

    @pytest.mark.asyncio
    async def test_chat_success(self):
        provider = OllamaProvider(model="llama3")
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "message": {"content": "This is a test response."},
            "prompt_eval_count": 15,
            "eval_count": 25,
            "total_duration": 1000000,
            "load_duration": 500000,
        }

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        provider._client = mock_client

        resp = await provider.chat([{"role": "user", "content": "Hello"}])
        assert resp.text == "This is a test response."
        assert resp.provider == "ollama"
        assert resp.model == "llama3"
        assert resp.usage["prompt_tokens"] == 15
        assert resp.usage["completion_tokens"] == 25
        assert resp.latency_ms > 0

    @pytest.mark.asyncio
    async def test_chat_failure_returns_empty(self):
        provider = OllamaProvider(model="llama3")
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=Exception("Connection refused"))
        provider._client = mock_client

        resp = await provider.chat([{"role": "user", "content": "test"}])
        assert resp.text == ""
        assert resp.provider == "ollama"

    @pytest.mark.asyncio
    async def test_health_check_success(self):
        provider = OllamaProvider()
        mock_resp = MagicMock()
        mock_resp.status_code = 200

        with patch("httpx.AsyncClient") as MockClient:
            mock_instance = AsyncMock()
            mock_instance.get = AsyncMock(return_value=mock_resp)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = mock_instance

            result = await provider.health_check()
            assert result is True

    @pytest.mark.asyncio
    async def test_health_check_failure(self):
        provider = OllamaProvider()
        with patch("httpx.AsyncClient") as MockClient:
            mock_instance = AsyncMock()
            mock_instance.get = AsyncMock(side_effect=Exception("Connection refused"))
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = mock_instance

            result = await provider.health_check()
            assert result is False

    def test_default_base_url(self):
        p = OllamaProvider()
        assert p.base_url == "http://localhost:11434"

    def test_custom_base_url_strips_trailing_slash(self):
        p = OllamaProvider(base_url="http://custom:11434/")
        assert p.base_url == "http://custom:11434"


# ---------------------------------------------------------------------------
# OpenAI Provider (mocked)
# ---------------------------------------------------------------------------


class TestOpenAIProvider:
    """Test OpenAIProvider with mocked openai client."""

    def test_init_with_explicit_key(self):
        p = OpenAIProvider(api_key="sk-test123", model="gpt-4o-mini")
        assert p.api_key == "sk-test123"
        assert p.model == "gpt-4o-mini"

    def test_init_resolves_env_key(self):
        os.environ["_TEST_OAI_KEY"] = "sk-from-env"
        p = OpenAIProvider(api_key="${_TEST_OAI_KEY}")
        assert p.api_key == "sk-from-env"
        del os.environ["_TEST_OAI_KEY"]

    @pytest.mark.asyncio
    async def test_chat_failure_returns_empty(self):
        """If the OpenAI client raises, we get an empty LLMResponse."""
        p = OpenAIProvider(api_key="sk-test")

        mock_completions = AsyncMock()
        mock_completions.create = AsyncMock(side_effect=Exception("API error"))
        mock_client = MagicMock()
        mock_client.chat = MagicMock()
        mock_client.chat.completions = mock_completions
        p._client = mock_client

        resp = await p.chat([{"role": "user", "content": "test"}])
        assert resp.text == ""
        assert resp.provider == "openai"


# ---------------------------------------------------------------------------
# Pipeline integration with provider
# ---------------------------------------------------------------------------


class TestPipelineProviderIntegration:
    """Test that the pipeline correctly accepts and uses LLM provider config."""

    def test_create_provider_from_yaml_for_pipeline(self):
        """Verify create_provider_from_yaml produces a valid provider for pipeline use."""
        config = {
            "llm": {
                "provider": "ollama",
                "ollama": {"base_url": "http://localhost:11434", "model": "llama3"},
            },
        }
        p = create_provider_from_yaml(config)
        assert isinstance(p, OllamaProvider)
        assert p.model == "llama3"

    def test_pipeline_config_with_all_providers(self):
        """All 7 provider types can be created from YAML config structure."""
        provider_configs = [
            {"llm": {"provider": "openai", "openai": {"api_key": "sk-test"}}},
            {"llm": {"provider": "ollama", "ollama": {"model": "llama3"}}},
            {"llm": {"provider": "anthropic", "anthropic": {"api_key": "ant-test"}}},
            {"llm": {"provider": "azure_openai", "azure_openai": {"api_key": "az", "endpoint": "https://x.openai.azure.com", "deployment": "gpt4"}}},
            {"llm": {"provider": "google_gemini", "google_gemini": {"api_key": "gm"}}},
            {"llm": {"provider": "huggingface", "huggingface": {"api_key": "hf"}}},
            {"llm": {"provider": "openai_compatible", "openai_compatible": {"base_url": "http://local:8080/v1"}}},
        ]
        for cfg in provider_configs:
            p = create_provider_from_yaml(cfg)
            assert isinstance(p, BaseLLMProvider), f"Failed for {cfg['llm']['provider']}"


# ---------------------------------------------------------------------------
# Multi-provider config switching
# ---------------------------------------------------------------------------


class TestProviderSwitching:
    """Test that different providers can be instantiated from same config."""

    def test_switch_providers(self):
        """Each provider type should instantiate cleanly."""
        configs = [
            {"provider": "openai", "api_key": "sk-test"},
            {"provider": "ollama", "model": "llama3"},
            {"provider": "anthropic", "api_key": "ant-test"},
            {"provider": "huggingface", "api_key": "hf-test"},
            {"provider": "openai_compatible", "base_url": "http://local:8080/v1"},
        ]
        providers = [create_provider(c) for c in configs]
        types_created = {type(p).__name__ for p in providers}
        assert "OpenAIProvider" in types_created
        assert "OllamaProvider" in types_created
        assert "AnthropicProvider" in types_created
        assert "HuggingFaceProvider" in types_created
        assert "OpenAICompatibleProvider" in types_created
