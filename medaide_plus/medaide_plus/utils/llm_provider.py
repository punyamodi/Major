"""
LLM Provider — Unified abstraction for multiple LLM backends.

Supports: OpenAI, Ollama, Anthropic, Azure OpenAI, Google Gemini,
HuggingFace Inference, and any OpenAI-compatible endpoint.

Usage:
    from medaide_plus.utils.llm_provider import create_provider, create_provider_from_yaml

    # From explicit config dict
    provider = create_provider({"provider": "ollama", "model": "llama3"})
    response = await provider.chat([{"role": "user", "content": "Hello"}])

    # From full config.yaml structure
    provider = create_provider_from_yaml(yaml_config)
"""

import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger("medaide_plus.llm_provider")


# ---------------------------------------------------------------------------
# Enums & Data Classes
# ---------------------------------------------------------------------------

class ProviderType(Enum):
    """Supported LLM provider backends."""

    OPENAI = "openai"
    OLLAMA = "ollama"
    ANTHROPIC = "anthropic"
    AZURE_OPENAI = "azure_openai"
    GOOGLE_GEMINI = "google_gemini"
    HUGGINGFACE = "huggingface"
    OPENAI_COMPATIBLE = "openai_compatible"


@dataclass
class LLMResponse:
    """Standardized response from any LLM provider.

    Attributes:
        text: The generated text content.
        model: Model identifier used for generation.
        provider: Name of the provider backend.
        usage: Token usage stats (prompt_tokens, completion_tokens).
        latency_ms: Round-trip latency in milliseconds.
        metadata: Any extra provider-specific metadata.
    """

    text: str
    model: str
    provider: str
    usage: Dict[str, int] = field(default_factory=dict)
    latency_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Abstract Base
# ---------------------------------------------------------------------------

class BaseLLMProvider(ABC):
    """Abstract base class for all LLM providers."""

    @abstractmethod
    async def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.3,
        **kwargs: Any,
    ) -> LLMResponse:
        """Send a chat completion request.

        Args:
            messages: List of message dicts with ``role`` and ``content`` keys.
            model: Override the default model for this request.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (0.0 – 2.0).
            **kwargs: Additional provider-specific parameters.

        Returns:
            An ``LLMResponse`` with the generated text and metadata.
        """
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the provider is available and responding.

        Returns:
            ``True`` if the provider is reachable, ``False`` otherwise.
        """
        ...

    def _resolve_env(self, value: Optional[str]) -> Optional[str]:
        """Resolve ``${ENV_VAR}`` placeholders in config values."""
        if value and isinstance(value, str) and value.startswith("${") and value.endswith("}"):
            env_name = value[2:-1]
            return os.environ.get(env_name)
        return value


# ---------------------------------------------------------------------------
# OpenAI Provider
# ---------------------------------------------------------------------------

class OpenAIProvider(BaseLLMProvider):
    """Provider for the OpenAI API (GPT-4o, GPT-4-turbo, GPT-3.5-turbo, etc.).

    Requires the ``openai`` package (already in requirements.txt).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o",
        **kwargs: Any,
    ) -> None:
        self.api_key = self._resolve_env(api_key) or os.environ.get("OPENAI_API_KEY")
        self.model = model
        self._client: Any = None

    def _get_client(self) -> Any:
        """Lazily initialise the async OpenAI client."""
        if self._client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError:
                raise ImportError("The 'openai' package is required for OpenAIProvider. pip install openai")
            self._client = AsyncOpenAI(api_key=self.api_key)
        return self._client

    async def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.3,
        **kwargs: Any,
    ) -> LLMResponse:
        """Send a chat completion request to the OpenAI API."""
        client = self._get_client()
        model = model or self.model
        start = time.time()
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs,
            )
            latency = (time.time() - start) * 1000
            choice = resp.choices[0]
            usage = {}
            if resp.usage:
                usage = {
                    "prompt_tokens": resp.usage.prompt_tokens or 0,
                    "completion_tokens": resp.usage.completion_tokens or 0,
                }
            return LLMResponse(
                text=choice.message.content or "",
                model=model,
                provider="openai",
                usage=usage,
                latency_ms=latency,
                metadata={"finish_reason": choice.finish_reason},
            )
        except Exception as exc:
            logger.error("OpenAI call failed: %s", exc)
            return LLMResponse(text="", model=model, provider="openai")

    async def health_check(self) -> bool:
        """Verify the OpenAI API is reachable by listing models."""
        try:
            client = self._get_client()
            await client.models.list()
            return True
        except Exception as exc:
            logger.warning("OpenAI health check failed: %s", exc)
            return False


# ---------------------------------------------------------------------------
# Ollama Provider
# ---------------------------------------------------------------------------

class OllamaProvider(BaseLLMProvider):
    """Provider for Ollama local models via its REST API.

    Uses ``httpx`` for async HTTP — no ``ollama`` package required.
    Supports: llama3, mistral, codellama, phi3, gemma, medllama2, meditron, etc.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3",
        num_ctx: int = 2048,
        **kwargs: Any,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        # num_ctx controls the KV-cache size. The default Ollama context (32k–40k)
        # consumes 3-4 GB of VRAM even on small models. Setting 2048 drops KV-cache
        # to ~200 MB, allowing a 8B Q4 model to fully load on an 8 GB GPU (RTX 3070).
        self.num_ctx = num_ctx

    def _sync_chat(
        self,
        model: str,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Synchronous httpx call — safe across event-loop boundaries."""
        import httpx
        with httpx.Client(timeout=180.0) as client:
            resp = client.post(f"{self.base_url}/api/chat", json=payload)
            resp.raise_for_status()
            return resp.json()

    async def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.3,
        **kwargs: Any,
    ) -> LLMResponse:
        """Send a chat completion request to the Ollama REST API.

        Uses a fresh synchronous httpx.Client per call (via run_in_executor)
        so it is safe across multiple asyncio.run() invocations that each
        create a new event loop.

        Endpoint: ``POST /api/chat`` (non-streaming).
        """
        import asyncio

        model = model or self.model
        start = time.time()

        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "think": False,   # Disable chain-of-thought for thinking models (e.g. Qwen3)
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
                "num_ctx": self.num_ctx,  # Small context = small KV cache = fits fully on GPU
            },
        }

        try:
            loop = asyncio.get_event_loop()
            data = await loop.run_in_executor(
                None, self._sync_chat, model, payload
            )
            msg = data.get("message", {})
            # thinking models (Qwen3) may put response in content when think=False;
            # fall back to the thinking field if content is empty.
            text = msg.get("content", "") or msg.get("thinking", "")
            # phi4-reasoning and similar models embed thinking in <think>...</think> tags
            # within content. Strip those blocks to get only the final answer.
            import re as _re
            text = _re.sub(r"<think>.*?</think>\s*", "", text, flags=_re.DOTALL).strip()
            latency = (time.time() - start) * 1000
            return LLMResponse(
                text=text,
                model=model,
                provider="ollama",
                usage={
                    "prompt_tokens": data.get("prompt_eval_count", 0),
                    "completion_tokens": data.get("eval_count", 0),
                },
                latency_ms=latency,
                metadata={
                    "total_duration": data.get("total_duration"),
                    "load_duration": data.get("load_duration"),
                },
            )
        except Exception as exc:
            logger.error("Ollama call failed: %s", exc)
            return LLMResponse(text="", model=model, provider="ollama")

    async def health_check(self) -> bool:
        """Check Ollama availability via ``GET /api/tags``."""
        import asyncio
        import httpx

        def _check() -> bool:
            try:
                with httpx.Client(timeout=5.0) as client:
                    resp = client.get(f"{self.base_url}/api/tags")
                    return resp.status_code == 200
            except Exception as exc:
                logger.warning("Ollama health check failed: %s", exc)
                return False

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _check)


# ---------------------------------------------------------------------------
# Anthropic Provider
# ---------------------------------------------------------------------------

class AnthropicProvider(BaseLLMProvider):
    """Provider for Anthropic Claude models.

    Attempts to use the official ``anthropic`` package; falls back to raw
    ``httpx`` REST calls if the package is not installed.
    """

    ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-3-sonnet-20240229",
        **kwargs: Any,
    ) -> None:
        self.api_key = self._resolve_env(api_key) or os.environ.get("ANTHROPIC_API_KEY")
        self.model = model
        self._client: Any = None
        self._use_sdk: bool = False

        try:
            import anthropic  # noqa: F401
            self._use_sdk = True
        except ImportError:
            logger.info("anthropic package not installed — using httpx REST fallback")

    def _get_sdk_client(self) -> Any:
        """Lazily initialise the async Anthropic SDK client."""
        if self._client is None:
            from anthropic import AsyncAnthropic
            self._client = AsyncAnthropic(api_key=self.api_key)
        return self._client

    async def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.3,
        **kwargs: Any,
    ) -> LLMResponse:
        """Send a chat completion request to Anthropic.

        Note: Anthropic treats the ``system`` role as a top-level parameter,
        not inside the messages list.
        """
        model = model or self.model
        start = time.time()

        # Extract system message if present
        system_text: Optional[str] = None
        filtered_msgs: List[Dict[str, str]] = []
        for msg in messages:
            if msg.get("role") == "system":
                system_text = msg.get("content", "")
            else:
                filtered_msgs.append(msg)

        if self._use_sdk:
            return await self._chat_sdk(filtered_msgs, system_text, model, max_tokens, temperature, start, **kwargs)
        return await self._chat_httpx(filtered_msgs, system_text, model, max_tokens, temperature, start, **kwargs)

    async def _chat_sdk(
        self,
        messages: List[Dict[str, str]],
        system: Optional[str],
        model: str,
        max_tokens: int,
        temperature: float,
        start: float,
        **kwargs: Any,
    ) -> LLMResponse:
        """Chat via the official anthropic SDK."""
        client = self._get_sdk_client()
        try:
            create_kwargs: Dict[str, Any] = {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            if system:
                create_kwargs["system"] = system
            resp = await client.messages.create(**create_kwargs, **kwargs)
            latency = (time.time() - start) * 1000
            text = ""
            if resp.content:
                text = resp.content[0].text
            usage = {}
            if resp.usage:
                usage = {
                    "prompt_tokens": resp.usage.input_tokens or 0,
                    "completion_tokens": resp.usage.output_tokens or 0,
                }
            return LLMResponse(
                text=text,
                model=model,
                provider="anthropic",
                usage=usage,
                latency_ms=latency,
                metadata={"stop_reason": resp.stop_reason},
            )
        except Exception as exc:
            logger.error("Anthropic SDK call failed: %s", exc)
            return LLMResponse(text="", model=model, provider="anthropic")

    async def _chat_httpx(
        self,
        messages: List[Dict[str, str]],
        system: Optional[str],
        model: str,
        max_tokens: int,
        temperature: float,
        start: float,
        **kwargs: Any,
    ) -> LLMResponse:
        """Chat via raw httpx REST calls (fallback when SDK not installed)."""
        import httpx

        headers = {
            "x-api-key": self.api_key or "",
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if system:
            payload["system"] = system

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(self.ANTHROPIC_API_URL, headers=headers, json=payload)
                resp.raise_for_status()
                data = resp.json()
                latency = (time.time() - start) * 1000
                text = ""
                content = data.get("content", [])
                if content:
                    text = content[0].get("text", "")
                usage_data = data.get("usage", {})
                return LLMResponse(
                    text=text,
                    model=model,
                    provider="anthropic",
                    usage={
                        "prompt_tokens": usage_data.get("input_tokens", 0),
                        "completion_tokens": usage_data.get("output_tokens", 0),
                    },
                    latency_ms=latency,
                    metadata={"stop_reason": data.get("stop_reason")},
                )
        except Exception as exc:
            logger.error("Anthropic httpx call failed: %s", exc)
            return LLMResponse(text="", model=model, provider="anthropic")

    async def health_check(self) -> bool:
        """Verify Anthropic API key is set (no free health endpoint exists)."""
        if not self.api_key:
            logger.warning("Anthropic health check failed: no API key configured")
            return False
        # Send a minimal request to verify connectivity
        try:
            resp = await self.chat(
                [{"role": "user", "content": "ping"}],
                max_tokens=1,
                temperature=0.0,
            )
            return bool(resp.text)
        except Exception as exc:
            logger.warning("Anthropic health check failed: %s", exc)
            return False


# ---------------------------------------------------------------------------
# Azure OpenAI Provider
# ---------------------------------------------------------------------------

class AzureOpenAIProvider(BaseLLMProvider):
    """Provider for Azure-hosted OpenAI models.

    Requires the ``openai`` package with Azure support.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        api_version: str = "2024-02-01",
        deployment_name: str = "gpt-4o",
        **kwargs: Any,
    ) -> None:
        self.api_key = self._resolve_env(api_key) or os.environ.get("AZURE_OPENAI_API_KEY")
        self.endpoint = self._resolve_env(endpoint) or os.environ.get("AZURE_OPENAI_ENDPOINT")
        self.api_version = api_version
        self.deployment_name = deployment_name
        self._client: Any = None

    def _get_client(self) -> Any:
        """Lazily initialise the async Azure OpenAI client."""
        if self._client is None:
            try:
                from openai import AsyncAzureOpenAI
            except ImportError:
                raise ImportError("The 'openai' package is required for AzureOpenAIProvider. pip install openai")
            self._client = AsyncAzureOpenAI(
                api_key=self.api_key,
                azure_endpoint=self.endpoint or "",
                api_version=self.api_version,
            )
        return self._client

    async def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.3,
        **kwargs: Any,
    ) -> LLMResponse:
        """Send a chat completion request to Azure OpenAI."""
        client = self._get_client()
        deployment = model or self.deployment_name
        start = time.time()
        try:
            resp = await client.chat.completions.create(
                model=deployment,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs,
            )
            latency = (time.time() - start) * 1000
            choice = resp.choices[0]
            usage = {}
            if resp.usage:
                usage = {
                    "prompt_tokens": resp.usage.prompt_tokens or 0,
                    "completion_tokens": resp.usage.completion_tokens or 0,
                }
            return LLMResponse(
                text=choice.message.content or "",
                model=deployment,
                provider="azure_openai",
                usage=usage,
                latency_ms=latency,
                metadata={"finish_reason": choice.finish_reason},
            )
        except Exception as exc:
            logger.error("Azure OpenAI call failed: %s", exc)
            return LLMResponse(text="", model=deployment, provider="azure_openai")

    async def health_check(self) -> bool:
        """Verify Azure OpenAI endpoint is reachable."""
        if not self.api_key or not self.endpoint:
            logger.warning("Azure OpenAI health check failed: missing api_key or endpoint")
            return False
        try:
            client = self._get_client()
            await client.models.list()
            return True
        except Exception as exc:
            logger.warning("Azure OpenAI health check failed: %s", exc)
            return False


# ---------------------------------------------------------------------------
# Google Gemini Provider
# ---------------------------------------------------------------------------

class GoogleGeminiProvider(BaseLLMProvider):
    """Provider for Google Gemini models.

    Attempts to use ``google.generativeai``; falls back to httpx REST calls.
    """

    GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-1.5-pro",
        **kwargs: Any,
    ) -> None:
        self.api_key = self._resolve_env(api_key) or os.environ.get("GOOGLE_API_KEY")
        self.model = model
        self._use_sdk: bool = False
        self._sdk_model: Any = None

        try:
            import google.generativeai  # noqa: F401
            self._use_sdk = True
        except ImportError:
            logger.info("google-generativeai not installed — using httpx REST fallback")

    def _get_sdk_model(self) -> Any:
        """Lazily configure and return a ``GenerativeModel``."""
        if self._sdk_model is None:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self._sdk_model = genai.GenerativeModel(self.model)
        return self._sdk_model

    async def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.3,
        **kwargs: Any,
    ) -> LLMResponse:
        """Send a chat completion request to Google Gemini."""
        model_name = model or self.model
        start = time.time()

        if self._use_sdk:
            return await self._chat_sdk(messages, model_name, max_tokens, temperature, start, **kwargs)
        return await self._chat_httpx(messages, model_name, max_tokens, temperature, start, **kwargs)

    async def _chat_sdk(
        self,
        messages: List[Dict[str, str]],
        model_name: str,
        max_tokens: int,
        temperature: float,
        start: float,
        **kwargs: Any,
    ) -> LLMResponse:
        """Chat via the google-generativeai SDK."""
        import asyncio

        try:
            sdk_model = self._get_sdk_model()

            # Convert OpenAI-style messages to Gemini contents
            contents: List[Dict[str, Any]] = []
            for msg in messages:
                role = msg.get("role", "user")
                if role == "system":
                    role = "user"
                elif role == "assistant":
                    role = "model"
                contents.append({"role": role, "parts": [{"text": msg.get("content", "")}]})

            generation_config = {
                "max_output_tokens": max_tokens,
                "temperature": temperature,
            }

            # The SDK's generate_content is synchronous; run in executor
            resp = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: sdk_model.generate_content(
                    contents,
                    generation_config=generation_config,
                ),
            )
            latency = (time.time() - start) * 1000
            text = resp.text if hasattr(resp, "text") else ""
            usage: Dict[str, int] = {}
            if hasattr(resp, "usage_metadata") and resp.usage_metadata:
                usage = {
                    "prompt_tokens": getattr(resp.usage_metadata, "prompt_token_count", 0),
                    "completion_tokens": getattr(resp.usage_metadata, "candidates_token_count", 0),
                }
            return LLMResponse(
                text=text,
                model=model_name,
                provider="google_gemini",
                usage=usage,
                latency_ms=latency,
            )
        except Exception as exc:
            logger.error("Google Gemini SDK call failed: %s", exc)
            return LLMResponse(text="", model=model_name, provider="google_gemini")

    async def _chat_httpx(
        self,
        messages: List[Dict[str, str]],
        model_name: str,
        max_tokens: int,
        temperature: float,
        start: float,
        **kwargs: Any,
    ) -> LLMResponse:
        """Chat via the Gemini REST API (fallback)."""
        import httpx

        if not self.api_key:
            logger.error("Google Gemini httpx fallback requires an API key")
            return LLMResponse(text="", model=model_name, provider="google_gemini")

        # Build contents array
        contents: List[Dict[str, Any]] = []
        for msg in messages:
            role = msg.get("role", "user")
            if role == "system":
                role = "user"
            elif role == "assistant":
                role = "model"
            contents.append({"role": role, "parts": [{"text": msg.get("content", "")}]})

        url = f"{self.GEMINI_API_URL}/{model_name}:generateContent?key={self.api_key}"
        payload = {
            "contents": contents,
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": temperature,
            },
        }

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(url, json=payload)
                resp.raise_for_status()
                data = resp.json()
                latency = (time.time() - start) * 1000
                text = ""
                candidates = data.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    if parts:
                        text = parts[0].get("text", "")
                usage_meta = data.get("usageMetadata", {})
                return LLMResponse(
                    text=text,
                    model=model_name,
                    provider="google_gemini",
                    usage={
                        "prompt_tokens": usage_meta.get("promptTokenCount", 0),
                        "completion_tokens": usage_meta.get("candidatesTokenCount", 0),
                    },
                    latency_ms=latency,
                )
        except Exception as exc:
            logger.error("Google Gemini httpx call failed: %s", exc)
            return LLMResponse(text="", model=model_name, provider="google_gemini")

    async def health_check(self) -> bool:
        """Verify Google Gemini API is reachable."""
        if not self.api_key:
            logger.warning("Google Gemini health check failed: no API key configured")
            return False
        import httpx

        try:
            url = f"{self.GEMINI_API_URL}?key={self.api_key}"
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(url)
                return resp.status_code == 200
        except Exception as exc:
            logger.warning("Google Gemini health check failed: %s", exc)
            return False


# ---------------------------------------------------------------------------
# HuggingFace Inference Provider
# ---------------------------------------------------------------------------

class HuggingFaceProvider(BaseLLMProvider):
    """Provider for the HuggingFace Inference API.

    Uses ``httpx`` to call the HF Inference endpoint.
    """

    HF_API_URL = "https://api-inference.huggingface.co/models"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        **kwargs: Any,
    ) -> None:
        self.api_key = self._resolve_env(api_key) or os.environ.get("HF_TOKEN")
        self.model = model

    async def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.3,
        **kwargs: Any,
    ) -> LLMResponse:
        """Send a chat completion request to HuggingFace Inference API.

        Uses the ``/models/{model}`` endpoint with chat-style payload.
        """
        import httpx

        model_id = model or self.model
        start = time.time()

        headers = {
            "Authorization": f"Bearer {self.api_key}" if self.api_key else "",
            "Content-type": "application/json",
        }

        # HF Inference API supports OpenAI-compatible chat format for
        # instruction-tuned models.
        payload: Dict[str, Any] = {
            "inputs": self._format_prompt(messages),
            "parameters": {
                "max_new_tokens": max_tokens,
                "temperature": max(temperature, 0.01),  # HF rejects 0.0
                "return_full_text": False,
            },
        }

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.post(
                    f"{self.HF_API_URL}/{model_id}",
                    headers=headers,
                    json=payload,
                )
                resp.raise_for_status()
                data = resp.json()
                latency = (time.time() - start) * 1000

                text = ""
                if isinstance(data, list) and data:
                    text = data[0].get("generated_text", "")
                elif isinstance(data, dict):
                    text = data.get("generated_text", "")

                return LLMResponse(
                    text=text,
                    model=model_id,
                    provider="huggingface",
                    usage={},
                    latency_ms=latency,
                )
        except Exception as exc:
            logger.error("HuggingFace call failed: %s", exc)
            return LLMResponse(text="", model=model_id, provider="huggingface")

    @staticmethod
    def _format_prompt(messages: List[Dict[str, str]]) -> str:
        """Convert chat messages to a single prompt string for HF models."""
        parts: List[str] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                parts.append(f"<|system|>\n{content}")
            elif role == "assistant":
                parts.append(f"<|assistant|>\n{content}")
            else:
                parts.append(f"<|user|>\n{content}")
        parts.append("<|assistant|>\n")
        return "\n".join(parts)

    async def health_check(self) -> bool:
        """Verify the HuggingFace Inference API is reachable for the model."""
        import httpx

        if not self.api_key:
            logger.warning("HuggingFace health check failed: no API key (HF_TOKEN) configured")
            return False
        headers = {"Authorization": f"Bearer {self.api_key}"}
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    f"https://huggingface.co/api/models/{self.model}",
                    headers=headers,
                )
                return resp.status_code == 200
        except Exception as exc:
            logger.warning("HuggingFace health check failed: %s", exc)
            return False


# ---------------------------------------------------------------------------
# OpenAI-Compatible Provider
# ---------------------------------------------------------------------------

class OpenAICompatibleProvider(BaseLLMProvider):
    """Provider for any OpenAI-compatible endpoint.

    Works with: vLLM, LM Studio, text-generation-webui, LocalAI,
    llama.cpp server, and any other server that implements the
    OpenAI chat completions API.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "not-needed",
        model: str = "local-model",
        **kwargs: Any,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self._client: Any = None

    def _get_client(self) -> Any:
        """Lazily initialise the async OpenAI client with a custom base URL."""
        if self._client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError:
                raise ImportError(
                    "The 'openai' package is required for OpenAICompatibleProvider. pip install openai"
                )
            self._client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
            )
        return self._client

    async def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.3,
        **kwargs: Any,
    ) -> LLMResponse:
        """Send a chat completion request to an OpenAI-compatible endpoint."""
        client = self._get_client()
        model = model or self.model
        start = time.time()
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs,
            )
            latency = (time.time() - start) * 1000
            choice = resp.choices[0]
            usage = {}
            if resp.usage:
                usage = {
                    "prompt_tokens": resp.usage.prompt_tokens or 0,
                    "completion_tokens": resp.usage.completion_tokens or 0,
                }
            return LLMResponse(
                text=choice.message.content or "",
                model=model,
                provider="openai_compatible",
                usage=usage,
                latency_ms=latency,
                metadata={"finish_reason": choice.finish_reason},
            )
        except Exception as exc:
            logger.error("OpenAI-compatible call failed: %s", exc)
            return LLMResponse(text="", model=model, provider="openai_compatible")

    async def health_check(self) -> bool:
        """Verify the OpenAI-compatible endpoint is reachable."""
        import httpx

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self.base_url}/models")
                return resp.status_code == 200
        except Exception as exc:
            logger.warning("OpenAI-compatible health check failed: %s", exc)
            return False


# ---------------------------------------------------------------------------
# Factory Functions
# ---------------------------------------------------------------------------

_PROVIDER_MAP: Dict[str, type] = {
    ProviderType.OPENAI.value: OpenAIProvider,
    ProviderType.OLLAMA.value: OllamaProvider,
    ProviderType.ANTHROPIC.value: AnthropicProvider,
    ProviderType.AZURE_OPENAI.value: AzureOpenAIProvider,
    ProviderType.GOOGLE_GEMINI.value: GoogleGeminiProvider,
    ProviderType.HUGGINGFACE.value: HuggingFaceProvider,
    ProviderType.OPENAI_COMPATIBLE.value: OpenAICompatibleProvider,
}


def create_provider(config: Dict[str, Any]) -> BaseLLMProvider:
    """Factory function to create the appropriate provider from a config dict.

    The ``config`` dict must contain a ``"provider"`` key whose value matches
    one of the :class:`ProviderType` enum values.  Remaining keys are passed
    as keyword arguments to the provider constructor.

    Args:
        config: Dictionary with ``provider`` key and provider-specific params.

    Returns:
        An instantiated :class:`BaseLLMProvider` subclass.

    Raises:
        ValueError: If the provider type is unknown.

    Example::

        provider = create_provider({
            "provider": "ollama",
            "base_url": "http://localhost:11434",
            "model": "llama3",
        })
    """
    provider_type = config.get("provider", "openai")
    provider_cls = _PROVIDER_MAP.get(provider_type)
    if provider_cls is None:
        supported = ", ".join(sorted(_PROVIDER_MAP.keys()))
        raise ValueError(
            f"Unknown LLM provider '{provider_type}'. Supported: {supported}"
        )

    # Pass everything except the 'provider' key to the constructor
    kwargs = {k: v for k, v in config.items() if k != "provider"}
    logger.info("Creating LLM provider: %s (model=%s)", provider_type, kwargs.get("model", "default"))
    return provider_cls(**kwargs)


def create_provider_from_yaml(config: Dict[str, Any]) -> BaseLLMProvider:
    """Create a provider from the full ``config.yaml`` structure.

    Reads the ``llm`` section of the configuration.  Falls back to ``openai``
    if the section is missing or the provider is unspecified.

    The selected provider's sub-dictionary is merged with the top-level
    ``provider`` key and passed to :func:`create_provider`.

    Args:
        config: The full parsed YAML config dictionary.

    Returns:
        An instantiated :class:`BaseLLMProvider` subclass.

    Example YAML::

        llm:
          provider: "ollama"
          ollama:
            base_url: "http://localhost:11434"
            model: "llama3"
    """
    llm_config = config.get("llm", {})
    provider_type = llm_config.get("provider", "openai")

    # Grab provider-specific settings
    provider_settings: Dict[str, Any] = llm_config.get(provider_type, {})
    provider_settings["provider"] = provider_type

    logger.info("Creating provider from YAML config: %s", provider_type)
    return create_provider(provider_settings)
