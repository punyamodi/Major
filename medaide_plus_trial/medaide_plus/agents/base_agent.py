"""
BaseAgent — Abstract base class for all MedAide+ agents.

Provides:
  - async analyze(query, context) -> AgentOutput interface
  - Multi-provider LLM integration (OpenAI, Ollama, Anthropic, Azure, Gemini, HF, etc.)
  - Confidence estimation
  - Claim extraction
"""

import logging
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger("medaide_plus.base_agent")


@dataclass
class AgentOutput:
    """Standardized output from a medical domain agent."""
    agent_name: str
    response: str
    confidence: float = 0.0
    latency_ms: float = 0.0
    intents: List[str] = field(default_factory=list)
    claims: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


class BaseAgent(ABC):
    """
    Abstract base class for MedAide+ medical domain agents.

    Subclasses must implement:
      - name (str property)
      - system_prompt (str property)
      - analyze(query, context) -> AgentOutput

    Supports multiple LLM backends via the LLMProvider abstraction:
      - OpenAI, Ollama, Anthropic, Azure OpenAI, Google Gemini,
        HuggingFace Inference, and any OpenAI-compatible endpoint.

    Args:
        config: Configuration dict (model, max_tokens, temperature, provider settings).
        llm_client: Optional pre-built OpenAI async client (legacy support).
        llm_provider: Optional LLMProvider instance (preferred over llm_client).
    """

    def __init__(
        self,
        config: Optional[Dict] = None,
        llm_client=None,
        llm_provider=None,
    ) -> None:
        self.config = config or {}
        self.model: str = self.config.get("openai_model", self.config.get("model", "gpt-4o"))
        self.max_tokens: int = self.config.get("max_tokens", 1024)
        self.temperature: float = self.config.get("temperature", 0.3)
        self._client = llm_client
        self._provider = llm_provider

    @property
    @abstractmethod
    def name(self) -> str:
        """Agent name identifier."""
        ...

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """System prompt defining agent specialty."""
        ...

    async def analyze(self, query: str, context: Optional[Dict] = None) -> AgentOutput:
        """
        Analyze a medical query and return a structured response.

        Args:
            query: Medical query string (may include patient history prefix).
            context: Optional context dict with intents, patient_id, etc.

        Returns:
            AgentOutput with response text, confidence, and extracted claims.
        """
        context = context or {}
        start_time = time.time()

        try:
            full_prompt = self._build_prompt(query, context)
            response_text = await self._call_llm(full_prompt)
            confidence = self._estimate_confidence(response_text, query)
            claims = self._extract_claims(response_text)
            latency_ms = (time.time() - start_time) * 1000

            return AgentOutput(
                agent_name=self.name,
                response=response_text,
                confidence=confidence,
                latency_ms=latency_ms,
                intents=context.get("intents", []),
                claims=claims,
                metadata={"model": self.model},
            )
        except Exception as e:
            logger.error(f"[{self.name}] Error during analysis: {e}")
            latency_ms = (time.time() - start_time) * 1000
            return AgentOutput(
                agent_name=self.name,
                response="",
                confidence=0.0,
                latency_ms=latency_ms,
                error=str(e),
            )

    async def _call_llm(self, prompt: str) -> str:
        """
        Call the LLM using the configured provider.

        Priority: LLMProvider → legacy OpenAI client → mock response.

        Args:
            prompt: Full prompt string.

        Returns:
            LLM response text.
        """
        # Priority 1: Use the new unified LLM provider
        if self._provider is not None:
            try:
                import asyncio

                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt},
                ]

                # Never substitute mock text during evaluation. If Ollama intermittently
                # returns an empty string, retry a few times to get a real completion.
                max_retries = int(self.config.get("empty_response_retries", 2))
                base_sleep = float(self.config.get("empty_response_retry_sleep_s", 0.5))

                for attempt in range(max_retries + 1):
                    response = await self._provider.chat(
                        messages=messages,
                        model=self.model,
                        max_tokens=self.max_tokens,
                        temperature=self.temperature,
                    )
                    text = (response.text or "").strip()
                    if text:
                        return text

                    if attempt < max_retries:
                        logger.warning(
                            f"[{self.name}] Provider returned empty response (attempt {attempt+1}/{max_retries+1}). Retrying..."
                        )
                        await asyncio.sleep(base_sleep * (attempt + 1))

                logger.warning(f"[{self.name}] Provider returned empty response after retries. Returning empty string.")
                return ""
            except Exception as e:
                logger.warning(f"[{self.name}] Provider call failed: {e}. Trying fallback.")

        # Priority 2: Legacy OpenAI client
        if self._client is None:
            try:
                import os
                from openai import AsyncOpenAI
                api_key = os.environ.get("OPENAI_API_KEY")
                if not api_key:
                    return self._mock_response(prompt)
                self._client = AsyncOpenAI(api_key=api_key)
            except ImportError:
                return self._mock_response(prompt)

        try:
            response = await self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"[{self.name}] LLM call failed: {e}. Using mock.")
            return self._mock_response(prompt)

    def _mock_response(self, prompt: str) -> str:
        """
        Generate a mock response for testing when LLM is unavailable.

        Args:
            prompt: Input prompt.

        Returns:
            Mock response string.
        """
        return (
            f"[{self.name} Mock Response] This is a test response for the query. "
            f"In production, this would be answered by {self.model} with domain expertise. "
            f"Please consult a qualified medical professional for actual medical advice."
        )

    def _build_prompt(self, query: str, context: Dict) -> str:
        """
        Build the full user prompt including context, KB evidence, and instructions.

        Args:
            query: Medical query.
            context: Context dict with intents, subqueries, tier, kb_evidence, etc.

        Returns:
            Formatted prompt string.
        """
        context_parts = []

        # Phase 7: Universal prompt building — all models get the same context.
        # Reference-aligned specialist prompts ensure responses match reference format
        # without needing size-based prompt stripping.

        # Patient history prefix (from M7 MIET)
        if context.get("context_prefix"):
            context_parts.append(context["context_prefix"])

        if context.get("intents"):
            context_parts.append(f"Detected intents: {', '.join(context['intents'])}")
        if context.get("subqueries"):
            context_parts.append(
                "Sub-questions: " + "; ".join(context["subqueries"][:3])
            )

        # KB-retrieved evidence (from M1 BM25) — top-1 only to keep prompt focused
        if context.get("kb_evidence"):
            top_evidence = context["kb_evidence"][0][:300]
            context_parts.append(f"Relevant medical knowledge: {top_evidence}")

        prompt = query
        if context_parts:
            prompt = f"Context:\n{chr(10).join(context_parts)}\n\nQuestion: {query}"

        # Instruct for focused, direct answer
        prompt += (
            "\n\nProvide a focused, directly relevant medical response. "
            "Be specific and concise. Do not repeat the question."
        )
        return prompt

    def _estimate_confidence(self, response: str, query: str) -> float:
        """
        Estimate response confidence using heuristic signals.

        Signals:
          - Response length (longer = more confident up to a threshold)
          - Hedging language (reduces confidence)
          - Medical terminology density
          - Presence of specific recommendations

        Args:
            response: Generated response text.
            query: Original query.

        Returns:
            Confidence score in [0, 1].
        """
        if not response or len(response) < 10:
            return 0.1

        score = 0.5  # Base score

        # Length signal: +0.1 for responses 100-500 chars, -0.1 for very short
        length = len(response)
        if 100 <= length <= 500:
            score += 0.1
        elif length > 500:
            score += 0.05
        elif length < 50:
            score -= 0.2

        # Hedging language (reduces confidence)
        hedging_phrases = [
            "i'm not sure", "i cannot", "unclear", "uncertain",
            "might be", "could be", "possibly", "maybe",
        ]
        hedge_count = sum(1 for p in hedging_phrases if p in response.lower())
        score -= 0.05 * hedge_count

        # Medical terminology (increases confidence)
        medical_terms = [
            "diagnosis", "symptom", "treatment", "medication", "dosage",
            "mg", "ml", "contraindicated", "recommended", "evidence",
        ]
        term_count = sum(1 for t in medical_terms if t in response.lower())
        score += 0.02 * min(term_count, 5)

        # Specific recommendations (increases confidence)
        recommendation_phrases = [
            "recommend", "should", "advised", "suggest", "prescribe",
        ]
        rec_count = sum(1 for p in recommendation_phrases if p in response.lower())
        score += 0.02 * min(rec_count, 3)

        return float(min(max(score, 0.1), 0.95))

    def _extract_claims(self, response: str) -> List[str]:
        """
        Extract verifiable medical claims from response text.

        Args:
            response: Response text.

        Returns:
            List of claim strings (sentences with medical assertions).
        """
        sentences = re.split(r"[.!?]+", response)
        claims = []
        claim_keywords = [
            "should", "must", "can", "may", "recommend", "effective",
            "causes", "treats", "indicated", "contraindicated", "dose",
            "mg", "daily", "avoid", "take",
        ]
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 20:
                continue
            if any(kw in sentence.lower() for kw in claim_keywords):
                claims.append(sentence)
        return claims[:8]
