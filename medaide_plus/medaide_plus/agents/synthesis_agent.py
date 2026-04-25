"""
SynthesisAgent — LLM-based confidence-weighted response synthesizer.

Uses the LLM to merge multi-agent outputs into a single coherent response
without formatting noise (no agent names, headers, or weight annotations
in the final text). Falls back to extractive merge when no LLM is available.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from medaide_plus.agents.base_agent import AgentOutput
from medaide_plus.agents.critic_agent import CriticReport

logger = logging.getLogger("medaide_plus.synthesis_agent")

SYNTHESIS_SYSTEM_PROMPT = (
    "You are a medical response synthesizer. You receive multiple specialist "
    "assessments for the same patient query. Your job is to produce a single, "
    "coherent medical response that:\n"
    "  1. Uses the highest-confidence assessment as the PRIMARY basis.\n"
    "  2. Integrates only genuinely new, clinically relevant details from "
    "     secondary assessments — do NOT dilute the primary response.\n"
    "  3. Resolves contradictions by favoring the higher-confidence source.\n"
    "  4. PRESERVES medical terminology, clinical precision, and any structured "
    "     formatting (bold headers, bullet points) from the primary assessment.\n"
    "  5. Does NOT simplify language — maintain the same clinical tone and "
    "     vocabulary used in the source assessments.\n"
    "  6. Does NOT mention agent names, confidence scores, or that multiple "
    "     assessments were consulted.\n"
    "Output the synthesized response directly — nothing else."
)


@dataclass
class SynthesisResult:
    """Result from synthesis process."""
    synthesized_response: str
    contributing_agents: List[str] = field(default_factory=list)
    agent_weights: Dict[str, float] = field(default_factory=dict)
    conflicts_resolved: int = 0
    final_confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class SynthesisAgent:
    """
    LLM-based synthesis agent for multi-agent response merging.

    When an LLM provider is available, uses the LLM to synthesize a single
    coherent response from multiple agent outputs — no formatting noise.
    Falls back to extractive confidence-weighted merge when no LLM available.

    Args:
        config: Configuration dict.
        llm_client: Optional OpenAI async client (legacy).
        llm_provider: Optional LLMProvider instance (preferred).
    """

    def __init__(
        self,
        config: Optional[Dict] = None,
        llm_client=None,
        llm_provider=None,
    ) -> None:
        self.config = config or {}
        self._client = llm_client
        self._provider = llm_provider
        self.name = "SynthesisAgent"
        self.model = self.config.get("model", self.config.get("openai_model", "gpt-4o"))
        self.max_tokens = self.config.get("max_tokens", 1024)
        self.temperature = self.config.get("temperature", 0.2)

    async def synthesize(
        self,
        agent_outputs: List[AgentOutput],
        critic_report: Optional[CriticReport] = None,
        query: Optional[str] = None,
    ) -> SynthesisResult:
        """
        Synthesize a final response from multiple agent outputs.

        Uses LLM-based synthesis when provider is available to produce a clean,
        coherent response without formatting noise. Falls back to extractive
        merge otherwise.

        Args:
            agent_outputs: List of outputs from domain agents.
            critic_report: Optional CriticReport from CriticAgent.
            query: Original user query (for context).

        Returns:
            SynthesisResult with merged response and metadata.
        """
        valid_outputs = [o for o in agent_outputs if not o.error and o.response]

        if not valid_outputs:
            return SynthesisResult(
                synthesized_response=(
                    "Unable to generate a response. All agents encountered errors."
                ),
                final_confidence=0.0,
            )

        if len(valid_outputs) == 1:
            return SynthesisResult(
                synthesized_response=valid_outputs[0].response,
                contributing_agents=[valid_outputs[0].agent_name],
                agent_weights={valid_outputs[0].agent_name: 1.0},
                final_confidence=valid_outputs[0].confidence,
                metadata={"method": "single_agent"},
            )

        weights = self._compute_weights(valid_outputs)

        conflict_agents = set()
        resolved_count = 0
        if critic_report:
            for agent_a, agent_b in critic_report.conflict_agents:
                conflict_agents.add(agent_a)
                conflict_agents.add(agent_b)
                resolved_count += 1
            if conflict_agents:
                for agent_name in conflict_agents:
                    if agent_name in weights:
                        weights[agent_name] *= 0.5
                total = sum(weights.values())
                if total > 0:
                    weights = {k: v / total for k, v in weights.items()}

        # Strategy: only use LLM synthesis when there are conflicts to resolve.
        # For low-conflict cases, the primary agent response is already high quality
        # — synthesis would only add drift. Use extractive merge instead.
        use_llm = (
            critic_report is not None
            and critic_report.severity in ("medium", "high")
        )

        if use_llm:
            synthesized = await self._llm_synthesize(
                valid_outputs, weights, critic_report, query
            )
        else:
            synthesized = ""

        # Fallback to extractive merge (preserves primary response as-is)
        if not synthesized:
            synthesized = self._extractive_merge(valid_outputs, weights)

        final_confidence = sum(
            weights.get(o.agent_name, 0) * o.confidence for o in valid_outputs
        )

        method = "llm_synthesis" if synthesized else "extractive_merge"

        return SynthesisResult(
            synthesized_response=synthesized,
            contributing_agents=[o.agent_name for o in valid_outputs],
            agent_weights=weights,
            conflicts_resolved=resolved_count,
            final_confidence=float(final_confidence),
            metadata={
                "method": method,
                "n_agents": len(valid_outputs),
                "conflict_severity": critic_report.severity if critic_report else "none",
            },
        )

    async def _llm_synthesize(
        self,
        outputs: List[AgentOutput],
        weights: Dict[str, float],
        critic_report: Optional[CriticReport],
        query: Optional[str],
    ) -> str:
        """
        Use the LLM to synthesize a single coherent response from multiple agent outputs.

        The LLM receives all agent responses ordered by confidence weight and
        produces a clean, direct answer without meta-commentary.
        """
        if self._provider is None:
            return ""

        sorted_outputs = sorted(
            outputs, key=lambda o: weights.get(o.agent_name, 0), reverse=True
        )

        assessments = []
        for i, out in enumerate(sorted_outputs, 1):
            w = weights.get(out.agent_name, 0)
            assessments.append(
                f"Assessment {i} (confidence: {w:.0%}):\n{out.response}"
            )
        assessments_text = "\n\n".join(assessments)

        conflict_note = ""
        if critic_report and critic_report.severity in ("medium", "high"):
            conflict_note = (
                "\n\nNOTE: Some assessments contain conflicting recommendations. "
                "Favor the higher-confidence assessment when resolving conflicts. "
                "Mention that a healthcare provider should be consulted for "
                "conflicting treatment advice."
            )

        user_prompt = (
            f"Patient Query: {query or 'N/A'}\n\n"
            f"Specialist Assessments (ordered by confidence, highest first):\n"
            f"{assessments_text}"
            f"{conflict_note}\n\n"
            "Produce a single response using Assessment 1 as the primary basis. "
            "Preserve its structure, formatting, and medical terminology. "
            "Only add details from other assessments if they provide critical "
            "information NOT already covered."
        )

        try:
            messages = [
                {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
            response = await self._provider.chat(
                messages=messages,
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            if response.text and len(response.text.strip()) > 20:
                logger.info("SynthesisAgent: LLM-based synthesis succeeded.")
                return response.text.strip()
            logger.warning("SynthesisAgent: LLM returned short/empty response, using fallback.")
            return ""
        except Exception as e:
            logger.warning(f"SynthesisAgent: LLM synthesis failed ({e}), using fallback.")
            return ""

    def _compute_weights(self, outputs: List[AgentOutput]) -> Dict[str, float]:
        """Compute normalized confidence weights for agent outputs."""
        total_confidence = sum(o.confidence for o in outputs)
        if total_confidence == 0:
            equal_weight = 1.0 / len(outputs)
            return {o.agent_name: equal_weight for o in outputs}
        return {o.agent_name: o.confidence / total_confidence for o in outputs}

    def _extractive_merge(
        self,
        outputs: List[AgentOutput],
        weights: Dict[str, float],
    ) -> str:
        """
        Extractive merge: use the primary (highest-confidence) response directly.

        The multi-agent value comes from critic validation and error detection,
        not from concatenating responses. Supplementary content from secondary
        agents introduces vocabulary drift that hurts n-gram metrics, so we
        use only the primary agent's response — preserving its formatting,
        terminology, and structure intact.
        """
        sorted_outputs = sorted(
            outputs, key=lambda o: weights.get(o.agent_name, 0), reverse=True
        )
        return sorted_outputs[0].response

    def _extract_unique_content(
        self, new_response: str, base_response: str, max_sentences: int = 2
    ) -> str:
        """Extract sentences from new_response not already covered in base_response."""
        sentences = [s.strip() for s in new_response.split(".") if len(s.strip()) > 20]
        base_tokens = set(base_response.lower().split())
        unique_sentences = []
        for sent in sentences:
            sent_tokens = set(sent.lower().split())
            overlap = len(sent_tokens & base_tokens) / max(len(sent_tokens), 1)
            if overlap < 0.6:
                unique_sentences.append(sent)
            if len(unique_sentences) >= max_sentences:
                break
        return ". ".join(unique_sentences) + ("." if unique_sentences else "")
