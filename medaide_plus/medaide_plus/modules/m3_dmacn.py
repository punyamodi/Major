"""
M3 DMACN — Dynamic Multi-Agent Critic Network

Implements:
  1. asyncio.gather() for parallel multi-agent execution
  2. CriticAgent: detects contradictions (drug dosage, treatment conflicts)
  3. SynthesisAgent: confidence-weighted merge, conflict resolution
  4. Per-agent confidence scoring

Reference: arXiv 2410.12532, Section 3.3
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("medaide_plus.m3_dmacn")


@dataclass
class AgentOutput:
    """Standardized output from a single agent."""
    agent_name: str
    response: str
    confidence: float = 0.0
    latency_ms: float = 0.0
    intents: List[str] = field(default_factory=list)
    claims: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class CriticReport:
    """Structured critique report from the CriticAgent."""
    conflicts: List[Dict[str, str]] = field(default_factory=list)
    severity: str = "low"  # "low" | "medium" | "high"
    conflict_agents: List[Tuple[str, str]] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class DMACNResult:
    """Final result from the DMACN module."""
    synthesized_response: str
    agent_outputs: List[AgentOutput] = field(default_factory=list)
    critic_report: Optional[CriticReport] = None
    final_confidence: float = 0.0
    agents_used: List[str] = field(default_factory=list)
    total_latency_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class DMACNModule:
    """
    Dynamic Multi-Agent Critic Network.

    Orchestrates parallel execution of multiple medical domain agents,
    applies critic-based conflict detection, and synthesizes a final
    confidence-weighted response.

    Args:
        agents: List of agent instances (must implement async analyze()).
        config: Configuration dict with DMACN parameters.
    """

    # Conflict detection patterns (drug dosage and treatment conflicts)
    DOSAGE_CONFLICT_PATTERNS = [
        r"(\d+)\s*mg.*?(\d+)\s*mg",  # Multiple dosage mentions
        r"take\s+(\d+).*?take\s+(\d+)",  # Conflicting take instructions
    ]
    TREATMENT_CONFLICT_KEYWORDS = [
        ("avoid", "recommend"),
        ("do not", "should"),
        ("contraindicated", "prescribed"),
        ("increase", "decrease"),
        ("stop", "continue"),
    ]

    def __init__(
        self,
        agents: Optional[List] = None,
        config: Optional[Dict] = None,
    ) -> None:
        self.agents = agents or []
        self.config = config or {}
        self.timeout: float = self.config.get("timeout", 60.0)
        self.confidence_weight: bool = self.config.get("confidence_weight", True)
        self.max_agents: int = self.config.get("max_agents", 4)

    async def run(
        self,
        query: str,
        context: Optional[Dict] = None,
        n_agents: int = 4,
    ) -> DMACNResult:
        """
        Execute the full DMACN pipeline asynchronously.

        Steps:
          1. Run up to n_agents in parallel via asyncio.gather()
          2. CriticAgent checks all outputs for contradictions
          3. SynthesisAgent merges outputs with confidence weighting

        Args:
            query: Medical query string.
            context: Optional context dict (patient history, intents, etc.).
            n_agents: Number of agents to activate (1, 2, or 4).

        Returns:
            DMACNResult with synthesized response and full metadata.
        """
        start_time = time.time()
        context = context or {}
        active_agents = self.agents[:min(n_agents, self.max_agents, len(self.agents))]

        if not active_agents:
            logger.warning("No agents available; returning empty result.")
            return DMACNResult(
                synthesized_response="No agents available to process query.",
                final_confidence=0.0,
            )

        logger.info(
            f"DMACN: running {len(active_agents)} agents in parallel "
            f"for query: {query[:60]}..."
        )

        # Step 1: Parallel agent execution
        agent_outputs = await self._run_agents_parallel(
            active_agents, query, context
        )

        # Step 2: Critic evaluation
        critic_report = self._run_critic(agent_outputs)

        # Step 3: Synthesis
        synthesized = self._synthesize(agent_outputs, critic_report)

        total_latency = (time.time() - start_time) * 1000

        return DMACNResult(
            synthesized_response=synthesized,
            agent_outputs=agent_outputs,
            critic_report=critic_report,
            final_confidence=self._compute_ensemble_confidence(agent_outputs),
            agents_used=[a.agent_name for a in agent_outputs if not a.error],
            total_latency_ms=total_latency,
            metadata={
                "n_conflicts": len(critic_report.conflicts),
                "severity": critic_report.severity,
                "n_agents_successful": sum(1 for a in agent_outputs if not a.error),
            },
        )

    async def _run_agents_parallel(
        self,
        agents: List,
        query: str,
        context: Dict,
    ) -> List[AgentOutput]:
        """
        Run agents in parallel with asyncio.gather() and individual timeouts.

        Args:
            agents: List of agent instances.
            query: Query string.
            context: Context dictionary.

        Returns:
            List of AgentOutput objects (one per agent).
        """
        async def run_single(agent) -> AgentOutput:
            agent_start = time.time()
            try:
                output = await asyncio.wait_for(
                    agent.analyze(query, context),
                    timeout=self.timeout,
                )
                output.latency_ms = (time.time() - agent_start) * 1000
                return output
            except asyncio.TimeoutError:
                logger.warning(f"Agent {getattr(agent, 'name', 'unknown')} timed out.")
                return AgentOutput(
                    agent_name=getattr(agent, "name", "unknown"),
                    response="",
                    confidence=0.0,
                    error="timeout",
                    latency_ms=(time.time() - agent_start) * 1000,
                )
            except Exception as e:
                logger.error(f"Agent error: {e}")
                return AgentOutput(
                    agent_name=getattr(agent, "name", "unknown"),
                    response="",
                    confidence=0.0,
                    error=str(e),
                    latency_ms=(time.time() - agent_start) * 1000,
                )

        tasks = [run_single(agent) for agent in agents]
        results = await asyncio.gather(*tasks, return_exceptions=False)
        return list(results)

    def _run_critic(self, agent_outputs: List[AgentOutput]) -> CriticReport:
        """
        Check agent outputs for contradictions and conflicts.

        Detects:
          - Drug dosage contradictions (conflicting dose values)
          - Treatment direction conflicts (one agent says avoid, another recommends)
          - Contradictory medication instructions

        Args:
            agent_outputs: List of outputs from parallel agents.

        Returns:
            CriticReport with structured conflict information.
        """
        import re
        conflicts: List[Dict[str, str]] = []
        conflict_agents: List[Tuple[str, str]] = []

        # Filter successful outputs
        valid_outputs = [o for o in agent_outputs if not o.error and o.response]
        if len(valid_outputs) < 2:
            return CriticReport(severity="low")

        # Pairwise conflict detection
        for i in range(len(valid_outputs)):
            for j in range(i + 1, len(valid_outputs)):
                out_i = valid_outputs[i]
                out_j = valid_outputs[j]

                # Check dosage conflicts
                dosage_conflict = self._check_dosage_conflict(
                    out_i.response, out_j.response
                )
                if dosage_conflict:
                    conflicts.append({
                        "type": "dosage",
                        "agent_a": out_i.agent_name,
                        "agent_b": out_j.agent_name,
                        "description": dosage_conflict,
                    })
                    conflict_agents.append((out_i.agent_name, out_j.agent_name))

                # Check treatment direction conflicts
                treatment_conflict = self._check_treatment_conflict(
                    out_i.response, out_j.response
                )
                if treatment_conflict:
                    conflicts.append({
                        "type": "treatment",
                        "agent_a": out_i.agent_name,
                        "agent_b": out_j.agent_name,
                        "description": treatment_conflict,
                    })

        # Determine severity
        severity = "low"
        if len(conflicts) >= 3:
            severity = "high"
        elif len(conflicts) >= 1:
            severity = "medium"

        # Generate recommendations
        recommendations = []
        for conflict in conflicts:
            if conflict["type"] == "dosage":
                recommendations.append(
                    f"Resolve dosage discrepancy between {conflict['agent_a']} "
                    f"and {conflict['agent_b']}. Consult authoritative drug reference."
                )
            elif conflict["type"] == "treatment":
                recommendations.append(
                    f"Reconcile treatment guidance from {conflict['agent_a']} "
                    f"and {conflict['agent_b']}."
                )

        logger.debug(
            f"Critic found {len(conflicts)} conflicts (severity={severity})."
        )
        return CriticReport(
            conflicts=conflicts,
            severity=severity,
            conflict_agents=conflict_agents,
            recommendations=recommendations,
        )

    def _check_dosage_conflict(self, text_a: str, text_b: str) -> Optional[str]:
        """
        Detect conflicting dosage recommendations.

        Args:
            text_a: First agent response.
            text_b: Second agent response.

        Returns:
            Conflict description string if found, else None.
        """
        import re
        # Extract dosages (e.g., "500mg", "200 mg")
        pattern = r"(\d+\.?\d*)\s*mg"
        doses_a = set(float(d) for d in re.findall(pattern, text_a.lower()))
        doses_b = set(float(d) for d in re.findall(pattern, text_b.lower()))

        if doses_a and doses_b:
            # Check for significant dosage differences (>2x ratio)
            all_doses = list(doses_a) + list(doses_b)
            if max(all_doses) > 2.0 * min(all_doses):
                return (
                    f"Dosage conflict: {doses_a} mg vs {doses_b} mg. "
                    f"Max/min ratio exceeds 2x."
                )
        return None

    def _check_treatment_conflict(self, text_a: str, text_b: str) -> Optional[str]:
        """
        Detect conflicting treatment directions.

        Args:
            text_a: First agent response.
            text_b: Second agent response.

        Returns:
            Conflict description string if found, else None.
        """
        text_a_lower = text_a.lower()
        text_b_lower = text_b.lower()

        for keyword_pos, keyword_neg in self.TREATMENT_CONFLICT_KEYWORDS:
            a_has_pos = keyword_pos in text_a_lower
            a_has_neg = keyword_neg in text_a_lower
            b_has_pos = keyword_pos in text_b_lower
            b_has_neg = keyword_neg in text_b_lower

            if (a_has_pos and b_has_neg) or (a_has_neg and b_has_pos):
                return (
                    f"Treatment direction conflict: one agent uses '{keyword_pos}', "
                    f"another uses '{keyword_neg}'."
                )
        return None

    def _synthesize(
        self,
        agent_outputs: List[AgentOutput],
        critic_report: CriticReport,
    ) -> str:
        """
        Synthesize a final response using confidence-weighted merging.

        For conflicting claims, the higher-confidence agent's response takes
        precedence. Non-conflicting content is merged additively.

        Args:
            agent_outputs: List of agent outputs.
            critic_report: Critic's conflict report.

        Returns:
            Synthesized response string.
        """
        valid_outputs = [o for o in agent_outputs if not o.error and o.response]
        if not valid_outputs:
            return "Unable to generate response — all agents failed."

        if len(valid_outputs) == 1:
            return valid_outputs[0].response

        # Sort by confidence (descending)
        sorted_outputs = sorted(valid_outputs, key=lambda o: o.confidence, reverse=True)

        if self.confidence_weight:
            response = self._confidence_weighted_merge(
                sorted_outputs, critic_report
            )
        else:
            # Simple: return highest-confidence response
            response = sorted_outputs[0].response

        return response

    def _confidence_weighted_merge(
        self,
        outputs: List[AgentOutput],
        critic_report: CriticReport,
    ) -> str:
        """
        Merge responses using confidence weights.

        Strategy:
          1. Start with the highest-confidence agent's response as base.
          2. Append unique high-confidence insights from other agents.
          3. Flag any sections involved in critic conflicts.

        Args:
            outputs: Sorted list of AgentOutput (descending confidence).
            critic_report: Critic's analysis.

        Returns:
            Merged response string.
        """
        conflict_agents = set()
        for ca, cb in critic_report.conflict_agents:
            conflict_agents.add(ca)
            conflict_agents.add(cb)

        # Build weighted response
        base_response = outputs[0].response
        supplementary_parts = []

        total_weight = sum(o.confidence for o in outputs)
        for output in outputs[1:]:
            weight = output.confidence / max(total_weight, 1e-9)
            if weight < 0.1:
                continue  # Skip very low-confidence agents

            # Add unique content not in base response
            unique_sentences = self._extract_unique_sentences(
                output.response, base_response
            )
            if unique_sentences and output.agent_name not in conflict_agents:
                supplementary_parts.append(
                    f"\n[Additional insight from {output.agent_name}]: "
                    + " ".join(unique_sentences)
                )

        # Add conflict warnings
        if critic_report.conflicts and critic_report.severity in ("medium", "high"):
            warning = (
                "\n⚠️ Note: Some recommendations from different agents may conflict. "
                "Please consult a medical professional for authoritative guidance."
            )
            supplementary_parts.append(warning)

        return base_response + "".join(supplementary_parts)

    def _extract_unique_sentences(
        self, new_text: str, base_text: str, similarity_threshold: float = 0.7
    ) -> List[str]:
        """
        Extract sentences from new_text that are not already in base_text.

        Uses simple word overlap to estimate uniqueness.

        Args:
            new_text: Text to extract unique sentences from.
            base_text: Existing base text to compare against.
            similarity_threshold: Maximum allowed overlap to be considered unique.

        Returns:
            List of unique sentence strings.
        """
        new_sentences = [s.strip() for s in new_text.split(".") if len(s.strip()) > 20]
        base_words = set(base_text.lower().split())
        unique = []
        for sent in new_sentences:
            sent_words = set(sent.lower().split())
            if not sent_words:
                continue
            overlap = len(sent_words & base_words) / len(sent_words)
            if overlap < similarity_threshold:
                unique.append(sent)
        return unique[:2]  # Limit to 2 supplementary sentences

    def _compute_ensemble_confidence(
        self, agent_outputs: List[AgentOutput]
    ) -> float:
        """
        Compute ensemble confidence as weighted average of agent confidences.

        Args:
            agent_outputs: List of all agent outputs.

        Returns:
            Ensemble confidence score in [0, 1].
        """
        valid = [o for o in agent_outputs if not o.error and o.confidence > 0]
        if not valid:
            return 0.0
        weights = [o.confidence for o in valid]
        total = sum(weights)
        if total == 0:
            return 0.0
        return float(sum(w * o.confidence for w, o in zip(weights, valid)) / total)
