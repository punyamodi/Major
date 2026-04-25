"""
CriticAgent — Detects contradictions and conflicts in multi-agent outputs.
Returns structured JSON critique reports.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from medaide_plus.agents.base_agent import AgentOutput

logger = logging.getLogger("medaide_plus.critic_agent")

CONFLICT_PATTERNS = [
    (r"take\s+(\d+(?:\.\d+)?)\s*mg", r"take\s+(\d+(?:\.\d+)?)\s*mg"),
    (r"(\d+)\s*times\s+(?:a|per)\s+day", r"(\d+)\s*times\s+(?:a|per)\s+day"),
]

CONTRADICTION_PAIRS = [
    ("avoid", "take"),
    ("contraindicated", "recommended"),
    ("do not use", "use"),
    ("stop", "continue"),
    ("increase", "decrease"),
    ("high dose", "low dose"),
]


@dataclass
class CriticReport:
    """Structured critique report from the CriticAgent."""
    conflicts: List[Dict[str, Any]] = field(default_factory=list)
    severity: str = "low"
    conflict_agents: List[Tuple[str, str]] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    summary: str = ""


class CriticAgent:
    """
    Critic agent that analyzes multi-agent outputs for contradictions.

    Detects:
      - Dosage conflicts (e.g., 100mg vs 500mg for same drug)
      - Treatment direction conflicts (avoid vs. recommend)
      - Logical contradictions between agent responses

    Args:
        config: Configuration dict.
        llm_client: Optional OpenAI async client for LLM-based critique.
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
        self.name = "CriticAgent"

    async def evaluate(self, agent_outputs: List[AgentOutput]) -> CriticReport:
        """
        Evaluate a list of agent outputs for conflicts and contradictions.

        Args:
            agent_outputs: Outputs from parallel domain agents.

        Returns:
            CriticReport with structured conflict information.
        """
        valid_outputs = [o for o in agent_outputs if not o.error and o.response]

        if len(valid_outputs) < 2:
            return CriticReport(
                severity="low",
                summary="Single agent output — no conflicts possible.",
            )

        conflicts: List[Dict[str, Any]] = []
        conflict_agents: List[Tuple[str, str]] = []

        for i in range(len(valid_outputs)):
            for j in range(i + 1, len(valid_outputs)):
                out_i = valid_outputs[i]
                out_j = valid_outputs[j]

                dosage_conflict = self._check_dosage_conflict(out_i, out_j)
                if dosage_conflict:
                    conflicts.append(dosage_conflict)
                    conflict_agents.append((out_i.agent_name, out_j.agent_name))

                treatment_conflict = self._check_treatment_conflict(out_i, out_j)
                if treatment_conflict:
                    conflicts.append(treatment_conflict)

                logical_conflict = self._check_logical_contradiction(out_i, out_j)
                if logical_conflict:
                    conflicts.append(logical_conflict)

        severity = self._compute_severity(conflicts)
        recommendations = self._generate_recommendations(conflicts)
        summary = self._generate_summary(conflicts, severity, valid_outputs)

        logger.info(
            f"CriticAgent: {len(conflicts)} conflicts detected (severity={severity}), "
            f"agents={[o.agent_name for o in valid_outputs]}."
        )

        return CriticReport(
            conflicts=conflicts,
            severity=severity,
            conflict_agents=conflict_agents,
            recommendations=recommendations,
            summary=summary,
        )

    def _check_dosage_conflict(
        self, out_a: AgentOutput, out_b: AgentOutput
    ) -> Optional[Dict[str, Any]]:
        """Check for conflicting dosage recommendations."""
        pattern = r"(\d+\.?\d*)\s*mg"
        doses_a = [float(d) for d in re.findall(pattern, out_a.response.lower())]
        doses_b = [float(d) for d in re.findall(pattern, out_b.response.lower())]

        if doses_a and doses_b:
            all_doses = doses_a + doses_b
            if max(all_doses) > 0 and max(all_doses) / max(min(all_doses), 0.01) > 2.0:
                return {
                    "type": "dosage",
                    "severity": "high",
                    "agent_a": out_a.agent_name,
                    "agent_b": out_b.agent_name,
                    "description": f"Dosage conflict: {set(doses_a)} mg vs {set(doses_b)} mg",
                    "detail": "Significant dosage discrepancy (>2x ratio) detected.",
                }
        return None

    def _check_treatment_conflict(
        self, out_a: AgentOutput, out_b: AgentOutput
    ) -> Optional[Dict[str, Any]]:
        """Check for contradictory treatment directions."""
        text_a = out_a.response.lower()
        text_b = out_b.response.lower()

        for keyword_neg, keyword_pos in CONTRADICTION_PAIRS:
            if keyword_neg in text_a and keyword_pos in text_b:
                return {
                    "type": "treatment_direction",
                    "severity": "medium",
                    "agent_a": out_a.agent_name,
                    "agent_b": out_b.agent_name,
                    "description": (
                        f"Treatment conflict: {out_a.agent_name} says '{keyword_neg}' "
                        f"but {out_b.agent_name} says '{keyword_pos}'."
                    ),
                }
            if keyword_pos in text_a and keyword_neg in text_b:
                return {
                    "type": "treatment_direction",
                    "severity": "medium",
                    "agent_a": out_a.agent_name,
                    "agent_b": out_b.agent_name,
                    "description": (
                        f"Treatment conflict: {out_a.agent_name} says '{keyword_pos}' "
                        f"but {out_b.agent_name} says '{keyword_neg}'."
                    ),
                }
        return None

    def _check_logical_contradiction(
        self, out_a: AgentOutput, out_b: AgentOutput
    ) -> Optional[Dict[str, Any]]:
        """Check for logical contradictions using negation patterns."""
        text_a = out_a.response.lower()
        text_b = out_b.response.lower()
        negation_words = ["not", "no", "never", "neither", "nor"]

        sentences_a = [s.strip() for s in re.split(r"[.!?]+", text_a) if s.strip()]
        sentences_b = [s.strip() for s in re.split(r"[.!?]+", text_b) if s.strip()]

        for sent_a in sentences_a[:5]:
            tokens_a = set(sent_a.split())
            has_neg_a = bool(tokens_a & set(negation_words))
            for sent_b in sentences_b[:5]:
                tokens_b = set(sent_b.split())
                has_neg_b = bool(tokens_b & set(negation_words))
                overlap = tokens_a & tokens_b - set(negation_words) - {"a", "the", "is", "are"}
                if len(overlap) >= 4 and has_neg_a != has_neg_b:
                    return {
                        "type": "logical",
                        "severity": "medium",
                        "agent_a": out_a.agent_name,
                        "agent_b": out_b.agent_name,
                        "description": (
                            f"Logical contradiction: '{sent_a[:80]}' vs '{sent_b[:80]}'"
                        ),
                    }
        return None

    def _compute_severity(self, conflicts: List[Dict]) -> str:
        """Compute overall severity from conflict list."""
        if not conflicts:
            return "low"
        high_count = sum(1 for c in conflicts if c.get("severity") == "high")
        if high_count >= 1 or len(conflicts) >= 3:
            return "high"
        if len(conflicts) >= 1:
            return "medium"
        return "low"

    def _generate_recommendations(self, conflicts: List[Dict]) -> List[str]:
        """Generate actionable recommendations for each conflict."""
        recs = []
        for conflict in conflicts:
            if conflict["type"] == "dosage":
                recs.append(
                    "Verify dosage with authoritative drug reference (FDA label, Lexicomp). "
                    "Consult prescribing physician."
                )
            elif conflict["type"] == "treatment_direction":
                recs.append(
                    "Reconcile conflicting treatment guidance by checking clinical guidelines. "
                    f"Agents: {conflict['agent_a']} vs {conflict['agent_b']}."
                )
            elif conflict["type"] == "logical":
                recs.append(
                    "Review logical inconsistency between agent outputs. "
                    "Human clinical review recommended."
                )
        return list(set(recs))

    def _generate_summary(
        self,
        conflicts: List[Dict],
        severity: str,
        outputs: List[AgentOutput],
    ) -> str:
        """Generate human-readable summary of critic findings."""
        if not conflicts:
            return (
                f"No conflicts detected among {len(outputs)} agent outputs. "
                f"Responses appear consistent."
            )
        return (
            f"Found {len(conflicts)} conflict(s) (severity: {severity}) "
            f"among {len(outputs)} agent outputs. "
            f"Types: {', '.join(set(c['type'] for c in conflicts))}. "
            f"Human review recommended for high-severity conflicts."
        )

    def to_json(self, report: CriticReport) -> str:
        """Serialize CriticReport to JSON string."""
        return json.dumps(
            {
                "conflicts": report.conflicts,
                "severity": report.severity,
                "conflict_agents": [list(pair) for pair in report.conflict_agents],
                "recommendations": report.recommendations,
                "summary": report.summary,
            },
            indent=2,
        )
