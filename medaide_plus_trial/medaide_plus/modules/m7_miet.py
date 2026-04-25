"""
M7 MIET - Multi-Turn Intent Evolution Tracking

Implements:
  Dialogue state: theta_t = (1-beta)*theta_{t-1} + beta*alpha_t  (beta=0.6)
  Dialogue summary buffer (deque, max 5 turns)
  inject_state(): bias current intent scores with history
  get_context_prefix(): format history for agent prompt
  MietModule with update(), inject_state(), add_to_buffer(), reset()

Reference: arXiv 2410.12532, Section 3.7
"""

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional

import numpy as np

logger = logging.getLogger("medaide_plus.m7_miet")

# All 18 intent labels (same order as m2_hdio.ALL_INTENTS)
ALL_INTENT_LABELS = [
    "Symptom_Triage", "Department_Suggestion", "Risk_Assessment", "Health_Inquiry",
    "Symptom_Analysis", "Etiology_Detection", "Test_Interpretation", "Differential_Diagnosis",
    "Drug_Counseling", "Dosage_Recommendation", "Contraindication_Check", "Drug_Interaction",
    "Prescription_Review", "Rehabilitation_Advice", "Progress_Tracking", "Care_Support",
    "Lifestyle_Guidance", "Follow_up_Scheduling",
]


@dataclass
class DialogueTurn:
    """Represents a single dialogue turn."""
    turn_id: int
    query: str
    response: str
    intent_scores: Dict[str, float] = field(default_factory=dict)
    top_intents: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MietState:
    """Current dialogue state for a session."""
    theta: np.ndarray
    turn_count: int = 0
    session_id: str = ""


class MietModule:
    """
    Multi-Turn Intent Evolution Tracking module.

    Maintains an exponentially smoothed dialogue state vector theta_t
    that tracks the evolution of intent focus across conversation turns.

    State update: theta_t = (1-beta) * theta_{t-1} + beta * alpha_t
    where alpha_t is the normalized intent score vector at turn t.

    Args:
        config: Configuration dict with MIET parameters.
    """

    def __init__(self, config: Optional[Dict] = None) -> None:
        self.config = config or {}
        self.beta: float = self.config.get("beta", 0.6)
        self.buffer_size: int = self.config.get("buffer_size", 5)
        self.state_dim: int = self.config.get("state_dim", len(ALL_INTENT_LABELS))

        # Per-session state storage
        self._states: Dict[str, MietState] = {}
        self._buffers: Dict[str, Deque[DialogueTurn]] = {}

    def _get_or_create_state(self, session_id: str) -> MietState:
        """Get or create dialogue state for a session."""
        if session_id not in self._states:
            self._states[session_id] = MietState(
                theta=np.zeros(self.state_dim, dtype=np.float32),
                turn_count=0,
                session_id=session_id,
            )
        return self._states[session_id]

    def _get_or_create_buffer(self, session_id: str) -> Deque[DialogueTurn]:
        """Get or create summary buffer for a session."""
        if session_id not in self._buffers:
            self._buffers[session_id] = deque(maxlen=self.buffer_size)
        return self._buffers[session_id]

    def update(
        self,
        session_id: str,
        intent_scores: Dict[str, float],
        query: str = "",
        response: str = "",
    ) -> MietState:
        """
        Update dialogue state with new turn's intent scores.

        State update rule:
          theta_t = (1 - beta) * theta_{t-1} + beta * alpha_t

        Args:
            session_id: Unique session/conversation identifier.
            intent_scores: Current turn's per-intent scores from M2 HDIO.
            query: Current turn's query text.
            response: Current turn's response text.

        Returns:
            Updated MietState.
        """
        state = self._get_or_create_state(session_id)

        # Build alpha_t: normalized intent score vector
        alpha_t = np.zeros(self.state_dim, dtype=np.float32)
        for i, intent_label in enumerate(ALL_INTENT_LABELS):
            if i < self.state_dim:
                alpha_t[i] = float(intent_scores.get(intent_label, 0.0))

        # Normalize alpha_t to sum to 1 (if non-zero)
        alpha_sum = alpha_t.sum()
        if alpha_sum > 0:
            alpha_t = alpha_t / alpha_sum

        # Exponential smoothing update
        state.theta = (1.0 - self.beta) * state.theta + self.beta * alpha_t
        state.turn_count += 1

        # Update buffer
        top_intents = self._top_k_intents(intent_scores, k=3)
        turn = DialogueTurn(
            turn_id=state.turn_count,
            query=query,
            response=response,
            intent_scores=intent_scores,
            top_intents=top_intents,
        )
        self.add_to_buffer(session_id, turn)

        logger.debug(
            f"MIET [{session_id}] turn {state.turn_count}: "
            f"top_intents={top_intents}, "
            f"theta_max={float(state.theta.max()):.3f}"
        )
        return state

    def inject_state(
        self,
        session_id: str,
        current_intent_scores: Dict[str, float],
        alpha_inject: float = 0.3,
    ) -> Dict[str, float]:
        """
        Bias current intent scores with the historical dialogue state.

        Biased score: score'_i = (1 - alpha) * score_i + alpha * theta_i

        Args:
            session_id: Session identifier.
            current_intent_scores: Current turn's raw intent scores.
            alpha_inject: Blending weight for historical state (0=no history, 1=all history).

        Returns:
            History-biased intent score dict.
        """
        state = self._get_or_create_state(session_id)

        if state.turn_count == 0:
            return current_intent_scores

        biased_scores: Dict[str, float] = {}
        for i, intent_label in enumerate(ALL_INTENT_LABELS):
            if i < self.state_dim:
                current_score = current_intent_scores.get(intent_label, 0.0)
                history_score = float(state.theta[i])
                biased_scores[intent_label] = (
                    (1.0 - alpha_inject) * current_score
                    + alpha_inject * history_score
                )
            else:
                biased_scores[intent_label] = current_intent_scores.get(intent_label, 0.0)

        return biased_scores

    def get_context_prefix(self, session_id: str, max_turns: int = 3) -> str:
        """
        Format recent dialogue history as a context prefix for agent prompts.

        Args:
            session_id: Session identifier.
            max_turns: Maximum number of recent turns to include.

        Returns:
            Formatted context prefix string.
        """
        buffer = self._get_or_create_buffer(session_id)
        if not buffer:
            return ""

        recent_turns = list(buffer)[-max_turns:]
        parts = ["[Conversation History]"]
        for turn in recent_turns:
            if turn.query:
                parts.append(f"  Turn {turn.turn_id} Query: {turn.query[:100]}")
            if turn.top_intents:
                parts.append(f"  Turn {turn.turn_id} Intents: {', '.join(turn.top_intents)}")
            if turn.response:
                parts.append(f"  Turn {turn.turn_id} Response: {turn.response[:150]}...")
        parts.append("[Current Query]")
        return "\n".join(parts) + "\n"

    def add_to_buffer(self, session_id: str, turn: DialogueTurn) -> None:
        """
        Add a dialogue turn to the session summary buffer.

        The buffer is a fixed-size deque (max buffer_size turns).

        Args:
            session_id: Session identifier.
            turn: DialogueTurn to add.
        """
        buffer = self._get_or_create_buffer(session_id)
        buffer.append(turn)

    def reset(self, session_id: str) -> None:
        """
        Reset the dialogue state and buffer for a session.

        Args:
            session_id: Session identifier to reset.
        """
        self._states[session_id] = MietState(
            theta=np.zeros(self.state_dim, dtype=np.float32),
            turn_count=0,
            session_id=session_id,
        )
        self._buffers[session_id] = deque(maxlen=self.buffer_size)
        logger.info(f"MIET: reset session {session_id}.")

    def get_state_summary(self, session_id: str) -> Dict[str, Any]:
        """
        Get a summary of the current dialogue state.

        Args:
            session_id: Session identifier.

        Returns:
            Dict with theta vector, turn count, top intents.
        """
        state = self._get_or_create_state(session_id)
        top_indices = np.argsort(state.theta)[::-1][:5]
        top_intent_states = [
            {"intent": ALL_INTENT_LABELS[int(i)], "weight": float(state.theta[int(i)])}
            for i in top_indices
            if int(i) < len(ALL_INTENT_LABELS)
        ]
        return {
            "session_id": session_id,
            "turn_count": state.turn_count,
            "top_intent_states": top_intent_states,
            "theta_entropy": float(self._entropy(state.theta)),
        }

    def _top_k_intents(self, intent_scores: Dict[str, float], k: int = 3) -> List[str]:
        """Return top-k intent labels by score."""
        sorted_intents = sorted(intent_scores.items(), key=lambda x: x[1], reverse=True)
        return [label for label, _ in sorted_intents[:k]]

    def _entropy(self, probs: np.ndarray) -> float:
        """Compute entropy of a probability distribution."""
        p = probs + 1e-9
        p = p / p.sum()
        return float(-np.sum(p * np.log(p)))

# Aliases for backward compatibility
MIETModule = MietModule
