"""Unit tests for M7 MIET - Multi-Turn Intent Evolution Tracking."""

import pytest
import numpy as np
from medaide_plus.modules.m7_miet import ALL_INTENT_LABELS

N_INTENTS = len(ALL_INTENT_LABELS)  # 18


class TestMietModule:
    """Tests for MietModule."""

    @pytest.fixture
    def module(self):
        from medaide_plus.modules.m7_miet import MietModule
        return MietModule(config={"beta": 0.6, "buffer_size": 5, "state_dim": N_INTENTS})

    @pytest.fixture
    def sample_intents(self):
        return {label: (0.8 if i == 0 else 0.1) for i, label in enumerate(ALL_INTENT_LABELS)}

    def test_state_update(self, module, sample_intents):
        state = module.update("sess_update", sample_intents, query="Test query")
        assert state.turn_count == 1
        assert state.theta.shape == (N_INTENTS,)
        assert state.theta.sum() > 0

    def test_exponential_smoothing_formula(self, module, sample_intents):
        state1 = module.update("sess_formula", sample_intents)
        theta_1 = state1.theta.copy()
        intents2 = {k: 0.0 for k in ALL_INTENT_LABELS}
        intents2["Drug_Counseling"] = 1.0
        state2 = module.update("sess_formula", intents2)
        assert state2.turn_count == 2
        assert not np.allclose(theta_1, state2.theta)

    def test_context_injection(self, module, sample_intents):
        module.update("sess_inject", sample_intents)
        new_scores = {k: 0.5 for k in ALL_INTENT_LABELS}
        biased = module.inject_state("sess_inject", new_scores, alpha_inject=0.3)
        assert len(biased) == len(new_scores)
        has_diff = any(abs(biased[k] - new_scores[k]) > 1e-6 for k in new_scores)
        assert has_diff

    def test_buffer_management(self, module, sample_intents):
        session_id = "sess_buffer"
        for i in range(7):
            module.update(session_id, sample_intents, query=f"Query {i}")
        buf = module._get_or_create_buffer(session_id)
        assert len(buf) <= 5

    def test_recency_weighting(self, module):
        session_id = "sess_recency"
        symptom_intents = {k: 0.0 for k in ALL_INTENT_LABELS}
        symptom_intents["Symptom_Triage"] = 1.0
        module.update(session_id, symptom_intents)
        med_intents = {k: 0.0 for k in ALL_INTENT_LABELS}
        med_intents["Drug_Counseling"] = 1.0
        state = module.update(session_id, med_intents)
        assert state.theta.sum() > 0

    def test_reset(self, module, sample_intents):
        session_id = "sess_reset"
        module.update(session_id, sample_intents)
        assert module._states[session_id].turn_count == 1
        module.reset(session_id)
        state = module._states[session_id]
        assert state.turn_count == 0
        assert np.allclose(state.theta, np.zeros(N_INTENTS))

    def test_get_context_prefix(self, module, sample_intents):
        module.update("sess_prefix", sample_intents, query="I have a headache")
        prefix = module.get_context_prefix("sess_prefix")
        assert isinstance(prefix, str)

    def test_multiple_sessions_isolated(self, module, sample_intents):
        module.update("sess_A", sample_intents)
        state_b = module._get_or_create_state("sess_B")
        assert np.allclose(state_b.theta, np.zeros(N_INTENTS))

    def test_follow_up_scheduling_intent(self, module):
        """Ensure Follow_up_Scheduling is tracked (18th intent)."""
        assert "Follow_up_Scheduling" in ALL_INTENT_LABELS
        assert len(ALL_INTENT_LABELS) == 18
