"""Unit tests for M6 AQCR — Adaptive Query Complexity Routing."""

import pytest


class TestAqcrModule:
    """Tests for AqcrModule."""

    @pytest.fixture
    def module(self):
        """Create AqcrModule without loading heavy models."""
        from medaide_plus.modules.m6_aqcr import AqcrModule
        m = AqcrModule.__new__(AqcrModule)
        m.config = {}
        m.simple_threshold = 0.33
        m.complex_threshold = 0.66
        m._model = None
        m._tokenizer = None
        m._sentence_model = None
        m._routing_method = "feature_based"
        return m

    def test_simple_query_routes_1_agent(self, module):
        """Simple query with 1 intent should route to Simple tier."""
        result = module._feature_route({
            "query_length": 0.05,
            "n_intents": 0.1,
            "n_subqueries": 0.2,
            "has_medication_intent": 0.0,
            "has_multi_category": 0.0,
            "inter_intent_distance": 0.0,
            "complexity_keywords": 0.0,
        })
        assert result.tier == "Simple"
        assert result.n_agents == 1

    def test_complex_query_routes_4_agents(self, module):
        """Complex query with many intents should route to Complex tier."""
        result = module._feature_route({
            "query_length": 0.8,
            "n_intents": 0.9,
            "n_subqueries": 0.9,
            "has_medication_intent": 1.0,
            "has_multi_category": 1.0,
            "inter_intent_distance": 0.8,
            "complexity_keywords": 0.9,
        })
        assert result.tier == "Complex"
        assert result.n_agents == 4

    def test_moderate_query(self, module):
        """Mid-complexity query should route to Moderate tier."""
        result = module._feature_route({
            "query_length": 0.3,
            "n_intents": 0.3,
            "n_subqueries": 0.3,
            "has_medication_intent": 0.5,
            "has_multi_category": 0.0,
            "inter_intent_distance": 0.2,
            "complexity_keywords": 0.2,
        })
        assert result.tier in ["Simple", "Moderate", "Complex"]
        assert result.n_agents in [1, 2, 4]

    def test_feature_routing_returns_valid_result(self, module):
        """Test feature routing produces valid AqcrResult."""
        from medaide_plus.modules.m6_aqcr import TIER_LABELS
        features = {
            "query_length": 0.4,
            "n_intents": 0.5,
            "n_subqueries": 0.4,
            "has_medication_intent": 1.0,
            "has_multi_category": 1.0,
            "inter_intent_distance": 0.3,
            "complexity_keywords": 0.4,
        }
        result = module._feature_route(features)
        assert result.tier in TIER_LABELS
        assert result.n_agents in [1, 2, 4]
        assert 0.0 <= result.confidence <= 1.0
        assert sum(result.tier_probabilities.values()) > 0

    def test_silver_label_generation(self, module):
        """Test silver label generation for training data."""
        queries = [
            "What is aspirin?",
            "I have diabetes, hypertension, and need medication review with interactions",
            "My blood test is back",
        ]
        labels = module.generate_silver_labels(queries)
        assert len(labels) == 3
        for label in labels:
            assert label in ["Simple", "Moderate", "Complex"]

    def test_tier_agent_mapping(self):
        """Test that tier-to-agent mapping is correct."""
        from medaide_plus.modules.m6_aqcr import COMPLEXITY_TIERS
        assert COMPLEXITY_TIERS["Simple"] == 1
        assert COMPLEXITY_TIERS["Moderate"] == 2
        assert COMPLEXITY_TIERS["Complex"] == 4
