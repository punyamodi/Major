"""Unit tests for M2 HDIO - Hierarchical Dual-level Intent Ontology."""

import os
import pytest
import numpy as np

os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["USE_TORCH"] = "1"


class TestHDIOModule:
    """Tests for HDIOModule (runs fully offline with TF-IDF fallback)."""

    @pytest.fixture(scope="class")
    def module(self):
        """Build HDIOModule in fully-offline TF-IDF mode (no network calls)."""
        from medaide_plus.modules.m2_hdio import (
            HDIOModule, GATLayer, ALL_INTENTS, INTENT_ONTOLOGY,
        )
        m = HDIOModule.__new__(HDIOModule)
        m.config = {}
        m.ood_threshold = 0.4
        m.n_categories = len(INTENT_ONTOLOGY)     # 4
        m.n_intents = len(ALL_INTENTS)             # 18
        m.gat_heads = 4
        m.gat_hidden = 128

        # TF-IDF encoder — _tfidf_encode always pads to _encoder_dim
        m._encoder_type = "tfidf"
        m._encoder_dim = 256
        m._tokenizer = None
        m._encoder = None

        # GAT layer
        m._gat_layer = GATLayer(in_features=256, out_features=128, n_heads=4)

        # Classification heads (shape: gat_hidden × n_categories/n_intents)
        rng = np.random.default_rng(0)
        m._category_head = rng.standard_normal((128, m.n_categories)).astype(np.float32) * 0.1
        m._intent_head   = rng.standard_normal((128, m.n_intents)).astype(np.float32) * 0.1

        # Intent graph adjacency matrix
        m._intent_graph = m._build_intent_graph()
        return m

    def test_symptom_query_routing(self, module):
        from medaide_plus.modules.m2_hdio import ALL_INTENTS
        result = module.classify("I have severe headache and fever for 3 days")
        assert result.top_category in ["Pre-Diagnosis", "Diagnosis"]
        assert len(result.intent_scores) == len(ALL_INTENTS)
        assert result.confidence >= 0.0

    def test_medication_query(self, module):
        result = module.classify("What is the correct dosage of ibuprofen? Drug interactions?")
        assert "Medication" in result.category_scores
        assert result.category_scores["Medication"] > 0

    def test_ood_detection(self, module):
        result = module.classify("What is the weather like today?")
        assert result.ood_score >= 0.0
        assert 0.0 <= result.confidence <= 1.0

    def test_multilabel_activation(self, module):
        from medaide_plus.modules.m2_hdio import ALL_INTENTS
        result = module.classify(
            "I have a headache and need medication dosage advice for aspirin"
        )
        assert len(result.intent_scores) == len(ALL_INTENTS)
        for intent, score in result.intent_scores.items():
            assert 0.0 <= score <= 1.0, f"{intent}: {score} out of range"

    def test_hierarchical_classification(self, module):
        result = module.classify("I need rehabilitation advice after my surgery")
        assert "Post-Diagnosis" in result.category_scores
        from medaide_plus.modules.m2_hdio import INTENT_ONTOLOGY
        post_intents = INTENT_ONTOLOGY["Post-Diagnosis"]
        max_intent = max(result.intent_scores.get(i, 0) for i in post_intents)
        assert abs(result.category_scores["Post-Diagnosis"] - max_intent) < 1e-6

    def test_intent_graph_structure(self, module):
        from medaide_plus.modules.m2_hdio import INTENT_ONTOLOGY, ALL_INTENTS
        n_total = len(INTENT_ONTOLOGY) + len(ALL_INTENTS)
        assert module._intent_graph.shape == (n_total, n_total)
        for i in range(n_total):
            assert module._intent_graph[i, i] == 1.0

    def test_all_intents_present(self, module):
        from medaide_plus.modules.m2_hdio import ALL_INTENTS
        result = module.classify("I have a medical question about my treatment")
        for intent in ALL_INTENTS:
            assert intent in result.intent_scores, f"Missing intent: {intent}"
        assert len(ALL_INTENTS) == 18

    def test_follow_up_scheduling_in_ontology(self):
        from medaide_plus.modules.m2_hdio import INTENT_ONTOLOGY, ALL_INTENTS
        assert "Follow_up_Scheduling" in ALL_INTENTS
        assert "Follow_up_Scheduling" in INTENT_ONTOLOGY["Post-Diagnosis"]
