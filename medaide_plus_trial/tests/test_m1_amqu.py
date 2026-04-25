"""Unit tests for M1 AMQU — Adaptive Multi-shot Query Understanding."""

import math
import time
import pytest
import numpy as np


class TestAMQUModule:
    """Tests for AMQUModule."""

    @pytest.fixture
    def module(self):
        """Create AMQUModule with minimal config (no heavy models)."""
        from medaide_plus.modules.m1_amqu import AMQUModule
        # Force rule-based mode by not loading models
        m = AMQUModule.__new__(AMQUModule)
        m.config = {}
        m.k_shots = 5
        m.consistency_threshold = 0.85
        m.min_count = 3
        m.recency_lambda = 0.1
        m.max_subqueries = 5
        m._model = None
        m._tokenizer = None
        m._sentence_model = None
        m._bm25 = None
        m._corpus = []
        m._corpus_timestamps = []
        return m

    def test_basic_decomposition(self, module):
        """Test that a complex query is decomposed into subqueries."""
        from medaide_plus.modules.m1_amqu import AMQUModule
        query = "I have a headache and also feel nauseous. What could be wrong?"
        result = module._rule_based_decomposition(query)
        assert len(result) >= 1
        for sq in result:
            assert len(sq.text) > 5
            assert sq.source == "rule_based"

    def test_consistency_filtering_basic(self, module):
        """Test that consistency filtering returns subqueries."""
        from medaide_plus.modules.m1_amqu import SubQuery
        sq1 = SubQuery(text="I have a headache", confidence=0.9, source="model")
        sq2 = SubQuery(text="I have a headache and fever", confidence=0.85, source="model")
        sq3 = SubQuery(text="What medication for pain?", confidence=0.8, source="model")
        candidate_sets = [[sq1, sq2], [sq1, sq3], [sq2, sq3]]
        result = module._filter_by_consistency(candidate_sets)
        assert isinstance(result, list)
        # Should return at least one subquery
        assert len(result) >= 1

    def test_recency_bm25(self, module):
        """Test recency-weighted BM25 scoring."""
        import time as t
        now = t.time()
        corpus = [
            "headache pain relief medication",
            "fever temperature management",
            "nausea vomiting treatment",
        ]
        timestamps = [now - 86400, now - 3600, now]  # 1 day, 1 hour, now
        module._build_bm25_index(corpus, timestamps)
        results = module._recency_weighted_bm25("headache pain", top_k=3)
        assert len(results) <= 3
        for doc, score in results:
            assert isinstance(score, float)
            assert score >= 0.0

    def test_handles_short_query(self, module):
        """Test that a short query is handled without errors."""
        from medaide_plus.modules.m1_amqu import AMQUModule
        result = module.run("headache")
        assert result.original_query == "headache"
        assert isinstance(result.subqueries, list)

    def test_handles_complex_query(self, module):
        """Test decomposition of a complex multi-part query."""
        query = (
            "I have been experiencing chest pain for 2 days, also have shortness of "
            "breath, and my blood pressure is high. Furthermore, I take metformin for "
            "diabetes. What should I do and which doctor should I see?"
        )
        candidate_sets = module._generate_subqueries(query, k=3)
        assert len(candidate_sets) > 0
        assert all(len(cs) > 0 for cs in candidate_sets)

    def test_cosine_sim_matrix(self, module):
        """Test cosine similarity matrix computation."""
        embeddings = np.array([
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ])
        sim = module._cosine_sim_matrix(embeddings)
        assert sim.shape == (3, 3)
        assert abs(sim[0, 1] - 1.0) < 1e-5
        assert abs(sim[0, 2]) < 1e-5

    def test_recency_weight_decay(self):
        """Test that recency weight decays exponentially."""
        from medaide_plus.modules.m1_amqu import AMQUModule
        lambda_ = 0.1
        T = 100.0
        t1, t2 = 90.0, 50.0
        w1 = math.exp(-lambda_ * (T - t1))
        w2 = math.exp(-lambda_ * (T - t2))
        assert w1 > w2
        assert abs(w1 - math.exp(-1.0)) < 1e-6

    def test_empty_corpus_bm25(self, module):
        """Test BM25 returns empty list when corpus is empty."""
        results = module._recency_weighted_bm25("test query", top_k=5)
        assert results == []
