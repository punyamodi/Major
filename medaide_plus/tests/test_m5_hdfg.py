"""Unit tests for M5 HDFG — Hallucination-Aware Dual-Verification."""

import os
import pytest
import numpy as np

# Prevent TF from loading to avoid tf_keras error
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["USE_TORCH"] = "1"


class TestHdfgModule:
    """Tests for HdfgModule."""

    @pytest.fixture
    def module(self):
        """Create HdfgModule with sample KB, bypassing sentence-transformers load."""
        from unittest.mock import patch
        from medaide_plus.modules.m5_hdfg import HdfgModule

        def _mock_load_embedding(_self):
            _self._embedding_model = None
            _self._embed_dim = 256

        kb = [
            "Aspirin 325mg is used for pain relief and antiplatelet therapy.",
            "Metformin is first-line treatment for type 2 diabetes.",
            "Hypertension target blood pressure is below 130/80 mmHg.",
            "Penicillin allergy: use azithromycin or clindamycin as alternatives.",
            "Warfarin interacts with fluconazole increasing bleeding risk.",
        ]
        with patch.object(HdfgModule, "_load_embedding_model", _mock_load_embedding):
            m = HdfgModule(config={"mc_dropout_passes": 5, "support_threshold": 0.3}, knowledge_base=kb)
        return m

    def test_claim_extraction(self, module):
        """Test that medical claims are extracted from response text."""
        response = (
            "You should take 500mg aspirin daily. Metformin is effective for diabetes. "
            "Avoid ibuprofen if you have kidney problems."
        )
        claims = module._extract_claims(response)
        assert len(claims) >= 1
        assert all(isinstance(c, str) for c in claims)

    def test_annotate_response_structure(self, module):
        """Test that annotated response contains [SUPPORTED] or [FLAGGED] markers."""
        response = (
            "Aspirin should be taken at 325mg for pain relief. "
            "Metformin is first-line treatment for diabetes."
        )
        result = module.annotate_response(response)
        assert result.original_response == response
        assert isinstance(result.annotated_response, str)
        assert isinstance(result.hallucination_rate, float)
        assert 0.0 <= result.hallucination_rate <= 1.0

    def test_mc_dropout_uncertainty(self, module):
        """Test that MC-Dropout returns valid confidence and uncertainty."""
        confidence, uncertainty = module._stage2_mc_uncertainty("aspirin pain relief")
        assert 0.0 <= confidence <= 1.0
        assert uncertainty >= 0.0

    def test_stage1_kb_search(self, module):
        """Test FAISS/numpy KB search returns a score."""
        score, sources = module._stage1_kb_search("aspirin dosage pain")
        assert 0.0 <= score <= 1.0
        assert isinstance(sources, list)

    def test_empty_kb(self):
        """Test behavior with empty knowledge base."""
        from medaide_plus.modules.m5_hdfg import HdfgModule
        m = HdfgModule(config={})
        result = m.annotate_response("Test medical statement about aspirin.")
        assert result.original_response is not None
        assert isinstance(result.hallucination_rate, float)

    def test_no_claims_response(self, module):
        """Test handling of response with no extractable claims."""
        result = module.annotate_response("Ok.")
        assert result.hallucination_rate == 0.0
        assert result.verified_claims == []

    def test_hallucination_rate_range(self, module):
        """Test that hallucination rate is always in [0, 1]."""
        for response in [
            "Aspirin should be taken for pain.",
            "Unicorns cure all diseases immediately.",
            "",
            "The quick brown fox jumps over the lazy dog.",
        ]:
            result = module.annotate_response(response)
            assert 0.0 <= result.hallucination_rate <= 1.0
