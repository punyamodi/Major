"""Integration tests for MedAide+ Pipeline."""

import os
import asyncio
import pytest

os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["USE_TORCH"] = "1"


class TestMedAidePlusPipeline:
    """Integration tests for the full pipeline."""

    @pytest.fixture
    def pipeline(self, tmp_path):
        """Create a minimal pipeline for testing (explicit mock mode)."""
        import yaml
        import json
        from medaide_plus.pipeline import MedAidePlusPipeline

        storage = str(tmp_path / "patient_graphs")
        kb_path = tmp_path / "knowledge_base.json"
        kb_path.write_text(
            json.dumps(
                {
                    "documents": [
                        {
                            "id": "kb_test_001",
                            "text": "Hypertension management includes lifestyle changes and antihypertensive therapy.",
                            "metadata": {"category": "Diagnosis"},
                        },
                        {
                            "id": "kb_test_002",
                            "text": "Metformin is first-line treatment for type 2 diabetes in most adults.",
                            "metadata": {"category": "Medication"},
                        },
                    ]
                }
            ),
            encoding="utf-8",
        )

        config = {
            "models": {"openai_model": "gpt-4o"},
            "modules": {
                "amqu": {"k_shots": 2, "consistency_threshold": 0.85, "min_count": 2},
                "hdio": {"ood_threshold": 0.4},
                "dmacn": {"timeout": 10},
                "plmm": {"graph_storage_path": storage},
                "hdfg": {"mc_dropout_passes": 3, "support_threshold": 0.3},
                "aqcr": {"simple_threshold": 0.33, "complex_threshold": 0.66},
                "miet": {"beta": 0.6, "buffer_size": 3},
            },
            "knowledge_base": {"kb_path": str(kb_path), "top_k": 3},
            "runtime": {"allow_mock_llm": True},
            "api": {"openai_model": "gpt-4o", "max_tokens": 512, "temperature": 0.3},
            "logging": {"level": "WARNING"},
        }
        config_path = tmp_path / "test_config.yaml"
        config_path.write_text(yaml.dump(config))
        return MedAidePlusPipeline(config_path=str(config_path), patient_id="test_patient")

    @pytest.mark.asyncio
    async def test_full_pipeline_run(self, pipeline):
        result = await pipeline.run(
            query="I have a headache and need pain medication advice.",
            patient_id="test_patient_001",
        )
        assert result.query == "I have a headache and need pain medication advice."
        assert result.patient_id == "test_patient_001"
        assert isinstance(result.final_response, str)
        assert len(result.final_response) > 0
        assert result.tier in ["Simple", "Moderate", "Complex"]
        assert result.n_agents_used in [1, 2, 4]
        assert 0.0 <= result.hallucination_rate <= 1.0
        assert result.latency_ms > 0

    @pytest.mark.asyncio
    async def test_pipeline_with_patient_history(self, pipeline):
        patient_id = "test_patient_history"
        await pipeline.run(query="I have diabetes and take metformin.", patient_id=patient_id)
        result2 = await pipeline.run(
            query="What other medications interact with my current drugs?",
            patient_id=patient_id,
        )
        assert result2.patient_id == patient_id
        assert isinstance(result2.final_response, str)

    @pytest.mark.asyncio
    async def test_pipeline_routing_tiers(self, pipeline):
        result_simple = await pipeline.run(query="What is aspirin?", patient_id="tier_test")
        result_complex = await pipeline.run(
            query=(
                "I have diabetes, hypertension, and coronary artery disease. "
                "I am taking metformin 1000mg, lisinopril 20mg, aspirin 81mg, "
                "and atorvastatin 40mg. I am experiencing dizziness and need a "
                "full medication review including interactions and dosage adjustments."
            ),
            patient_id="tier_test",
        )
        assert isinstance(result_simple.final_response, str)
        assert isinstance(result_complex.final_response, str)

    @pytest.mark.asyncio
    async def test_pipeline_metadata_completeness(self, pipeline):
        result = await pipeline.run(
            query="I have chest pain and shortness of breath.",
            patient_id="metadata_test_patient",
        )
        assert "amqu" in result.metadata
        assert "hdio" in result.metadata
        assert "aqcr" in result.metadata
        assert "hdfg" in result.metadata

    @pytest.mark.asyncio
    async def test_pipeline_multiturn_session(self, pipeline):
        patient_id = "multi_turn_patient"
        queries = [
            "I have a persistent cough.",
            "What tests should I get?",
            "My chest X-ray is normal. What next?",
        ]
        results = []
        for query in queries:
            result = await pipeline.run(
                query=query, patient_id=patient_id, session_id="session_multi"
            )
            results.append(result)
        assert len(results) == 3
        for r in results:
            assert isinstance(r.final_response, str)

    def test_run_sync_wrapper(self, pipeline):
        result = pipeline.run_sync(
            query="What is hypertension?", patient_id="sync_test_patient"
        )
        assert isinstance(result.final_response, str)
        assert result.tier in ["Simple", "Moderate", "Complex"]
