"""
MedAide+ Main Pipeline

Orchestrates all 7 modules in a complete end-to-end medical query processing pipeline.

Pipeline steps:
  Step 0: M4 inject patient history
  Step 1: M1 decompose query
  Step 2: M2 + M7 classify intents, update dialogue state
  Step 3: M6 route by complexity
  Step 4: M3 run agents in parallel
  Step 5: M5 verify output
  Step 6: Update M4 and M7 state
"""

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from medaide_plus.agents.critic_agent import CriticAgent
from medaide_plus.agents.diagnosis_agent import DiagnosisAgent
from medaide_plus.agents.medication_agent import MedicationAgent
from medaide_plus.agents.post_diagnosis_agent import PostDiagnosisAgent
from medaide_plus.agents.pre_diagnosis_agent import PreDiagnosisAgent
from medaide_plus.agents.synthesis_agent import SynthesisAgent
from medaide_plus.knowledge_base.kb_manager import KBManager
from medaide_plus.modules.m1_amqu import AMQUModule
from medaide_plus.modules.m2_hdio import HDIOModule
from medaide_plus.modules.m3_dmacn import DMACNModule
from medaide_plus.modules.m4_plmm import PLMMModule
from medaide_plus.modules.m5_hdfg import HdfgModule
from medaide_plus.modules.m6_aqcr import AqcrModule
from medaide_plus.modules.m7_miet import MietModule
from medaide_plus.utils.logger import get_logger, setup_logging

logger = get_logger("pipeline")


def _create_llm_provider(config: Dict):
    """
    Create an LLM provider from the pipeline config.

    Reads the 'llm' section of config.yaml and instantiates the
    appropriate provider. Falls back gracefully if provider module
    is unavailable.

    Args:
        config: Full config dict.

    Returns:
        BaseLLMProvider instance or None if unavailable.
    """
    try:
        from medaide_plus.utils.llm_provider import create_provider_from_yaml
        return create_provider_from_yaml(config)
    except ImportError:
        logger.warning("LLM provider module not available. Using legacy OpenAI client.")
        return None
    except Exception as e:
        logger.warning(f"Failed to create LLM provider: {e}. Using legacy OpenAI client.")
        return None


@dataclass
class PipelineResult:
    """Complete result from the MedAide+ pipeline."""
    query: str
    patient_id: str
    final_response: str
    annotated_response: str
    confidence: float = 0.0
    tier: str = "Simple"
    n_agents_used: int = 1
    intents: List[str] = field(default_factory=list)
    top_category: str = ""
    subqueries: List[str] = field(default_factory=list)
    hallucination_rate: float = 0.0
    verified_claims: List[Dict] = field(default_factory=list)
    agent_names: List[str] = field(default_factory=list)
    latency_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class MedAidePlusPipeline:
    """
    MedAide+ End-to-End Pipeline.

    Orchestrates 7 modules and 4-6 agents to process medical queries
    with patient history awareness, intent classification, complexity
    routing, hallucination verification, and multi-turn tracking.

    Args:
        config_path: Path to config/config.yaml.
        patient_id: Default patient ID (can be overridden per query).
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        patient_id: Optional[str] = None,
        llm_provider=None,
    ) -> None:
        self.default_patient_id = patient_id or "default_patient"
        self.config = self._load_config(config_path)

        setup_logging(
            level=self.config.get("logging", {}).get("level", "INFO"),
        )
        logger.info("Initializing MedAide+ Pipeline...")

        # Initialize LLM provider (multi-backend support)
        self._llm_provider = llm_provider or _create_llm_provider(self.config)
        if self._llm_provider:
            logger.info(f"LLM provider initialized: {type(self._llm_provider).__name__}")
        else:
            logger.info("No LLM provider configured. Using legacy OpenAI client / mock.")

        # Load knowledge base
        self.kb_manager = KBManager(config=self.config.get("knowledge_base", {}))
        self._init_knowledge_base()

        # Initialize 7 modules
        logger.info("Loading modules...")
        module_configs = self.config.get("modules", {})

        self.m1_amqu = AMQUModule(
            config=module_configs.get("amqu", {}),
            corpus=self.kb_manager.get_all_texts(),
        )
        self.m2_hdio = HDIOModule(config=module_configs.get("hdio", {}))
        self.m4_plmm = PLMMModule(
            config=module_configs.get("plmm", {}),
            storage_path=module_configs.get("plmm", {}).get(
                "graph_storage_path", "data/patient_graphs"
            ),
        )
        self.m5_hdfg = HdfgModule(
            config=module_configs.get("hdfg", {}),
            knowledge_base=self.kb_manager.get_all_texts(),
        )
        self.m6_aqcr = AqcrModule(config=module_configs.get("aqcr", {}))
        self.m7_miet = MietModule(config=module_configs.get("miet", {}))

        # Initialize agents with provider
        api_config = self.config.get("api", {})
        llm_config = self.config.get("llm", {})
        active_provider = llm_config.get("provider", "openai")
        provider_settings = llm_config.get(active_provider, {})

        agent_config = {
            "openai_model": provider_settings.get("model", api_config.get("openai_model", "gpt-4o")),
            "model": provider_settings.get("model", api_config.get("openai_model", "gpt-4o")),
            "max_tokens": provider_settings.get("max_tokens", api_config.get("max_tokens", 1024)),
            "temperature": provider_settings.get("temperature", api_config.get("temperature", 0.3)),
        }
        self.agents = [
            PreDiagnosisAgent(config=agent_config, llm_provider=self._llm_provider),
            DiagnosisAgent(config=agent_config, llm_provider=self._llm_provider),
            MedicationAgent(config=agent_config, llm_provider=self._llm_provider),
            PostDiagnosisAgent(config=agent_config, llm_provider=self._llm_provider),
        ]
        self.critic_agent = CriticAgent(config=agent_config, llm_provider=self._llm_provider)
        self.synthesis_agent = SynthesisAgent(config=agent_config, llm_provider=self._llm_provider)

        # Initialize DMACN with agents
        self.m3_dmacn = DMACNModule(
            agents=self.agents,
            config=module_configs.get("dmacn", {}),
        )

        logger.info("MedAide+ Pipeline initialized successfully.")

    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load YAML configuration file."""
        if config_path is None:
            # Try default locations
            candidates = [
                "config/config.yaml",
                Path(__file__).parent.parent / "config" / "config.yaml",
            ]
            for candidate in candidates:
                if Path(candidate).exists():
                    config_path = str(candidate)
                    break

        if config_path and Path(config_path).exists():
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded config from {config_path}.")
            return config or {}
        else:
            logger.warning("No config file found. Using defaults.")
            return {}

    def _init_knowledge_base(self) -> None:
        """Initialize knowledge base with sample data if empty."""
        kb_path = self.config.get("knowledge_base", {}).get(
            "kb_path", "data/knowledge_base.json"
        )
        if Path(kb_path).exists():
            self.kb_manager.load(kb_path)
        else:
            logger.info("Seeding KB with sample medical guidelines.")
            sample_docs = KBManager.create_sample_kb()
            texts = [d["text"] for d in sample_docs]
            metas = [d.get("metadata", {}) for d in sample_docs]
            self.kb_manager.add_documents(texts, metas)

    async def run(
        self,
        query: str,
        patient_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> PipelineResult:
        """
        Execute the full MedAide+ pipeline for a single query.

        Steps:
          0. M4: Inject patient history into query
          1. M1: Decompose query into subqueries
          2. M2: Classify intents
          3. M7: Update dialogue state, inject history bias
          4. M6: Route by complexity → n_agents
          5. M3: Run n_agents in parallel
          6. Critic + Synthesis
          7. M5: Verify output for hallucinations
          8. M4: Update patient graph from response
          9. M7: Update session buffer

        Args:
            query: Patient's medical query.
            patient_id: Patient identifier (uses default if None).
            session_id: Dialogue session ID (uses patient_id if None).

        Returns:
            PipelineResult with full metadata.
        """
        start_time = time.time()
        patient_id = patient_id or self.default_patient_id
        session_id = session_id or patient_id

        logger.info(f"Pipeline: processing query for patient={patient_id}")
        logger.debug(f"Query: {query[:80]}...")

        # ── Step 0: M4 — Inject patient history ──────────────────────────
        plmm_result = self.m4_plmm.inject_history(query, patient_id)
        enriched_query = plmm_result.enriched_query

        # ── Step 1: M1 — Decompose query ─────────────────────────────────
        amqu_result = self.m1_amqu.run(enriched_query)
        subqueries = [sq.text for sq in amqu_result.subqueries]

        # ── Step 2: M2 — Classify intents ────────────────────────────────
        hdio_result = self.m2_hdio.classify(enriched_query)
        raw_intent_scores = hdio_result.intent_scores
        top_intents = hdio_result.top_intents
        top_category = hdio_result.top_category

        # ── Step 3: M7 — Update dialogue state + inject history bias ──────
        self.m7_miet.update(
            session_id=session_id,
            intent_scores=raw_intent_scores,
            query=query,
        )
        biased_intent_scores = self.m7_miet.inject_state(session_id, raw_intent_scores)
        context_prefix = self.m7_miet.get_context_prefix(session_id)

        # Re-select intents from biased scores
        top_intents_biased = sorted(
            biased_intent_scores, key=biased_intent_scores.get, reverse=True
        )[:3]

        # ── Step 4: M6 — Route by complexity ─────────────────────────────
        aqcr_result = self.m6_aqcr.route(
            query=enriched_query,
            intents=top_intents_biased,
            subqueries=subqueries,
        )
        n_agents = aqcr_result.n_agents
        tier = aqcr_result.tier

        logger.info(f"Routing to tier={tier} ({n_agents} agents).")

        # ── Step 5: M3 — Run agents in parallel ──────────────────────────
        # Collect KB-retrieved evidence from M1's BM25 results
        kb_evidence = [doc for doc, score in amqu_result.bm25_weighted_docs[:3]] if amqu_result.bm25_weighted_docs else []

        agent_context = {
            "intents": top_intents_biased,
            "subqueries": subqueries,
            "patient_id": patient_id,
            "tier": tier,
            "context_prefix": context_prefix,
            "kb_evidence": kb_evidence,
        }

        dmacn_result = await self.m3_dmacn.run(
            query=enriched_query,
            context=agent_context,
            n_agents=n_agents,
        )

        # ── Step 6: Critic + Synthesis ────────────────────────────────────
        critic_report = await self.critic_agent.evaluate(dmacn_result.agent_outputs)
        synthesis_result = await self.synthesis_agent.synthesize(
            agent_outputs=dmacn_result.agent_outputs,
            critic_report=critic_report,
            query=query,
        )
        final_response = synthesis_result.synthesized_response

        # ── Step 7: M5 — Verify for hallucinations ───────────────────────
        hdfg_result = self.m5_hdfg.annotate_response(final_response, query=query)
        annotated_response = hdfg_result.annotated_response

        # ── Step 8: M4 — Update patient graph ────────────────────────────
        self.m4_plmm.update_from_response(
            response=final_response,
            patient_id=patient_id,
            query=query,
        )

        # ── Step 9: M7 — Update buffer with response ─────────────────────
        buffer = self.m7_miet._get_or_create_buffer(session_id)
        if buffer:
            last_turn = list(buffer)[-1]
            last_turn.response = final_response[:200]

        latency_ms = (time.time() - start_time) * 1000

        logger.info(
            f"Pipeline complete: tier={tier}, n_agents={n_agents}, "
            f"confidence={synthesis_result.final_confidence:.3f}, "
            f"hallucination_rate={hdfg_result.hallucination_rate:.3f}, "
            f"latency={latency_ms:.0f}ms"
        )

        return PipelineResult(
            query=query,
            patient_id=patient_id,
            final_response=final_response,
            annotated_response=annotated_response,
            confidence=synthesis_result.final_confidence,
            tier=tier,
            n_agents_used=n_agents,
            intents=top_intents_biased,
            top_category=top_category,
            subqueries=subqueries,
            hallucination_rate=hdfg_result.hallucination_rate,
            verified_claims=[
                {
                    "claim": c.claim[:100],
                    "supported": c.supported,
                    "confidence": c.confidence,
                }
                for c in hdfg_result.verified_claims[:5]
            ],
            agent_names=dmacn_result.agents_used,
            latency_ms=latency_ms,
            metadata={
                "amqu": amqu_result.processing_metadata,
                "hdio": hdio_result.metadata,
                "aqcr": {"tier": tier, "confidence": aqcr_result.confidence},
                "hdfg": hdfg_result.metadata,
                "critic_severity": critic_report.severity,
            },
        )

    def run_sync(
        self,
        query: str,
        patient_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> PipelineResult:
        """Synchronous wrapper around async run()."""
        return asyncio.run(self.run(query, patient_id, session_id))
