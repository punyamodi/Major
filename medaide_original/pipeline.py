"""
MedAide Original Pipeline (M1 + M2 + M3).

Recreates the original 3-module architecture used before MedAide+ enhancements.
"""

import asyncio
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

ROOT = Path(__file__).resolve().parent.parent
MEDAIDE_PLUS_ROOT = ROOT / "medaide_plus"
if str(MEDAIDE_PLUS_ROOT) not in sys.path:
    sys.path.insert(0, str(MEDAIDE_PLUS_ROOT))

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
from medaide_plus.utils.logger import get_logger, setup_logging
from medaide_plus.utils.llm_provider import create_provider_from_yaml

logger = get_logger("original_pipeline")


@dataclass
class OriginalPipelineResult:
    query: str
    patient_id: str
    final_response: str
    confidence: float = 0.0
    tier: str = "Simple"
    n_agents_used: int = 1
    intents: List[str] = field(default_factory=list)
    top_category: str = ""
    subqueries: List[str] = field(default_factory=list)
    agent_names: List[str] = field(default_factory=list)
    latency_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class MedAideOriginalPipeline:
    """
    Simulated original MedAide architecture:
    - M1 AMQU query decomposition
    - M2 HDIO intent classification
    - M3 DMACN agent execution (fixed 1-agent mode)
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        patient_id: Optional[str] = None,
        llm_provider=None,
    ) -> None:
        self.default_patient_id = patient_id or "original_patient"
        self.config = self._load_config(config_path)
        self._resolve_runtime_paths(self.config)
        runtime_cfg = self.config.get("runtime", {})
        self.allow_mock_llm = bool(runtime_cfg.get("allow_mock_llm", False))
        self.allow_provider_fallback = bool(runtime_cfg.get("allow_provider_fallback", False))

        setup_logging(level=self.config.get("logging", {}).get("level", "INFO"))
        logger.info("Initializing MedAide Original Pipeline...")

        self._llm_provider = llm_provider or self._create_llm_provider(self.config)
        if self._llm_provider is None and not self.allow_mock_llm:
            raise RuntimeError(
                "LLM provider initialization failed for medaide_original. "
                "Configure a valid provider or explicitly set runtime.allow_mock_llm=true for tests."
            )

        self.kb_manager = KBManager(config=self.config.get("knowledge_base", {}))
        self._init_knowledge_base()

        module_configs = self.config.get("modules", {})
        self.m1_amqu = AMQUModule(
            config=module_configs.get("amqu", {}),
            corpus=self.kb_manager.get_all_texts(),
        )
        self.m2_hdio = HDIOModule(config=module_configs.get("hdio", {}))

        llm_config = self.config.get("llm", {})
        active_provider = llm_config.get("provider", "openai")
        provider_settings = llm_config.get(active_provider, {})
        api_config = self.config.get("api", {})
        agent_config = {
            "openai_model": provider_settings.get("model", api_config.get("openai_model", "gpt-4o")),
            "model": provider_settings.get("model", api_config.get("openai_model", "gpt-4o")),
            "max_tokens": provider_settings.get("max_tokens", api_config.get("max_tokens", 1024)),
            "temperature": provider_settings.get("temperature", api_config.get("temperature", 0.3)),
            "allow_mock_llm": self.allow_mock_llm,
            "allow_provider_fallback": self.allow_provider_fallback,
        }

        self.agents = [
            PreDiagnosisAgent(config=agent_config, llm_provider=self._llm_provider),
            DiagnosisAgent(config=agent_config, llm_provider=self._llm_provider),
            MedicationAgent(config=agent_config, llm_provider=self._llm_provider),
            PostDiagnosisAgent(config=agent_config, llm_provider=self._llm_provider),
        ]
        self.critic_agent = CriticAgent(config=agent_config, llm_provider=self._llm_provider)
        self.synthesis_agent = SynthesisAgent(config=agent_config, llm_provider=self._llm_provider)
        self.m3_dmacn = DMACNModule(
            agents=self.agents,
            config=module_configs.get("dmacn", {}),
        )
        logger.info("MedAide Original Pipeline initialized successfully.")

    def _create_llm_provider(self, config: Dict[str, Any]):
        try:
            return create_provider_from_yaml(config)
        except Exception as e:
            logger.warning(f"Failed to initialize LLM provider: {e}. Falling back to legacy path.")
            return None

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        candidates = []
        if config_path:
            candidates.append(Path(config_path))
        candidates.append(Path(__file__).resolve().parent / "config.yaml")
        candidates.append(MEDAIDE_PLUS_ROOT / "config" / "config.yaml")

        for candidate in candidates:
            if candidate.exists():
                with open(candidate, "r", encoding="utf-8") as f:
                    return yaml.safe_load(f) or {}

        logger.warning("No config file found for MedAide Original. Using defaults.")
        return {}

    def _resolve_runtime_paths(self, config: Dict[str, Any]) -> None:
        kb_cfg = config.setdefault("knowledge_base", {})
        for key in ("kb_path", "faiss_index_path"):
            value = kb_cfg.get(key)
            if isinstance(value, str) and value and not Path(value).is_absolute():
                kb_cfg[key] = str((ROOT / value).resolve())

        log_cfg = config.setdefault("logging", {})
        log_file = log_cfg.get("file")
        if isinstance(log_file, str) and log_file and not Path(log_file).is_absolute():
            log_cfg["file"] = str((ROOT / log_file).resolve())

    def _init_knowledge_base(self) -> None:
        kb_path = self.config.get("knowledge_base", {}).get("kb_path", "")
        kb_file = Path(kb_path)
        if not kb_path or not kb_file.exists():
            raise FileNotFoundError(
                f"Knowledge base file not found for medaide_original: {kb_file}. "
                "Production mode requires persisted KB data."
            )
        loaded = self.kb_manager.load(str(kb_file))
        if loaded == 0:
            raise ValueError(
                f"Knowledge base is empty or invalid for medaide_original: {kb_file}."
            )

    async def run(
        self,
        query: str,
        patient_id: Optional[str] = None,
    ) -> OriginalPipelineResult:
        start = time.time()
        patient_id = patient_id or self.default_patient_id

        amqu_result = self.m1_amqu.run(query)
        subqueries = [sq.text for sq in amqu_result.subqueries]

        hdio_result = self.m2_hdio.classify(query)
        top_intents = hdio_result.top_intents[:3]

        agent_context = {
            "intents": top_intents,
            "subqueries": subqueries,
            "patient_id": patient_id,
            "tier": "Simple",
            "context_prefix": "",
        }
        dmacn_result = await self.m3_dmacn.run(
            query=query,
            context=agent_context,
            n_agents=1,
        )

        critic_report = await self.critic_agent.evaluate(dmacn_result.agent_outputs)
        synthesis_result = await self.synthesis_agent.synthesize(
            agent_outputs=dmacn_result.agent_outputs,
            critic_report=critic_report,
            query=query,
        )
        latency_ms = (time.time() - start) * 1000

        return OriginalPipelineResult(
            query=query,
            patient_id=patient_id,
            final_response=synthesis_result.synthesized_response,
            confidence=synthesis_result.final_confidence,
            tier="Simple",
            n_agents_used=1,
            intents=top_intents,
            top_category=hdio_result.top_category,
            subqueries=subqueries,
            agent_names=dmacn_result.agents_used,
            latency_ms=latency_ms,
            metadata={
                "amqu": amqu_result.processing_metadata,
                "hdio": hdio_result.metadata,
                "critic_severity": critic_report.severity,
            },
        )

    def run_sync(
        self,
        query: str,
        patient_id: Optional[str] = None,
    ) -> OriginalPipelineResult:
        return asyncio.run(self.run(query=query, patient_id=patient_id))
