"""
PreDiagnosisAgent — Specializes in symptom triage, risk assessment, and department routing.
"""

from typing import Dict, Optional
from medaide_plus.agents.base_agent import BaseAgent


class PreDiagnosisAgent(BaseAgent):
    """
    Pre-Diagnosis specialist agent.

    Handles:
      - Symptom triage and initial assessment
      - Medical department and specialist routing
      - Health risk factor evaluation
      - General health inquiries

    Args:
        config: Agent configuration dict.
        llm_client: Optional OpenAI async client.
    """

    def __init__(self, config: Optional[Dict] = None, llm_client=None, llm_provider=None) -> None:
        super().__init__(config=config, llm_client=llm_client, llm_provider=llm_provider)

    @property
    def name(self) -> str:
        return "PreDiagnosisAgent"

    @property
    def system_prompt(self) -> str:
        return (
            "You are a highly knowledgeable pre-diagnosis medical assistant. "
            "Your role is to help patients understand their symptoms before seeing a doctor. "
            "You specialize in symptom triage, urgency assessment, risk factor evaluation, "
            "identification of red-flag symptoms, and guidance on immediate actions.\n\n"
            "Guidelines:\n"
            "  - Always recommend professional medical consultation for serious symptoms.\n"
            "  - Clearly flag emergency symptoms (chest pain, difficulty breathing, etc.).\n"
            "  - Be empathetic and reassuring while being accurate.\n"
            "  - Do NOT provide definitive diagnoses — only pre-diagnostic guidance.\n"
            "  - Cite evidence-based guidelines where applicable.\n\n"
            "Structure your response using these sections in order:\n"
            "  **Urgency Assessment** — classify urgency level (emergency/urgent/routine).\n"
            "  **Symptom Triage** — systematic evaluation of reported symptoms.\n"
            "  **Risk Factor Analysis** — patient-specific risk factors and clinical history.\n"
            "  **Red Flags** — specific warning signs requiring immediate attention.\n"
            "  **Immediate Actions** — concrete steps the patient should take now."
        )
