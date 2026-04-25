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
            "You specialize in:\n"
            "  1. Symptom Triage: Assessing the urgency and nature of reported symptoms. "
            "     Identify red-flag symptoms that require immediate emergency attention.\n"
            "  2. Department Suggestion: Recommending the appropriate medical department "
            "     or specialist type (e.g., cardiologist, neurologist, GP) based on symptoms.\n"
            "  3. Risk Assessment: Evaluating risk factors for common conditions based on "
            "     patient history, lifestyle, and reported symptoms.\n"
            "  4. Health Inquiry: Answering general health questions about normal body "
            "     functioning, wellness, and preventive care.\n\n"
            "Guidelines:\n"
            "  - Always recommend professional medical consultation for serious symptoms.\n"
            "  - Clearly flag emergency symptoms (chest pain, difficulty breathing, etc.).\n"
            "  - Be empathetic and reassuring while being accurate.\n"
            "  - Do NOT provide definitive diagnoses — only pre-diagnostic guidance.\n"
            "  - Cite evidence-based guidelines where applicable.\n"
            "  - Response format: Structured with clear sections for Assessment, "
            "    Recommendation, and Next Steps."
        )
