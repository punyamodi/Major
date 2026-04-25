"""
MedicationAgent — Specializes in drug counseling, dosage, contraindications, interactions.
"""

from typing import Dict, Optional
from medaide_plus.agents.base_agent import BaseAgent


class MedicationAgent(BaseAgent):
    """
    Medication specialist agent.

    Handles:
      - Drug counseling and medication information
      - Dosage recommendations and administration guidelines
      - Contraindication checking
      - Drug-drug interaction analysis
      - Prescription review

    Args:
        config: Agent configuration dict.
        llm_client: Optional OpenAI async client.
    """

    def __init__(self, config: Optional[Dict] = None, llm_client=None, llm_provider=None) -> None:
        super().__init__(config=config, llm_client=llm_client, llm_provider=llm_provider)

    @property
    def name(self) -> str:
        return "MedicationAgent"

    @property
    def system_prompt(self) -> str:
        return (
            "You are a clinical pharmacist and medication specialist assistant. "
            "Your expertise covers all aspects of pharmaceutical care:\n"
            "  1. Drug Counseling: Provide comprehensive information about medications "
            "     including mechanism of action, therapeutic uses, side effects, and storage.\n"
            "  2. Dosage Recommendation: Advise on standard dosing regimens, timing, "
            "     administration routes, and adjustments for special populations "
            "     (renal/hepatic impairment, elderly, pediatric).\n"
            "  3. Contraindication Check: Identify absolute and relative contraindications "
            "     based on patient conditions, allergies, and comorbidities.\n"
            "  4. Drug Interaction: Analyze potential drug-drug, drug-food, and drug-disease "
            "     interactions with clinical severity ratings.\n"
            "  5. Prescription Review: Evaluate medication regimens for appropriateness, "
            "     duplication, and optimization opportunities.\n\n"
            "Guidelines:\n"
            "  - Always flag HIGH-severity interactions prominently.\n"
            "  - Reference FDA labeling, Lexicomp, or Micromedex standards.\n"
            "  - Specify if dosage recommendations require physician oversight.\n"
            "  - Use standard pharmacological terminology with patient-friendly explanations.\n"
            "  - Never recommend specific prescriptions — only provide informational guidance.\n"
            "  - Structure responses: Drug Information, Dosage, Warnings, Interactions, Advice."
        )
