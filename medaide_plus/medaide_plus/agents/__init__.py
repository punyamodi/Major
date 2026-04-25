"""
MedAide+ Agents Package

Specialized medical domain agents:
- BaseAgent: Abstract base with LLM integration
- PreDiagnosisAgent: Symptom triage, risk assessment
- DiagnosisAgent: Differential diagnosis, etiology
- MedicationAgent: Drug counseling, interactions
- PostDiagnosisAgent: Rehabilitation, lifestyle guidance
- CriticAgent: Contradiction detection
- SynthesisAgent: Confidence-weighted response merging
"""

from medaide_plus.agents.base_agent import BaseAgent, AgentOutput
from medaide_plus.agents.pre_diagnosis_agent import PreDiagnosisAgent
from medaide_plus.agents.diagnosis_agent import DiagnosisAgent
from medaide_plus.agents.medication_agent import MedicationAgent
from medaide_plus.agents.post_diagnosis_agent import PostDiagnosisAgent
from medaide_plus.agents.critic_agent import CriticAgent
from medaide_plus.agents.synthesis_agent import SynthesisAgent

__all__ = [
    "BaseAgent",
    "AgentOutput",
    "PreDiagnosisAgent",
    "DiagnosisAgent",
    "MedicationAgent",
    "PostDiagnosisAgent",
    "CriticAgent",
    "SynthesisAgent",
]
