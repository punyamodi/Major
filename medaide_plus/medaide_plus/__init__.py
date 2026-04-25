"""
MedAide+ — An improved LLM-based medical multi-agent framework.

Extends the original MedAide system (Fudan University, arXiv 2410.12532) with
7 new modules for enhanced query understanding, intent classification, multi-agent
orchestration, patient memory, hallucination verification, complexity routing,
and multi-turn dialogue tracking.
"""

__version__ = "1.0.0"
__author__ = "MedAide+ Team"
__description__ = "Improved LLM-based medical multi-agent framework"

from medaide_plus.pipeline import MedAidePlusPipeline

__all__ = ["MedAidePlusPipeline"]
