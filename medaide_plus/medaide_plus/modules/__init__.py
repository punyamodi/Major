"""
MedAide+ Modules Package

Seven enhancement modules:
- M1 AMQU: Adaptive Multi-shot Query Understanding
- M2 HDIO: Hierarchical Dual-level Intent Ontology
- M3 DMACN: Dynamic Multi-Agent Critic Network
- M4 PLMM: Persistent Longitudinal Medical Memory
- M5 HDFG: Hallucination-Aware Dual-Verification with Factual Grounding
- M6 AQCR: Adaptive Query Complexity Routing
- M7 MIET: Multi-Turn Intent Evolution Tracking
"""

from medaide_plus.modules.m1_amqu import AMQUModule
from medaide_plus.modules.m2_hdio import HDIOModule
from medaide_plus.modules.m3_dmacn import DMACNModule
from medaide_plus.modules.m4_plmm import PLMMModule
from medaide_plus.modules.m5_hdfg import HdfgModule
from medaide_plus.modules.m6_aqcr import AqcrModule
from medaide_plus.modules.m7_miet import MietModule

__all__ = [
    "AMQUModule",
    "HDIOModule",
    "DMACNModule",
    "PLMMModule",
    "HdfgModule",
    "AqcrModule",
    "MietModule",
]
