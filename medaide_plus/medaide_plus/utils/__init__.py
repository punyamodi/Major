"""
MedAide+ Utilities Package

Components:
- logger: Structured logging setup
- metrics: Evaluation metrics (BLEU, ROUGE, BERTScore, hallucination rate)
"""

from medaide_plus.utils.logger import get_logger, setup_logging
from medaide_plus.utils.metrics import (
    compute_bleu,
    compute_rouge,
    compute_meteor,
    compute_bert_score,
    compute_hallucination_rate,
    compute_latency,
    compute_multiturn_consistency,
)

__all__ = [
    "get_logger",
    "setup_logging",
    "compute_bleu",
    "compute_rouge",
    "compute_meteor",
    "compute_bert_score",
    "compute_hallucination_rate",
    "compute_latency",
    "compute_multiturn_consistency",
]
