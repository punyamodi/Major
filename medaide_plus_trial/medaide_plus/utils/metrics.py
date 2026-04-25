"""
MedAide+ Evaluation Metrics

Implements standard NLP evaluation metrics plus domain-specific metrics
for hallucination rate, latency, and multi-turn consistency.
"""

import time
import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger("medaide_plus.metrics")


def compute_bleu(
    pred: str,
    ref: str,
    weights: Tuple[float, ...] = (0.25, 0.25, 0.25, 0.25),
) -> float:
    """
    Compute BLEU score between prediction and reference.

    Args:
        pred: Predicted text string.
        ref: Reference text string.
        weights: N-gram weights for BLEU-1 through BLEU-4.

    Returns:
        BLEU score in range [0, 1].
    """
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        import nltk
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt", quiet=True)

        pred_tokens = pred.lower().split()
        ref_tokens = ref.lower().split()
        smoother = SmoothingFunction().method1
        return float(sentence_bleu([ref_tokens], pred_tokens, weights=weights, smoothing_function=smoother))
    except ImportError:
        # Fallback: simple unigram overlap
        pred_tokens = set(pred.lower().split())
        ref_tokens = set(ref.lower().split())
        if not ref_tokens:
            return 0.0
        return len(pred_tokens & ref_tokens) / len(ref_tokens)


def compute_rouge(
    pred: str,
    ref: str,
    rouge_types: List[str] = ("rouge1", "rouge2", "rougeL"),
) -> Dict[str, float]:
    """
    Compute ROUGE scores between prediction and reference.

    Args:
        pred: Predicted text string.
        ref: Reference text string.
        rouge_types: ROUGE variants to compute.

    Returns:
        Dict mapping rouge type to F1 score.
    """
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(list(rouge_types), use_stemmer=True)
        scores = scorer.score(ref, pred)
        return {k: v.fmeasure for k, v in scores.items()}
    except ImportError:
        # Fallback: token overlap
        pred_tokens = pred.lower().split()
        ref_tokens = ref.lower().split()
        overlap = len(set(pred_tokens) & set(ref_tokens))
        denom = max(len(ref_tokens), 1)
        f1 = overlap / denom
        return {rt: f1 for rt in rouge_types}


def compute_meteor(pred: str, ref: str) -> float:
    """
    Compute METEOR score between prediction and reference.

    Args:
        pred: Predicted text string.
        ref: Reference text string.

    Returns:
        METEOR score in range [0, 1].
    """
    try:
        import nltk
        from nltk.translate.meteor_score import meteor_score
        try:
            nltk.data.find("wordnet")
        except LookupError:
            nltk.download("wordnet", quiet=True)
        try:
            nltk.data.find("omw-1.4")
        except LookupError:
            nltk.download("omw-1.4", quiet=True)

        return float(meteor_score([ref.split()], pred.split()))
    except (ImportError, Exception) as e:
        logger.debug(f"METEOR fallback due to: {e}")
        # Fallback: harmonic mean of precision and recall
        pred_tokens = pred.lower().split()
        ref_tokens = ref.lower().split()
        overlap = len(set(pred_tokens) & set(ref_tokens))
        precision = overlap / max(len(pred_tokens), 1)
        recall = overlap / max(len(ref_tokens), 1)
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)


def compute_bert_score(
    pred: str,
    ref: str,
    model_type: str = "bert-base-uncased",
) -> Dict[str, float]:
    """
    Compute BERTScore between prediction and reference.

    Args:
        pred: Predicted text string.
        ref: Reference text string.
        model_type: Pre-trained BERT model to use.

    Returns:
        Dict with 'precision', 'recall', 'f1' keys.
    """
    try:
        from bert_score import score as bert_score_fn
        P, R, F1 = bert_score_fn([pred], [ref], model_type=model_type, verbose=False)
        return {
            "precision": float(P[0]),
            "recall": float(R[0]),
            "f1": float(F1[0]),
        }
    except (ImportError, Exception) as e:
        logger.debug(f"BERTScore fallback due to: {e}")
        # Fallback: token-level cosine-like similarity
        pred_tokens = set(pred.lower().split())
        ref_tokens = set(ref.lower().split())
        overlap = len(pred_tokens & ref_tokens)
        precision = overlap / max(len(pred_tokens), 1)
        recall = overlap / max(len(ref_tokens), 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        return {"precision": precision, "recall": recall, "f1": f1}


def compute_hallucination_rate(
    verified_claims: List[Dict[str, Union[str, bool]]],
) -> float:
    """
    Compute hallucination rate from verified claim annotations.

    Args:
        verified_claims: List of dicts with keys:
            - 'claim' (str): The claim text.
            - 'supported' (bool): Whether the claim is factually supported.

    Returns:
        Hallucination rate in range [0, 1] (fraction of unsupported claims).
    """
    if not verified_claims:
        return 0.0
    flagged = sum(1 for c in verified_claims if not c.get("supported", True))
    return flagged / len(verified_claims)


def compute_latency(start: float, end: float) -> Dict[str, float]:
    """
    Compute latency statistics between start and end timestamps.

    Args:
        start: Start timestamp (from time.time()).
        end: End timestamp (from time.time()).

    Returns:
        Dict with 'total_seconds' and 'milliseconds' keys.
    """
    elapsed = end - start
    return {
        "total_seconds": round(elapsed, 4),
        "milliseconds": round(elapsed * 1000, 2),
    }


def compute_multiturn_consistency(
    responses: List[str],
) -> float:
    """
    Compute consistency score across multiple conversation turns.

    Uses pairwise cosine similarity of TF-IDF vectors as a proxy for
    semantic consistency across responses.

    Args:
        responses: List of response strings across turns.

    Returns:
        Mean pairwise similarity in range [0, 1].
    """
    if len(responses) < 2:
        return 1.0

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        vectorizer = TfidfVectorizer(stop_words="english", max_features=500)
        tfidf_matrix = vectorizer.fit_transform(responses)
        sim_matrix = cosine_similarity(tfidf_matrix)

        # Average upper-triangle (pairwise, excluding self-similarity)
        n = len(responses)
        total, count = 0.0, 0
        for i in range(n):
            for j in range(i + 1, n):
                total += sim_matrix[i, j]
                count += 1
        return float(total / count) if count > 0 else 1.0

    except (ImportError, Exception) as e:
        logger.debug(f"Consistency fallback due to: {e}")
        # Fallback: Jaccard similarity between consecutive responses
        similarities = []
        for i in range(len(responses) - 1):
            a = set(responses[i].lower().split())
            b = set(responses[i + 1].lower().split())
            union = a | b
            if union:
                similarities.append(len(a & b) / len(union))
        return float(np.mean(similarities)) if similarities else 1.0


class LatencyTimer:
    """Context manager for measuring execution latency."""

    def __init__(self, name: str = "operation"):
        self.name = name
        self.start: float = 0.0
        self.end: float = 0.0
        self.result: Dict[str, float] = {}

    def __enter__(self) -> "LatencyTimer":
        self.start = time.time()
        return self

    def __exit__(self, *args) -> None:
        self.end = time.time()
        self.result = compute_latency(self.start, self.end)
        logger.debug(f"{self.name} latency: {self.result['milliseconds']}ms")
