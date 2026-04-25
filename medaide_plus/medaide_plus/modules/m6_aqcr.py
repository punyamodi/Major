"""
M6 AQCR - Adaptive Query Complexity Routing

Implements:
  RoBERTa-base classifier (3 classes: Simple/Moderate/Complex)
  Feature-based routing fallback: n_intents, query_length, inter-intent cosine distance
  Tier -> n_agents: Simple=1, Moderate=2, Complex=4
  Training function for silver label generation

Reference: arXiv 2410.12532, Section 3.6
"""

import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("medaide_plus.m6_aqcr")

COMPLEXITY_TIERS = {
    "Simple": 1,
    "Moderate": 2,
    "Complex": 4,
}

TIER_LABELS = ["Simple", "Moderate", "Complex"]


@dataclass
class AqcrResult:
    """Result from AQCR routing decision."""
    tier: str
    n_agents: int
    confidence: float
    routing_method: str
    features: Dict[str, float]
    tier_probabilities: Dict[str, float]


class AqcrModule:
    """
    Adaptive Query Complexity Routing module.

    Routes medical queries to an appropriate number of agents based on
    estimated complexity: Simple (1), Moderate (2), or Complex (4).

    Args:
        config: Configuration dict with AQCR parameters.
    """

    def __init__(self, config: Optional[Dict] = None) -> None:
        self.config = config or {}
        self.simple_threshold: float = self.config.get("simple_threshold", 0.33)
        self.complex_threshold: float = self.config.get("complex_threshold", 0.66)

        self._model = None
        self._tokenizer = None
        self._sentence_model = None
        self._routing_method = "feature_based"

        self._load_models()

    def _load_models(self) -> None:
        """Load sentence-transformer for features. RoBERTa is disabled
        because the model requires fine-tuning on medical complexity data
        to produce meaningful routing. The feature-based fallback is more
        reliable for un-tuned deployments."""
        # RoBERTa disabled — would need fine-tuning on medical complexity labels
        self._model = None
        self._tokenizer = None
        self._routing_method = "feature_based"

        try:
            from sentence_transformers import SentenceTransformer
            self._sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            pass

    def route(
        self,
        query: str,
        intents: Optional[List[str]] = None,
        subqueries: Optional[List[str]] = None,
    ) -> AqcrResult:
        """
        Route a query to the appropriate complexity tier.

        Args:
            query: Medical query string.
            intents: Detected intent labels (from M2 HDIO).
            subqueries: Decomposed subqueries (from M1 AMQU).

        Returns:
            AqcrResult with tier, n_agents, and routing metadata.
        """
        intents = intents or []
        subqueries = subqueries or [query]

        features = self._extract_features(query, intents, subqueries)

        if self._model and self._tokenizer and self._routing_method == "roberta":
            result = self._roberta_route(query, features)
        else:
            result = self._feature_route(features)

        logger.info(
            f"AQCR: query routed to '{result.tier}' tier "
            f"({result.n_agents} agents, method={result.routing_method})."
        )
        return result

    def _extract_features(
        self,
        query: str,
        intents: List[str],
        subqueries: List[str],
    ) -> Dict[str, float]:
        """
        Extract routing features from query and detected intents.

        Features:
          - query_length: normalized query character count
          - n_intents: number of detected intents (normalized)
          - n_subqueries: number of decomposed subqueries
          - inter_intent_distance: mean pairwise cosine distance between intents
          - has_medication_intent: 1.0 if medication intents detected
          - has_multi_category: 1.0 if intents span multiple categories

        Args:
            query: Original query.
            intents: List of detected intent labels.
            subqueries: List of decomposed subqueries.

        Returns:
            Feature dict.
        """
        from medaide_plus.modules.m2_hdio import INTENT_TO_CATEGORY

        features: Dict[str, float] = {}

        features["query_length"] = min(len(query) / 500.0, 1.0)
        features["n_intents"] = min(len(intents) / 5.0, 1.0)
        features["n_subqueries"] = min(len(subqueries) / 5.0, 1.0)

        has_medication = any(
            "medication" in INTENT_TO_CATEGORY.get(i, "").lower()
            or "drug" in i.lower()
            or "dosage" in i.lower()
            for i in intents
        )
        features["has_medication_intent"] = 1.0 if has_medication else 0.0

        categories = set(INTENT_TO_CATEGORY.get(i, "Unknown") for i in intents)
        features["has_multi_category"] = 1.0 if len(categories) > 1 else 0.0

        if len(intents) >= 2 and self._sentence_model:
            try:
                intent_embeddings = self._sentence_model.encode(
                    [i.replace("_", " ") for i in intents],
                    show_progress_bar=False,
                )
                norms = np.linalg.norm(intent_embeddings, axis=1, keepdims=True)
                norms = np.where(norms == 0, 1e-9, norms)
                normed = intent_embeddings / norms
                sim_matrix = normed @ normed.T
                n = len(intents)
                distances = []
                for i in range(n):
                    for j in range(i + 1, n):
                        distances.append(1.0 - sim_matrix[i, j])
                features["inter_intent_distance"] = float(np.mean(distances))
            except Exception:
                features["inter_intent_distance"] = 0.0
        else:
            features["inter_intent_distance"] = 0.0

        complexity_keywords = [
            "interaction", "contraindication", "combination", "multiple",
            "chronic", "comorbidity", "differential", "complicated",
        ]
        kw_matches = sum(1 for kw in complexity_keywords if kw in query.lower())
        features["complexity_keywords"] = min(kw_matches / 3.0, 1.0)

        return features

    def _feature_route(self, features: Dict[str, float]) -> AqcrResult:
        """
        Feature-based complexity routing (fallback).

        Computes a complexity score as a weighted sum of features.
        Maps score to tier: [0, 0.33) -> Simple, [0.33, 0.66) -> Moderate, [0.66, 1] -> Complex.

        Args:
            features: Extracted feature dict.

        Returns:
            AqcrResult.
        """
        weights = {
            "query_length": 0.15,
            "n_intents": 0.25,
            "n_subqueries": 0.20,
            "has_medication_intent": 0.10,
            "has_multi_category": 0.15,
            "inter_intent_distance": 0.05,
            "complexity_keywords": 0.10,
        }

        complexity_score = sum(
            weights.get(k, 0.0) * v for k, v in features.items()
        )
        complexity_score = float(np.clip(complexity_score, 0.0, 1.0))

        if complexity_score < self.simple_threshold:
            tier = "Simple"
        elif complexity_score < self.complex_threshold:
            tier = "Moderate"
        else:
            tier = "Complex"

        tier_probs = {
            "Simple": max(0.0, 1.0 - complexity_score / self.simple_threshold)
            if complexity_score < self.simple_threshold
            else 0.1,
            "Moderate": 0.8 if tier == "Moderate" else 0.1,
            "Complex": complexity_score if tier == "Complex" else 0.1,
        }
        total = sum(tier_probs.values())
        tier_probs = {k: v / total for k, v in tier_probs.items()}

        return AqcrResult(
            tier=tier,
            n_agents=COMPLEXITY_TIERS[tier],
            confidence=float(max(tier_probs.values())),
            routing_method="feature_based",
            features=features,
            tier_probabilities=tier_probs,
        )

    def _roberta_route(self, query: str, features: Dict[str, float]) -> AqcrResult:
        """
        RoBERTa-based complexity classification.

        Args:
            query: Query text.
            features: Pre-computed features (used for confidence adjustment).

        Returns:
            AqcrResult.
        """
        import torch

        inputs = self._tokenizer(
            query, return_tensors="pt", max_length=512,
            truncation=True, padding=True,
        )
        with torch.no_grad():
            logits = self._model(**inputs).logits

        probs = torch.softmax(logits, dim=-1).squeeze().numpy()
        tier_probs = {TIER_LABELS[i]: float(probs[i]) for i in range(3)}
        tier = TIER_LABELS[int(np.argmax(probs))]
        confidence = float(np.max(probs))

        return AqcrResult(
            tier=tier,
            n_agents=COMPLEXITY_TIERS[tier],
            confidence=confidence,
            routing_method="roberta",
            features=features,
            tier_probabilities=tier_probs,
        )

    def generate_silver_labels(
        self, queries: List[str], intent_lists: Optional[List[List[str]]] = None
    ) -> List[str]:
        """
        Generate silver complexity labels for a list of queries using features.

        Useful for creating training data for fine-tuning the RoBERTa classifier.

        Args:
            queries: List of query strings.
            intent_lists: Optional list of intent lists per query.

        Returns:
            List of complexity tier labels.
        """
        labels = []
        for i, query in enumerate(queries):
            intents = intent_lists[i] if intent_lists and i < len(intent_lists) else []
            result = self._feature_route(
                self._extract_features(query, intents, [query])
            )
            labels.append(result.tier)
        return labels

# Aliases for backward compatibility
AQCRModule = AqcrModule
AQCRResult = AqcrResult
TIER_AGENTS = {k.lower(): v for k, v in COMPLEXITY_TIERS.items()}

