"""
M1 AMQU — Adaptive Multi-shot Query Understanding

Replaces the original MedAide CFG parser with:
  1. Flan-T5-base for subquery decomposition (with rule-based fallback)
  2. K=5 multi-shot consistency filtering via cosine similarity clustering
  3. Recency-weighted BM25: w(t) = exp(-λ(T-t)) with λ=0.1

Reference: arXiv 2410.12532 (MedAide), Section 3.1
"""

import re
import math
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("medaide_plus.m1_amqu")


@dataclass
class SubQuery:
    """Represents a decomposed subquery with metadata."""
    text: str
    confidence: float = 1.0
    source: str = "model"  # "model" | "rule_based"
    timestamp: float = 0.0  # Unix timestamp for recency weighting


@dataclass
class AMQUResult:
    """Result from AMQU module processing."""
    original_query: str
    subqueries: List[SubQuery] = field(default_factory=list)
    bm25_weighted_docs: List[Tuple[str, float]] = field(default_factory=list)
    processing_metadata: Dict = field(default_factory=dict)


class AMQUModule:
    """
    Adaptive Multi-shot Query Understanding module.

    Decomposes complex medical queries into structured subqueries using
    Flan-T5-base with consistency filtering, then applies recency-weighted
    BM25 to retrieve relevant knowledge base passages.

    Args:
        config: Configuration dict with AMQU parameters.
        corpus: Optional list of document strings for BM25 indexing.
    """

    # Regex patterns for rule-based fallback decomposition
    CONJUNCTION_PATTERNS = [
        r"\band\b",
        r"\balso\b",
        r"\bfurthermore\b",
        r"\bin addition\b",
        r"\bmoreover\b",
    ]
    QUESTION_PATTERNS = [
        r"(?<=[.?!])\s+(?=[A-Z])",  # Sentence boundary
        r"\bwhat\b|\bwhy\b|\bhow\b|\bwhen\b|\bwhere\b|\bwhich\b",
    ]

    def __init__(
        self,
        config: Optional[Dict] = None,
        corpus: Optional[List[str]] = None,
    ) -> None:
        self.config = config or {}
        self.k_shots: int = self.config.get("k_shots", 5)
        self.consistency_threshold: float = self.config.get("consistency_threshold", 0.85)
        self.min_count: int = self.config.get("min_count", 3)
        self.recency_lambda: float = self.config.get("recency_lambda", 0.1)
        self.max_subqueries: int = self.config.get("max_subqueries", 5)

        self._model = None
        self._tokenizer = None
        self._sentence_model = None
        self._bm25 = None
        self._corpus = corpus or []
        self._corpus_timestamps: List[float] = []

        self._load_models()
        if self._corpus:
            self._build_bm25_index(self._corpus)

    def _load_models(self) -> None:
        """Load Flan-T5 and sentence-transformer models with graceful fallback."""
        # Load Flan-T5 for decomposition
        try:
            from transformers import T5ForConditionalGeneration, T5Tokenizer
            import torch
            model_name = "google/flan-t5-base"
            logger.info(f"Loading Flan-T5 model: {model_name}")
            self._tokenizer = T5Tokenizer.from_pretrained(model_name)
            self._model = T5ForConditionalGeneration.from_pretrained(model_name).cpu()  # CPU; leave GPU for Ollama
            self._model.eval()
            logger.info("Flan-T5 loaded successfully.")
        except Exception as e:
            logger.warning(f"Could not load Flan-T5 ({e}). Using rule-based fallback.")
            self._model = None
            self._tokenizer = None

        # Load sentence-transformer for consistency filtering
        try:
            from sentence_transformers import SentenceTransformer
            self._sentence_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        except Exception as e:
            logger.warning(f"Could not load sentence-transformer ({e}). Using TF-IDF fallback.")
            self._sentence_model = None

    def _build_bm25_index(
        self,
        corpus: List[str],
        timestamps: Optional[List[float]] = None,
    ) -> None:
        """
        Build BM25 index over the provided corpus.

        Args:
            corpus: List of document strings.
            timestamps: Optional list of Unix timestamps per document.
                        If None, all documents get timestamp 0.
        """
        import time
        self._corpus = corpus
        self._corpus_timestamps = timestamps or [0.0] * len(corpus)

        try:
            from rank_bm25 import BM25Okapi
            tokenized = [doc.lower().split() for doc in corpus]
            self._bm25 = BM25Okapi(tokenized)
            logger.debug(f"BM25 index built over {len(corpus)} documents.")
        except ImportError:
            logger.warning("rank-bm25 not available. BM25 retrieval disabled.")
            self._bm25 = None

    def run(self, query: str, top_k: int = 5) -> AMQUResult:
        """
        Full AMQU pipeline: decompose → filter → BM25 retrieve.

        Args:
            query: The original user medical query.
            top_k: Number of top BM25 documents to retrieve.

        Returns:
            AMQUResult with subqueries and weighted documents.
        """
        logger.info(f"AMQU processing query: {query[:80]}...")
        result = AMQUResult(original_query=query)

        # Step 1: Generate K candidate subquery sets
        candidate_sets = self._generate_subqueries(query, k=self.k_shots)

        # Step 2: Filter by consistency
        filtered_subqueries = self._filter_by_consistency(candidate_sets)
        result.subqueries = filtered_subqueries[:self.max_subqueries]

        # Step 3: Recency-weighted BM25 retrieval
        if self._bm25 and self._corpus:
            combined_query = " ".join(sq.text for sq in result.subqueries)
            result.bm25_weighted_docs = self._recency_weighted_bm25(
                combined_query, top_k=top_k
            )

        result.processing_metadata = {
            "n_candidates": sum(len(s) for s in candidate_sets),
            "n_filtered": len(result.subqueries),
            "model_used": "flan-t5" if self._model else "rule_based",
        }
        logger.info(
            f"AMQU complete: {len(result.subqueries)} subqueries, "
            f"{len(result.bm25_weighted_docs)} docs retrieved."
        )
        return result

    def _generate_subqueries(self, query: str, k: int = 5) -> List[List[SubQuery]]:
        """
        Generate K sets of candidate subqueries using Flan-T5 or rule-based fallback.

        Args:
            query: Original query string.
            k: Number of candidate sets to generate.

        Returns:
            List of k lists of SubQuery objects.
        """
        if self._model and self._tokenizer:
            return self._model_based_decomposition(query, k)
        else:
            # All k shots use the same rule-based result (deterministic)
            rule_result = self._rule_based_decomposition(query)
            return [rule_result] * k

    def _model_based_decomposition(self, query: str, k: int) -> List[List[SubQuery]]:
        """Use Flan-T5 to generate k diverse decompositions via sampling."""
        import torch
        prompt = (
            f"Decompose the following medical question into simpler sub-questions. "
            f"List each sub-question on a new line.\n\nQuestion: {query}\n\nSub-questions:"
        )
        inputs = self._tokenizer(
            prompt, return_tensors="pt", max_length=512, truncation=True
        )

        candidate_sets: List[List[SubQuery]] = []
        with torch.no_grad():
            for _ in range(k):
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                )
                decoded = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
                subqueries = self._parse_decomposition_output(decoded, source="model")
                candidate_sets.append(subqueries)

        return candidate_sets

    def _rule_based_decomposition(self, query: str) -> List[SubQuery]:
        """
        Rule-based query decomposition as fallback.

        Splits on conjunctions and sentence boundaries to identify
        independent medical sub-questions.

        Args:
            query: Query to decompose.

        Returns:
            List of SubQuery objects.
        """
        # Split on conjunctions
        parts = [query]
        for pattern in self.CONJUNCTION_PATTERNS:
            new_parts = []
            for part in parts:
                new_parts.extend(re.split(pattern, part, flags=re.IGNORECASE))
            parts = new_parts

        # Further split on sentence boundaries
        final_parts = []
        for part in parts:
            sentences = re.split(r"[.?!]+", part.strip())
            final_parts.extend([s.strip() for s in sentences if len(s.strip()) > 10])

        if not final_parts:
            final_parts = [query]

        return [
            SubQuery(text=p, confidence=0.8, source="rule_based")
            for p in final_parts[:self.max_subqueries]
        ]

    def _parse_decomposition_output(self, text: str, source: str = "model") -> List[SubQuery]:
        """
        Parse model output into SubQuery objects.

        Args:
            text: Raw model output string.
            source: Origin label ('model' or 'rule_based').

        Returns:
            List of parsed SubQuery objects.
        """
        lines = [
            line.strip().lstrip("0123456789.-) ")
            for line in text.split("\n")
            if len(line.strip()) > 10
        ]
        if not lines:
            return [SubQuery(text=text.strip(), confidence=0.9, source=source)]

        return [SubQuery(text=line, confidence=0.9, source=source) for line in lines[:self.max_subqueries]]

    def _filter_by_consistency(
        self, candidate_sets: List[List[SubQuery]]
    ) -> List[SubQuery]:
        """
        Filter subqueries by cross-candidate consistency.

        Clusters subqueries from all K sets using cosine similarity.
        Retains subqueries that appear in at least min_count clusters
        with cosine similarity ≥ consistency_threshold.

        Args:
            candidate_sets: K lists of SubQuery objects.

        Returns:
            Filtered list of consistent SubQuery objects.
        """
        # Flatten all subqueries
        all_subqueries = [sq for sq_list in candidate_sets for sq in sq_list]
        if not all_subqueries:
            return []

        texts = [sq.text for sq in all_subqueries]

        # Compute embeddings
        embeddings = self._get_embeddings(texts)

        # Cluster by cosine similarity
        n = len(embeddings)
        if n == 1:
            return all_subqueries

        # Compute pairwise cosine similarity
        sim_matrix = self._cosine_sim_matrix(embeddings)

        # Greedy clustering: each unassigned item starts a new cluster
        cluster_ids = [-1] * n
        cluster_count = 0
        for i in range(n):
            if cluster_ids[i] == -1:
                cluster_ids[i] = cluster_count
                for j in range(i + 1, n):
                    if cluster_ids[j] == -1 and sim_matrix[i, j] >= self.consistency_threshold:
                        cluster_ids[j] = cluster_count
                cluster_count += 1

        # Count cluster sizes
        cluster_sizes: Dict[int, int] = {}
        cluster_representatives: Dict[int, SubQuery] = {}
        for i, cid in enumerate(cluster_ids):
            cluster_sizes[cid] = cluster_sizes.get(cid, 0) + 1
            if cid not in cluster_representatives:
                cluster_representatives[cid] = all_subqueries[i]

        # Retain clusters meeting min_count threshold
        consistent = [
            cluster_representatives[cid]
            for cid, size in cluster_sizes.items()
            if size >= min(self.min_count, len(candidate_sets))
        ]

        # If nothing passes the filter, return the first set (fallback)
        if not consistent:
            logger.debug("Consistency filter returned nothing; using first candidate set.")
            return candidate_sets[0] if candidate_sets else []

        logger.debug(
            f"Consistency filter: {n} candidates → {len(consistent)} retained "
            f"(threshold={self.consistency_threshold}, min_count={self.min_count})."
        )
        return consistent

    def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Compute text embeddings using sentence-transformer or TF-IDF fallback.

        Args:
            texts: List of strings to embed.

        Returns:
            2D numpy array of shape (len(texts), embedding_dim).
        """
        if self._sentence_model:
            embeddings = self._sentence_model.encode(texts, show_progress_bar=False)
            return np.array(embeddings)
        else:
            # TF-IDF fallback
            from sklearn.feature_extraction.text import TfidfVectorizer
            vectorizer = TfidfVectorizer(max_features=256)
            try:
                matrix = vectorizer.fit_transform(texts)
                return matrix.toarray()
            except Exception:
                # Last resort: one-hot bag of words
                return np.eye(len(texts))

    def _cosine_sim_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Compute pairwise cosine similarity matrix.

        Args:
            embeddings: 2D array of shape (n, d).

        Returns:
            Symmetric similarity matrix of shape (n, n).
        """
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-9, norms)
        normed = embeddings / norms
        return normed @ normed.T

    def _recency_weighted_bm25(
        self, query: str, top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Recency-weighted BM25 retrieval.

        Computes: score(d, q) = BM25(d, q) × exp(-λ(T - t_d))
        where T is the maximum timestamp, t_d is document timestamp,
        and λ=0.1 controls recency decay.

        Args:
            query: Query string for BM25 scoring.
            top_k: Number of top documents to return.

        Returns:
            List of (document_text, weighted_score) tuples, sorted by score.
        """
        if not self._bm25 or not self._corpus:
            return []

        import time
        T = max(self._corpus_timestamps) if self._corpus_timestamps else time.time()

        tokenized_query = query.lower().split()
        bm25_scores = self._bm25.get_scores(tokenized_query)

        # Apply recency weighting
        weighted_scores: List[Tuple[str, float]] = []
        for i, (doc, score) in enumerate(zip(self._corpus, bm25_scores)):
            t_d = self._corpus_timestamps[i] if i < len(self._corpus_timestamps) else 0.0
            recency_weight = math.exp(-self.recency_lambda * (T - t_d))
            weighted = float(score) * recency_weight
            weighted_scores.append((doc, weighted))

        # Sort by weighted score descending
        weighted_scores.sort(key=lambda x: x[1], reverse=True)
        return weighted_scores[:top_k]

    def update_corpus(
        self,
        corpus: List[str],
        timestamps: Optional[List[float]] = None,
    ) -> None:
        """
        Update the BM25 index with a new corpus.

        Args:
            corpus: New list of document strings.
            timestamps: Optional timestamps for recency weighting.
        """
        self._build_bm25_index(corpus, timestamps)
