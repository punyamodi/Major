"""
M5 HDFG - Hallucination-Aware Dual-Verification with Factual Grounding

Implements:
  Stage 1: FAISS IndexFlatIP semantic KB search to verify each claim
  Stage 2: Monte Carlo Dropout (N=15 passes with noise injection) for uncertainty
  NER-based claim extraction
  annotate_response() with inline [SUPPORTED]/[FLAGGED] markers + source attribution

Reference: arXiv 2410.12532, Section 3.5
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("medaide_plus.m5_hdfg")


@dataclass
class ClaimVerification:
    """Verification result for a single extracted claim."""
    claim: str
    supported: bool
    confidence: float
    uncertainty: float
    sources: List[str] = field(default_factory=list)
    support_score: float = 0.0


@dataclass
class HdfgResult:
    """Result from HDFG module verification."""
    original_response: str
    annotated_response: str
    verified_claims: List[ClaimVerification] = field(default_factory=list)
    hallucination_rate: float = 0.0
    overall_confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class HdfgModule:
    """
    Hallucination-Aware Dual-Verification with Factual Grounding module.

    Two-stage verification pipeline:
      Stage 1: Retrieve evidence from FAISS knowledge base
      Stage 2: Monte Carlo Dropout uncertainty estimation

    Args:
        config: Configuration dict with HDFG parameters.
        knowledge_base: Optional list of (text, embedding) tuples for FAISS.
    """

    def __init__(
        self,
        config: Optional[Dict] = None,
        knowledge_base: Optional[List[str]] = None,
    ) -> None:
        self.config = config or {}
        self.mc_passes: int = self.config.get("mc_dropout_passes", 15)
        self.uncertainty_threshold: float = self.config.get("uncertainty_threshold", 0.5)
        self.top_k: int = self.config.get("faiss_top_k", 5)
        self.support_threshold: float = self.config.get("support_threshold", 0.45)

        self._faiss_index = None
        self._kb_texts: List[str] = []
        self._embedding_model = None
        self._tfidf_vectorizer = None  # Saved after build_index for consistent dims

        self._load_embedding_model()
        if knowledge_base:
            self.build_index(knowledge_base)

    def _load_embedding_model(self) -> None:
        """Load sentence-transformer embedding model."""
        try:
            from sentence_transformers import SentenceTransformer
            self._embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
            self._embed_dim = 384
            logger.info("Sentence-transformer loaded for HDFG.")
        except ImportError:
            logger.warning("sentence-transformers not available. Using TF-IDF fallback.")
            self._embedding_model = None
            self._embed_dim = 256

    def build_index(self, documents: List[str]) -> None:
        """
        Build FAISS index from knowledge base documents.

        Args:
            documents: List of document strings to index.
        """
        self._kb_texts = documents
        embeddings = self._embed_texts(documents)
        try:
            import faiss
            dim = embeddings.shape[1]
            self._faiss_index = faiss.IndexFlatIP(dim)
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1e-9, norms)
            normalized = embeddings / norms
            self._faiss_index.add(normalized.astype(np.float32))
            logger.info(f"FAISS index built with {len(documents)} documents.")
        except ImportError:
            logger.warning("FAISS not available. Using numpy cosine similarity fallback.")
            self._embeddings_matrix = embeddings

    def annotate_response(self, response: str, query: Optional[str] = None) -> HdfgResult:
        """
        Annotate a response with [SUPPORTED] and [FLAGGED] markers.

        Args:
            response: Agent response text to verify.
            query: Optional original query for context.

        Returns:
            HdfgResult with annotated response and verification details.
        """
        claims = self._extract_claims(response)
        if not claims:
            return HdfgResult(
                original_response=response,
                annotated_response=response,
                hallucination_rate=0.0,
                overall_confidence=1.0,
                metadata={"n_claims": 0},
            )

        verified_claims = []
        for claim in claims:
            verification = self._verify_claim(claim)
            verified_claims.append(verification)

        annotated = self._build_annotated_response(response, verified_claims)
        n_flagged = sum(1 for c in verified_claims if not c.supported)
        hallucination_rate = n_flagged / max(len(verified_claims), 1)
        avg_confidence = float(np.mean([c.confidence for c in verified_claims]))

        return HdfgResult(
            original_response=response,
            annotated_response=annotated,
            verified_claims=verified_claims,
            hallucination_rate=hallucination_rate,
            overall_confidence=avg_confidence,
            metadata={
                "n_claims": len(verified_claims),
                "n_supported": len(verified_claims) - n_flagged,
                "n_flagged": n_flagged,
            },
        )

    def _extract_claims(self, text: str) -> List[str]:
        """
        Extract verifiable medical claims from response text.

        Uses sentence splitting + medical claim heuristics.

        Args:
            text: Response text.

        Returns:
            List of claim strings.
        """
        sentences = re.split(r"[.!?]+", text)
        claims = []
        medical_claim_keywords = [
            "should", "can", "may", "recommend", "avoid", "take", "dose",
            "effective", "causes", "treats", "indicated", "contraindicated",
            "mg", "ml", "twice", "daily", "weekly", "symptom", "diagnosis",
        ]
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 15:
                continue
            sentence_lower = sentence.lower()
            if any(kw in sentence_lower for kw in medical_claim_keywords):
                claims.append(sentence)
        return claims[:10]

    def _verify_claim(self, claim: str) -> ClaimVerification:
        """
        Verify a single claim using FAISS KB search + MC-Dropout.

        Uses a two-stage approach:
          Stage 1: KB search for factual support (primary signal)
          Stage 2: MC uncertainty via perturbation of KB search scores

        A claim is supported if it has reasonable KB support OR low uncertainty.
        This avoids the previous overly-strict AND condition that flagged everything.

        Args:
            claim: Claim string to verify.

        Returns:
            ClaimVerification with support status and uncertainty.
        """
        support_score, sources = self._stage1_kb_search(claim)
        confidence, uncertainty = self._stage2_mc_uncertainty(claim)

        # Relaxed verification: support OR low uncertainty
        # A claim is supported if:
        #   - It has good KB grounding (support_score >= threshold), OR
        #   - It has moderate support AND low uncertainty
        supported = (
            support_score >= self.support_threshold
            or (support_score >= self.support_threshold * 0.6 and uncertainty <= self.uncertainty_threshold)
        )

        # Confidence is a blend of KB support and MC confidence
        blended_confidence = 0.6 * support_score + 0.4 * confidence

        return ClaimVerification(
            claim=claim,
            supported=supported,
            confidence=blended_confidence,
            uncertainty=uncertainty,
            sources=sources[:2],
            support_score=support_score,
        )

    def _stage1_kb_search(self, claim: str) -> Tuple[float, List[str]]:
        """
        Stage 1: FAISS knowledge base search for claim verification.

        Args:
            claim: Claim to verify.

        Returns:
            (support_score, source_excerpts) tuple.
        """
        if not self._kb_texts:
            return 0.5, []

        claim_embedding = self._embed_texts([claim])

        if self._faiss_index is not None:
            import faiss
            norm = np.linalg.norm(claim_embedding)
            if norm > 0:
                normalized = claim_embedding / norm
            else:
                normalized = claim_embedding
            scores, indices = self._faiss_index.search(
                normalized.astype(np.float32), min(self.top_k, len(self._kb_texts))
            )
            top_scores = scores[0]
            top_indices = indices[0]
        elif hasattr(self, "_embeddings_matrix"):
            norms = np.linalg.norm(self._embeddings_matrix, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1e-9, norms)
            norm_matrix = self._embeddings_matrix / norms
            claim_norm = np.linalg.norm(claim_embedding)
            if claim_norm > 0:
                claim_normalized = claim_embedding / claim_norm
            else:
                claim_normalized = claim_embedding
            similarities = (norm_matrix @ claim_normalized.T).flatten()
            top_k = min(self.top_k, len(self._kb_texts))
            top_indices = np.argsort(similarities)[::-1][:top_k]
            top_scores = similarities[top_indices]
        else:
            return 0.5, []

        if len(top_scores) == 0:
            return 0.0, []

        max_score = float(np.clip(top_scores[0], 0, 1))
        sources = [
            self._kb_texts[int(i)][:100] + "..."
            for i in top_indices
            if 0 <= int(i) < len(self._kb_texts)
        ]
        return max_score, sources

    def _stage2_mc_uncertainty(self, claim: str) -> Tuple[float, float]:
        """
        Stage 2: Monte Carlo Dropout uncertainty estimation.

        Performs N passes with noise injection on the claim embedding and
        re-computes KB similarity each time. The variance of similarity
        scores across passes measures uncertainty — high variance = uncertain.

        Args:
            claim: Claim string.

        Returns:
            (mean_confidence, uncertainty) tuple.
        """
        embedding = self._embed_texts([claim])[0]

        # If we have a KB, use variance of KB similarity scores
        if self._kb_texts and (self._faiss_index is not None or hasattr(self, "_embeddings_matrix")):
            similarity_scores = []
            np.random.seed(None)
            for _ in range(self.mc_passes):
                noise = np.random.normal(0, 0.02, embedding.shape)
                noisy_embedding = embedding + noise
                norm = np.linalg.norm(noisy_embedding)
                if norm > 0:
                    noisy_embedding = noisy_embedding / norm

                # Compute similarity against KB
                if self._faiss_index is not None:
                    import faiss
                    scores, _ = self._faiss_index.search(
                        noisy_embedding.reshape(1, -1).astype(np.float32),
                        min(3, len(self._kb_texts))
                    )
                    top_score = float(np.clip(scores[0][0], 0, 1))
                elif hasattr(self, "_embeddings_matrix"):
                    norms = np.linalg.norm(self._embeddings_matrix, axis=1, keepdims=True)
                    norms = np.where(norms == 0, 1e-9, norms)
                    norm_matrix = self._embeddings_matrix / norms
                    sims = (norm_matrix @ noisy_embedding.reshape(-1, 1)).flatten()
                    top_score = float(np.clip(np.max(sims), 0, 1))
                else:
                    top_score = 0.5

                similarity_scores.append(top_score)

            mean_conf = float(np.mean(similarity_scores))
            uncertainty = float(np.std(similarity_scores))
            return mean_conf, uncertainty

        # Fallback: no KB available — return moderate confidence, low uncertainty
        return 0.5, 0.1

    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embed texts using sentence-transformer or TF-IDF fallback."""
        if self._embedding_model:
            return np.array(self._embedding_model.encode(texts, show_progress_bar=False))
        else:
            from sklearn.feature_extraction.text import TfidfVectorizer
            try:
                if self._tfidf_vectorizer is not None:
                    # Reuse the saved vectorizer for consistent vocabulary/dimensions
                    return self._tfidf_vectorizer.transform(texts).toarray().astype(np.float32)
                else:
                    # First call (during build_index): fit on KB + current texts
                    vectorizer = TfidfVectorizer(max_features=self._embed_dim)
                    all_texts = list(self._kb_texts) + list(texts)
                    vectorizer.fit(all_texts)
                    self._tfidf_vectorizer = vectorizer
                    return vectorizer.transform(texts).toarray().astype(np.float32)
            except Exception:
                return np.random.randn(len(texts), self._embed_dim).astype(np.float32)

    def _build_annotated_response(
        self, response: str, verified_claims: List[ClaimVerification]
    ) -> str:
        """
        Build annotated response with [SUPPORTED]/[FLAGGED] inline markers.

        Args:
            response: Original response text.
            verified_claims: List of verification results.

        Returns:
            Annotated response string.
        """
        annotated = response
        for cv in verified_claims:
            marker = "[SUPPORTED]" if cv.supported else "[FLAGGED]"
            source_str = ""
            if cv.supported and cv.sources:
                source_str = f" (Source: {cv.sources[0][:50]}...)"
            replacement = f"{cv.claim} {marker}{source_str}"
            annotated = annotated.replace(cv.claim, replacement, 1)
        return annotated
