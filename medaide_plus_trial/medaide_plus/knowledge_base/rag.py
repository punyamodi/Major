"""
RAGRetriever — Hybrid retrieval with sentence-transformer + FAISS + BM25 fallback.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("medaide_plus.rag")


@dataclass
class Document:
    """A knowledge base document."""
    id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None


@dataclass
class RetrievalResult:
    """Result from RAG retrieval."""
    documents: List[Document] = field(default_factory=list)
    scores: List[float] = field(default_factory=list)
    method: str = "hybrid"


class RAGRetriever:
    """
    Hybrid Retrieval-Augmented Generation retriever.

    Combines:
      - Sentence-transformer dense embeddings + FAISS index
      - BM25 sparse retrieval as fallback

    Args:
        config: Configuration dict.
        model_name: Sentence-transformer model name.
    """

    def __init__(
        self,
        config: Optional[Dict] = None,
        model_name: str = "all-MiniLM-L6-v2",
    ) -> None:
        self.config = config or {}
        self.model_name = model_name
        self.top_k: int = self.config.get("top_k", 5)
        self.chunk_size: int = self.config.get("chunk_size", 512)

        self._documents: List[Document] = []
        self._embedding_model = None
        self._faiss_index = None
        self._bm25 = None
        self._embed_dim = 384

        self._load_models()

    def _load_models(self) -> None:
        """Load sentence-transformer and initialize indices."""
        try:
            from sentence_transformers import SentenceTransformer
            self._embedding_model = SentenceTransformer(self.model_name, device="cpu")
            self._embed_dim = self._embedding_model.get_sentence_embedding_dimension()
            logger.info(f"Sentence-transformer loaded: {self.model_name}")
        except Exception as e:
            logger.warning(f"Sentence-transformer unavailable: {e}")

    def add_documents(self, documents: List[str], metadata: Optional[List[Dict]] = None) -> None:
        """
        Add documents to the retrieval index.

        Args:
            documents: List of document text strings.
            metadata: Optional metadata list (one dict per document).
        """
        metadata = metadata or [{}] * len(documents)
        new_docs = []
        for i, (text, meta) in enumerate(zip(documents, metadata)):
            doc_id = meta.get("id", f"doc_{len(self._documents) + i}")
            doc = Document(id=doc_id, text=text, metadata=meta)
            new_docs.append(doc)
            self._documents.append(doc)

        self._build_dense_index()
        self._build_sparse_index()
        logger.info(f"Added {len(new_docs)} documents. Total: {len(self._documents)}.")

    def _build_dense_index(self) -> None:
        """Build FAISS dense index from document embeddings."""
        if not self._documents:
            return
        texts = [d.text for d in self._documents]
        embeddings = self._embed(texts)
        for i, doc in enumerate(self._documents):
            doc.embedding = embeddings[i]

        try:
            import faiss
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1e-9, norms)
            normalized = (embeddings / norms).astype(np.float32)
            self._faiss_index = faiss.IndexFlatIP(self._embed_dim)
            self._faiss_index.add(normalized)
            logger.debug(f"FAISS index built: {self._faiss_index.ntotal} vectors.")
        except ImportError:
            logger.warning("FAISS not available. Dense retrieval will use numpy.")

    def _build_sparse_index(self) -> None:
        """Build BM25 sparse index."""
        try:
            from rank_bm25 import BM25Okapi
            tokenized = [d.text.lower().split() for d in self._documents]
            self._bm25 = BM25Okapi(tokenized)
            logger.debug("BM25 index built.")
        except ImportError:
            logger.warning("rank-bm25 not available. BM25 retrieval disabled.")

    def retrieve(self, query: str, top_k: Optional[int] = None) -> RetrievalResult:
        """
        Dense-only retrieval using FAISS or numpy.

        Args:
            query: Query string.
            top_k: Number of documents to retrieve.

        Returns:
            RetrievalResult.
        """
        k = top_k or self.top_k
        if not self._documents:
            return RetrievalResult(method="dense")

        query_embedding = self._embed([query])[0]

        if self._faiss_index is not None:
            return self._faiss_retrieve(query_embedding, k)
        else:
            return self._numpy_retrieve(query_embedding, k)

    def _faiss_retrieve(self, query_embedding: np.ndarray, k: int) -> RetrievalResult:
        """Retrieve using FAISS index."""
        import faiss
        norm = np.linalg.norm(query_embedding)
        if norm > 0:
            q_norm = (query_embedding / norm).reshape(1, -1).astype(np.float32)
        else:
            q_norm = query_embedding.reshape(1, -1).astype(np.float32)

        scores, indices = self._faiss_index.search(q_norm, min(k, len(self._documents)))
        docs = [self._documents[int(i)] for i in indices[0] if 0 <= int(i) < len(self._documents)]
        scs = [float(s) for s in scores[0]]
        return RetrievalResult(documents=docs, scores=scs, method="dense_faiss")

    def _numpy_retrieve(self, query_embedding: np.ndarray, k: int) -> RetrievalResult:
        """Retrieve using numpy cosine similarity."""
        embeddings = np.array([d.embedding for d in self._documents if d.embedding is not None])
        if len(embeddings) == 0:
            return RetrievalResult(method="numpy")

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-9, norms)
        normed = embeddings / norms
        q_norm = query_embedding / max(np.linalg.norm(query_embedding), 1e-9)
        similarities = normed @ q_norm
        top_indices = np.argsort(similarities)[::-1][:k]
        docs = [self._documents[int(i)] for i in top_indices]
        scs = [float(similarities[int(i)]) for i in top_indices]
        return RetrievalResult(documents=docs, scores=scs, method="dense_numpy")

    def bm25_retrieve(self, query: str, top_k: Optional[int] = None) -> RetrievalResult:
        """BM25 sparse retrieval."""
        k = top_k or self.top_k
        if not self._bm25 or not self._documents:
            return RetrievalResult(method="bm25")

        tokenized = query.lower().split()
        scores = self._bm25.get_scores(tokenized)
        top_indices = np.argsort(scores)[::-1][:k]
        docs = [self._documents[int(i)] for i in top_indices]
        scs = [float(scores[int(i)]) for i in top_indices]
        return RetrievalResult(documents=docs, scores=scs, method="bm25")

    def hybrid_retrieve(
        self, query: str, top_k: Optional[int] = None, dense_weight: float = 0.7
    ) -> RetrievalResult:
        """
        Hybrid retrieval: weighted combination of dense + BM25 scores.

        Args:
            query: Query string.
            top_k: Number of results.
            dense_weight: Weight for dense scores (1-dense_weight for BM25).

        Returns:
            RetrievalResult with hybrid-ranked documents.
        """
        k = top_k or self.top_k
        dense_result = self.retrieve(query, top_k=len(self._documents))
        bm25_result = self.bm25_retrieve(query, top_k=len(self._documents))

        if not dense_result.documents and not bm25_result.documents:
            return RetrievalResult(method="hybrid")

        # Combine scores
        score_map: Dict[str, float] = {}

        # Normalize and add dense scores
        dense_scores = np.array(dense_result.scores) if dense_result.scores else np.array([])
        if len(dense_scores) > 0:
            max_dense = max(abs(dense_scores.max()), 1e-9)
            for doc, score in zip(dense_result.documents, dense_result.scores):
                score_map[doc.id] = score_map.get(doc.id, 0) + dense_weight * score / max_dense

        # Normalize and add BM25 scores
        bm25_scores = np.array(bm25_result.scores) if bm25_result.scores else np.array([])
        if len(bm25_scores) > 0:
            max_bm25 = max(bm25_scores.max(), 1e-9)
            for doc, score in zip(bm25_result.documents, bm25_result.scores):
                score_map[doc.id] = (
                    score_map.get(doc.id, 0) + (1 - dense_weight) * score / max_bm25
                )

        # Sort and return top_k
        sorted_ids = sorted(score_map, key=score_map.get, reverse=True)[:k]
        doc_lookup = {d.id: d for d in dense_result.documents + bm25_result.documents}
        result_docs = [doc_lookup[did] for did in sorted_ids if did in doc_lookup]
        result_scores = [score_map[did] for did in sorted_ids if did in doc_lookup]

        return RetrievalResult(documents=result_docs, scores=result_scores, method="hybrid")

    def _embed(self, texts: List[str]) -> np.ndarray:
        """Embed texts using sentence-transformer or TF-IDF fallback."""
        if self._embedding_model:
            return np.array(self._embedding_model.encode(texts, show_progress_bar=False))
        else:
            from sklearn.feature_extraction.text import TfidfVectorizer
            try:
                all_texts = [d.text for d in self._documents] + texts
                vec = TfidfVectorizer(max_features=self._embed_dim)
                vec.fit(all_texts)
                return vec.transform(texts).toarray().astype(np.float32)
            except Exception:
                return np.zeros((len(texts), self._embed_dim), dtype=np.float32)
