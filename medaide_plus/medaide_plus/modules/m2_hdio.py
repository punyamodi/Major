"""
M2 HDIO — Hierarchical Dual-level Intent Ontology

Implements:
  1. BioBERT-based encoder (dmis-lab/biobert-base-cased-v1.2)
  2. Graph Attention Network (GAT) over intent hierarchy
  3. Two-level classification: 4 parent categories × 17 leaf intents
  4. Per-intent sigmoid scores (multi-label) — NOT global softmax
  5. OOD (out-of-distribution) detection with threshold routing

Reference: arXiv 2410.12532, Section 3.2
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("medaide_plus.m2_hdio")

# Full intent ontology: category → list of intents
INTENT_ONTOLOGY: Dict[str, List[str]] = {
    "Pre-Diagnosis": [
        "Symptom_Triage",
        "Department_Suggestion",
        "Risk_Assessment",
        "Health_Inquiry",
    ],
    "Diagnosis": [
        "Symptom_Analysis",
        "Etiology_Detection",
        "Test_Interpretation",
        "Differential_Diagnosis",
    ],
    "Medication": [
        "Drug_Counseling",
        "Dosage_Recommendation",
        "Contraindication_Check",
        "Drug_Interaction",
        "Prescription_Review",
    ],
    "Post-Diagnosis": [
        "Rehabilitation_Advice",
        "Progress_Tracking",
        "Care_Support",
        "Lifestyle_Guidance",
        "Follow_up_Scheduling",
    ],
}

# Flat list of all intents for indexing
ALL_INTENTS: List[str] = [
    intent
    for intents in INTENT_ONTOLOGY.values()
    for intent in intents
]

# Category membership for each intent
INTENT_TO_CATEGORY: Dict[str, str] = {
    intent: cat
    for cat, intents in INTENT_ONTOLOGY.items()
    for intent in intents
}

# Intent keyword lookup for rule-based fallback
INTENT_KEYWORDS: Dict[str, List[str]] = {
    "Symptom_Triage": ["symptom", "feel", "pain", "ache", "hurt", "discomfort"],
    "Department_Suggestion": ["doctor", "department", "specialist", "consult", "refer"],
    "Risk_Assessment": ["risk", "chance", "probability", "predisposition", "susceptible"],
    "Health_Inquiry": ["what is", "explain", "healthy", "wellness", "normal"],
    "Symptom_Analysis": ["analyze", "assess symptom", "combination", "pattern"],
    "Etiology_Detection": ["cause", "why", "reason", "trigger", "origin"],
    "Test_Interpretation": ["test result", "lab", "blood test", "imaging", "biopsy", "report"],
    "Differential_Diagnosis": ["could be", "possible diagnosis", "rule out", "differential"],
    "Drug_Counseling": ["medication", "drug", "medicine", "pill", "tablet"],
    "Dosage_Recommendation": ["dose", "dosage", "how much", "how often", "frequency"],
    "Contraindication_Check": ["contraindication", "avoid", "should not", "allergy", "adverse"],
    "Drug_Interaction": ["interaction", "combine", "together", "mix", "concurrent"],
    "Prescription_Review": ["prescription", "review", "current meds", "regimen"],
    "Rehabilitation_Advice": ["rehabilitation", "recovery", "physical therapy", "exercise", "recover"],
    "Progress_Tracking": ["progress", "improvement", "getting better", "tracking", "monitor"],
    "Care_Support": ["support", "cope", "manage", "living with", "deal with"],
    "Lifestyle_Guidance": ["diet", "lifestyle", "exercise", "sleep", "stress", "nutrition"],
    "Follow_up_Scheduling": ["appointment", "follow up", "schedule", "next visit", "check up"],
}


@dataclass
class HDIOResult:
    """Result from HDIO module classification."""
    category_scores: Dict[str, float] = field(default_factory=dict)
    intent_scores: Dict[str, float] = field(default_factory=dict)
    top_category: str = ""
    top_intents: List[str] = field(default_factory=list)
    is_ood: bool = False
    ood_score: float = 0.0
    confidence: float = 0.0
    metadata: Dict = field(default_factory=dict)


class GATLayer:
    """
    Manual Graph Attention Network layer (no torch_geometric dependency).

    Implements: h'_i = σ(Σ_j α_ij W h_j)
    where attention α_ij uses softmax over neighborhood.
    """

    def __init__(self, in_features: int, out_features: int, n_heads: int = 4):
        self.in_features = in_features
        self.out_features = out_features
        self.n_heads = n_heads
        # Initialize weights (will be set from encoder dim in practice)
        np.random.seed(42)
        head_dim = out_features // n_heads
        self.W = np.random.randn(in_features, out_features) * 0.01
        self.a = np.random.randn(2 * out_features) * 0.01

    def forward(
        self,
        node_features: np.ndarray,
        adj_matrix: np.ndarray,
    ) -> np.ndarray:
        """
        Forward pass.

        Args:
            node_features: (n_nodes, in_features)
            adj_matrix: (n_nodes, n_nodes) binary adjacency

        Returns:
            Updated features (n_nodes, out_features)
        """
        h = node_features @ self.W  # (n, out_features)
        n = h.shape[0]

        # Attention mechanism
        attention = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if adj_matrix[i, j] > 0 or i == j:
                    concat = np.concatenate([h[i], h[j]])
                    e_ij = np.dot(self.a, concat)
                    attention[i, j] = np.exp(e_ij)

        # Normalize rows
        row_sums = attention.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1e-9, row_sums)
        attention = attention / row_sums

        # Aggregate
        output = attention @ h  # (n, out_features)
        return np.maximum(output, 0)  # ReLU activation


class HDIOModule:
    """
    Hierarchical Dual-level Intent Ontology module.

    Two-pass classification:
      Pass 1: BioBERT embedding → GAT → category scores (4 categories)
      Pass 2: Category-conditioned → leaf intent scores (17 intents)

    Both passes use per-intent sigmoid (multi-label), not softmax.

    Args:
        config: Configuration dict with HDIO parameters.
    """

    def __init__(self, config: Optional[Dict] = None) -> None:
        self.config = config or {}
        self.ood_threshold: float = self.config.get("ood_threshold", 0.30)
        self.n_categories: int = len(INTENT_ONTOLOGY)  # 4
        self.n_intents: int = len(ALL_INTENTS)          # 18
        self.gat_heads: int = self.config.get("gat_heads", 4)
        self.gat_hidden: int = self.config.get("gat_hidden", 128)
        self.use_biobert: bool = bool(self.config.get("use_biobert", True))
        self.use_sentence_transformer: bool = bool(
            self.config.get("use_sentence_transformer", True)
        )
        self.use_gat: bool = bool(self.config.get("use_gat", True))

        self._encoder = None
        self._gat_layer = None
        self._intent_graph: Optional[np.ndarray] = None

        self._load_models()
        self._intent_graph = self._build_intent_graph()

    def _load_models(self) -> None:
        """Load BioBERT encoder with fallback to sentence-transformer."""
        self._encoder_type = None
        self._encoder_dim = None
        if self.use_biobert:
            try:
                from transformers import AutoTokenizer, AutoModel
                import torch
                model_name = "dmis-lab/biobert-base-cased-v1.2"
                logger.info(f"Loading BioBERT: {model_name}")
                self._tokenizer = AutoTokenizer.from_pretrained(model_name)
                self._bert_model = AutoModel.from_pretrained(model_name)
                self._bert_model.eval()
                self._encoder_type = "biobert"
                self._encoder_dim = 768
                logger.info("BioBERT loaded successfully.")
            except Exception as e:
                logger.warning(f"BioBERT unavailable ({e}). Trying sentence-transformer.")
        else:
            logger.info("BioBERT disabled; skipping transformer encoder.")

        if self._encoder_type == "biobert":
            return
        if self.use_sentence_transformer:
            try:
                from sentence_transformers import SentenceTransformer
                self._sent_model = SentenceTransformer("all-MiniLM-L6-v2")
                self._encoder_type = "sentence_transformer"
                self._encoder_dim = 384
                logger.info("Sentence-transformer loaded as BioBERT fallback.")
            except Exception as e2:
                logger.warning(f"Sentence-transformer also unavailable ({e2}). Using TF-IDF.")
                self._encoder_type = "tfidf"
                self._encoder_dim = 256
        else:
            logger.info("Sentence-transformer disabled; using TF-IDF.")
            self._encoder_type = "tfidf"
            self._encoder_dim = 256

        # Initialize GAT layer
        self._gat_layer = GATLayer(
            in_features=self._encoder_dim,
            out_features=self.gat_hidden,
            n_heads=self.gat_heads,
        )

        # Linear classification heads (random init — would be fine-tuned in practice)
        np.random.seed(0)
        self._category_head = np.random.randn(self.gat_hidden, self.n_categories) * 0.1
        self._intent_head = np.random.randn(self.gat_hidden, self.n_intents) * 0.1

    def _build_intent_graph(self) -> np.ndarray:
        """
        Build adjacency matrix for the intent hierarchy graph.

        Connects:
          - Each category node to its leaf intent nodes
          - All leaf intents within a category (intra-category edges)

        Returns:
            (n_categories + n_intents) × (n_categories + n_intents) adjacency matrix.
        """
        n_total = self.n_categories + self.n_intents
        adj = np.zeros((n_total, n_total))

        intent_offset = self.n_categories
        for cat_idx, (cat, intents) in enumerate(INTENT_ONTOLOGY.items()):
            for local_intent_idx, intent in enumerate(intents):
                global_intent_idx = ALL_INTENTS.index(intent) + intent_offset
                # Category → Intent edge (bidirectional)
                adj[cat_idx, global_intent_idx] = 1.0
                adj[global_intent_idx, cat_idx] = 1.0

            # Intra-category edges between all intent pairs
            intent_indices = [
                ALL_INTENTS.index(intent) + intent_offset for intent in intents
            ]
            for i in intent_indices:
                for j in intent_indices:
                    if i != j:
                        adj[i, j] = 1.0

        # Self-loops
        np.fill_diagonal(adj, 1.0)
        return adj

    def _encode_text(self, text: str) -> np.ndarray:
        """
        Encode text to a fixed-size embedding vector.

        Args:
            text: Input text string.

        Returns:
            1D embedding array of shape (encoder_dim,).
        """
        if self._encoder_type == "biobert":
            return self._biobert_encode(text)
        elif self._encoder_type == "sentence_transformer":
            return self._sent_model.encode([text])[0]
        else:
            return self._tfidf_encode(text)

    def _biobert_encode(self, text: str) -> np.ndarray:
        """Encode text using BioBERT [CLS] pooling."""
        import torch
        inputs = self._tokenizer(
            text, return_tensors="pt", max_length=512, truncation=True, padding=True
        )
        with torch.no_grad():
            outputs = self._bert_model(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
        return cls_embedding.astype(np.float32)

    def _tfidf_encode(self, text: str) -> np.ndarray:
        """TF-IDF bag-of-words encoding as last-resort fallback."""
        tokens = text.lower().split()
        vocab = list(set(
            word
            for keywords in INTENT_KEYWORDS.values()
            for word in " ".join(keywords).split()
        ))
        # Always pad to _encoder_dim so downstream layers receive consistent shape
        vec = np.zeros(self._encoder_dim, dtype=np.float32)
        for i, word in enumerate(vocab[:self._encoder_dim]):
            if word in tokens:
                vec[i] = tokens.count(word) / max(len(tokens), 1)
        return vec

    def classify(self, text: str) -> HDIOResult:
        """
        Classify a query through the two-level intent hierarchy.

        Args:
            text: Input medical query string.

        Returns:
            HDIOResult with category and intent scores, OOD flag.
        """
        logger.debug(f"HDIO classifying: {text[:60]}...")

        # Step 1: Encode query
        query_embedding = self._encode_text(text)

        # Step 2: Build graph node features
        # Category nodes: mean of member intent keyword embeddings
        # Intent nodes: query-relevant attention scores (simplified)
        if self.use_gat:
            n_total = self.n_categories + self.n_intents
            node_features = np.zeros((n_total, self._encoder_dim), dtype=np.float32)

            # Query embedding for all nodes (broadcast — in practice nodes would have own embeddings)
            for i in range(n_total):
                node_features[i] = query_embedding

            # Step 3: GAT forward pass
            gat_out = self._gat_layer.forward(node_features, self._intent_graph)

            # Normalize gat output
            gat_mean = gat_out.mean(axis=0)  # (gat_hidden,)
            gat_norm = np.linalg.norm(gat_mean)
            if gat_norm > 0:
                gat_mean = gat_mean / gat_norm
        else:
            logger.info("GAT disabled; using keyword-only intent scoring.")
            gat_mean = np.zeros(self.gat_hidden, dtype=np.float32)

        # Step 4: Compute per-intent sigmoid scores (multi-label)
        # Use keyword matching + embedding similarity hybrid
        intent_scores = self._compute_intent_scores(text, gat_mean)
        category_scores = self._aggregate_category_scores(intent_scores)

        # Step 5: OOD detection
        max_intent_score = max(intent_scores.values()) if intent_scores else 0.0
        is_ood = max_intent_score < self.ood_threshold
        ood_score = self._compute_ood_score(intent_scores)

        # Step 6: Top results
        top_category = max(category_scores, key=category_scores.get) if category_scores else "Unknown"
        top_intents = sorted(intent_scores, key=intent_scores.get, reverse=True)[:3]
        top_intents = [i for i in top_intents if intent_scores[i] >= self.ood_threshold]

        result = HDIOResult(
            category_scores=category_scores,
            intent_scores=intent_scores,
            top_category=top_category,
            top_intents=top_intents,
            is_ood=is_ood,
            ood_score=ood_score,
            confidence=max_intent_score,
            metadata={
                "encoder": self._encoder_type,
                "n_intents_activated": len(top_intents),
            },
        )
        logger.debug(
            f"HDIO result: category={top_category}, intents={top_intents}, "
            f"is_ood={is_ood}, confidence={max_intent_score:.3f}"
        )
        return result

    def _compute_intent_scores(
        self, text: str, gat_embedding: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute per-intent sigmoid scores using keyword matching + embedding head.

        Each intent score is computed as:
          score = sigmoid(keyword_match_score + embedding_projection)

        Args:
            text: Input query text.
            gat_embedding: GAT output embedding (gat_hidden,).

        Returns:
            Dict mapping intent name → sigmoid score in (0, 1).
        """
        text_lower = text.lower()
        scores: Dict[str, float] = {}

        for intent in ALL_INTENTS:
            # Keyword matching score
            keywords = INTENT_KEYWORDS.get(intent, [])
            match_count = sum(1 for kw in keywords if kw in text_lower)
            keyword_score = match_count / max(len(keywords), 1)

            # Embedding-based score (dot product with intent head)
            if len(gat_embedding) == self.gat_hidden:
                intent_idx = ALL_INTENTS.index(intent)
                if intent_idx < self._intent_head.shape[1]:
                    embed_score = float(gat_embedding @ self._intent_head[:, intent_idx])
                else:
                    embed_score = 0.0
            else:
                embed_score = 0.0

            # Combine: sigmoid(2 * keyword_score + 0.5 * embed_score)
            logit = 2.0 * keyword_score + 0.3 * embed_score
            sigmoid_score = 1.0 / (1.0 + np.exp(-logit))
            scores[intent] = float(sigmoid_score)

        return scores

    def _aggregate_category_scores(
        self, intent_scores: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Aggregate leaf intent scores to parent category scores.

        Category score = max of member intent scores.

        Args:
            intent_scores: Dict of intent → score.

        Returns:
            Dict of category → aggregated score.
        """
        category_scores: Dict[str, float] = {}
        for cat, intents in INTENT_ONTOLOGY.items():
            member_scores = [intent_scores.get(intent, 0.0) for intent in intents]
            category_scores[cat] = float(max(member_scores)) if member_scores else 0.0
        return category_scores

    def _compute_ood_score(self, intent_scores: Dict[str, float]) -> float:
        """
        Compute OOD score: 1 - max_intent_confidence.

        A high OOD score indicates the query likely falls outside the
        known intent space.

        Args:
            intent_scores: Dict of intent → score.

        Returns:
            OOD score in [0, 1].
        """
        if not intent_scores:
            return 1.0
        max_score = max(intent_scores.values())
        return float(1.0 - max_score)
