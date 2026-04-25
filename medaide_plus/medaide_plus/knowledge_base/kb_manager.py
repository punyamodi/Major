"""
KBManager — Knowledge base loading, saving, and index management.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from medaide_plus.knowledge_base.rag import RAGRetriever

logger = logging.getLogger("medaide_plus.kb_manager")


class KBManager:
    """
    Knowledge Base Manager.

    Handles loading, saving, and managing the medical knowledge base
    used by the HDFG verification module and RAG retriever.

    Args:
        config: Configuration dict.
        kb_path: Path to knowledge base JSON file.
    """

    def __init__(
        self,
        config: Optional[Dict] = None,
        kb_path: Optional[str] = None,
    ) -> None:
        self.config = config or {}
        self.kb_path = Path(kb_path or self.config.get("kb_path", "data/knowledge_base.json"))
        self._documents: List[Dict[str, Any]] = []
        self._retriever: Optional[RAGRetriever] = None

    def load(self, path: Optional[str] = None) -> int:
        """
        Load knowledge base from JSON file.

        Expected format: list of {"id": str, "text": str, "metadata": dict}

        Args:
            path: Optional override path. Uses self.kb_path if not provided.

        Returns:
            Number of documents loaded.
        """
        load_path = Path(path) if path else self.kb_path
        if not load_path.exists():
            logger.warning(f"KB file not found: {load_path}. Starting with empty KB.")
            return 0

        with open(load_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            self._documents = data
        elif isinstance(data, dict) and "documents" in data:
            self._documents = data["documents"]
        else:
            logger.error("Invalid KB format. Expected list or {'documents': [...]}")
            return 0

        logger.info(f"Loaded {len(self._documents)} documents from {load_path}.")
        self._build_retriever()
        return len(self._documents)

    def save(self, path: Optional[str] = None) -> None:
        """
        Save knowledge base to JSON file.

        Args:
            path: Optional override path.
        """
        save_path = Path(path) if path else self.kb_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump({"documents": self._documents}, f, indent=2)
        logger.info(f"Saved {len(self._documents)} documents to {save_path}.")

    def add_documents(
        self,
        texts: List[str],
        metadata: Optional[List[Dict]] = None,
    ) -> None:
        """
        Add new documents to the knowledge base.

        Args:
            texts: Document text strings.
            metadata: Optional metadata per document.
        """
        metadata = metadata or [{}] * len(texts)
        for i, (text, meta) in enumerate(zip(texts, metadata)):
            doc_id = meta.get("id", f"doc_{len(self._documents)}")
            self._documents.append({"id": doc_id, "text": text, "metadata": meta})

        if self._retriever:
            self._retriever.add_documents(texts, metadata)
        else:
            self._build_retriever()

        logger.info(f"Added {len(texts)} documents. Total: {len(self._documents)}.")

    def get_retriever(self) -> RAGRetriever:
        """Get or build the RAG retriever."""
        if self._retriever is None:
            self._build_retriever()
        return self._retriever

    def get_all_texts(self) -> List[str]:
        """Return all document texts."""
        return [d["text"] for d in self._documents]

    def _build_retriever(self) -> None:
        """Build RAGRetriever from current documents."""
        if not self._documents:
            self._retriever = RAGRetriever(config=self.config)
            return

        texts = [d["text"] for d in self._documents]
        metas = [d.get("metadata", {}) for d in self._documents]
        for i, (doc, meta) in enumerate(zip(self._documents, metas)):
            if "id" not in meta:
                meta["id"] = doc.get("id", f"doc_{i}")

        self._retriever = RAGRetriever(config=self.config)
        self._retriever.add_documents(texts, metas)

    def seed_from_medaide_guidelines(self, guidelines: List[Dict]) -> None:
        """
        Seed the KB from MedAide-format guideline entries.

        Expected format: [{"category": str, "intent": str, "guideline": str}]

        Args:
            guidelines: List of guideline dicts.
        """
        texts = []
        metas = []
        for g in guidelines:
            text = (
                f"[{g.get('category', 'General')}] "
                f"[{g.get('intent', 'General')}] "
                f"{g.get('guideline', '')}"
            )
            texts.append(text)
            metas.append({
                "id": f"guideline_{len(self._documents) + len(texts)}",
                "source": "medaide_guidelines",
                "category": g.get("category", ""),
                "intent": g.get("intent", ""),
            })
        self.add_documents(texts, metas)

    @staticmethod
    def create_sample_kb() -> List[Dict[str, str]]:
        """Create a minimal sample medical knowledge base for testing."""
        return [
            {
                "id": "kb_001",
                "text": (
                    "Aspirin 325mg is indicated for pain relief, fever reduction, "
                    "and antiplatelet therapy. Standard adult dose is 325-650mg every 4-6 hours "
                    "as needed. Contraindicated in patients with aspirin allergy or active peptic ulcer."
                ),
                "metadata": {"category": "Medication", "drug": "aspirin"},
            },
            {
                "id": "kb_002",
                "text": (
                    "Hypertension management: First-line treatment includes ACE inhibitors "
                    "(e.g., lisinopril 10-40mg daily), ARBs (e.g., losartan 50-100mg daily), "
                    "thiazide diuretics, and calcium channel blockers. Target BP <130/80 mmHg."
                ),
                "metadata": {"category": "Diagnosis", "condition": "hypertension"},
            },
            {
                "id": "kb_003",
                "text": (
                    "Type 2 diabetes: First-line medication is metformin 500-2000mg daily. "
                    "Monitor HbA1c every 3 months. Target HbA1c <7% for most adults. "
                    "Lifestyle modifications including diet and exercise are essential."
                ),
                "metadata": {"category": "Medication", "condition": "diabetes"},
            },
            {
                "id": "kb_004",
                "text": (
                    "Chest pain differential diagnosis includes: acute coronary syndrome, "
                    "pulmonary embolism, aortic dissection, pneumothorax, esophageal spasm, "
                    "and musculoskeletal pain. Urgent ECG and troponin levels required."
                ),
                "metadata": {"category": "Diagnosis", "symptom": "chest_pain"},
            },
            {
                "id": "kb_005",
                "text": (
                    "Post-MI rehabilitation: Cardiac rehabilitation reduces mortality by 25%. "
                    "Progressive exercise starting 1-2 weeks post-discharge. "
                    "Target 150 minutes moderate aerobic activity per week. "
                    "Mediterranean diet recommended."
                ),
                "metadata": {"category": "Post-Diagnosis", "condition": "post_mi"},
            },
        ]
