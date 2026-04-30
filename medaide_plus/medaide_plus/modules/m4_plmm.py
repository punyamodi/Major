"""
M4 PLMM — Persistent Longitudinal Medical Memory

Implements per-patient knowledge graphs using NetworkX DiGraph:
  - Node types: Symptom, Diagnosis, Medication, Allergy, Procedure
  - Timestamped nodes and edges
  - inject_history(): prepend relevant subgraph to query
  - update_from_response(): NER → merge into patient graph
  - JSON-based persistence per patient

Reference: arXiv 2410.12532, Section 3.4
"""

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx

logger = logging.getLogger("medaide_plus.m4_plmm")


@dataclass
class MedicalEntity:
    """Represents a medical entity extracted from text."""
    text: str
    entity_type: str  # Symptom | Diagnosis | Medication | Allergy | Procedure
    confidence: float = 1.0
    timestamp: float = field(default_factory=time.time)
    source: str = "extracted"  # "extracted" | "manual" | "imported"


@dataclass
class PLMMResult:
    """Result from PLMM module operations."""
    enriched_query: str
    patient_id: str
    relevant_history: List[Dict[str, Any]] = field(default_factory=list)
    n_history_nodes: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


# Regex patterns for entity extraction fallback (when spaCy is unavailable)
ENTITY_PATTERNS: Dict[str, List[str]] = {
    "Medication": [
        r"\b(aspirin|ibuprofen|acetaminophen|paracetamol|metformin|lisinopril|"
        r"atorvastatin|omeprazole|amoxicillin|ciprofloxacin|metoprolol|"
        r"amlodipine|sertraline|prednisone|warfarin|insulin|levothyroxine|"
        r"hydrochlorothiazide|gabapentin|losartan)\b",
        r"\b\w+cillin\b|\b\w+mycin\b|\b\w+pril\b|\b\w+sartan\b|\b\w+statin\b",
    ],
    "Symptom": [
        r"\b(fever|headache|cough|fatigue|nausea|vomiting|dizziness|"
        r"chest pain|shortness of breath|diarrhea|constipation|"
        r"abdominal pain|back pain|joint pain|muscle ache|sore throat|"
        r"runny nose|rash|swelling|palpitation|insomnia|anxiety|depression)\b",
    ],
    "Diagnosis": [
        r"\b(diabetes|hypertension|asthma|pneumonia|bronchitis|"
        r"arthritis|osteoporosis|hypothyroidism|hyperthyroidism|"
        r"heart failure|atrial fibrillation|stroke|anemia|"
        r"chronic kidney disease|liver cirrhosis|copd|"
        r"gastroesophageal reflux|irritable bowel syndrome|migraine)\b",
    ],
    "Allergy": [
        r"\ballerg(?:ic|y|ies)\s+to\s+(\w+)\b",
        r"\b(penicillin|sulfa|latex|peanut|shellfish)\s+allergy\b",
    ],
    "Procedure": [
        r"\b(surgery|operation|biopsy|colonoscopy|endoscopy|"
        r"appendectomy|cholecystectomy|bypass|stent|transplant|"
        r"chemotherapy|radiation|dialysis|catheterization)\b",
    ],
}


class PLMMModule:
    """
    Persistent Longitudinal Medical Memory module.

    Maintains per-patient knowledge graphs in NetworkX DiGraph format,
    persisted as JSON files. Supports history injection and graph updates.

    Args:
        config: Configuration dict with PLMM parameters.
        storage_path: Directory path for patient graph JSON files.
    """

    def __init__(
        self,
        config: Optional[Dict] = None,
        storage_path: Optional[str] = None,
    ) -> None:
        self.config = config or {}
        self.storage_path = Path(
            storage_path or self.config.get("graph_storage_path", "data/patient_graphs")
        )
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.max_nodes: int = self.config.get("max_nodes", 500)

        # In-memory cache of patient graphs
        self._graphs: Dict[str, nx.DiGraph] = {}

        # Load spaCy NER model
        self._nlp = None
        self._load_nlp()

    def _load_nlp(self) -> None:
        """Load spaCy NER model with fallback to regex."""
        try:
            import spacy
            try:
                self._nlp = spacy.load("en_core_web_sm")
                logger.info("spaCy en_core_web_sm loaded for NER.")
            except OSError:
                logger.warning(
                    "spaCy model en_core_web_sm not found. "
                    "Run: python -m spacy download en_core_web_sm. "
                    "Using regex fallback."
                )
        except ImportError:
            logger.warning("spaCy not installed. Using regex NER fallback.")

    def get_or_create_graph(self, patient_id: str) -> nx.DiGraph:
        """
        Get existing patient graph or create a new empty one.

        Args:
            patient_id: Unique patient identifier.

        Returns:
            NetworkX DiGraph for the patient.
        """
        if patient_id not in self._graphs:
            saved = self._load_graph(patient_id)
            if saved:
                self._graphs[patient_id] = saved
            else:
                self._graphs[patient_id] = nx.DiGraph(
                    patient_id=patient_id,
                    created_at=time.time(),
                )
        return self._graphs[patient_id]

    def inject_history(self, query: str, patient_id: str, top_k: int = 5) -> PLMMResult:
        """
        Enrich the query with relevant patient history from the knowledge graph.

        Retrieves the most relevant nodes (by type and keyword match)
        and prepends a summary to the query.

        Args:
            query: Original user query.
            patient_id: Patient identifier.
            top_k: Maximum number of history nodes to include.

        Returns:
            PLMMResult with enriched query and history details.
        """
        graph = self.get_or_create_graph(patient_id)

        if graph.number_of_nodes() == 0:
            return PLMMResult(
                enriched_query=query,
                patient_id=patient_id,
                relevant_history=[],
                n_history_nodes=0,
                metadata={"status": "no_history"},
            )

        # Find relevant nodes based on query keyword overlap
        relevant_nodes = self._find_relevant_nodes(graph, query, top_k=top_k)

        # Format history summary
        history_parts = []
        history_records = []
        for node_id, node_data in relevant_nodes:
            entity_type = node_data.get("type", "Unknown")
            entity_text = node_data.get("text", node_id)
            timestamp = node_data.get("timestamp", 0)
            date_str = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d") if timestamp else "unknown"
            history_parts.append(f"{entity_type}: {entity_text} (recorded: {date_str})")
            history_records.append({
                "type": entity_type,
                "text": entity_text,
                "date": date_str,
                "node_id": node_id,
            })

        if history_parts:
            history_prefix = (
                f"[Patient History - ID: {patient_id}]\n"
                + "\n".join(f"  • {h}" for h in history_parts)
                + "\n\n[Current Query]\n"
            )
            enriched_query = history_prefix + query
        else:
            enriched_query = query

        return PLMMResult(
            enriched_query=enriched_query,
            patient_id=patient_id,
            relevant_history=history_records,
            n_history_nodes=len(relevant_nodes),
            metadata={"total_nodes": graph.number_of_nodes()},
        )

    def update_from_response(
        self,
        response: str,
        patient_id: str,
        query: Optional[str] = None,
    ) -> int:
        """
        Extract medical entities from response and merge into patient graph.

        Args:
            response: Agent response text to extract entities from.
            patient_id: Patient identifier.
            query: Optional original query (also mined for entities).

        Returns:
            Number of new nodes added to the graph.
        """
        graph = self.get_or_create_graph(patient_id)

        # Extract entities from both response and query
        entities: List[MedicalEntity] = self._extract_medical_entities(response)
        if query:
            entities += self._extract_medical_entities(query)

        # Deduplicate
        seen: Set[str] = set()
        unique_entities = []
        for ent in entities:
            key = f"{ent.entity_type}::{ent.text.lower()}"
            if key not in seen:
                seen.add(key)
                unique_entities.append(ent)

        # Merge into graph
        n_added = 0
        for entity in unique_entities:
            node_id = f"{entity.entity_type}_{entity.text.lower().replace(' ', '_')}"
            if not graph.has_node(node_id):
                if graph.number_of_nodes() < self.max_nodes:
                    graph.add_node(
                        node_id,
                        text=entity.text,
                        type=entity.entity_type,
                        confidence=entity.confidence,
                        timestamp=entity.timestamp,
                        source=entity.source,
                    )
                    n_added += 1
                    logger.debug(
                        f"Added node: {node_id} to patient {patient_id}'s graph."
                    )
            else:
                # Update existing node with latest timestamp
                graph.nodes[node_id]["timestamp"] = max(
                    graph.nodes[node_id].get("timestamp", 0), entity.timestamp
                )

        # Add temporal edges between co-occurring entities
        self._add_cooccurrence_edges(graph, unique_entities)

        # Persist updated graph
        if n_added > 0:
            self._save_graph(patient_id, graph)

        logger.info(
            f"PLMM: added {n_added} new nodes to patient {patient_id}'s graph "
            f"(total: {graph.number_of_nodes()})."
        )
        return n_added

    def _extract_medical_entities(self, text: str) -> List[MedicalEntity]:
        """
        Extract medical entities using spaCy NER (or regex fallback).

        Args:
            text: Text to extract entities from.

        Returns:
            List of MedicalEntity objects.
        """
        entities: List[MedicalEntity] = []

        if self._nlp:
            entities += self._spacy_extract(text)

        # Always also apply regex patterns (catches domain-specific terms)
        entities += self._regex_extract(text)
        return entities

    def _spacy_extract(self, text: str) -> List[MedicalEntity]:
        """Extract entities using spaCy."""
        doc = self._nlp(text)
        entities = []
        spacy_type_map = {
            "DISEASE": "Diagnosis",
            "CHEMICAL": "Medication",
            "DRUG": "Medication",
            "SYMPTOM": "Symptom",
            "PROCEDURE": "Procedure",
            "PERSON": None,
            "ORG": None,
        }
        for ent in doc.ents:
            mapped_type = spacy_type_map.get(ent.label_)
            if mapped_type:
                entities.append(
                    MedicalEntity(
                        text=ent.text,
                        entity_type=mapped_type,
                        confidence=0.85,
                        source="spacy",
                    )
                )
        return entities

    def _regex_extract(self, text: str) -> List[MedicalEntity]:
        """Extract entities using regex pattern matching."""
        entities = []
        text_lower = text.lower()
        for entity_type, patterns in ENTITY_PATTERNS.items():
            for pattern in patterns:
                matches = re.findall(pattern, text_lower, re.IGNORECASE)
                for match in matches:
                    if isinstance(match, tuple):
                        match = match[0]
                    if len(match) > 2:
                        entities.append(
                            MedicalEntity(
                                text=match.strip(),
                                entity_type=entity_type,
                                confidence=0.7,
                                source="regex",
                            )
                        )
        return entities

    def _find_relevant_nodes(
        self, graph: nx.DiGraph, query: str, top_k: int = 5
    ) -> List[Tuple[str, Dict]]:
        """
        Find the most relevant patient history nodes for a given query.

        Relevance = keyword overlap between query tokens and node text,
        weighted by recency (more recent = higher weight).

        Args:
            graph: Patient's knowledge graph.
            query: Current query string.
            top_k: Maximum nodes to return.

        Returns:
            List of (node_id, node_data) tuples sorted by relevance.
        """
        query_tokens = set(query.lower().split())
        current_time = time.time()

        scored_nodes = []
        for node_id, node_data in graph.nodes(data=True):
            node_text = node_data.get("text", "").lower()
            node_tokens = set(node_text.split())

            # Keyword overlap score
            overlap = len(query_tokens & node_tokens) / max(len(node_tokens), 1)

            # Recency score: nodes updated within last 30 days score higher
            node_time = node_data.get("timestamp", 0)
            age_days = (current_time - node_time) / 86400 if node_time else 365
            recency_score = max(0.0, 1.0 - age_days / 365)

            # Priority for high-importance types
            type_weight = {
                "Allergy": 1.5, "Diagnosis": 1.2, "Medication": 1.1,
                "Symptom": 1.0, "Procedure": 0.9,
            }.get(node_data.get("type", ""), 1.0)

            total_score = (overlap * 2.0 + recency_score * 0.5) * type_weight
            scored_nodes.append((node_id, node_data, total_score))

        # Sort by score and return top_k
        scored_nodes.sort(key=lambda x: x[2], reverse=True)
        return [(nid, ndata) for nid, ndata, _ in scored_nodes[:top_k]]

    def _add_cooccurrence_edges(
        self,
        graph: nx.DiGraph,
        entities: List[MedicalEntity],
    ) -> None:
        """
        Add co-occurrence edges between entities extracted from the same text.

        For medical relationships (e.g., Symptom → Diagnosis → Medication).

        Args:
            graph: Patient graph to update.
            entities: List of co-occurring entities.
        """
        current_time = time.time()
        node_ids = []
        for entity in entities:
            node_id = f"{entity.entity_type}_{entity.text.lower().replace(' ', '_')}"
            if graph.has_node(node_id):
                node_ids.append((node_id, entity.entity_type))

        # Add edges between entity pairs with relationship labels
        for i in range(len(node_ids)):
            for j in range(i + 1, len(node_ids)):
                nid_i, type_i = node_ids[i]
                nid_j, type_j = node_ids[j]
                if not graph.has_edge(nid_i, nid_j):
                    edge_type = f"{type_i}_to_{type_j}"
                    graph.add_edge(
                        nid_i, nid_j,
                        relationship=edge_type,
                        timestamp=current_time,
                    )

    def _save_graph(self, patient_id: str, graph: nx.DiGraph) -> None:
        """
        Serialize and save a patient's graph to JSON.

        Args:
            patient_id: Patient identifier.
            graph: NetworkX DiGraph to save.
        """
        safe_id = re.sub(r"[^a-zA-Z0-9_-]", "_", patient_id)
        filepath = self.storage_path / f"patient_{safe_id}.json"
        try:
            data = nx.node_link_data(graph, edges="edges")
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)
            logger.debug(f"Saved patient graph to {filepath}.")
        except Exception as e:
            logger.error(f"Failed to save patient graph: {e}")

    def _load_graph(self, patient_id: str) -> Optional[nx.DiGraph]:
        """
        Load a patient's graph from JSON if it exists.

        Args:
            patient_id: Patient identifier.

        Returns:
            Loaded DiGraph or None if file not found.
        """
        safe_id = re.sub(r"[^a-zA-Z0-9_-]", "_", patient_id)
        filepath = self.storage_path / f"patient_{safe_id}.json"
        if not filepath.exists():
            return None
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            edges_key = "edges" if "edges" in data else "links"
            graph = nx.node_link_graph(
                data,
                directed=True,
                multigraph=False,
                edges=edges_key,
            )
            logger.debug(
                f"Loaded patient graph from {filepath} "
                f"({graph.number_of_nodes()} nodes)."
            )
            return graph
        except Exception as e:
            logger.error(f"Failed to load patient graph: {e}")
            return None

    def delete_patient(self, patient_id: str) -> bool:
        """
        Delete a patient's graph from memory and disk.

        Args:
            patient_id: Patient identifier.

        Returns:
            True if deleted successfully, False otherwise.
        """
        if patient_id in self._graphs:
            del self._graphs[patient_id]

        safe_id = re.sub(r"[^a-zA-Z0-9_-]", "_", patient_id)
        filepath = self.storage_path / f"patient_{safe_id}.json"
        if filepath.exists():
            filepath.unlink()
            logger.info(f"Deleted patient graph for {patient_id}.")
            return True
        return False

    def get_graph_summary(self, patient_id: str) -> Dict[str, Any]:
        """
        Get a statistical summary of a patient's knowledge graph.

        Args:
            patient_id: Patient identifier.

        Returns:
            Dict with node count, edge count, node type breakdown, etc.
        """
        graph = self.get_or_create_graph(patient_id)
        type_counts: Dict[str, int] = {}
        for _, node_data in graph.nodes(data=True):
            t = node_data.get("type", "Unknown")
            type_counts[t] = type_counts.get(t, 0) + 1

        return {
            "patient_id": patient_id,
            "n_nodes": graph.number_of_nodes(),
            "n_edges": graph.number_of_edges(),
            "node_types": type_counts,
            "is_connected": nx.is_weakly_connected(graph) if graph.number_of_nodes() > 0 else False,
        }
