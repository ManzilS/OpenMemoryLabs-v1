"""
Chunk-Entity Provenance Index
==============================
Bidirectional mapping between chunks and their REBEL-extracted entities.
Built at ingest time, used at query time for Graph-Traced Context Chains.

Data structure:
    chunk_to_entities:  {chunk_id: set(entity_str, ...)}
    entity_to_chunks:   {entity_str: set(chunk_id, ...)}
"""

import pickle
import logging
from pathlib import Path
from typing import Dict, Set, List, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)


class ProvenanceIndex:
    """Bidirectional chunk ↔ entity provenance index."""

    def __init__(self, artifacts_dir: Path):
        self.artifacts_dir = artifacts_dir
        self.index_path = artifacts_dir / "provenance_index.pkl"
        self.chunk_to_entities: Dict[str, Set[str]] = defaultdict(set)
        self.entity_to_chunks: Dict[str, Set[str]] = defaultdict(set)

    def add_triples(self, chunk_id: str, triples: List[Tuple[str, str, str]]):
        """Register triples extracted from a specific chunk."""
        for subj, pred, obj in triples:
            s = subj.strip().lower()
            o = obj.strip().lower()
            if s:
                self.chunk_to_entities[chunk_id].add(s)
                self.entity_to_chunks[s].add(chunk_id)
            if o:
                self.chunk_to_entities[chunk_id].add(o)
                self.entity_to_chunks[o].add(chunk_id)

    def get_entities_for_chunk(self, chunk_id: str) -> Set[str]:
        """Get all entities mentioned in a chunk."""
        return self.chunk_to_entities.get(chunk_id, set())

    def get_chunks_for_entity(self, entity: str) -> Set[str]:
        """Get all chunks that mention an entity."""
        return self.entity_to_chunks.get(entity.lower().strip(), set())

    def get_chunks_for_entities(self, entities: List[str]) -> Dict[str, Set[str]]:
        """Get chunk sets for multiple entities."""
        return {e: self.get_chunks_for_entity(e) for e in entities}

    def find_bridge_chunks(
        self,
        retrieved_chunk_ids: List[str],
        max_bridges: int = 3,
    ) -> List[Tuple[str, float]]:
        """
        Find bridge chunks that connect retrieved chunks via shared entities.

        A bridge chunk is one that:
        1. Was NOT already retrieved
        2. Shares entities with 2+ retrieved chunks
        3. Scored by: number of entity connections to retrieved set

        Returns:
            List of (chunk_id, bridge_score) sorted by score descending
        """
        retrieved_set = set(retrieved_chunk_ids)

        # Collect all entities from retrieved chunks
        retrieved_entities: Dict[str, Set[str]] = {}
        for cid in retrieved_chunk_ids:
            entities = self.get_entities_for_chunk(cid)
            for e in entities:
                if e not in retrieved_entities:
                    retrieved_entities[e] = set()
                retrieved_entities[e].add(cid)

        # Find candidate bridge chunks
        candidate_scores: Dict[str, float] = defaultdict(float)
        candidate_connections: Dict[str, Set[str]] = defaultdict(set)

        for entity, source_chunks in retrieved_entities.items():
            # Find other chunks mentioning this entity
            entity_chunks = self.get_chunks_for_entity(entity)
            for candidate_cid in entity_chunks:
                if candidate_cid in retrieved_set:
                    continue  # Skip already retrieved
                # Score: how many retrieved chunks share entities with this candidate
                candidate_connections[candidate_cid].update(source_chunks)
                candidate_scores[candidate_cid] += 1.0

        # Boost chunks that connect to multiple retrieved chunks
        bridges = []
        for cid, score in candidate_scores.items():
            n_connections = len(candidate_connections[cid])
            if n_connections >= 2:
                # Strong bridge: connects 2+ retrieved chunks
                bridge_score = score * n_connections
                bridges.append((cid, bridge_score))
            elif n_connections == 1 and score >= 2:
                # Weak bridge: shares 2+ entities with one chunk
                bridges.append((cid, score * 0.5))

        # Sort by score, return top bridges
        bridges.sort(key=lambda x: -x[1])
        return bridges[:max_bridges]

    def stats(self) -> dict:
        """Return index statistics."""
        return {
            "chunks_indexed": len(self.chunk_to_entities),
            "unique_entities": len(self.entity_to_chunks),
            "avg_entities_per_chunk": (
                sum(len(v) for v in self.chunk_to_entities.values())
                / max(len(self.chunk_to_entities), 1)
            ),
        }

    def save(self):
        """Persist index to disk."""
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        data = {
            "chunk_to_entities": dict(self.chunk_to_entities),
            "entity_to_chunks": dict(self.entity_to_chunks),
        }
        with open(self.index_path, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"Provenance index saved: {self.stats()}")

    def load(self) -> bool:
        """Load index from disk."""
        if not self.index_path.exists():
            return False
        try:
            with open(self.index_path, "rb") as f:
                data = pickle.load(f)
            self.chunk_to_entities = defaultdict(set, {
                k: set(v) for k, v in data["chunk_to_entities"].items()
            })
            self.entity_to_chunks = defaultdict(set, {
                k: set(v) for k, v in data["entity_to_chunks"].items()
            })
            logger.info(f"Provenance index loaded: {self.stats()}")
            return True
        except Exception as e:
            logger.error(f"Failed to load provenance index: {e}")
            return False
