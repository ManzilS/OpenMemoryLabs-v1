"""
Graph-Traced Context Chains (GTCC)
====================================
Novel retrieval strategy that uses REBEL entity provenance to discover
bridge chunks that connect retrieved chunks via shared entities.

Zero LLM calls at both ingest and query time.

Architecture:
    1. Standard hybrid retrieval → top-k chunks
    2. Look up entities in each retrieved chunk (provenance index)
    3. Find bridge chunks sharing entities with 2+ retrieved chunks
    4. Order all chunks by narrative position (graph traversal order)
    5. Return coherent context chain
"""

import logging
from pathlib import Path
from typing import List, Tuple, Optional

from oml.retrieval.provenance_index import ProvenanceIndex

logger = logging.getLogger(__name__)


class GTCCRetriever:
    """
    Graph-Traced Context Chain retriever.
    Uses the provenance index to expand retrieval results with bridge chunks.
    """

    def __init__(self, artifacts_dir: Path, max_bridges: int = 3):
        self.artifacts_dir = artifacts_dir
        self.max_bridges = max_bridges
        self.provenance = ProvenanceIndex(artifacts_dir)
        self._loaded = False

    def load(self) -> bool:
        """Load the provenance index."""
        if not self._loaded:
            self._loaded = self.provenance.load()
        return self._loaded

    def expand_results(
        self,
        retrieved_chunk_ids: List[str],
        max_bridges: Optional[int] = None,
    ) -> List[Tuple[str, float, str]]:
        """
        Expand retrieval results with bridge chunks.

        Args:
            retrieved_chunk_ids: Chunk IDs from standard retrieval
            max_bridges: Override default max bridges

        Returns:
            List of (chunk_id, score, source) tuples where source is
            'retrieved' or 'bridge'
        """
        if not self.load():
            logger.warning("[GTCC] No provenance index found — skipping expansion")
            return [(cid, 1.0, "retrieved") for cid in retrieved_chunk_ids]

        bridges = max_bridges or self.max_bridges

        # Find bridge chunks
        bridge_chunks = self.provenance.find_bridge_chunks(
            retrieved_chunk_ids, max_bridges=bridges
        )

        # Build combined result list
        results = []

        # Add original retrieved chunks
        for i, cid in enumerate(retrieved_chunk_ids):
            # Score decreases by position
            score = 1.0 - (i * 0.05)
            results.append((cid, score, "retrieved"))

        # Add bridge chunks
        for cid, bridge_score in bridge_chunks:
            # Normalize bridge score relative to retrieved scores
            normalized_score = min(bridge_score / 10.0, 0.85)
            results.append((cid, normalized_score, "bridge"))

        # Sort by: original retrieved first (by position), then bridges by score
        # This preserves retrieval ranking while appending bridge context
        return results

    def get_entity_context(self, chunk_ids: List[str]) -> str:
        """
        Build a compact entity-relationship summary for the retrieved chunks.
        Shows which entities connect which chunks.
        """
        if not self._loaded:
            return ""

        entity_chunks = {}
        for cid in chunk_ids:
            entities = self.provenance.get_entities_for_chunk(cid)
            for e in entities:
                if e not in entity_chunks:
                    entity_chunks[e] = []
                entity_chunks[e].append(cid)

        # Only include entities that appear in 2+ chunks (connecting entities)
        connecting = {
            e: chunks for e, chunks in entity_chunks.items()
            if len(chunks) >= 2
        }

        if not connecting:
            return ""

        lines = ["[ENTITY CONNECTIONS]"]
        for entity, chunks in sorted(connecting.items(), key=lambda x: -len(x[1])):
            lines.append(f"- '{entity}' connects {len(chunks)} chunks")

        return "\n".join(lines)

    def stats(self) -> dict:
        """Return GTCC statistics."""
        if not self._loaded:
            self.load()
        return {
            "provenance_loaded": self._loaded,
            **self.provenance.stats(),
        }
