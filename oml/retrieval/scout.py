"""
ScoutRetriever — Relation-first graph traversal for TEEG memory retrieval.
===========================================================================

The Scout is deliberately different from standard RAG retrieval:

  Standard RAG:    query -> cosine similarity -> top-k statistically similar chunks
  Scout:           query -> seed notes (small vector search) -> graph walk -> exact context

The Scout refuses to load large amounts of text based on keyword overlap.  It
only retrieves notes that are *logically connected* to the seed set via
explicit graph edges.  When no graph path exists it falls back gracefully to
vector similarity -- but always prefers the auditable graph path.

Traversal strategy
------------------
1. **Seed phase** -- find 1-3 seed notes via vector search (or keyword fallback).
2. **Walk phase** -- BFS/DFS from each seed, following edges up to ``max_hops``
   deep.  Edge weight acts as a priority weight: higher-weight edges are
   explored first.
3. **Filter phase** -- remove archived notes; deduplicate by note_id.
4. **Rank phase** -- rank returned notes by:
     (a) hops from seed (closer = higher priority)
     (b) edge weight along the path
     (c) note confidence score
5. **Pack phase** -- serialize the top-k notes in TOON and return both the
   structured list and a ready-to-inject context string.

Usage
-----
    scout = ScoutRetriever(store)
    results = scout.search("Who created the creature?", top_k=5)
    for note, score, hops in results:
        print(note.content, score, hops)

    context_str = scout.build_context("Who created the creature?", top_k=5)
    # context_str is TOON-encoded, ready to paste into a prompt
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

from oml.memory.atomic_note import AtomicNote
from oml.memory.importance import ImportanceScorer
from oml.memory import toon
from oml.retrieval.techniques.graph_walker import GraphWalker
from oml.retrieval.techniques.vector_seeder import VectorSeeder
from oml.storage.teeg_store import TEEGStore

logger = logging.getLogger(__name__)

# Each result item: (note, combined_score, hops_from_seed)
ScoutResult = Tuple[AtomicNote, float, int]


class ScoutRetriever:
    """
    Relation-first graph traversal retriever for TEEG memory.

    Parameters
    ----------
    store:
        The :class:`~oml.storage.teeg_store.TEEGStore` containing notes and the graph.
    seed_k:
        Number of seed notes to find via vector/keyword search.
    max_hops:
        Maximum BFS depth when walking the graph from seeds.
    hop_decay:
        Score multiplier applied per hop (e.g. 0.8 means 2 hops -> 0.64x score).
    """

    def __init__(
        self,
        store: TEEGStore,
        seed_k: int = 3,
        max_hops: int = 2,
        hop_decay: float = 0.8,
        use_importance: bool = True,
        record_access: bool = True,
    ):
        self.store = store
        self.seed_k = seed_k
        self.max_hops = max_hops
        self.hop_decay = hop_decay
        self.use_importance = use_importance
        self.record_access = record_access
        self._scorer: Optional[ImportanceScorer] = None

        # Composed technique modules
        self._seeder = VectorSeeder(
            store=store,
            seed_k=seed_k,
            use_importance=use_importance,
        )
        self._walker = GraphWalker(
            store=store,
            max_hops=max_hops,
            hop_decay=hop_decay,
            use_importance=use_importance,
        )

    # -- public API ------------------------------------------------------------

    def search(self, query: str, top_k: int = 5) -> List[ScoutResult]:
        """
        Return the top-k most relevant active notes for ``query``.

        Uses relation-first graph traversal from seed notes.  Each result is
        a ``(AtomicNote, score, hops)`` tuple.
        """
        # 1. Seed phase -- delegate to VectorSeeder
        seed_results = self._seeder.find_seeds(query)
        if not seed_results:
            logger.debug("[Scout] No seed notes found; returning empty")
            return []

        # 2. Walk phase -- delegate to GraphWalker
        visited = self._walker.walk(seed_results)

        # 3. Assemble and rank results
        results: List[ScoutResult] = []
        for note_id, (score, hops) in visited.items():
            note = self.store.get(note_id)
            if note and note.active:
                results.append((note, score, hops))

        # Apply final importance boost to ranking score
        if self.use_importance:
            scorer = self._get_scorer()
            results = [
                (note, score * (0.7 + 0.3 * scorer.score(note)), hops)
                for note, score, hops in results
            ]

        # Sort: seeds first (hops=0), then by score descending
        results.sort(key=lambda r: (r[2], -r[1]))
        top = results[:top_k]

        # Record access for each returned note (feeds ImportanceScorer on next query)
        if self.record_access:
            for note, _, _ in top:
                self.store.record_access(note.note_id)

        return top

    def _get_scorer(self) -> ImportanceScorer:
        """Lazy-initialise the shared ImportanceScorer."""
        if self._scorer is None:
            self._scorer = ImportanceScorer(self.store)
        return self._scorer

    def build_context(
        self,
        query: str,
        top_k: int = 5,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Build a TOON-encoded context string ready to inject into an LLM prompt.

        Each retrieved note is serialized in TOON format and prefixed with
        a hop-distance annotation so the LLM knows how directly relevant it is.

        Args:
            query: The user query.
            top_k: Maximum number of notes to include.
            max_tokens: Optional token cap; notes are dropped from the bottom
                        if adding them would exceed the budget.

        Returns:
            A multiline TOON string wrapped in ``[TEEG MEMORY]`` / ``[/TEEG MEMORY]``.
        """
        results = self.search(query, top_k=top_k)
        if not results:
            return "[TEEG MEMORY]\n(no relevant memory found)\n[/TEEG MEMORY]"

        lines = ["[TEEG MEMORY]"]
        total_tokens = len("[TEEG MEMORY]\n[/TEEG MEMORY]") // 4  # rough overhead

        for note, score, hops in results:
            label = "seed" if hops == 0 else f"hop-{hops}"
            entry = f"--- [{label}  score={score:.3f}] ---\n{note.to_toon()}"
            entry_tokens = toon.token_count_estimate(entry)

            if max_tokens is not None and total_tokens + entry_tokens > max_tokens:
                break

            lines.append(entry)
            total_tokens += entry_tokens

        lines.append("[/TEEG MEMORY]")
        return "\n".join(lines)

    def explain(self, query: str, top_k: int = 5) -> str:
        """
        Return a human-readable explanation of the traversal for debugging.

        Shows which notes were seeds, which were found via graph walk, and
        the relation chain that led to each.
        """
        results = self.search(query, top_k=top_k)
        if not results:
            return "Scout found no relevant notes."

        lines = [f"Scout results for: {query!r}", ""]
        for i, (note, score, hops) in enumerate(results, 1):
            source = "SEED" if hops == 0 else f"GRAPH (hop {hops})"
            lines.append(
                f"{i}. [{source}] score={score:.3f}  id={note.note_id}"
            )
            lines.append(f"   content: {note.content[:100]}")
            if note.tags:
                lines.append(f"   tags: {', '.join(note.tags)}")
            lines.append("")

        # Show edge summary
        graph = self.store.get_graph()
        result_ids = {r[0].note_id for r in results}
        relevant_edges = [
            (u, v, d)
            for u, v, d in graph.edges(data=True)
            if u in result_ids or v in result_ids
        ]
        if relevant_edges:
            lines.append("Graph edges in result set:")
            for u, v, d in relevant_edges:
                rel = d.get("relation", "?")
                w = d.get("weight", 1.0)
                lines.append(f"  {u[:12]}... --[{rel} w={w:.1f}]--> {v[:12]}...")

        return "\n".join(lines)

    def stats(self) -> dict:
        """Return Scout configuration and store statistics."""
        return {
            "seed_k": self.seed_k,
            "max_hops": self.max_hops,
            "hop_decay": self.hop_decay,
            **self.store.stats(),
        }
