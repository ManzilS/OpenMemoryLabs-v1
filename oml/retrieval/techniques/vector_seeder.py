"""VectorSeeder -- Seed-discovery phase for ScoutRetriever.

Finds initial seed notes via vector search (or keyword fallback) and
optionally re-ranks them using importance scoring.

Extracted from :class:`~oml.retrieval.scout.ScoutRetriever` so it can be
tested and composed independently.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

from oml.memory.atomic_note import AtomicNote
from oml.memory.importance import ImportanceScorer
from oml.storage.teeg_store import TEEGStore

logger = logging.getLogger(__name__)


class VectorSeeder:
    """Find seed notes for graph traversal via vector/keyword search.

    Parameters
    ----------
    store:
        The :class:`~oml.storage.teeg_store.TEEGStore` containing notes.
    seed_k:
        Number of seed notes to return after re-ranking.
    use_importance:
        If ``True``, fetch ``2 * seed_k`` candidates and re-rank by
        importance-weighted similarity before returning the top ``seed_k``.
    """

    def __init__(
        self,
        store: TEEGStore,
        seed_k: int = 3,
        use_importance: bool = True,
    ) -> None:
        self.store = store
        self.seed_k = seed_k
        self.use_importance = use_importance
        self._scorer: Optional[ImportanceScorer] = None

    def find_seeds(self, query: str) -> List[Tuple[AtomicNote, float]]:
        """Return up to ``seed_k`` seed notes ranked by relevance.

        Each item is ``(note, similarity_score)``.  When importance
        re-ranking is enabled the similarity score is blended with the
        importance score before sorting.

        Returns an empty list when the store has no matching notes.
        """
        fetch_k = self.seed_k * 2 if self.use_importance else self.seed_k
        seed_results = self.store.vector_search(query, top_k=fetch_k)
        if not seed_results:
            logger.debug("[VectorSeeder] No seed notes found; returning empty")
            return []

        if self.use_importance:
            scorer = self._get_scorer()
            seed_results = sorted(
                seed_results,
                key=lambda r: r[1] * (0.4 + 0.6 * scorer.score(r[0])),
                reverse=True,
            )[: self.seed_k]

        return seed_results

    def _get_scorer(self) -> ImportanceScorer:
        """Lazy-initialise the shared ImportanceScorer."""
        if self._scorer is None:
            self._scorer = ImportanceScorer(self.store)
        return self._scorer
