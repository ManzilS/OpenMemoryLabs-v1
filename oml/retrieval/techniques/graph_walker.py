"""GraphWalker -- BFS graph-walk phase for ScoutRetriever.

Traverses the TEEG graph from seed notes, following edges up to
``max_hops`` deep with score decay per hop.

Extracted from :class:`~oml.retrieval.scout.ScoutRetriever` so it can be
tested and composed independently.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Dict, List, Tuple

from oml.memory.atomic_note import AtomicNote
from oml.memory.importance import ImportanceScorer
from oml.storage.teeg_store import TEEGStore

logger = logging.getLogger(__name__)


class GraphWalker:
    """BFS graph traversal from seed notes.

    Parameters
    ----------
    store:
        The :class:`~oml.storage.teeg_store.TEEGStore` containing notes
        and the NetworkX graph.
    max_hops:
        Maximum BFS depth when walking the graph from seeds.
    hop_decay:
        Score multiplier applied per hop (e.g. 0.8 means 2 hops = 0.64x).
    use_importance:
        If ``True``, blend importance into the seed score before walking.
    """

    def __init__(
        self,
        store: TEEGStore,
        max_hops: int = 2,
        hop_decay: float = 0.8,
        use_importance: bool = True,
    ) -> None:
        self.store = store
        self.max_hops = max_hops
        self.hop_decay = hop_decay
        self.use_importance = use_importance
        self._scorer: ImportanceScorer | None = None

    def walk(
        self, seeds: List[Tuple[AtomicNote, float]]
    ) -> Dict[str, Tuple[float, int]]:
        """BFS from *seeds*, returning ``{note_id: (best_score, min_hops)}``.

        Parameters
        ----------
        seeds:
            List of ``(note, similarity_score)`` tuples produced by
            :class:`~oml.retrieval.techniques.vector_seeder.VectorSeeder`.

        Returns
        -------
        dict[str, tuple[float, int]]
            Mapping from ``note_id`` to ``(score, hops)`` for every active
            note reachable within ``max_hops`` of a seed.
        """
        visited: Dict[str, Tuple[float, int]] = {}

        for seed_note, seed_sim in seeds:
            if not seed_note.active:
                continue

            imp = self._get_scorer().score(seed_note) if self.use_importance else 1.0
            seed_score = seed_sim * seed_note.confidence * (0.5 + 0.5 * imp)

            # Register seed
            if seed_note.note_id not in visited or visited[seed_note.note_id][0] < seed_score:
                visited[seed_note.note_id] = (seed_score, 0)

            # BFS queue: (note_id, current_score, hops)
            queue: deque[Tuple[str, float, int]] = deque(
                [(seed_note.note_id, seed_score, 0)]
            )
            seen_in_walk = {seed_note.note_id}

            while queue:
                current_id, current_score, hops = queue.popleft()
                if hops >= self.max_hops:
                    continue

                graph = self.store.get_graph()
                if current_id not in graph:
                    continue

                # Explore neighbours (outgoing + incoming edges)
                edges = list(graph.out_edges(current_id, data=True)) + \
                        list(graph.in_edges(current_id, data=True))

                # Sort by edge weight descending (prefer high-confidence links)
                edges.sort(key=lambda e: e[2].get("weight", 1.0), reverse=True)

                for u, v, data in edges:
                    neighbour_id = v if u == current_id else u
                    if neighbour_id in seen_in_walk:
                        continue
                    seen_in_walk.add(neighbour_id)

                    neighbour_note = self.store.get(neighbour_id)
                    if neighbour_note is None or not neighbour_note.active:
                        continue

                    edge_weight = data.get("weight", 1.0)
                    next_score = current_score * self.hop_decay * edge_weight
                    next_hops = hops + 1

                    prev = visited.get(neighbour_id)
                    if prev is None or prev[0] < next_score:
                        visited[neighbour_id] = (next_score, next_hops)

                    queue.append((neighbour_id, next_score, next_hops))

        return visited

    def _get_scorer(self) -> ImportanceScorer:
        """Lazy-initialise the shared ImportanceScorer."""
        if self._scorer is None:
            self._scorer = ImportanceScorer(self.store)
        return self._scorer
