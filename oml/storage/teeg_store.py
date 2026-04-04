"""
TEEGStore — Persistent storage for the TEEG memory system.
===========================================================

Layer 1: Note storage
    Primary storage is a JSON-Lines file (one TOON-serialized AtomicNote per line)
    in ``<artifacts_dir>/teeg_notes.jsonl``.  This keeps the store zero-dependency
    by default — no LanceDB required for basic operation.

    When LanceDB is available, notes are *also* embedded and stored as a LanceDB
    table for vector-similarity fallback during Scout retrieval.

Layer 2: Relation graph
    Explicit edges between notes are persisted as a NetworkX DiGraph in
    ``<artifacts_dir>/teeg_graph.pkl``.  Each edge carries:
      - ``relation``  — predicate string (e.g. "contradicts", "supports", "extends")
      - ``weight``    — float [0, 1] confidence of the relation

Public API
----------
    store = TEEGStore("teeg_store/")
    store.add(note)
    store.add_edge(from_id, to_id, relation="extends", weight=0.9)
    notes = store.get_active()
    similar = store.vector_search("who created the creature", top_k=5)
    store.save()
    store.load()
"""

from __future__ import annotations

import json
import logging
import pickle
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import networkx as nx

from oml.memory.atomic_note import AtomicNote
from oml.utils.device import resolve_device

logger = logging.getLogger(__name__)

# Resolve once at import time so all TEEGStore instances share the same device
_DEVICE = resolve_device("auto")

_NOTES_FILE = "teeg_notes.jsonl"
_GRAPH_FILE = "teeg_graph.pkl"
_LANCE_TABLE = "teeg_notes"


class TEEGStore:
    """
    Dual-layer store: flat JSON-Lines for notes + NetworkX graph for edges.

    The vector-search capability is optional and activates automatically when
    LanceDB and sentence-transformers are available.
    """

    def __init__(self, artifacts_dir: str | Path = "teeg_store"):
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        self._notes: Dict[str, AtomicNote] = {}   # note_id → AtomicNote
        self._graph: nx.DiGraph = nx.DiGraph()

        # Optional vector index (sentence-transformers)
        self._embedder = None
        self._vectors: Dict[str, list] = {}        # note_id → embedding list

        self.load()

    # ── note CRUD ────────────────────────────────────────────────────────────

    def add(self, note: AtomicNote) -> None:
        """Upsert a note. If note_id already exists it is overwritten."""
        self._notes[note.note_id] = note
        self._graph.add_node(note.note_id, content=note.content, active=note.active)
        logger.debug(f"[TEEGStore] Added note {note.note_id}")

    def get(self, note_id: str) -> Optional[AtomicNote]:
        """Retrieve a note by ID (active or archived)."""
        return self._notes.get(note_id)

    def get_active(self) -> List[AtomicNote]:
        """Return all non-superseded notes."""
        return [n for n in self._notes.values() if n.active]

    def get_all(self) -> List[AtomicNote]:
        """Return every note including archived ones."""
        return list(self._notes.values())

    def archive(self, note_id: str) -> None:
        """Mark a note as inactive (soft-delete / superseded).

        Sets ``active=False`` and stamps ``archived_at`` with the current UTC
        time so ``vector_search_warm`` can bound the resurrection window.
        """
        if note_id in self._notes:
            note = self._notes[note_id]
            note.active = False
            note.archived_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
            self._graph.nodes[note_id]["active"] = False
            # Remove from vector index — archived notes must not appear in hot search
            self._vectors.pop(note_id, None)
            logger.info(f"[TEEGStore] Archived note {note_id}")

    def unarchive(self, note_id: str) -> bool:
        """Restore an archived note to active status (idempotent).

        Re-embeds the note in the vector index if the embedder is available.
        Called by :class:`~oml.memory.evolver.MemoryEvolver` when a SUPPORTS
        verdict on an archived note raises its confidence above the archive
        threshold (resurrection path).

        Returns
        -------
        bool
            ``True`` if the note was actually restored; ``False`` if it was
            already active or does not exist (no-op).
        """
        note = self._notes.get(note_id)
        if note is None or note.active:
            return False   # already active or missing — idempotent no-op
        note.active = True
        note.archived_at = ""
        self._graph.nodes[note_id]["active"] = True
        # Re-add to vector index if not already present
        if note_id not in self._vectors:
            self._embed_single(note)
        logger.info(f"[TEEGStore] Resurrected note {note_id}")
        return True

    def record_access(self, note_id: str) -> None:
        """Increment ``access_count`` and refresh ``last_accessed`` timestamp.

        Called by :class:`~oml.retrieval.scout.ScoutRetriever` each time a note
        is returned in a search result.  The updated fields feed into
        :class:`~oml.memory.importance.ImportanceScorer` on the next query.

        This method modifies the in-memory note but does *not* persist to disk;
        call :meth:`save` to write the updated counts.
        """
        note = self._notes.get(note_id)
        if note is not None:
            note.access_count += 1
            note.last_accessed = datetime.now(timezone.utc).isoformat(timespec="seconds")

    def count(self) -> int:
        """Total number of notes (active + archived)."""
        return len(self._notes)

    def active_count(self) -> int:
        """Number of active (non-archived) notes."""
        return sum(1 for n in self._notes.values() if n.active)

    # ── graph edges ──────────────────────────────────────────────────────────

    def add_edge(
        self,
        from_id: str,
        to_id: str,
        relation: str = "related",
        weight: float = 1.0,
    ) -> None:
        """Add a directed labelled edge between two notes.

        Both nodes must already exist (call :meth:`add` first).
        """
        if from_id not in self._graph or to_id not in self._graph:
            logger.warning(
                f"[TEEGStore] Cannot add edge {from_id!r} → {to_id!r}: node(s) missing"
            )
            return
        self._graph.add_edge(from_id, to_id, relation=relation, weight=weight)

    def get_edges(self, note_id: str) -> List[Tuple[str, str, dict]]:
        """Return all edges incident to a note (outgoing + incoming).

        Returns a list of ``(from_id, to_id, edge_data)`` tuples.
        """
        out = [(note_id, v, d) for v, d in self._graph[note_id].items()]
        inc = [(u, note_id, d) for u, d in self._graph.pred[note_id].items()]
        return out + inc

    def neighbors(self, note_id: str, relation: Optional[str] = None) -> List[str]:
        """Return IDs of directly connected notes, optionally filtered by relation."""
        result = []
        for _, v, data in self.get_edges(note_id):
            if relation is None or data.get("relation") == relation:
                result.append(v)
        return result

    def get_graph(self) -> nx.DiGraph:
        """Return the underlying NetworkX graph (read-only by convention)."""
        return self._graph

    # ── vector search (optional) ─────────────────────────────────────────────

    def _ensure_embedder(self) -> bool:
        """Lazy-load the sentence-transformer embedder onto the auto-detected device."""
        if self._embedder is not None:
            return True
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"[TEEGStore] Loading embedder on {_DEVICE}...")
            self._embedder = SentenceTransformer("all-MiniLM-L6-v2", device=_DEVICE)
            return True
        except Exception as exc:
            logger.warning(f"[TEEGStore] Vector search unavailable: {exc}")
            return False

    def build_vector_index(self) -> None:
        """Embed all active notes. Called after bulk ingestion."""
        if not self._ensure_embedder():
            return
        active = self.get_active()
        if not active:
            return
        texts = [n.embedding_text() for n in active]
        embeddings = self._embedder.encode(texts, show_progress_bar=False)
        self._vectors = {
            n.note_id: emb.tolist()
            for n, emb in zip(active, embeddings)
        }
        logger.info(f"[TEEGStore] Built vector index for {len(self._vectors)} notes")

    def _embed_single(self, note: "AtomicNote") -> None:  # type: ignore[name-defined]
        """Embed a single note and add it to the in-memory vector index.

        Called by :meth:`unarchive` to restore a resurrected note to the hot
        index without rebuilding the entire index.  No-op if the embedder is
        unavailable.
        """
        if not self._ensure_embedder():
            return
        try:
            emb = self._embedder.encode([note.embedding_text()], show_progress_bar=False)[0]
            self._vectors[note.note_id] = emb.tolist()
        except Exception as exc:
            logger.warning(f"[TEEGStore] Could not embed resurrected note {note.note_id}: {exc}")

    def vector_search(self, query: str, top_k: int = 5) -> List[Tuple[AtomicNote, float]]:
        """
        Return the top-k active notes most semantically similar to ``query``.

        Returns a list of ``(AtomicNote, cosine_similarity)`` tuples, sorted
        descending by similarity.

        Falls back to keyword matching (checks if query words appear in note
        content) when the embedder is unavailable or the index is empty.
        """
        if self._vectors and self._ensure_embedder():
            return self._vector_search_dense(query, top_k)
        return self._keyword_fallback(query, top_k)

    def _vector_search_dense(
        self, query: str, top_k: int
    ) -> List[Tuple[AtomicNote, float]]:
        import numpy as np

        q_emb = self._embedder.encode([query], show_progress_bar=False)[0]

        scores: List[Tuple[AtomicNote, float]] = []
        for note_id, vec in self._vectors.items():
            note = self._notes.get(note_id)
            if note is None or not note.active:
                continue
            n_emb = np.array(vec)
            # Cosine similarity
            denom = (np.linalg.norm(q_emb) * np.linalg.norm(n_emb))
            sim = float(np.dot(q_emb, n_emb) / denom) if denom > 0 else 0.0
            scores.append((note, sim))

        scores.sort(key=lambda x: -x[1])
        return scores[:top_k]

    def _keyword_fallback(
        self, query: str, top_k: int
    ) -> List[Tuple[AtomicNote, float]]:
        """Simple keyword overlap fallback when embedder is unavailable."""
        words = {w.lower().strip(".,?!") for w in query.split() if len(w) > 2}
        scored: List[Tuple[AtomicNote, float]] = []
        for note in self.get_active():
            note_words = set(note.content.lower().split())
            note_words.update(k.lower() for k in note.keywords)
            overlap = len(words & note_words)
            if overlap > 0:
                scored.append((note, float(overlap) / len(words)))
        scored.sort(key=lambda x: -x[1])
        return scored[:top_k]

    def vector_search_warm(
        self,
        query: str,
        top_k: int = 3,
        warm_days: int = 30,
    ) -> List[Tuple[AtomicNote, float]]:
        """Search recently-archived notes for potential resurrection candidates.

        Only considers notes that:
          - are inactive (``active=False``)
          - were archived within ``warm_days`` days
          - have ``confidence > 0.05`` (above absolute floor — deeply dead notes
            are not worth resurrecting)

        Uses keyword overlap (same as ``_keyword_fallback``) — warm notes are
        not in the dense vector index.  Returns an empty list if no warm notes
        match.

        This is intentionally a small, bounded search: warm notes are few
        (bounded by ingest_rate × warm_days × survival_rate) so the O(n) scan
        is cheap even without a vector index.

        Parameters
        ----------
        query:
            Embedding text of the new note being ingested.
        top_k:
            Maximum warm candidates to return.
        warm_days:
            Grace period in days.  Notes archived longer ago are ignored.
        """
        cutoff = (
            datetime.now(timezone.utc) - timedelta(days=warm_days)
        ).isoformat(timespec="seconds")

        words = {w.lower().strip(".,?!") for w in query.split() if len(w) > 2}
        if not words:
            return []

        scored: List[Tuple[AtomicNote, float]] = []
        for note in self._notes.values():
            if note.active:
                continue
            if not note.archived_at or note.archived_at < cutoff:
                continue
            if note.confidence <= 0.05:
                continue  # below absolute floor — skip
            note_words = set(note.content.lower().split())
            note_words.update(k.lower() for k in note.keywords)
            overlap = len(words & note_words)
            if overlap > 0:
                scored.append((note, float(overlap) / len(words)))

        scored.sort(key=lambda x: -x[1])
        return scored[:top_k]

    # ── persistence ──────────────────────────────────────────────────────────

    def save(self) -> None:
        """Persist notes (JSON-Lines) and graph (pickle) to disk."""
        notes_path = self.artifacts_dir / _NOTES_FILE
        with open(notes_path, "w", encoding="utf-8") as f:
            for note in self._notes.values():
                f.write(json.dumps(note.to_dict()) + "\n")

        graph_path = self.artifacts_dir / _GRAPH_FILE
        with open(graph_path, "wb") as f:
            pickle.dump(self._graph, f)

        logger.info(
            f"[TEEGStore] Saved {len(self._notes)} notes, "
            f"{self._graph.number_of_edges()} edges"
        )

    def load(self) -> None:
        """Load notes and graph from disk (no-op if files don't exist yet)."""
        notes_path = self.artifacts_dir / _NOTES_FILE
        if notes_path.exists():
            with open(notes_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                        note = AtomicNote.from_dict(d)
                        self._notes[note.note_id] = note
                        self._graph.add_node(
                            note.note_id,
                            content=note.content,
                            active=note.active,
                        )
                    except Exception as exc:
                        logger.warning(f"[TEEGStore] Skipped malformed note: {exc}")

        graph_path = self.artifacts_dir / _GRAPH_FILE
        if graph_path.exists():
            try:
                with open(graph_path, "rb") as f:
                    self._graph = pickle.load(f)
            except Exception as exc:
                logger.warning(f"[TEEGStore] Could not load graph: {exc}")

        logger.debug(
            f"[TEEGStore] Loaded {len(self._notes)} notes, "
            f"{self._graph.number_of_edges()} edges"
        )

    def stats(self) -> dict:
        """Return store statistics."""
        return {
            "total_notes": self.count(),
            "active_notes": self.active_count(),
            "archived_notes": self.count() - self.active_count(),
            "graph_nodes": self._graph.number_of_nodes(),
            "graph_edges": self._graph.number_of_edges(),
            "vector_index_size": len(self._vectors),
        }
