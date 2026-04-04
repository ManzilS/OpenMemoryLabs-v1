"""
Belief Propagation — single-hop directed confidence propagation.
================================================================

Non-recursive, direction-enforced (out-edges only) propagation of
confidence deltas to direct graph neighbours.  Persists a queue file
(``teeg_propagation_queue.jsonl``) that survives crashes.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from oml.memory.atomic_note import AtomicNote
from oml.storage.teeg_store import TEEGStore

logger = logging.getLogger(__name__)

# ── Constants (shared with confidence_engine) ────────────────────────────────
_ARCHIVE_THRESHOLD  : float = 0.15
_CONFIDENCE_STEP    : float = 0.05
_PROPAGATION_FACTOR : float = 0.30


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _round_confidence(value: float) -> float:
    """Round to nearest _CONFIDENCE_STEP, clamped to [0.05, 0.95]."""
    rounded = round(value / _CONFIDENCE_STEP) * _CONFIDENCE_STEP
    return max(0.05, min(0.95, rounded))


class BeliefPropagator:
    """Single-hop directed belief propagation over the TEEG graph.

    Parameters
    ----------
    store:
        The :class:`~oml.storage.teeg_store.TEEGStore` to read from and write to.
    queue_path:
        Optional explicit path for the propagation queue file.  Defaults to
        ``store.artifacts_dir / "teeg_propagation_queue.jsonl"``.
    """

    def __init__(self, store: TEEGStore, queue_path: Path | None = None):
        self.store = store
        self._queue_path = queue_path or store.artifacts_dir / "teeg_propagation_queue.jsonl"

    @property
    def queue_path(self) -> Path:
        return self._queue_path

    def propagate_single_hop(self, note: AtomicNote, delta: float) -> None:
        """Apply a dampened confidence delta to direct out-edge neighbours.

        Strictly one hop only -- no recursion.  Direction-enforced (out-edges
        only) to prevent A->B->A feedback cycles even in cyclic graphs.
        """
        graph = self.store.get_graph()
        if note.note_id not in graph:
            return

        for u, v, data in graph.out_edges(note.note_id, data=True):
            neighbour = self.store.get(v)
            if neighbour is None or not neighbour.active:
                continue
            edge_weight = data.get("weight", 1.0)
            propagated  = delta * _PROPAGATION_FACTOR * edge_weight
            neighbour.confidence = _round_confidence(
                neighbour.confidence + propagated
            )
            if neighbour.confidence < _ARCHIVE_THRESHOLD:
                self.store.archive(neighbour.note_id)
                logger.info(
                    "[MemoryEvolver] Propagation archived %r (conf=%.2f)",
                    neighbour.note_id, neighbour.confidence,
                )

    def add_to_propagation_queue(self, note_id: str, delta: float) -> None:
        """Append a pending confidence delta to the persistent queue file."""
        entry = json.dumps({
            "note_id":   note_id,
            "delta":     delta,
            "queued_at": _utcnow_iso(),
            "done":      False,
        })
        with open(self._queue_path, "a", encoding="utf-8") as f:
            f.write(entry + "\n")

    def propagation_sweep(self) -> int:
        """Apply all pending single-hop belief propagation deltas.

        Reads ``teeg_propagation_queue.jsonl``, coalesces deltas by note_id,
        applies directed single-hop propagation to each note's out-neighbours,
        then rewrites the queue with all entries marked done.

        Returns
        -------
        int
            Number of notes whose confidence was updated by propagation.
        """
        if not self._queue_path.exists():
            return 0

        entries: List[dict] = []
        with open(self._queue_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

        pending = [e for e in entries if not e.get("done", False)]
        if not pending:
            return 0

        # Coalesce multiple deltas for the same note into one application
        coalesced: dict[str, float] = {}
        for entry in pending:
            nid = entry.get("note_id", "")
            coalesced[nid] = coalesced.get(nid, 0.0) + entry.get("delta", 0.0)

        count = 0
        for note_id, total_delta in coalesced.items():
            note = self.store.get(note_id)
            if note and note.active:
                self.propagate_single_hop(note, total_delta)
                count += 1

        # Mark all entries done (rewrite file)
        now = _utcnow_iso()
        with open(self._queue_path, "w", encoding="utf-8") as f:
            for entry in entries:
                entry["done"] = True
                entry["processed_at"] = now
                f.write(json.dumps(entry) + "\n")

        logger.info("[MemoryEvolver] Propagation sweep: %d notes updated", count)
        return count
