"""oml/memory/importance.py — Importance scoring for AtomicNote retrieval.

Computes a composite importance score in [0, 1] for each note using a formula
inspired by the Ebbinghaus forgetting curve, Zipf access frequency, and graph
centrality:

  importance = confidence × recency_factor × frequency_factor × link_bonus

Where:
  recency_factor   = exp(−decay_const × days_since_last_access)
                     Ebbinghaus exponential decay; score halves every
                     ``HALF_LIFE_DAYS`` days of non-access (default: 30).
  frequency_factor  Logarithmically-scaled access count with a non-zero
                    floor (``FREQ_EPSILON``) so brand-new notes are not
                    penalised to zero.
  link_bonus        Graph-degree centrality bonus; a note connected to more
                    notes is assumed to be more "central" knowledge.

Usage
-----
    scorer = ImportanceScorer(store)

    # Score a single note
    score = scorer.score(note)                # float in [0, 1]

    # Re-rank a list of notes
    ranked = scorer.rank(notes)               # sorted descending

    # Batch-score every note in the store
    scores: dict[str, float] = scorer.score_all()

Design notes
------------
* The scorer is *stateless* — it reads live data from the store and note
  objects; it does not mutate anything.
* Access tracking (incrementing ``access_count`` / ``last_accessed``) is the
  responsibility of the caller (``TEEGStore.record_access()``).
* ``link_bonus`` is a multiplicative modifier in [1.0, 1.5], so it can push
  already-high scores past 1.0 before the final clamp — i.e. highly-connected,
  high-confidence notes reliably reach 1.0.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Dict, List

from oml.memory.atomic_note import AtomicNote

if TYPE_CHECKING:
    from oml.storage.teeg_store import TEEGStore


# ── tunables ─────────────────────────────────────────────────────────────────

HALF_LIFE_DAYS: float = 30.0
"""Ebbinghaus half-life: score halves for every ``HALF_LIFE_DAYS`` days of
non-access.  Increase for longer-lived memories (e.g. reference documents);
decrease for short-lived session memories."""

FREQ_EPSILON: float = 0.2
"""Non-zero floor for frequency_factor so new notes (access_count=0) are not
scored at zero.  Value in (0, 1); default 0.2 gives new notes 20 % of the
maximum frequency contribution."""

FREQ_MAX_COUNT: int = 100
"""Access count at which frequency_factor saturates (reaches 1.0).  Beyond
this threshold further accesses have no additional effect on the score."""

MAX_DEGREE: int = 20
"""Maximum graph degree used for link_bonus.  Degrees above this value are
capped so that highly-connected hub nodes do not dominate retrieval."""


# ── helpers ──────────────────────────────────────────────────────────────────


def _days_since(iso_ts: str) -> float:
    """Return the number of days since an ISO-8601 timestamp (or 0 on error)."""
    if not iso_ts:
        return 0.0
    try:
        dt = datetime.fromisoformat(iso_ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        delta = datetime.now(timezone.utc) - dt
        return max(0.0, delta.total_seconds() / 86_400.0)
    except (ValueError, TypeError):
        return 0.0


def _reference_timestamp(note: AtomicNote) -> str:
    """Return the most recent of last_accessed and created_at."""
    if note.last_accessed:
        return note.last_accessed
    return note.created_at


# ── scorer ───────────────────────────────────────────────────────────────────


class ImportanceScorer:
    """Composite importance scorer for AtomicNotes.

    Parameters
    ----------
    store:
        ``TEEGStore`` whose NetworkX graph is used for the link-centrality
        bonus.  Pass ``None`` to disable the link bonus (all notes get
        ``link_bonus = 1.0``).
    half_life_days:
        Recency decay half-life in days.  A note last accessed exactly
        ``half_life_days`` ago receives ``recency_factor = 0.5``.
    """

    def __init__(
        self,
        store: "TEEGStore | None" = None,
        half_life_days: float = HALF_LIFE_DAYS,
    ) -> None:
        self.store = store
        self.half_life_days = half_life_days
        # Pre-compute decay constant: recency = exp(-k*t), k = ln2 / half_life
        self._decay_k: float = math.log(2) / max(half_life_days, 1e-9)

    # ── public API ────────────────────────────────────────────────────────────

    def score(self, note: AtomicNote) -> float:
        """Return a composite importance score in [0, 1].

        Archived (inactive) notes always return ``0.0``.

        The four sub-components are:

        * **confidence** — self-reported correctness of the fact [0, 1].
        * **recency** — Ebbinghaus decay since last access.
        * **frequency** — log-scaled access count with non-zero floor.
        * **link_bonus** — multiplicative graph-degree centrality modifier
          in [1.0, 1.5].

        Returns
        -------
        float
            Score in [0, 1]; higher = more important.
        """
        if not note.active:
            return 0.0

        conf = float(note.confidence)
        recency = self._recency(note)
        freq = self._frequency(note)
        link = self._link_bonus(note.note_id)

        raw = conf * recency * freq * link
        return max(0.0, min(1.0, raw))

    def rank(self, notes: List[AtomicNote]) -> List[AtomicNote]:
        """Return ``notes`` sorted by importance score (descending)."""
        return sorted(notes, key=self.score, reverse=True)

    def top_k(self, notes: List[AtomicNote], k: int) -> List[AtomicNote]:
        """Return the *k* most important notes from ``notes``."""
        return self.rank(notes)[:k]

    def score_all(self) -> Dict[str, float]:
        """Score every note in the attached store.

        Returns
        -------
        dict[str, float]
            Mapping of ``note_id → importance_score`` for all notes.

        Raises
        ------
        RuntimeError
            If no store was provided at construction time.
        """
        if self.store is None:
            raise RuntimeError("ImportanceScorer was created without a store.")
        return {n.note_id: self.score(n) for n in self.store.get_all()}

    def explain(self, note: AtomicNote) -> dict:
        """Return a breakdown of all sub-components for debugging.

        Example output::

            {
              "note_id": "teeg-abc123",
              "final_score": 0.71,
              "confidence": 0.9,
              "recency": 0.97,
              "frequency": 0.38,
              "link_bonus": 1.25,
            }
        """
        conf = float(note.confidence)
        recency = self._recency(note)
        freq = self._frequency(note)
        link = self._link_bonus(note.note_id)
        raw = conf * recency * freq * link
        return {
            "note_id": note.note_id,
            "final_score": round(max(0.0, min(1.0, raw)), 4),
            "confidence": round(conf, 4),
            "recency": round(recency, 4),
            "frequency": round(freq, 4),
            "link_bonus": round(link, 4),
            "days_since_access": round(_days_since(_reference_timestamp(note)), 2),
            "access_count": note.access_count,
        }

    # ── sub-components ────────────────────────────────────────────────────────

    def _recency(self, note: AtomicNote) -> float:
        """Ebbinghaus exponential decay: exp(−k × days_since_last_access)."""
        days = _days_since(_reference_timestamp(note))
        return math.exp(-self._decay_k * days)

    def _frequency(self, note: AtomicNote) -> float:
        """Log-normalized access count with non-zero floor for new notes.

        Maps access_count ∈ [0, FREQ_MAX_COUNT] to [FREQ_EPSILON, 1.0].
        """
        count = max(0, int(note.access_count))
        if count == 0:
            return FREQ_EPSILON
        # Logarithmic ramp from FREQ_EPSILON → 1.0 over [1, FREQ_MAX_COUNT]
        log_ratio = math.log(1 + count) / math.log(1 + FREQ_MAX_COUNT)
        return FREQ_EPSILON + (1.0 - FREQ_EPSILON) * min(1.0, log_ratio)

    def _link_bonus(self, note_id: str) -> float:
        """Graph-degree centrality bonus in [1.0, 1.5].

        Isolated notes (degree=0) get 1.0; notes with ``MAX_DEGREE`` or more
        edges get 1.5.  The bonus is capped so hub nodes don't dominate.
        """
        if self.store is None:
            return 1.0
        graph = self.store.get_graph()
        if note_id not in graph:
            return 1.0
        degree = graph.degree(note_id)
        return 1.0 + 0.5 * min(degree, MAX_DEGREE) / MAX_DEGREE
