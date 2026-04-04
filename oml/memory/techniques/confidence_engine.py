"""
Confidence Engine — Bayesian confidence decay and boost application.
====================================================================

Applies judge verdicts via confidence decay rather than immediate hard
archival:
  - CONTRADICTS: existing note confidence decays proportionally to
    STRENGTH x AUTHORITY x current_confidence.  Archived only when
    confidence falls below _ARCHIVE_THRESHOLD.
  - EXTENDS / SUPPORTS: small logistic boost (saturates near 1.0).
  - UNRELATED: no change.
"""

from __future__ import annotations

import logging

from oml.memory.atomic_note import AtomicNote
from oml.storage.teeg_store import TEEGStore

logger = logging.getLogger(__name__)

# ── Verdict constants ─────────────────────────────────────────────────────────
_CONTRADICTS = "CONTRADICTS"
_EXTENDS     = "EXTENDS"
_SUPPORTS    = "SUPPORTS"

# ── Confidence decay (Bayesian update) ───────────────────────────────────────
# BASE_CONTRADICT = 0.90  ->  invariant A: one full-strength hit archives (1.0 -> 0.10)
# BASE_SUPPORT    = 0.10  ->  gentle logistic boost; saturates naturally near 1.0
_BASE_CONTRADICT    : float = 0.90
_BASE_SUPPORT       : float = 0.10
_ARCHIVE_THRESHOLD  : float = 0.15   # archive when confidence falls below this
_CONFIDENCE_STEP    : float = 0.05   # discretise to avoid false-precision artifacts
_PROPAGATION_FACTOR : float = 0.30   # fraction of delta forwarded to direct neighbours


def _round_confidence(value: float) -> float:
    """Round to nearest _CONFIDENCE_STEP, clamped to [0.05, 0.95]."""
    rounded = round(value / _CONFIDENCE_STEP) * _CONFIDENCE_STEP
    return max(0.05, min(0.95, rounded))


class ConfidenceEngine:
    """Applies judge verdicts using Bayesian confidence decay and boost.

    Parameters
    ----------
    store:
        The :class:`~oml.storage.teeg_store.TEEGStore` to read from and write to.
    add_to_propagation_queue:
        Callback function to queue belief propagation deltas.  Signature:
        ``(note_id: str, delta: float) -> None``.
    """

    def __init__(
        self,
        store: TEEGStore,
        add_to_propagation_queue=None,
    ):
        self.store = store
        self._add_to_propagation_queue = add_to_propagation_queue

    def apply_verdict(
        self,
        new_note: AtomicNote,
        existing_note: AtomicNote,
        relation: str,
        reason: str,
        strength: float = 1.0,
        authority: float = 1.0,
    ) -> None:
        """Apply the judge verdict using confidence decay.

        CONTRADICTS
        -----------
        Decays existing_note.confidence proportionally to strength x authority x
        current_confidence (logistic curve -- high-confidence notes require more
        contradictions to archive; fading notes decay quickly).  Archives only
        when confidence falls below _ARCHIVE_THRESHOLD.

        EXTENDS / SUPPORTS
        ------------------
        Applies a small logistic confidence boost to the existing note.  If the
        existing note is in the warm store (archived), a strong enough SUPPORTS
        verdict can resurrect it via ``store.unarchive()``.

        Belief propagation
        ------------------
        Non-trivial confidence changes are written to the propagation queue for
        lazy single-hop application in the next ``propagation_sweep()`` call.

        Parameters
        ----------
        strength:
            How strongly the relation holds (0.0-1.0).  Defaults to 1.0 for
            backward-compatible single-judge path.
        authority:
            How credible the new note's source is (0.0-1.0).  Defaults to 1.0.
        """
        if relation == _CONTRADICTS:
            if not existing_note.active:
                # Already archived -- no further decay needed
                logger.debug(
                    "[MemoryEvolver] Skipping CONTRADICTS for archived %r",
                    existing_note.note_id,
                )
                return

            raw_delta = _BASE_CONTRADICT * strength * authority * existing_note.confidence
            delta     = _round_confidence(raw_delta)
            new_conf  = _round_confidence(existing_note.confidence - delta)
            existing_note.confidence = new_conf

            new_note.supersedes = existing_note.note_id

            if existing_note.confidence < _ARCHIVE_THRESHOLD:
                self.store.archive(existing_note.note_id)
                logger.info(
                    "[MemoryEvolver] %r archived (conf=%.2f) reason=%r",
                    existing_note.note_id, existing_note.confidence, reason,
                )
            else:
                # Still active but weakened -- propagate damage to neighbours
                if self._add_to_propagation_queue:
                    self._add_to_propagation_queue(
                        existing_note.note_id, -delta * _PROPAGATION_FACTOR
                    )
                logger.info(
                    "[MemoryEvolver] %r decayed to conf=%.2f (not archived) reason=%r",
                    existing_note.note_id, existing_note.confidence, reason,
                )

        elif relation == _EXTENDS:
            self.store.add_edge(
                new_note.note_id, existing_note.note_id,
                relation="extends", weight=0.9,
            )
            # Small boost: being extended is weak corroboration
            if existing_note.active:
                raw_delta = _BASE_SUPPORT * 0.5 * strength * authority * (
                    1.0 - existing_note.confidence
                )
                delta    = _round_confidence(raw_delta)
                new_conf = _round_confidence(existing_note.confidence + delta)
                existing_note.confidence = new_conf
            logger.info(
                "[MemoryEvolver] %r extends %r  reason=%r",
                new_note.note_id, existing_note.note_id, reason,
            )

        elif relation == _SUPPORTS:
            raw_delta = _BASE_SUPPORT * strength * authority * (
                1.0 - existing_note.confidence
            )
            delta    = _round_confidence(raw_delta)
            new_conf = _round_confidence(existing_note.confidence + delta)
            existing_note.confidence = new_conf

            if not existing_note.active:
                # Warm-store note: attempt resurrection if confidence recovered
                if existing_note.confidence >= _ARCHIVE_THRESHOLD:
                    resurrected = self.store.unarchive(existing_note.note_id)
                    if resurrected:
                        logger.info(
                            "[MemoryEvolver] Resurrected %r (conf=%.2f)",
                            existing_note.note_id, existing_note.confidence,
                        )
            else:
                # Normal supports: queue propagation boost
                if self._add_to_propagation_queue:
                    self._add_to_propagation_queue(
                        existing_note.note_id, delta * _PROPAGATION_FACTOR
                    )

            self.store.add_edge(
                new_note.note_id, existing_note.note_id,
                relation="supports", weight=0.6,
            )
            logger.debug(
                "[MemoryEvolver] %r supports %r",
                new_note.note_id, existing_note.note_id,
            )

        # UNRELATED -> no edge, no confidence change
