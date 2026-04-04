"""
MemoryEvolver — Active memory evolution via LLM-driven contradiction resolution.
=================================================================================

The key insight of the TEEG system is that contradictions should be resolved
**at write time**, not at query time.  When a new observation arrives, the
Evolver proactively:

  1. Finds existing notes (active + recently-archived warm store) semantically
     related to the new note.
  2. Runs a cheap Stage 1 pre-screen on ALL pairs — a fast 3B model that
     answers only YES / SCOPE? / NO.  Stage 1 is recall-biased: "when in doubt,
     say YES."  A false-positive (YES on a non-contradiction) is safe; a
     false-negative (NO on a real contradiction) is permanent.
  3. Escalates only YES / SCOPE? pairs to the full Stage 2 judge, which
     classifies CONTRADICTS / EXTENDS / SUPPORTS / UNRELATED and rates
     STRENGTH, AUTHORITY, and SCOPE_MATCH.
  4. Applies the verdict via confidence decay rather than immediate hard
     archival:
       - CONTRADICTS: existing note confidence decays proportionally to
         STRENGTH × AUTHORITY × current_confidence.  The note is archived
         only when confidence falls below _ARCHIVE_THRESHOLD.
       - EXTENDS / SUPPORTS: existing note confidence receives a small
         logistic boost (saturates near 1.0).
       - UNRELATED: no change.
  5. Queues single-hop directed belief propagation (non-recursive, no cycles)
     for background application via ``propagation_sweep()``.

Architecture (v2 — decomposed)
-------------------------------
MemoryEvolver is a thin *composer* that delegates to four atomic technique
modules:

  - :class:`~oml.memory.techniques.stage1_prescreen.Stage1PreScreen`
  - :class:`~oml.memory.techniques.stage2_judge.Stage2Judge`
  - :class:`~oml.memory.techniques.confidence_engine.ConfidenceEngine`
  - :class:`~oml.memory.techniques.belief_propagation.BeliefPropagator`

All original public method signatures (``evolve()``, ``propagation_sweep()``)
and internal method names (``_parse_stage1_verdict``, ``_build_judge_prompt``,
``_apply``, etc.) are preserved as thin wrappers for backward compatibility.

Fallback Safety
---------------
- Stage 1 failure → defaults to YES (recall-biased)
- Stage 2 / LLM failure → defaults to SUPPORTS (archive-safe conservative)
- Unresolvable parse → SUPPORTS
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple

from oml.memory.atomic_note import AtomicNote
from oml.memory.techniques.belief_propagation import BeliefPropagator
from oml.memory.techniques.confidence_engine import ConfidenceEngine
from oml.memory.techniques.stage1_prescreen import Stage1PreScreen
from oml.memory.techniques.stage2_judge import Stage2Judge
from oml.storage.teeg_store import TEEGStore

logger = logging.getLogger(__name__)

# ── Verdict constants (re-exported for backward compatibility) ───────────────
_CONTRADICTS = "CONTRADICTS"
_EXTENDS     = "EXTENDS"
_SUPPORTS    = "SUPPORTS"
_UNRELATED   = "UNRELATED"
_VALID_RELATIONS = {_CONTRADICTS, _EXTENDS, _SUPPORTS, _UNRELATED}

# ── Candidate search ──────────────────────────────────────────────────────────
_MAX_CANDIDATES  = 5    # Stage 2 judge candidates (active notes)
_WARM_CANDIDATES = 3    # Warm-store resurrection candidates
_WARM_STORE_DAYS = 30   # Archived notes searchable for resurrection

# ── Confidence decay constants (re-exported for backward compatibility) ──────
_BASE_CONTRADICT    : float = 0.90
_BASE_SUPPORT       : float = 0.10
_ARCHIVE_THRESHOLD  : float = 0.15
_CONFIDENCE_STEP    : float = 0.05
_PROPAGATION_FACTOR : float = 0.30

# ── Stage 1 fuzzy parser patterns (re-exported for backward compatibility) ───
_SCOPE_PATTERNS = Stage1PreScreen.__module__ and [
    r'\bscope\b', r'\bcontext\b', r'\bcondition\b', r'\baltitude\b',
    r'\bhowever\b', r'\bcaveat\b', r'\bdepend', r'\bqualif',
    r'\bspecific\b', r'\bvary\b', r'\bvaries\b', r'\bdifferent\b',
    r'\bbut\b',
]
_CONTRADICTION_PATTERNS: List[str] = [
    r'\bcontradict', r'\bconflict', r'\boppos', r'\brefut',
    r'\bdisagree\b', r'\binconsist', r'\bincompat',
]
_NO_PATTERNS: List[str] = [
    r'\bunrelated\b', r'\bsupport', r'\belaborat', r'\bextend',
    r'\bcorroborat', r'\bconsistent\b', r'\bagree\b',
    r'\bsame\s+(fact|claim|idea|thing)\b',
]


def _utcnow_iso() -> str:
    from oml.memory.techniques.belief_propagation import _utcnow_iso as _impl
    return _impl()


def _round_confidence(value: float) -> float:
    """Round to nearest _CONFIDENCE_STEP, clamped to [0.05, 0.95]."""
    from oml.memory.techniques.confidence_engine import _round_confidence as _impl
    return _impl(value)


class MemoryEvolver:
    """
    Resolves contradictions and links new notes to existing memory at write time.

    Thin composer that delegates to four atomic technique modules:
    Stage1PreScreen, Stage2Judge, ConfidenceEngine, and BeliefPropagator.

    Parameters
    ----------
    store:
        The :class:`~oml.storage.teeg_store.TEEGStore` to read from and write to.
    model_name:
        Stage 2 (full judge) LLM provider string.
    similarity_top_k:
        How many active-note candidates to send to Stage 2 per evolution cycle.
    stage1_model_name:
        Stage 1 (cheap pre-screen) LLM.  Defaults to ``model_name`` if None —
        shares the same client, which is correct for single-model setups.
        Set to a smaller/faster model (e.g. ``"lmstudio:qwen-1.5b"``) to gate
        Stage 2 cheaply.
    """

    def __init__(
        self,
        store: TEEGStore,
        model_name: str = "mock",
        similarity_top_k: int = _MAX_CANDIDATES,
        stage1_model_name: Optional[str] = None,
    ):
        self.store = store
        self.model_name = model_name
        self.stage1_model_name = stage1_model_name or model_name
        self.similarity_top_k = similarity_top_k
        self._llm = None        # lazy Stage 2 / fallback LLM
        self._stage1_llm = None # lazy Stage 1 LLM (may alias self._llm)

        # ── Technique delegates ───────────────────────────────────────────
        self._stage1 = Stage1PreScreen()
        self._stage2 = Stage2Judge()
        self._propagator = BeliefPropagator(store)
        self._confidence = ConfidenceEngine(
            store,
            add_to_propagation_queue=self._propagator.add_to_propagation_queue,
        )
        self._queue_path: Path = self._propagator.queue_path

    # ── LLM sync: keep technique delegates in sync with injected _llm ────

    def _sync_llm_to_delegates(self):
        """Push the current _llm to technique delegates that need it."""
        if self._llm is not None:
            self._stage1.llm = self._llm
            self._stage2.llm = self._llm

    # ── public API ────────────────────────────────────────────────────────────

    def evolve(self, new_note: AtomicNote) -> None:
        """Ingest a new note into the store, resolving any conflicts first.

        Steps:
          1. Find candidates (active hot-store + warm archived-store).
          2. Persist the new note so its graph node exists before add_edge calls.
          3. Two-stage judge: Stage 1 fast pre-screen → Stage 2 full judge on
             escalated pairs only.
          4. Apply verdicts via confidence decay; queue belief propagation.
        """
        candidates = self._find_candidates(new_note)

        # Persist FIRST so graph node exists before any add_edge() calls.
        self.store.add(new_note)

        if not candidates:
            pass
        elif len(candidates) == 1:
            results = self._judge_batch([new_note], candidates)
            relation, reason, strength, authority = results[0]
            self._apply(new_note, candidates[0], relation, reason, strength, authority)
        else:
            pairs_new = [new_note] * len(candidates)
            results = self._judge_batch(pairs_new, candidates)
            for existing_note, (relation, reason, strength, authority) in zip(candidates, results):
                self._apply(new_note, existing_note, relation, reason, strength, authority)

        logger.info(
            "[MemoryEvolver] Stored note %r after evaluating %d candidate(s)",
            new_note.note_id, len(candidates),
        )

    def evolve_batch(self, notes: List[AtomicNote]) -> None:
        """Evolve a list of notes sequentially.

        Each note is evolved *after* the previous one is stored, so later notes
        in the batch can conflict with earlier ones in the same batch.
        """
        for note in notes:
            self.evolve(note)

    def propagation_sweep(self) -> int:
        """Apply all pending single-hop belief propagation deltas.

        Delegates to :meth:`BeliefPropagator.propagation_sweep`.

        Returns
        -------
        int
            Number of notes whose confidence was updated by propagation.
        """
        return self._propagator.propagation_sweep()

    # ── Stage 1: thin wrappers ────────────────────────────────────────────────

    def _parse_stage1_verdict(self, response: str) -> str:
        """Fuzzy-parse Stage 1 output → YES / SCOPE? / NO.  (Thin wrapper.)"""
        return self._stage1.parse_stage1_verdict(response)

    def _build_stage1_prompt(
        self, new_note: AtomicNote, existing_note: AtomicNote
    ) -> str:
        """Ultra-light Stage 1 prompt.  (Thin wrapper.)"""
        return self._stage1.build_stage1_prompt(new_note, existing_note)

    def _run_stage1(
        self,
        new_notes: List[AtomicNote],
        existing_notes: List[AtomicNote],
    ) -> List[str]:
        """Run Stage 1 fast pre-screen on all pairs.  (Thin wrapper.)"""
        self._sync_llm_to_delegates()
        # Ensure the stage1 LLM is set (may differ from stage2)
        stage1_llm = self._get_stage1_llm()
        self._stage1.llm = stage1_llm
        return self._stage1.screen(new_notes, existing_notes)

    # ── Stage 2: thin wrappers ────────────────────────────────────────────────

    def _judge_batch(
        self,
        new_notes: List[AtomicNote],
        existing_notes: List[AtomicNote],
    ) -> List[Tuple[str, str, float, float]]:
        """Two-stage batch judge.  (Thin wrapper.)"""
        self._sync_llm_to_delegates()
        # Ensure stage1 LLM is set for the prescreen step
        stage1_llm = self._get_stage1_llm()
        self._stage1.llm = stage1_llm
        # Ensure stage2 LLM is set
        stage2_llm = self._get_llm()
        self._stage2.llm = stage2_llm
        return self._stage2.judge_batch(
            new_notes, existing_notes,
            stage1_prescreen=self._stage1,
            similarity_top_k=self.similarity_top_k,
        )

    def _judge(
        self, new_note: AtomicNote, existing_note: AtomicNote
    ) -> Tuple[str, str]:
        """Single-pair fallback judge (no Stage 1).  Returns (relation, reason).

        Kept for backward compatibility with tests that call ``_judge`` directly.
        """
        self._sync_llm_to_delegates()
        stage2_llm = self._get_llm()
        self._stage2.llm = stage2_llm
        return self._stage2.judge(new_note, existing_note)

    def _judge_full(
        self, new_note: AtomicNote, existing_note: AtomicNote
    ) -> Tuple[str, str, float, float]:
        """Single-pair full judge returning (relation, reason, strength, authority)."""
        self._sync_llm_to_delegates()
        stage2_llm = self._get_llm()
        self._stage2.llm = stage2_llm
        return self._stage2.judge_full(new_note, existing_note)

    def _build_judge_prompt(
        self, new_note: AtomicNote, existing_note: AtomicNote
    ) -> str:
        """Build the Stage 2 TOON-encoded judge prompt.  (Thin wrapper.)"""
        return self._stage2.build_judge_prompt(new_note, existing_note)

    def _parse_verdict(self, response: str) -> Tuple[str, str]:
        """Parse Stage 2 response → (relation, reason).  Backward-compat alias."""
        return self._stage2.parse_verdict(response)

    def _parse_verdict_full(
        self, response: str
    ) -> Tuple[str, str, float, float]:
        """Parse Stage 2 response → (relation, reason, strength, authority)."""
        return self._stage2.parse_verdict_full(response)

    # ── Verdict application (thin wrapper) ────────────────────────────────────

    def _apply(
        self,
        new_note: AtomicNote,
        existing_note: AtomicNote,
        relation: str,
        reason: str,
        strength: float = 1.0,
        authority: float = 1.0,
    ) -> None:
        """Apply the judge verdict using confidence decay.  (Thin wrapper.)"""
        self._confidence.apply_verdict(
            new_note, existing_note, relation, reason, strength, authority
        )

    # ── Belief propagation (thin wrappers) ────────────────────────────────────

    def _propagate_single_hop(self, note: AtomicNote, delta: float) -> None:
        """Apply a dampened confidence delta to direct out-edge neighbours."""
        self._propagator.propagate_single_hop(note, delta)

    def _add_to_propagation_queue(self, note_id: str, delta: float) -> None:
        """Append a pending confidence delta to the persistent queue file."""
        self._propagator.add_to_propagation_queue(note_id, delta)

    # ── Candidate discovery ───────────────────────────────────────────────────

    def _find_candidates(self, new_note: AtomicNote) -> List[AtomicNote]:
        """Return candidates from both hot (active) and warm (recently-archived) stores.

        Hot candidates: top-k active notes by vector/keyword similarity.
        Warm candidates: recently-archived notes eligible for resurrection.
        Merged and deduplicated; hot candidates take priority.
        """
        # Hot store
        hot_results = self.store.vector_search(
            new_note.embedding_text(), top_k=self.similarity_top_k
        )
        hot = [n for n, _s in hot_results if n.note_id != new_note.note_id]

        # Warm store (resurrection candidates)
        warm_results = self.store.vector_search_warm(
            new_note.embedding_text(),
            top_k=_WARM_CANDIDATES,
            warm_days=_WARM_STORE_DAYS,
        )
        seen = {n.note_id for n in hot}
        warm = [n for n, _s in warm_results
                if n.note_id != new_note.note_id and n.note_id not in seen]

        return (hot + warm)[: self.similarity_top_k]

    # ── LLM client access ─────────────────────────────────────────────────────

    def _get_llm(self):
        if self._llm is None:
            from oml.llm.factory import get_llm_client
            self._llm = get_llm_client(self.model_name)
        return self._llm

    def _get_stage1_llm(self):
        """Return the Stage 1 LLM client.

        If ``stage1_model_name == model_name`` (the default), reuse the Stage 2
        client so test mocks that inject ``self._llm`` are automatically shared.
        """
        if self.stage1_model_name == self.model_name:
            return self._get_llm()
        if self._stage1_llm is None:
            from oml.llm.factory import get_llm_client
            self._stage1_llm = get_llm_client(self.stage1_model_name)
        return self._stage1_llm

    # ── diagnostics ──────────────────────────────────────────────────────────

    def audit(self) -> dict:
        """Return a summary of the evolution state of the store."""
        notes            = self.store.get_all()
        active           = [n for n in notes if n.active]
        archived         = [n for n in notes if not n.active]
        supersession_chains = [n for n in notes if n.supersedes]
        graph            = self.store.get_graph()
        edges_by_relation: dict[str, int] = {}
        for _, _, data in graph.edges(data=True):
            rel = data.get("relation", "unknown")
            edges_by_relation[rel] = edges_by_relation.get(rel, 0) + 1

        # Propagation queue depth
        queue_depth = 0
        if self._queue_path.exists():
            with open(self._queue_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        e = json.loads(line.strip())
                        if not e.get("done", False):
                            queue_depth += 1
                    except json.JSONDecodeError:
                        pass

        return {
            "active_notes":         len(active),
            "archived_notes":       len(archived),
            "supersession_chains":  len(supersession_chains),
            "edges_by_relation":    edges_by_relation,
            "propagation_queue":    queue_depth,
        }
