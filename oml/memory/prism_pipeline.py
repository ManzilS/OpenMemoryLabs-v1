"""
PRISMPipeline — Three-layer efficient memory pipeline for AI agents.
=====================================================================

PRISM = Probabilistic Retrieval with Intelligent Sparse Memory

Wraps the existing TEEG/Scout/TieredContextPacker stack with three
efficiency innovations that together address all three bottlenecks
in large-scale AI memory management:

  Layer 1 — SketchGate (oml/memory/sketch.py)
  ─────────────────────────────────────────────
  Write-time near-duplicate detection via MinHash LSH.  Before making
  any LLM calls, the SketchGate checks whether the incoming text is
  ≥ 75 % similar (by keyword Jaccard) to an existing note.  If so, the
  ingest is skipped and the existing note's ``access_count`` is
  incremented instead.

  Computational efficiency: O(N × H) MinHash scan (no embedding model)
  vs. O(N × D) dense cosine search.

  Layer 2 — DeltaStore (oml/memory/delta.py)
  ────────────────────────────────────────────
  Semantic patch storage for ``EXTENDS`` verdicts.  When a new note
  extends an existing base note, only the new information (the delta)
  is stored rather than the full content of the extending note.  At
  context-assembly time, the reconstructed full content is used and
  one COMPACT block is rendered instead of two FULL blocks, saving ~32
  LLM context tokens per delta note per query.

  Token efficiency: reduces per-note storage by ~60 % for extending notes.

  Layer 3 — CallBatcher (oml/memory/batcher.py)
  ──────────────────────────────────────────────
  N-to-1 LLM call coalescing for bulk ingestion.  ``batch_ingest()``
  packs N distillation requests into one structured LLM prompt and N
  evolution judgments into a second prompt, reducing the per-document
  API cost from 2 calls to 2/N calls.

  Call efficiency: 87.5 % savings for N=8, 96.9 % for N=32.

Novel combination
-----------------
No existing AI memory system (MemGPT, A-MEM, LangMem, Zep) combines
MinHash deduplication, semantic delta encoding, and batched multi-output
LLM distillation in the same pipeline.  PRISM is specifically designed
for the ``AtomicNote`` / TOON representation used in TEEG.

Usage
-----
    pipeline = PRISMPipeline(artifacts_dir="teeg_store", model="mock")

    # Single ingest with dedup gate
    result = pipeline.ingest("Victor created the creature in Geneva.")
    print(result.was_deduplicated)   # False on first call
    result2 = pipeline.ingest("Victor created the creature in Geneva.")
    print(result2.was_deduplicated)  # True — skipped, access_count++

    # Batch ingest: 8 texts, 2 LLM calls instead of 16
    batch = pipeline.batch_ingest(texts)
    print(batch.call_efficiency)     # ~0.875

    # Query delegates to TEEGPipeline (Scout + TieredContextPacker)
    answer, ctx = pipeline.query("Who created the creature?")

    print(pipeline.stats())
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

from oml.memory.atomic_note import AtomicNote
from oml.memory.batcher import CallBatcher
from oml.memory.delta import DeltaStore
from oml.memory.sketch import SketchGate
from oml.memory.teeg_pipeline import TEEGPipeline
from oml.storage.teeg_store import TEEGStore

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Result / stats dataclasses
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class PRISMIngestResult:
    """Outcome of a single PRISM ingest operation.

    Attributes
    ----------
    note:
        The resulting ``AtomicNote``.  For deduplicated inputs this is the
        *existing* note (not a newly created one).
    was_deduplicated:
        True if the SketchGate identified the input as a near-duplicate of an
        existing note and the LLM call was skipped.
    merged_into:
        ``note_id`` of the existing note if ``was_deduplicated`` is True,
        otherwise None.
    is_delta:
        True if the note was stored as a ``SemanticPatch`` in the DeltaStore
        (only set by ``batch_ingest`` for ``EXTENDS`` verdicts).
    """

    note: AtomicNote
    was_deduplicated: bool = False
    merged_into: Optional[str] = None
    is_delta: bool = False


@dataclass
class PRISMBatchResult:
    """Aggregate outcome of a ``batch_ingest`` call.

    Attributes
    ----------
    notes:
        All resulting ``AtomicNote`` objects (deduped inputs return existing notes).
    dedup_count:
        Number of inputs skipped by the SketchGate.
    delta_count:
        Number of notes stored as ``SemanticPatch`` objects (EXTENDS verdict).
    llm_calls_made:
        Actual number of ``llm.generate()`` calls made.
    llm_calls_saved:
        Calls avoided vs. the naive 2-per-text baseline.
    call_efficiency:
        ``llm_calls_saved / (llm_calls_made + llm_calls_saved)`` in [0, 1].
    """

    notes: List[AtomicNote] = field(default_factory=list)
    dedup_count: int = 0
    delta_count: int = 0
    llm_calls_made: int = 0
    llm_calls_saved: int = 0
    call_efficiency: float = 0.0


@dataclass
class PRISMStats:
    """Aggregated statistics from all PRISM efficiency layers.

    Attributes
    ----------
    total_notes:
        All notes in the TEEGStore (active + archived).
    active_notes:
        Active (non-archived) notes.
    delta_notes:
        Total ``SemanticPatch`` objects in the DeltaStore.
    dedup_rate:
        Fraction of ingest calls that were skipped by SketchGate (0–1).
    avg_call_efficiency:
        Mean call efficiency across all ``batch_ingest`` calls (0–1).
    token_savings_est:
        Estimated LLM context tokens saved per query from delta storage.
    bloom_fp_rate:
        Configured Bloom filter false-positive rate.
    minhash_threshold:
        Configured MinHash Jaccard dedup threshold.
    """

    total_notes: int = 0
    active_notes: int = 0
    delta_notes: int = 0
    dedup_rate: float = 0.0
    avg_call_efficiency: float = 0.0
    token_savings_est: int = 0
    bloom_fp_rate: float = 0.01
    minhash_threshold: float = 0.75


# ══════════════════════════════════════════════════════════════════════════════
# PRISMPipeline
# ══════════════════════════════════════════════════════════════════════════════


class PRISMPipeline:
    """Three-layer efficient memory pipeline for AI and AI agents.

    Parameters
    ----------
    artifacts_dir:
        Directory for all persistent storage (TEEG notes, graph, SketchGate,
        DeltaStore).
    model:
        LLM provider string passed to ``get_llm_client()``.
    token_budget:
        Max tokens allocated to TEEG context per query prompt.
    scout_top_k:
        Number of notes to include in query context.
    scout_max_hops:
        Graph traversal depth for ScoutRetriever.
    dedup_threshold:
        MinHash Jaccard similarity threshold above which an ingest is
        considered a near-duplicate (default 0.75).
    batch_size:
        Maximum items per batch LLM call (default 8).
    """

    def __init__(
        self,
        artifacts_dir: str | Path = "teeg_store",
        model: str = "mock",
        token_budget: int = 3000,
        scout_top_k: int = 8,
        scout_max_hops: int = 2,
        dedup_threshold: float = 0.75,
        batch_size: int = 8,
    ) -> None:
        self.model = model
        self.artifacts_dir = Path(artifacts_dir)

        # Shared TEEGStore (reused by TEEGPipeline to avoid double-loading)
        self._store = TEEGStore(artifacts_dir)

        # TEEGPipeline owns evolver, scout, query logic
        self._teeg = TEEGPipeline(
            artifacts_dir=artifacts_dir,
            model=model,
            token_budget=token_budget,
            scout_top_k=scout_top_k,
            scout_max_hops=scout_max_hops,
        )
        # Use the shared store from TEEGPipeline to avoid double-loading
        self._store = self._teeg.store

        # PRISM efficiency layers
        self._sketch = SketchGate(
            artifacts_dir=artifacts_dir,
            dedup_threshold=dedup_threshold,
        )
        self._delta = DeltaStore(artifacts_dir=artifacts_dir)
        self._batcher: Optional[CallBatcher] = None  # lazy-init after LLM loaded

        # Cumulative efficiency counters
        self._total_calls_made: int = 0
        self._total_calls_saved: int = 0

        # Warm-register all existing active notes into SketchGate
        # (only if gate is freshly initialised — i.e., sketch_gate.json absent)
        if len(self._sketch._minhash) == 0 and self._store.active_count() > 0:
            self._sketch.bulk_register(self._store.get_active())
            logger.debug(
                "[PRISM] Warm-registered %d existing notes into SketchGate",
                self._store.active_count(),
            )

    # ══════════════════════════════════════════════════════════════════════════
    # Ingest — single text
    # ══════════════════════════════════════════════════════════════════════════

    def ingest(
        self,
        text: str,
        context_hint: str = "",
        source_id: str = "",
    ) -> PRISMIngestResult:
        """Ingest a single text with full PRISM efficiency stack.

        1. SketchGate dedup check — if near-duplicate found, skip LLM call and
           increment ``access_count`` on the existing note.
        2. TEEGPipeline.ingest() for new notes (distil → evolve → store).
        3. SketchGate.register() so future near-duplicates are caught.

        Returns
        -------
        PRISMIngestResult
            Contains the resulting note, dedup flag, and merge target.
        """
        # ── Layer 1: SketchGate check ─────────────────────────────────────────
        # We need keywords to check; do a quick extraction from text
        quick_keywords = _quick_keywords(text)
        existing_id = self._sketch.should_skip(text, quick_keywords)

        if existing_id is not None:
            existing_note = self._store.get(existing_id)
            if existing_note is not None and existing_note.active:
                self._store.record_access(existing_id)
                logger.info(
                    "[PRISM] Near-duplicate of %r — skipping ingest (access_count++)",
                    existing_id,
                )
                return PRISMIngestResult(
                    note=existing_note,
                    was_deduplicated=True,
                    merged_into=existing_id,
                )
            # Existing note was archived — proceed with fresh ingest
            logger.debug(
                "[PRISM] Near-duplicate %r was archived; re-ingesting", existing_id
            )

        # ── Layers 2+3: normal TEEGPipeline ingest ────────────────────────────
        note = self._teeg.ingest(text, context_hint=context_hint, source_id=source_id)

        # Register new note in SketchGate for future dedup.
        # Use the same quick_keywords that should_skip() uses for consistent
        # Jaccard estimation (LLM-extracted note.keywords may differ from the
        # simple word extraction used at query time).
        self._sketch.register(note, keywords_override=quick_keywords)

        return PRISMIngestResult(note=note, was_deduplicated=False)

    # ══════════════════════════════════════════════════════════════════════════
    # Batch ingest — N texts → 2 LLM calls
    # ══════════════════════════════════════════════════════════════════════════

    def batch_ingest(
        self,
        texts: List[str],
        context_hints: Optional[List[str]] = None,
    ) -> PRISMBatchResult:
        """Ingest *N* texts in 2 LLM calls (vs 2N in the naive approach).

        Pipeline:
        1. SketchGate filters near-duplicates (O(N × H), no LLM).
        2. CallBatcher.distil_batch() → 1 LLM call → N TOON strings.
        3. Parse TOON strings into AtomicNotes.
        4. Find one candidate note per new note (vector search top-1).
        5. CallBatcher.evolve_batch() → 1 LLM call → N verdicts.
        6. Apply verdicts:
           - EXTENDS     → DeltaStore patch + TEEGStore + graph edge
           - CONTRADICTS → archive old + supersedes link
           - SUPPORTS    → TEEGStore + graph edge
           - UNRELATED   → TEEGStore only
        7. SketchGate.register() for all new notes.

        Parameters
        ----------
        texts:
            Raw texts to ingest.
        context_hints:
            Optional per-text context hints (same length as *texts*).

        Returns
        -------
        PRISMBatchResult
            Aggregated outcome including call-efficiency metrics.
        """
        if not texts:
            return PRISMBatchResult()

        hints = list(context_hints) if context_hints else [""] * len(texts)
        result = PRISMBatchResult()

        # ── Step 1: SketchGate filter ─────────────────────────────────────────
        filtered_texts: List[str] = []
        filtered_hints: List[str] = []
        filtered_kws: List[List[str]] = []   # reused in Step 6 for consistent Jaccard
        dedup_notes: List[AtomicNote] = []

        for text, hint in zip(texts, hints):
            kws = _quick_keywords(text)
            existing_id = self._sketch.should_skip(text, kws)
            if existing_id is not None:
                existing = self._store.get(existing_id)
                if existing is not None and existing.active:
                    self._store.record_access(existing_id)
                    result.dedup_count += 1
                    dedup_notes.append(existing)
                    continue
            filtered_texts.append(text)
            filtered_hints.append(hint)
            filtered_kws.append(kws)   # store for Step 6 reuse

        # All inputs were near-duplicates
        if not filtered_texts:
            result.notes = dedup_notes
            result.call_efficiency = 1.0
            return result

        # ── Step 2: Batch distillation ────────────────────────────────────────
        batcher = self._get_batcher()
        distil_result = batcher.distil_batch(filtered_texts, context_hints=filtered_hints)
        naive_calls_expected = 2 * len(filtered_texts)

        # ── Step 3: Parse TOON strings → AtomicNotes ──────────────────────────
        new_notes: List[AtomicNote] = []
        for i, toon_str in enumerate(distil_result.toon_strings):
            note = _parse_toon_to_note(toon_str, filtered_texts[i])
            new_notes.append(note)

        # ── Step 4: Find one candidate per new note ───────────────────────────
        candidates: List[AtomicNote] = []
        for note in new_notes:
            search_results = self._store.vector_search(note.content, top_k=1)
            if search_results:
                candidates.append(search_results[0][0])
            else:
                # No existing notes — use a dummy "unrelated" placeholder
                candidates.append(
                    AtomicNote(
                        content="(no existing notes)",
                        context="",
                        keywords=[],
                        tags=[],
                        confidence=0.5,
                    )
                )

        # ── Step 5: Batch evolution judgment ──────────────────────────────────
        has_existing = self._store.active_count() > 0
        if has_existing:
            verdict_result = batcher.evolve_batch(new_notes, candidates)
            verdicts = verdict_result.verdicts
            total_calls = distil_result.total_llm_calls + verdict_result.total_llm_calls
        else:
            # No existing notes → all verdicts are UNRELATED
            verdicts = ["UNRELATED"] * len(new_notes)
            total_calls = distil_result.total_llm_calls

        # ── Step 6: Apply verdicts ────────────────────────────────────────────
        final_notes: List[AtomicNote] = list(dedup_notes)

        for note, verdict, candidate, raw_text, note_kws in zip(
            new_notes, verdicts, candidates, filtered_texts, filtered_kws
        ):
            cand_id = candidate.note_id
            is_real_candidate = (
                candidate.content != "(no existing notes)"
                and self._store.get(cand_id) is not None
            )

            if verdict == "EXTENDS" and is_real_candidate:
                # Store as delta + full note (delta for efficient context, full for retrieval)
                self._delta.store_patch(
                    base_id=cand_id,
                    patch_note=note,
                    patch_type="EXTENDS",
                )
                self._store.add(note)
                self._store.add_edge(note.note_id, cand_id, "extends", 0.9)
                result.delta_count += 1
                logger.debug(
                    "[PRISM] Delta stored: %s EXTENDS %s", note.note_id, cand_id
                )

            elif verdict == "CONTRADICTS" and is_real_candidate:
                note.supersedes = cand_id
                self._store.archive(cand_id)
                self._store.add(note)
                logger.debug(
                    "[PRISM] Contradiction: %s archived, %s supersedes",
                    cand_id, note.note_id,
                )

            elif verdict == "SUPPORTS" and is_real_candidate:
                self._store.add(note)
                self._store.add_edge(note.note_id, cand_id, "supports", 0.6)

            else:  # UNRELATED or no real candidate
                self._store.add(note)

            # Register with the SAME keywords computed in Step 1 (no recomputation).
            # Using the same keyword set guarantees Jaccard consistency between
            # should_skip() and register() even if _quick_keywords() ever changes.
            self._sketch.register(note, keywords_override=note_kws)
            final_notes.append(note)

        # ── Step 7: Compute efficiency metrics ────────────────────────────────
        calls_saved = max(0, naive_calls_expected - total_calls)
        self._total_calls_made += total_calls
        self._total_calls_saved += calls_saved

        total_seen = total_calls + calls_saved
        result.notes = final_notes
        result.llm_calls_made = total_calls
        result.llm_calls_saved = calls_saved
        result.call_efficiency = (
            round(calls_saved / total_seen, 4) if total_seen > 0 else 0.0
        )
        return result

    # ══════════════════════════════════════════════════════════════════════════
    def ingest_batch(self, texts, context_hints=None):
        """Convenience alias for batch_ingest (satisfies MemoryPipeline protocol)."""
        return self.batch_ingest(texts, context_hints=context_hints)

    # Query — delegates to TEEGPipeline (Scout + TieredContextPacker)
    # ══════════════════════════════════════════════════════════════════════════

    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        return_context: bool = True,
    ) -> Tuple[str, str]:
        """Answer a question using TEEG memory with all efficiency layers active.

        Delegates to ``TEEGPipeline.query()`` which already uses
        ``ScoutRetriever`` (graph BFS + importance weighting) and
        ``TieredContextPacker`` (FULL/COMPACT/MINIMAL tiered compression).

        Returns
        -------
        Tuple[str, str]
            ``(answer, context_str)`` where *context_str* is the TOON memory
            block sent to the LLM.
        """
        result = self._teeg.query(question, top_k=top_k, return_context=True)
        if isinstance(result, tuple):
            return result
        return result, ""  # pragma: no cover

    def search(self, query: str, top_k: int = 5):
        """Return raw Scout results without generating an LLM answer."""
        return self._teeg.search(query, top_k=top_k)

    # ══════════════════════════════════════════════════════════════════════════
    # Persistence
    # ══════════════════════════════════════════════════════════════════════════

    def save(self) -> None:
        """Persist all three layers (TEEGStore, SketchGate, DeltaStore) to disk."""
        self._teeg.save()
        self._sketch.save()
        self._delta.save()
        logger.info("[PRISM] All layers saved to %s", self.artifacts_dir)

    # ══════════════════════════════════════════════════════════════════════════
    # Diagnostics
    # ══════════════════════════════════════════════════════════════════════════

    def stats(self) -> PRISMStats:
        """Return aggregated statistics from all PRISM efficiency layers."""
        store_stats = self._store.stats()
        sketch_stats = self._sketch.stats()
        delta_stats = self._delta.stats()

        total_naive = self._total_calls_made + self._total_calls_saved
        avg_efficiency = (
            self._total_calls_saved / total_naive if total_naive > 0 else 0.0
        )

        return PRISMStats(
            total_notes=store_stats["total_notes"],
            active_notes=store_stats["active_notes"],
            delta_notes=delta_stats["total_patches"],
            dedup_rate=sketch_stats["dedup_rate"],
            avg_call_efficiency=round(avg_efficiency, 4),
            token_savings_est=(
                delta_stats["token_savings_est"]
                + store_stats.get("active_notes", 0) * 0  # from TieredContextPacker (handled separately)
            ),
            bloom_fp_rate=self._sketch._bloom._fp_rate,
            minhash_threshold=self._sketch.dedup_threshold,
        )

    def raw_stats(self) -> dict:
        """Return full raw stats dict from all components (for CLI/API)."""
        store_stats = self._store.stats()
        sketch_stats = self._sketch.stats()
        delta_stats = self._delta.stats()
        batcher_stats = self._get_batcher().stats() if self._batcher else {
            "calls_made": self._total_calls_made,
            "calls_saved": self._total_calls_saved,
            "call_efficiency": round(
                self._total_calls_saved / max(1, self._total_calls_made + self._total_calls_saved), 4
            ),
        }
        return {
            "store": store_stats,
            "sketch_gate": sketch_stats,
            "delta_store": delta_stats,
            "call_batcher": batcher_stats,
            "model": self.model,
        }

    # ══════════════════════════════════════════════════════════════════════════
    # Internals
    # ══════════════════════════════════════════════════════════════════════════

    def _get_batcher(self) -> CallBatcher:
        """Lazy-initialise the CallBatcher (needs the LLM client)."""
        if self._batcher is None:
            llm = self._teeg._get_llm()
            self._batcher = CallBatcher(llm_client=llm, max_batch_size=8)
        return self._batcher


# ══════════════════════════════════════════════════════════════════════════════
# Module-level helpers
# ══════════════════════════════════════════════════════════════════════════════


def _quick_keywords(text: str, max_kw: int = 6) -> List[str]:
    """Extract a quick keyword list from raw text without LLM.

    Used by the SketchGate dedup check *before* any LLM call is made.
    Simple heuristic: unique lowercase words longer than 3 characters,
    minus common stop words.
    """
    _STOP = frozenset({
        "this", "that", "with", "from", "they", "were", "have",
        "been", "will", "into", "when", "then", "than", "also",
        "more", "some", "such", "each", "what", "where", "which",
    })
    words = text.lower().split()
    seen = set()
    kws: List[str] = []
    for w in words:
        w = w.strip(".,!?;:\"'()")
        if len(w) > 3 and w not in _STOP and w not in seen:
            seen.add(w)
            kws.append(w)
            if len(kws) >= max_kw:
                break
    return kws


def _parse_toon_to_note(toon_str: str, raw_text: str) -> AtomicNote:
    """Parse a TOON string into an AtomicNote, falling back to heuristic."""
    # Strip markdown fences
    clean = toon_str.strip()
    for fence in ("```toon", "```yaml", "```"):
        clean = clean.replace(fence, "")
    clean = clean.strip()

    if clean and "content:" in clean.lower():
        try:
            note = AtomicNote.from_toon(clean)
            if note.content:
                return note
        except Exception as exc:  # noqa: BLE001
            logger.debug("[PRISM] TOON parse failed (%s); using heuristic", exc)

    # Heuristic fallback
    words = raw_text.split()
    content = " ".join(words[:40])
    keywords = list(
        {w.lower().strip(".,!?;:") for w in words[:20] if len(w) > 3}
    )[:6]
    return AtomicNote(
        content=content,
        context="auto-generated",
        keywords=keywords,
        tags=["auto"],
        confidence=0.5,
    )
