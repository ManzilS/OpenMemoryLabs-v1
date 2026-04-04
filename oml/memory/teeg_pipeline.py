"""
TEEGPipeline — End-to-end facade for the TEEG memory system.
=============================================================

Orchestrates the three-layer TEEG architecture:

  1. **Ingest** raw text → LLM distills it into an AtomicNote → MemoryEvolver
     resolves contradictions and links to related notes → TEEGStore persists.

  2. **Query** → ScoutRetriever traverses graph → TOON-encoded context packed
     into LLM prompt → response generated.

The pipeline is designed to be usable both programmatically and from the CLI
(see ``oml/cli.py``).

Usage
-----
    # Quick demo (mock LLM, no GPU)
    pipeline = TEEGPipeline(artifacts_dir="teeg_store", model="mock")
    pipeline.ingest("Victor Frankenstein spent two years assembling the creature.")
    pipeline.ingest("The creature was brought to life in a stormy November night.")
    pipeline.ingest("Victor fled his laboratory immediately after the creature awoke.")
    response = pipeline.query("What did Victor do when the creature came to life?")
    print(response)

    # With a real LLM
    pipeline = TEEGPipeline(
        artifacts_dir="teeg_store",
        model="ollama:qwen3:4b",
        token_budget=2000,
    )
    pipeline.ingest_batch(["fact 1", "fact 2", "fact 3"])
    pipeline.save()   # persist to disk
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

from oml.memory.atomic_note import AtomicNote
from oml.memory.compressor import TieredContextPacker
from oml.memory.importance import ImportanceScorer
from oml.storage.teeg_store import TEEGStore
from oml.memory.evolver import MemoryEvolver
from oml.retrieval.scout import ScoutRetriever, ScoutResult
from oml.memory.techniques.llm_distiller import LLMDistiller
from oml.memory.techniques.heuristic_distiller import HeuristicDistiller
from oml.memory.techniques.answer_generator import AnswerGenerator

logger = logging.getLogger(__name__)


class TEEGPipeline:
    """
    End-to-end TEEG memory pipeline: Observe → Evolve → Query.

    Parameters
    ----------
    artifacts_dir:
        Directory for persistent storage (notes + graph).
    model:
        LLM provider string used for both note distillation and generation.
    token_budget:
        Max tokens to allocate to TEEG context in each query prompt.
    scout_top_k:
        Number of notes to include in query context.
    scout_max_hops:
        Graph traversal depth for Scout retrieval.
    """

    def __init__(
        self,
        artifacts_dir: str | Path = "teeg_store",
        model: str = "mock",
        token_budget: int = 3000,
        scout_top_k: int = 8,
        scout_max_hops: int = 2,
    ):
        self.model = model
        self.token_budget = token_budget

        self.store = TEEGStore(artifacts_dir)
        self.evolver = MemoryEvolver(self.store, model_name=model)
        self.scout = ScoutRetriever(
            self.store,
            max_hops=scout_max_hops,
        )

        self._llm = None   # lazy-loaded

        # Technique modules are created lazily (need _llm)
        self._distiller: Optional[LLMDistiller] = None
        self._answer_gen: Optional[AnswerGenerator] = None

    # ── technique accessors (lazy, share _llm) ────────────────────────────────

    def _get_distiller(self) -> LLMDistiller:
        if self._distiller is None or self._distiller._llm is not self._get_llm():
            self._distiller = LLMDistiller(self._get_llm())
        return self._distiller

    def _get_answer_generator(self) -> AnswerGenerator:
        if self._answer_gen is None or self._answer_gen._llm is not self._get_llm():
            self._answer_gen = AnswerGenerator(self._get_llm())
        return self._answer_gen

    # ── ingestion ─────────────────────────────────────────────────────────────

    def ingest(
        self,
        raw_text: str,
        context_hint: str = "",
        source_id: str = "",
    ) -> AtomicNote:
        """
        Distil ``raw_text`` into an AtomicNote and evolve memory.

        The LLM is asked to produce a structured AtomicNote in TOON format.
        If the LLM call fails (e.g. model="mock"), the pipeline falls back to
        a heuristic note derived directly from the raw text.

        Args:
            raw_text:     The observation / document excerpt to memorise.
            context_hint: Optional free-text hint about the source.
            source_id:    Optional ID linking back to the originating document.

        Returns:
            The :class:`AtomicNote` that was ingested and stored.
        """
        note = self._distil(raw_text, context_hint, source_id)
        self.evolver.evolve(note)
        logger.info(f"[TEEGPipeline] Ingested note {note.note_id!r}")
        return note

    def ingest_batch(
        self,
        texts: List[str],
        context_hint: str = "",
    ) -> List[AtomicNote]:
        """Ingest a list of raw texts and return the resulting notes."""
        notes = []
        for text in texts:
            note = self.ingest(text, context_hint=context_hint)
            notes.append(note)
        return notes

    def ingest_note(self, note: AtomicNote) -> AtomicNote:
        """
        Ingest a pre-constructed AtomicNote directly (skip distillation).

        Useful when the caller already has structured data or is migrating
        from another memory system.
        """
        self.evolver.evolve(note)
        return note

    # ── query ─────────────────────────────────────────────────────────────────

    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        return_context: bool = False,
    ) -> str | Tuple[str, str]:
        """
        Answer a question using TEEG memory.

        Steps:
          1. Scout retrieves logically-connected notes.
          2. Notes are TOON-encoded into a context block (token-budget-aware).
          3. Context + question are sent to the LLM.
          4. LLM response returned.

        Args:
            question:       The question to answer.
            top_k:          Override default scout_top_k.
            return_context: If True, return ``(answer, context_str)`` tuple.

        Returns:
            LLM answer string, or ``(answer, context)`` if ``return_context``.
        """
        k = top_k or self.scout.seed_k * 2
        scout_results = self.scout.search(question, top_k=k)

        # Use TieredContextPacker for importance-weighted compression
        if scout_results:
            scorer = ImportanceScorer(self.store)
            importance = scorer.score_all()
            packer = TieredContextPacker(budget=self.token_budget)
            context_str = packer.pack(scout_results, importance_scores=importance)
        else:
            context_str = "[TEEG MEMORY]\n(no relevant memory found)\n[/TEEG MEMORY]"

        answer = self._get_answer_generator().generate(question, context_str)

        if return_context:
            return answer, context_str
        return answer

    def search(self, query: str, top_k: int = 5) -> List[ScoutResult]:
        """
        Return raw Scout results without generating an LLM answer.

        Useful for inspection, evaluation, and programmatic use.
        """
        return self.scout.search(query, top_k=top_k)

    # ── persistence ──────────────────────────────────────────────────────────

    def save(self) -> None:
        """Persist the store (notes + graph) to disk."""
        self.store.save()
        logger.info("[TEEGPipeline] Store saved")

    def rebuild_vector_index(self) -> None:
        """Rebuild the in-memory vector index from the current note set."""
        self.store.build_vector_index()

    # ── diagnostics ──────────────────────────────────────────────────────────

    def stats(self) -> dict:
        """Return pipeline and store statistics."""
        return {
            "model": self.model,
            "token_budget": self.token_budget,
            **self.store.stats(),
            "evolution_audit": self.evolver.audit(),
        }

    def explain_query(self, question: str, top_k: int = 5) -> str:
        """Return a human-readable explanation of how Scout would answer."""
        return self.scout.explain(question, top_k=top_k)

    # ── internals (backward-compat wrappers) ──────────────────────────────────

    def _get_llm(self):
        if self._llm is None:
            from oml.llm.factory import get_llm_client
            self._llm = get_llm_client(self.model)
        return self._llm

    def _distil(
        self, raw_text: str, context_hint: str, source_id: str
    ) -> AtomicNote:
        """
        Ask the LLM to distil raw_text into a structured AtomicNote.

        Delegates to :class:`LLMDistiller`; falls back to
        :class:`HeuristicDistiller` on any failure.
        """
        try:
            distiller = self._get_distiller()
            note = distiller.distil(raw_text, context_hint, source_id)
        except Exception as exc:
            logger.warning(f"[TEEGPipeline] Distil LLM failed ({exc}); using heuristic")
            note = self._heuristic_note(raw_text, context_hint, source_id)

        if source_id:
            note.source_ids = [source_id]
        return note

    def _build_distil_prompt(self, raw_text: str, context_hint: str) -> str:
        """Backward-compat wrapper — delegates to LLMDistiller."""
        return LLMDistiller._build_distil_prompt(raw_text, context_hint)

    def _parse_distil_response(
        self, response: str, raw_text: str, source_id: str
    ) -> AtomicNote:
        """Backward-compat wrapper — delegates to LLMDistiller."""
        return LLMDistiller._parse_distil_response(response, raw_text, source_id)

    @staticmethod
    def _heuristic_note(
        raw_text: str, context_hint: str, source_id: str
    ) -> AtomicNote:
        """Backward-compat wrapper — delegates to HeuristicDistiller."""
        return HeuristicDistiller.distil(raw_text, context_hint, source_id)

    @staticmethod
    def _build_query_prompt(question: str, context_str: str) -> str:
        """Backward-compat wrapper — delegates to AnswerGenerator."""
        return AnswerGenerator._build_query_prompt(question, context_str)
