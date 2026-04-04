"""tests/test_memory_efficiency.py — Tests for TEEG memory efficiency components.

Covers:
  - ImportanceScorer: recency decay, frequency floor, link bonus, archive=0
  - TieredContextPacker: tier assignment, token savings, budget enforcement
  - MemoryConsolidator: cluster detection, archival, dry-run, token savings
  - AtomicNote: new access_count / last_accessed fields, round-trip persistence
  - TEEGStore: record_access() increments correctly
  - Integration: Scout access tracking, pipeline tiered context
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import pytest

from oml.memory.atomic_note import AtomicNote
from oml.memory.compressor import (
    Tier,
    TieredContextPacker,
    encode_compact,
    encode_minimal,
    estimate_tokens,
    tier_token_cost,
)
from oml.memory.consolidator import ConsolidationResult, MemoryConsolidator
from oml.memory.importance import ImportanceScorer
from oml.storage.teeg_store import TEEGStore


# ── helpers ───────────────────────────────────────────────────────────────────


def _note(content: str, keywords=None, tags=None, confidence=0.9, active=True) -> AtomicNote:
    return AtomicNote(
        content=content,
        context="test",
        keywords=keywords or ["alpha", "beta"],
        tags=tags or ["test"],
        confidence=confidence,
        active=active,
    )


def _store_with_notes(notes: list[AtomicNote]) -> tuple[TEEGStore, Path]:
    """Create a temp TEEGStore pre-loaded with *notes*."""
    tmpdir = tempfile.mkdtemp()
    store = TEEGStore(artifacts_dir=tmpdir)
    for n in notes:
        store.add(n)
    return store, Path(tmpdir)


# ── AtomicNote: new fields ────────────────────────────────────────────────────


class TestAtomicNoteUsageFields:
    def test_default_access_count_is_zero(self):
        note = _note("test content")
        assert note.access_count == 0

    def test_default_last_accessed_is_empty(self):
        note = _note("test content")
        assert note.last_accessed == ""

    def test_to_dict_includes_access_fields(self):
        note = _note("content")
        note.access_count = 5
        note.last_accessed = "2026-01-01T00:00:00"
        d = note.to_dict()
        assert d["access_count"] == 5
        assert d["last_accessed"] == "2026-01-01T00:00:00"

    def test_from_dict_restores_access_fields(self):
        d = {
            "note_id": "teeg-abc",
            "content": "restored",
            "context": "",
            "keywords": "",
            "tags": "",
            "created_at": "2026-01-01T00:00:00",
            "supersedes": "",
            "confidence": 0.8,
            "source_ids": "",
            "active": True,
            "access_count": 7,
            "last_accessed": "2026-02-01T00:00:00",
        }
        note = AtomicNote.from_dict(d)
        assert note.access_count == 7
        assert note.last_accessed == "2026-02-01T00:00:00"

    def test_from_dict_handles_missing_access_fields(self):
        """Legacy dicts without access_count/last_accessed default gracefully."""
        d = {
            "note_id": "teeg-legacy",
            "content": "old note",
            "context": "",
            "keywords": "",
            "tags": "",
            "created_at": "2025-01-01T00:00:00",
            "supersedes": "",
            "confidence": 0.9,
            "source_ids": "",
            "active": True,
        }
        note = AtomicNote.from_dict(d)
        assert note.access_count == 0
        assert note.last_accessed == ""

    def test_to_toon_does_not_include_access_fields(self):
        """access_count / last_accessed must NOT appear in TOON (LLM context)."""
        note = _note("content")
        note.access_count = 99
        note.last_accessed = "2026-01-01T00:00:00"
        toon_str = note.to_toon()
        assert "access_count" not in toon_str
        assert "last_accessed" not in toon_str


# ── TEEGStore.record_access ───────────────────────────────────────────────────


class TestRecordAccess:
    def test_record_access_increments_count(self):
        store, _ = _store_with_notes([_note("a")])
        note = store.get_active()[0]
        nid = note.note_id
        store.record_access(nid)
        assert store.get(nid).access_count == 1

    def test_record_access_twice(self):
        store, _ = _store_with_notes([_note("a")])
        nid = store.get_active()[0].note_id
        store.record_access(nid)
        store.record_access(nid)
        assert store.get(nid).access_count == 2

    def test_record_access_sets_last_accessed(self):
        store, _ = _store_with_notes([_note("a")])
        nid = store.get_active()[0].note_id
        store.record_access(nid)
        assert store.get(nid).last_accessed != ""

    def test_record_access_missing_id_noop(self):
        store, _ = _store_with_notes([])
        # Should not raise
        store.record_access("nonexistent-id")

    def test_record_access_persists_on_save(self, tmp_path):
        store = TEEGStore(artifacts_dir=tmp_path)
        note = _note("persistent")
        store.add(note)
        store.record_access(note.note_id)
        store.save()

        store2 = TEEGStore(artifacts_dir=tmp_path)
        reloaded = store2.get(note.note_id)
        assert reloaded.access_count == 1


# ── ImportanceScorer ──────────────────────────────────────────────────────────


class TestImportanceScorer:
    def test_archived_note_scores_zero(self):
        note = _note("archived", active=False)
        store, _ = _store_with_notes([note])
        scorer = ImportanceScorer(store)
        assert scorer.score(note) == 0.0

    def test_active_note_scores_positive(self):
        note = _note("active note")
        store, _ = _store_with_notes([note])
        scorer = ImportanceScorer(store)
        assert scorer.score(note) > 0.0

    def test_score_in_unit_interval(self):
        for conf in [0.1, 0.5, 0.9]:
            note = _note("test", confidence=conf)
            store, _ = _store_with_notes([note])
            scorer = ImportanceScorer(store)
            score = scorer.score(note)
            assert 0.0 <= score <= 1.0, f"score {score} out of range for conf={conf}"

    def test_higher_confidence_scores_higher(self):
        low = _note("low", confidence=0.2)
        high = _note("high", confidence=0.9)
        store, _ = _store_with_notes([low, high])
        scorer = ImportanceScorer(store)
        assert scorer.score(high) > scorer.score(low)

    def test_accessed_note_scores_higher_than_new(self):
        new_note = _note("brand new")
        accessed = _note("often accessed")
        accessed.access_count = 20
        store, _ = _store_with_notes([new_note, accessed])
        scorer = ImportanceScorer(store)
        # accessed note should beat brand-new note at same confidence
        assert scorer.score(accessed) > scorer.score(new_note)

    def test_frequency_floor_nonzero(self):
        """New notes (access_count=0) get a non-zero frequency factor."""
        note = _note("new")
        store, _ = _store_with_notes([note])
        scorer = ImportanceScorer(store)
        # Must be positive even at zero accesses
        assert scorer.score(note) > 0.0

    def test_link_bonus_increases_score(self):
        isolated = _note("isolated")
        connected = _note("connected")
        store, _ = _store_with_notes([isolated, connected])
        extra = _note("extra")
        store.add(extra)
        store.add_edge(connected.note_id, extra.note_id, relation="supports")
        scorer = ImportanceScorer(store)
        # connected note has degree=1; isolated has degree=0
        assert scorer.score(connected) > scorer.score(isolated)

    def test_rank_returns_descending_order(self):
        notes = [_note(f"note {i}", confidence=i * 0.1) for i in range(1, 6)]
        store, _ = _store_with_notes(notes)
        scorer = ImportanceScorer(store)
        ranked = scorer.rank(notes)
        scores = [scorer.score(n) for n in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_score_all_returns_all_note_ids(self):
        notes = [_note(f"note {i}") for i in range(4)]
        store, _ = _store_with_notes(notes)
        scorer = ImportanceScorer(store)
        all_scores = scorer.score_all()
        assert len(all_scores) == 4
        for nid in all_scores:
            assert 0.0 <= all_scores[nid] <= 1.0

    def test_score_all_raises_without_store(self):
        scorer = ImportanceScorer(store=None)
        with pytest.raises(RuntimeError, match="without a store"):
            scorer.score_all()

    def test_explain_returns_all_components(self):
        note = _note("explain me")
        store, _ = _store_with_notes([note])
        scorer = ImportanceScorer(store)
        info = scorer.explain(note)
        for key in ("note_id", "final_score", "confidence", "recency",
                    "frequency", "link_bonus", "days_since_access", "access_count"):
            assert key in info


# ── TieredContextPacker / compressor ─────────────────────────────────────────


class TestCompressor:
    def test_encode_compact_shorter_than_full(self):
        note = _note("Victor Frankenstein assembled the creature from corpse parts")
        full = note.to_toon()
        compact = encode_compact(note)
        assert len(compact) < len(full)

    def test_encode_minimal_shortest(self):
        note = _note("Victor Frankenstein assembled the creature")
        compact = encode_compact(note)
        minimal = encode_minimal(note)
        assert len(minimal) < len(compact)

    def test_encode_compact_contains_content(self):
        note = _note("The creature learned language secretly")
        compact = encode_compact(note)
        assert "The creature learned language secretly" in compact

    def test_encode_minimal_is_single_line(self):
        note = _note("short fact")
        minimal = encode_minimal(note)
        assert "\n" not in minimal

    def test_encode_compact_omits_timestamps(self):
        note = _note("fact")
        compact = encode_compact(note)
        assert "created_at" not in compact

    def test_encode_compact_omits_active_field(self):
        note = _note("fact")
        compact = encode_compact(note)
        assert "active:" not in compact

    def test_tier_token_cost_ordering(self):
        note = _note("longer fact that takes up some tokens here")
        full = tier_token_cost(note, Tier.FULL)
        compact = tier_token_cost(note, Tier.COMPACT)
        minimal = tier_token_cost(note, Tier.MINIMAL)
        assert full > compact > minimal

    def test_packer_respects_budget(self):
        notes = [_note(f"note {i} with some content about frankenstein and monsters") for i in range(10)]
        store, _ = _store_with_notes(notes)
        results = [(n, 0.9, 0) for n in notes]
        budget = 200
        packer = TieredContextPacker(budget=budget)
        context = packer.pack(results)
        tokens = estimate_tokens(context)
        # Allow 5% overshoot due to rounding; budget is a soft cap
        assert tokens <= budget * 1.05

    def test_packer_wraps_in_teeg_tags(self):
        note = _note("fact")
        results = [(note, 0.9, 0)]
        packer = TieredContextPacker(budget=2000)
        context = packer.pack(results)
        assert context.startswith("[TEEG MEMORY]")
        assert context.endswith("[/TEEG MEMORY]")

    def test_packer_empty_results(self):
        packer = TieredContextPacker(budget=2000)
        context = packer.pack([])
        assert "no relevant memory found" in context

    def test_packer_stats_tokens_less_than_all_full(self):
        notes = [_note(f"note {i}") for i in range(8)]
        store, _ = _store_with_notes(notes)
        results = [(n, float(i) / 8, 0) for i, n in enumerate(reversed(notes))]
        packer = TieredContextPacker(budget=5000)
        stats = packer.stats(results)
        # At default fractions (25% FULL, 50% COMPACT, 25% MINIMAL)
        # packed tokens should be less than all-FULL
        all_full_tokens = sum(tier_token_cost(n, Tier.FULL) for n, _, _ in results)
        assert stats.tokens_used < all_full_tokens

    def test_packer_tier_distribution(self):
        notes = [_note(f"note {i}") for i in range(8)]
        results = [(n, float(i), 0) for i, n in enumerate(notes)]
        packer = TieredContextPacker(budget=5000)
        stats = packer.stats(results)
        # With 8 notes: ceil(8*0.25)=2 FULL, ceil(8*0.75)=6 compact cutoff
        assert stats.full_count >= 1
        assert stats.compact_count + stats.minimal_count >= 1


# ── MemoryConsolidator ────────────────────────────────────────────────────────


class TestMemoryConsolidator:
    def _make_cluster(self, store: TEEGStore, n: int, shared_kw: list[str]) -> list[AtomicNote]:
        notes = []
        for i in range(n):
            note = AtomicNote(
                content=f"Cluster fact {i}",
                context="test",
                keywords=shared_kw + [f"unique_{i}"],
                tags=["cluster"],
                confidence=0.8,
            )
            store.add(note)
            notes.append(note)
        return notes

    def test_consolidate_archives_cluster_notes(self, tmp_path):
        store = TEEGStore(artifacts_dir=tmp_path)
        cluster = self._make_cluster(store, 3, ["creature", "victor", "frankenstein"])
        consolidator = MemoryConsolidator(store, model_name="mock", min_cluster_size=3)
        result = consolidator.consolidate()
        assert result.notes_archived == 3
        for note in cluster:
            assert not store.get(note.note_id).active

    def test_consolidate_creates_summary_note(self, tmp_path):
        store = TEEGStore(artifacts_dir=tmp_path)
        self._make_cluster(store, 3, ["creature", "victor", "creation"])
        consolidator = MemoryConsolidator(store, model_name="mock", min_cluster_size=3)
        active_before = store.active_count()
        result = consolidator.consolidate()
        # One new summary note added, 3 archived → net = active_before - 3 + 1
        assert store.active_count() == active_before - 3 + result.summaries_created

    def test_consolidate_returns_correct_stats(self, tmp_path):
        store = TEEGStore(artifacts_dir=tmp_path)
        self._make_cluster(store, 4, ["alpha", "beta", "gamma"])
        consolidator = MemoryConsolidator(store, model_name="mock", min_cluster_size=3)
        result = consolidator.consolidate()
        assert result.clusters_found >= 1
        assert result.notes_archived >= 3
        assert result.summaries_created >= 1
        assert result.token_savings_est > 0

    def test_consolidate_token_savings_formula(self, tmp_path):
        from oml.memory.consolidator import TOKENS_PER_NOTE_FULL
        store = TEEGStore(artifacts_dir=tmp_path)
        self._make_cluster(store, 5, ["alpha", "beta", "gamma"])
        consolidator = MemoryConsolidator(store, model_name="mock", min_cluster_size=3)
        result = consolidator.consolidate()
        expected = max(0, result.notes_archived - result.summaries_created) * TOKENS_PER_NOTE_FULL
        assert result.token_savings_est == expected

    def test_dry_run_does_not_modify_store(self, tmp_path):
        store = TEEGStore(artifacts_dir=tmp_path)
        self._make_cluster(store, 3, ["alpha", "beta", "gamma"])
        active_before = store.active_count()
        consolidator = MemoryConsolidator(store, model_name="mock", min_cluster_size=3)
        result = consolidator.dry_run()
        assert store.active_count() == active_before  # unchanged
        assert result.clusters_found >= 1
        assert result.notes_archived >= 1

    def test_small_cluster_skipped(self, tmp_path):
        store = TEEGStore(artifacts_dir=tmp_path)
        # Only 2 notes — below min_cluster_size=3
        self._make_cluster(store, 2, ["alpha", "beta", "gamma"])
        consolidator = MemoryConsolidator(store, model_name="mock", min_cluster_size=3)
        result = consolidator.consolidate()
        assert result.notes_archived == 0
        assert result.summaries_created == 0

    def test_consolidate_adds_consolidates_edges(self, tmp_path):
        store = TEEGStore(artifacts_dir=tmp_path)
        self._make_cluster(store, 3, ["creature", "victor", "lab"])
        consolidator = MemoryConsolidator(store, model_name="mock", min_cluster_size=3)
        consolidator.consolidate()
        # Find the summary note (only active note after consolidation)
        active = store.get_active()
        assert len(active) == 1
        summary = active[0]
        # Summary should have edges to archived notes
        graph = store.get_graph()
        out_edges = list(graph.out_edges(summary.note_id, data=True))
        assert len(out_edges) == 3
        for _, _, data in out_edges:
            assert data.get("relation") == "consolidates"

    def test_graph_edge_connection_triggers_cluster(self, tmp_path):
        store = TEEGStore(artifacts_dir=tmp_path)
        # 3 notes with NO shared keywords, but connected by edges
        notes = []
        for i in range(3):
            n = AtomicNote(
                content=f"Edge-connected fact {i}",
                keywords=[f"unique_kw_{i}"],
                tags=["test"],
                confidence=0.7,
            )
            store.add(n)
            notes.append(n)
        # Connect them in a chain
        store.add_edge(notes[0].note_id, notes[1].note_id, "supports")
        store.add_edge(notes[1].note_id, notes[2].note_id, "extends")

        consolidator = MemoryConsolidator(store, model_name="mock", min_cluster_size=3)
        result = consolidator.consolidate()
        assert result.clusters_found >= 1
        assert result.notes_archived == 3


# ── Integration: Scout access tracking ───────────────────────────────────────


class TestScoutIntegration:
    def test_scout_records_access_on_search(self, tmp_path):
        from oml.retrieval.scout import ScoutRetriever
        store = TEEGStore(artifacts_dir=tmp_path)
        note = AtomicNote(
            content="Victor Frankenstein created the creature",
            keywords=["victor", "frankenstein", "creature"],
            tags=["creation"],
        )
        store.add(note)
        scout = ScoutRetriever(store, record_access=True)
        results = scout.search("frankenstein creature", top_k=3)
        if results:
            found_note = results[0][0]
            assert found_note.access_count >= 1

    def test_scout_no_access_recording_when_disabled(self, tmp_path):
        from oml.retrieval.scout import ScoutRetriever
        store = TEEGStore(artifacts_dir=tmp_path)
        note = AtomicNote(
            content="Victor Frankenstein created the creature",
            keywords=["victor", "frankenstein", "creature"],
            tags=["creation"],
        )
        store.add(note)
        scout = ScoutRetriever(store, record_access=False)
        scout.search("frankenstein creature", top_k=3)
        # access_count should remain 0
        assert store.get(note.note_id).access_count == 0


# ── Integration: TEEGPipeline tiered context ──────────────────────────────────


class TestPipelineIntegration:
    def test_pipeline_query_returns_teeg_context_tags(self, tmp_path):
        from oml.memory.teeg_pipeline import TEEGPipeline
        pipeline = TEEGPipeline(artifacts_dir=str(tmp_path), model="mock")
        pipeline.ingest("Victor Frankenstein created the creature.")
        pipeline.ingest("The creature was brought to life in November.")
        _, context = pipeline.query("Who created the creature?", return_context=True)
        assert "[TEEG MEMORY]" in context
        assert "[/TEEG MEMORY]" in context

    def test_pipeline_stats_include_efficiency_data(self, tmp_path):
        from oml.memory.teeg_pipeline import TEEGPipeline
        pipeline = TEEGPipeline(artifacts_dir=str(tmp_path), model="mock")
        pipeline.ingest("Test fact.")
        stats = pipeline.stats()
        assert "active_notes" in stats
        assert "total_notes" in stats
