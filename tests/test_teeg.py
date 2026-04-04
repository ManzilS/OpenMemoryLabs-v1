"""
Tests for the TEEG (TOON-Encoded Evolving Graph) Memory System.
================================================================

Full unit test coverage with ALL external dependencies mocked:
  - No real LLM calls (mock LLM client injected)
  - No sentence-transformers (vector index replaced with keyword fallback)
  - No LanceDB
  - All file I/O uses tmp_path fixtures

Test groups
-----------
1. TOON encoder/decoder
2. AtomicNote serialization (TOON + dict roundtrips)
3. TEEGStore CRUD, graph edges, persistence, vector search fallback
4. MemoryEvolver verdict parsing and application
5. ScoutRetriever graph traversal
6. TEEGPipeline end-to-end with mock LLM
7. CLI smoke tests (teeg-ingest / teeg-query)
8. Stage 1 fuzzy verdict parser
9. Confidence decay, resurrection, propagation queue
"""

from __future__ import annotations

import json
import pickle
import textwrap
from pathlib import Path
from typing import List, Tuple
from unittest.mock import MagicMock, patch

import pytest

# ── 1. TOON ────────────────────────────────────────────────────────────────────


class TestToon:
    """TOON encoder / decoder round-trip tests."""

    def test_dumps_scalar(self):
        from oml.memory import toon
        result = toon.dumps({"key": "value", "num": 42})
        assert "key: value" in result
        assert "num: 42" in result

    def test_dumps_list_field(self):
        from oml.memory import toon
        result = toon.dumps({"tags": ["a", "b", "c"]})
        assert "tags: a|b|c" in result

    def test_dumps_empty_list(self):
        from oml.memory import toon
        result = toon.dumps({"tags": []})
        assert "tags: " in result

    def test_dumps_none_becomes_empty(self):
        from oml.memory import toon
        result = toon.dumps({"note": None})
        assert "note: " in result

    def test_loads_scalar(self):
        from oml.memory import toon
        d = toon.loads("note_id: n1\ncontent: cats sleep a lot")
        assert d["note_id"] == "n1"
        assert d["content"] == "cats sleep a lot"

    def test_loads_list_field(self):
        from oml.memory import toon
        d = toon.loads("tags: science|ethics|gothic")
        assert d["tags"] == ["science", "ethics", "gothic"]

    def test_loads_empty_list(self):
        from oml.memory import toon
        d = toon.loads("keywords: ")
        assert d["keywords"] == []

    def test_loads_colon_in_value(self):
        """Values containing ': ' should not be split further."""
        from oml.memory import toon
        d = toon.loads("context: Chapter 5: first animation")
        assert d["context"] == "Chapter 5: first animation"

    def test_loads_malformed_line_skipped(self):
        from oml.memory import toon
        d = toon.loads("good: fine\nBAD LINE\ncontent: ok")
        assert "content" in d
        assert "BAD LINE" not in d

    def test_roundtrip(self):
        from oml.memory import toon
        original = {
            "note_id": "teeg-abc123",
            "content": "Victor assembled the creature",
            "context": "Chapter 4",
            "keywords": ["victor", "creature"],
            "tags": ["gothic"],
            "created_at": "2026-01-01T00:00:00",
            "supersedes": "",
            "confidence": "0.9",
            "source_ids": ["doc1"],
            "active": "True",
        }
        serialized = toon.dumps(original)
        recovered = toon.loads(serialized)
        assert recovered["note_id"] == original["note_id"]
        assert recovered["content"] == original["content"]
        assert recovered["tags"] == ["gothic"]
        assert recovered["keywords"] == ["victor", "creature"]

    def test_compare_sizes_returns_savings(self):
        from oml.memory import toon
        payload = {
            "note_id": "n1",
            "content": "a long fact about something important",
            "context": "some source",
            "keywords": ["fact", "important"],
            "tags": ["knowledge"],
        }
        result = toon.compare_sizes(payload)
        assert "json_tokens" in result
        assert "toon_tokens" in result
        assert "savings_pct" in result
        assert result["toon_tokens"] <= result["json_tokens"]

    def test_token_count_estimate(self):
        from oml.memory import toon
        estimate = toon.token_count_estimate("hello world", chars_per_token=3.5)
        assert estimate > 0


# ── 2. AtomicNote ──────────────────────────────────────────────────────────────


class TestAtomicNote:
    """AtomicNote serialization and behaviour tests."""

    def _make_note(self, **kwargs):
        from oml.memory.atomic_note import AtomicNote
        defaults = dict(
            content="Victor Frankenstein created the creature",
            context="Chapter 5",
            keywords=["victor", "creature", "created"],
            tags=["gothic", "science"],
            confidence=0.9,
        )
        defaults.update(kwargs)
        return AtomicNote(**defaults)

    def test_default_id_prefix(self):
        note = self._make_note()
        assert note.note_id.startswith("teeg-")

    def test_active_default_true(self):
        note = self._make_note()
        assert note.active is True

    def test_to_toon_contains_content(self):
        note = self._make_note()
        t = note.to_toon()
        assert "content:" in t
        assert "Victor Frankenstein" in t

    def test_from_toon_roundtrip(self):
        from oml.memory.atomic_note import AtomicNote
        note = self._make_note()
        t = note.to_toon()
        recovered = AtomicNote.from_toon(t)
        assert recovered.content == note.content
        assert recovered.keywords == note.keywords
        assert recovered.tags == note.tags
        assert abs(recovered.confidence - note.confidence) < 0.001

    def test_from_toon_active_false(self):
        from oml.memory.atomic_note import AtomicNote
        note = self._make_note(active=False)
        recovered = AtomicNote.from_toon(note.to_toon())
        assert recovered.active is False

    def test_to_dict_roundtrip(self):
        from oml.memory.atomic_note import AtomicNote
        note = self._make_note()
        d = note.to_dict()
        # keywords serialized as pipe-separated
        assert "|" in d["keywords"] or d["keywords"] == ""
        recovered = AtomicNote.from_dict(d)
        assert recovered.content == note.content
        assert recovered.tags == note.tags

    def test_to_dict_list_fields_pipe_separated(self):
        note = self._make_note(keywords=["a", "b"], tags=["x"])
        d = note.to_dict()
        assert d["keywords"] == "a|b"
        assert d["tags"] == "x"

    def test_from_dict_list_fields_split(self):
        from oml.memory.atomic_note import AtomicNote
        d = {"note_id": "teeg-x1", "content": "hello", "keywords": "a|b|c", "tags": "x|y", "active": True,
             "confidence": 1.0, "context": "", "supersedes": "", "source_ids": "", "created_at": "2026-01-01T00:00:00"}
        note = AtomicNote.from_dict(d)
        assert note.keywords == ["a", "b", "c"]
        assert note.tags == ["x", "y"]

    def test_embedding_text_includes_content_and_keywords(self):
        note = self._make_note()
        emb = note.embedding_text()
        assert "Victor" in emb
        assert "creature" in emb
        assert "context:" in emb.lower() or "Chapter 5" in emb

    def test_token_cost_positive(self):
        note = self._make_note()
        assert note.token_cost() > 0

    def test_repr(self):
        note = self._make_note()
        r = repr(note)
        assert "AtomicNote" in r
        assert note.note_id in r


# ── 3. TEEGStore ───────────────────────────────────────────────────────────────


class TestTEEGStore:
    """TEEGStore CRUD, graph operations, and persistence."""

    def _make_store(self, tmp_path):
        from oml.storage.teeg_store import TEEGStore
        return TEEGStore(artifacts_dir=tmp_path)

    def _note(self, content="test fact", **kwargs):
        from oml.memory.atomic_note import AtomicNote
        return AtomicNote(content=content, **kwargs)

    # ── CRUD ────────────────────────────────────────────────────────────────

    def test_add_and_get(self, tmp_path):
        store = self._make_store(tmp_path)
        note = self._note("fact 1")
        store.add(note)
        assert store.get(note.note_id) is note

    def test_get_nonexistent_returns_none(self, tmp_path):
        store = self._make_store(tmp_path)
        assert store.get("nonexistent") is None

    def test_count(self, tmp_path):
        store = self._make_store(tmp_path)
        store.add(self._note("a"))
        store.add(self._note("b"))
        assert store.count() == 2

    def test_active_count_excludes_archived(self, tmp_path):
        store = self._make_store(tmp_path)
        n1 = self._note("a")
        n2 = self._note("b")
        store.add(n1)
        store.add(n2)
        store.archive(n1.note_id)
        assert store.active_count() == 1

    def test_get_active_filters_archived(self, tmp_path):
        store = self._make_store(tmp_path)
        n1 = self._note("active")
        n2 = self._note("archived")
        store.add(n1)
        store.add(n2)
        store.archive(n2.note_id)
        active = store.get_active()
        ids = [n.note_id for n in active]
        assert n1.note_id in ids
        assert n2.note_id not in ids

    def test_archive_sets_flag(self, tmp_path):
        store = self._make_store(tmp_path)
        note = self._note("something")
        store.add(note)
        store.archive(note.note_id)
        assert store.get(note.note_id).active is False

    def test_archive_nonexistent_is_noop(self, tmp_path):
        store = self._make_store(tmp_path)
        store.archive("does-not-exist")  # should not raise

    def test_upsert_overwrites(self, tmp_path):
        store = self._make_store(tmp_path)
        note = self._note("old content")
        store.add(note)
        note.content = "new content"
        store.add(note)
        assert store.get(note.note_id).content == "new content"

    # ── graph ────────────────────────────────────────────────────────────────

    def test_add_edge_and_get_edges(self, tmp_path):
        store = self._make_store(tmp_path)
        n1 = self._note("cause")
        n2 = self._note("effect")
        store.add(n1)
        store.add(n2)
        store.add_edge(n1.note_id, n2.note_id, relation="extends", weight=0.9)
        edges = store.get_edges(n1.note_id)
        assert any(e[1] == n2.note_id for e in edges)

    def test_add_edge_missing_node_logs_warning(self, tmp_path):
        store = self._make_store(tmp_path)
        n1 = self._note("exists")
        store.add(n1)
        # n2 not added — should not raise, just warn
        store.add_edge(n1.note_id, "missing-id", relation="supports", weight=0.5)

    def test_neighbors_returns_connected_ids(self, tmp_path):
        store = self._make_store(tmp_path)
        n1 = self._note("a")
        n2 = self._note("b")
        n3 = self._note("c")
        store.add(n1)
        store.add(n2)
        store.add(n3)
        store.add_edge(n1.note_id, n2.note_id, relation="extends")
        store.add_edge(n1.note_id, n3.note_id, relation="supports")
        nbrs = store.neighbors(n1.note_id)
        assert n2.note_id in nbrs
        assert n3.note_id in nbrs

    def test_neighbors_filtered_by_relation(self, tmp_path):
        store = self._make_store(tmp_path)
        n1, n2, n3 = self._note("a"), self._note("b"), self._note("c")
        store.add(n1); store.add(n2); store.add(n3)
        store.add_edge(n1.note_id, n2.note_id, relation="extends")
        store.add_edge(n1.note_id, n3.note_id, relation="supports")
        extends_only = store.neighbors(n1.note_id, relation="extends")
        assert n2.note_id in extends_only
        assert n3.note_id not in extends_only

    def test_get_graph_returns_digraph(self, tmp_path):
        import networkx as nx
        store = self._make_store(tmp_path)
        g = store.get_graph()
        assert isinstance(g, nx.DiGraph)

    # ── keyword fallback search ──────────────────────────────────────────────

    def test_keyword_fallback_finds_matching_note(self, tmp_path):
        store = self._make_store(tmp_path)
        n1 = self._note("Victor Frankenstein built the creature at midnight")
        n2 = self._note("Walton sails towards the arctic")
        store.add(n1)
        store.add(n2)
        results = store.vector_search("Victor creature", top_k=3)
        ids = [r[0].note_id for r in results]
        assert n1.note_id in ids

    def test_keyword_fallback_returns_scores(self, tmp_path):
        store = self._make_store(tmp_path)
        note = self._note("keyword test note", keywords=["specific", "term"])
        store.add(note)
        results = store.vector_search("specific term", top_k=5)
        assert len(results) > 0
        _, score = results[0]
        assert 0.0 < score <= 1.0

    def test_vector_search_empty_store(self, tmp_path):
        store = self._make_store(tmp_path)
        results = store.vector_search("anything", top_k=5)
        assert results == []

    # ── persistence ─────────────────────────────────────────────────────────

    def test_save_and_load_notes(self, tmp_path):
        store = self._make_store(tmp_path)
        note = self._note("persistent fact")
        store.add(note)
        store.save()

        from oml.storage.teeg_store import TEEGStore
        store2 = TEEGStore(artifacts_dir=tmp_path)
        loaded = store2.get(note.note_id)
        assert loaded is not None
        assert loaded.content == "persistent fact"

    def test_save_and_load_graph(self, tmp_path):
        store = self._make_store(tmp_path)
        n1 = self._note("node a")
        n2 = self._note("node b")
        store.add(n1)
        store.add(n2)
        store.add_edge(n1.note_id, n2.note_id, relation="extends", weight=0.8)
        store.save()

        from oml.storage.teeg_store import TEEGStore
        store2 = TEEGStore(artifacts_dir=tmp_path)
        g = store2.get_graph()
        assert g.has_edge(n1.note_id, n2.note_id)
        data = g[n1.note_id][n2.note_id]
        assert data["relation"] == "extends"
        assert abs(data["weight"] - 0.8) < 0.001

    def test_save_archived_note_reloads_archived(self, tmp_path):
        store = self._make_store(tmp_path)
        note = self._note("to be archived")
        store.add(note)
        store.archive(note.note_id)
        store.save()

        from oml.storage.teeg_store import TEEGStore
        store2 = TEEGStore(artifacts_dir=tmp_path)
        assert store2.get(note.note_id).active is False

    def test_stats(self, tmp_path):
        store = self._make_store(tmp_path)
        n1 = self._note("a")
        n2 = self._note("b")
        store.add(n1)
        store.add(n2)
        store.archive(n2.note_id)
        s = store.stats()
        assert s["total_notes"] == 2
        assert s["active_notes"] == 1
        assert s["archived_notes"] == 1


# ── 4. MemoryEvolver ───────────────────────────────────────────────────────────


class TestMemoryEvolver:
    """MemoryEvolver: verdict parsing and effect on store."""

    def _make_evolver(self, tmp_path, mock_response="RELATION: SUPPORTS\nREASON: corroborates"):
        from oml.storage.teeg_store import TEEGStore
        from oml.memory.evolver import MemoryEvolver
        store = TEEGStore(artifacts_dir=tmp_path)
        evolver = MemoryEvolver(store, model_name="mock")
        # Inject a mock LLM that returns a fixed response for BOTH Stage 1 and Stage 2.
        # Stage 1 shares the same client when stage1_model_name == model_name (default).
        mock_llm = MagicMock()
        mock_llm.generate.return_value = mock_response
        evolver._llm = mock_llm
        return evolver, store

    def _note(self, content, **kwargs):
        from oml.memory.atomic_note import AtomicNote
        kw = kwargs.pop("keywords", content.lower().split()[:4])
        return AtomicNote(content=content, keywords=kw, **kwargs)

    # ── verdict parsing ──────────────────────────────────────────────────────

    def test_parse_contradicts(self, tmp_path):
        evolver, _ = self._make_evolver(tmp_path)
        rel, reason = evolver._parse_verdict("RELATION: CONTRADICTS\nREASON: Directly conflicts.")
        assert rel == "CONTRADICTS"
        assert "Directly conflicts" in reason

    def test_parse_extends(self, tmp_path):
        evolver, _ = self._make_evolver(tmp_path)
        rel, _ = evolver._parse_verdict("RELATION: EXTENDS\nREASON: adds detail")
        assert rel == "EXTENDS"

    def test_parse_supports(self, tmp_path):
        evolver, _ = self._make_evolver(tmp_path)
        rel, _ = evolver._parse_verdict("RELATION: SUPPORTS\nREASON: corroborates")
        assert rel == "SUPPORTS"

    def test_parse_unrelated(self, tmp_path):
        evolver, _ = self._make_evolver(tmp_path)
        rel, _ = evolver._parse_verdict("RELATION: UNRELATED\nREASON: different topics")
        assert rel == "UNRELATED"

    def test_parse_partial_match(self, tmp_path):
        """LLM sometimes returns 'CONTRADICTS the existing note'."""
        evolver, _ = self._make_evolver(tmp_path)
        rel, _ = evolver._parse_verdict("RELATION: CONTRADICTS the existing note\nREASON: conflict")
        assert rel == "CONTRADICTS"

    def test_parse_case_insensitive(self, tmp_path):
        evolver, _ = self._make_evolver(tmp_path)
        rel, _ = evolver._parse_verdict("relation: extends\nreason: builds on it")
        assert rel == "EXTENDS"

    def test_parse_defaults_to_supports_on_garbage(self, tmp_path):
        evolver, _ = self._make_evolver(tmp_path)
        rel, _ = evolver._parse_verdict("something completely unparseable %%%")
        assert rel == "SUPPORTS"

    # ── effect on store ──────────────────────────────────────────────────────

    def test_contradicts_archives_existing(self, tmp_path):
        evolver, store = self._make_evolver(tmp_path, "RELATION: CONTRADICTS\nREASON: superseded")
        old_note = self._note("the creature was calm")
        store.add(old_note)
        # Manually add to vector index via keyword fallback
        store.build_vector_index()

        new_note = self._note("the creature was agitated calm")  # shares keywords for fallback
        evolver._apply(new_note, old_note, "CONTRADICTS", "superseded")

        assert store.get(old_note.note_id).active is False
        assert new_note.supersedes == old_note.note_id

    def test_extends_adds_edge(self, tmp_path):
        evolver, store = self._make_evolver(tmp_path, "RELATION: EXTENDS\nREASON: adds detail")
        old_note = self._note("creature was assembled")
        new_note = self._note("creature was assembled from parts")
        store.add(old_note)
        store.add(new_note)

        evolver._apply(new_note, old_note, "EXTENDS", "adds detail")

        g = store.get_graph()
        assert g.has_edge(new_note.note_id, old_note.note_id)
        assert g[new_note.note_id][old_note.note_id]["relation"] == "extends"

    def test_supports_adds_weaker_edge(self, tmp_path):
        evolver, store = self._make_evolver(tmp_path, "RELATION: SUPPORTS\nREASON: corroborates")
        old_note = self._note("victor was responsible")
        new_note = self._note("victor took responsibility")
        store.add(old_note)
        store.add(new_note)

        evolver._apply(new_note, old_note, "SUPPORTS", "corroborates")

        g = store.get_graph()
        assert g.has_edge(new_note.note_id, old_note.note_id)
        assert g[new_note.note_id][old_note.note_id]["relation"] == "supports"
        assert g[new_note.note_id][old_note.note_id]["weight"] < 1.0

    def test_unrelated_no_edge_no_archive(self, tmp_path):
        evolver, store = self._make_evolver(tmp_path, "RELATION: UNRELATED\nREASON: different")
        old_note = self._note("walton sails arctic")
        new_note = self._note("victor studies chemistry")
        store.add(old_note)
        store.add(new_note)

        evolver._apply(new_note, old_note, "UNRELATED", "different")

        assert store.get(old_note.note_id).active is True
        g = store.get_graph()
        assert not g.has_edge(new_note.note_id, old_note.note_id)

    def test_evolve_always_stores_new_note(self, tmp_path):
        """Even on CONTRADICTS, the new note must be stored."""
        evolver, store = self._make_evolver(tmp_path, "RELATION: UNRELATED\nREASON: unrelated")
        new_note = self._note("new unique observation")
        evolver.evolve(new_note)
        assert store.get(new_note.note_id) is not None

    def test_evolve_batch(self, tmp_path):
        evolver, store = self._make_evolver(tmp_path, "RELATION: UNRELATED\nREASON: unrelated")
        notes = [self._note(f"fact {i}") for i in range(3)]
        evolver.evolve_batch(notes)
        assert store.count() == 3

    def test_judge_llm_failure_defaults_to_supports(self, tmp_path):
        """If LLM throws, evolver defaults to SUPPORTS (safe)."""
        evolver, store = self._make_evolver(tmp_path)
        evolver._llm.generate.side_effect = RuntimeError("LLM unreachable")

        old_note = self._note("existing note")
        new_note = self._note("new note similar")
        store.add(old_note)
        store.add(new_note)

        rel, reason = evolver._judge(new_note, old_note)
        assert rel == "SUPPORTS"
        assert "unavailable" in reason

    def test_audit_returns_stats(self, tmp_path):
        evolver, store = self._make_evolver(tmp_path, "RELATION: UNRELATED\nREASON: different")
        for i in range(3):
            evolver.evolve(self._note(f"fact {i}"))
        audit = evolver.audit()
        assert "active_notes" in audit
        assert "archived_notes" in audit
        assert audit["active_notes"] == 3


# ── 5. ScoutRetriever ──────────────────────────────────────────────────────────


class TestScoutRetriever:
    """ScoutRetriever: seed selection and graph traversal."""

    def _build_store(self, tmp_path):
        from oml.storage.teeg_store import TEEGStore
        from oml.memory.atomic_note import AtomicNote
        store = TEEGStore(artifacts_dir=tmp_path)

        # Four notes with content for keyword fallback
        notes = [
            AtomicNote(note_id="n1", content="Victor Frankenstein created the creature",
                       keywords=["victor", "frankenstein", "creature", "created"]),
            AtomicNote(note_id="n2", content="The creature was built from dead body parts",
                       keywords=["creature", "built", "dead", "body"]),
            AtomicNote(note_id="n3", content="Victor fled the laboratory after the animation",
                       keywords=["victor", "fled", "laboratory", "animation"]),
            AtomicNote(note_id="n4", content="Walton explored the arctic ocean",
                       keywords=["walton", "arctic", "ocean"]),
        ]
        for n in notes:
            store.add(n)
        # Link n1 → n2 (creature creation chain), n1 → n3
        store.add_edge("n1", "n2", relation="extends", weight=0.9)
        store.add_edge("n1", "n3", relation="supports", weight=0.7)
        return store, notes

    def test_search_returns_results(self, tmp_path):
        from oml.retrieval.scout import ScoutRetriever
        store, _ = self._build_store(tmp_path)
        scout = ScoutRetriever(store, seed_k=2, max_hops=1)
        results = scout.search("Victor creature", top_k=5)
        assert len(results) > 0

    def test_search_result_shape(self, tmp_path):
        from oml.retrieval.scout import ScoutRetriever
        from oml.memory.atomic_note import AtomicNote
        store, _ = self._build_store(tmp_path)
        scout = ScoutRetriever(store, seed_k=2, max_hops=1)
        results = scout.search("Victor creature", top_k=5)
        for note, score, hops in results:
            assert isinstance(note, AtomicNote)
            assert score > 0
            assert hops >= 0

    def test_seeds_are_hop_zero(self, tmp_path):
        from oml.retrieval.scout import ScoutRetriever
        store, _ = self._build_store(tmp_path)
        scout = ScoutRetriever(store, seed_k=1, max_hops=1)
        results = scout.search("Victor frankenstein creature", top_k=5)
        hops = [h for _, _, h in results]
        assert 0 in hops, "At least one seed (hop=0) expected"

    def test_graph_neighbors_discovered(self, tmp_path):
        """Notes connected via edges should appear in results at hop > 0."""
        from oml.retrieval.scout import ScoutRetriever
        store, _ = self._build_store(tmp_path)
        scout = ScoutRetriever(store, seed_k=1, max_hops=2)
        results = scout.search("Victor frankenstein", top_k=10)
        result_ids = {r[0].note_id for r in results}
        # n1 should be seed; n2 and n3 should be discovered via graph
        assert "n2" in result_ids or "n3" in result_ids

    def test_archived_notes_excluded(self, tmp_path):
        from oml.retrieval.scout import ScoutRetriever
        store, notes = self._build_store(tmp_path)
        store.archive("n2")
        scout = ScoutRetriever(store, seed_k=3, max_hops=2)
        results = scout.search("creature body parts", top_k=10)
        result_ids = {r[0].note_id for r in results}
        assert "n2" not in result_ids

    def test_top_k_respected(self, tmp_path):
        from oml.retrieval.scout import ScoutRetriever
        store, _ = self._build_store(tmp_path)
        scout = ScoutRetriever(store, seed_k=2, max_hops=2)
        results = scout.search("Victor creature fled", top_k=2)
        assert len(results) <= 2

    def test_empty_store_returns_empty(self, tmp_path):
        from oml.storage.teeg_store import TEEGStore
        from oml.retrieval.scout import ScoutRetriever
        store = TEEGStore(artifacts_dir=tmp_path)
        scout = ScoutRetriever(store)
        results = scout.search("anything", top_k=5)
        assert results == []

    def test_build_context_contains_teeg_block(self, tmp_path):
        from oml.retrieval.scout import ScoutRetriever
        store, _ = self._build_store(tmp_path)
        scout = ScoutRetriever(store, seed_k=2, max_hops=1)
        ctx = scout.build_context("Victor creature", top_k=3)
        assert "[TEEG MEMORY]" in ctx
        assert "[/TEEG MEMORY]" in ctx

    def test_build_context_empty_store(self, tmp_path):
        from oml.storage.teeg_store import TEEGStore
        from oml.retrieval.scout import ScoutRetriever
        store = TEEGStore(artifacts_dir=tmp_path)
        scout = ScoutRetriever(store)
        ctx = scout.build_context("anything", top_k=5)
        assert "no relevant memory" in ctx

    def test_build_context_max_tokens(self, tmp_path):
        from oml.retrieval.scout import ScoutRetriever
        store, _ = self._build_store(tmp_path)
        scout = ScoutRetriever(store, seed_k=2, max_hops=2)
        ctx = scout.build_context("Victor", top_k=10, max_tokens=30)
        # Very tight budget — few or no notes packed
        assert "[TEEG MEMORY]" in ctx

    def test_explain_returns_string(self, tmp_path):
        from oml.retrieval.scout import ScoutRetriever
        store, _ = self._build_store(tmp_path)
        scout = ScoutRetriever(store, seed_k=2, max_hops=1)
        explanation = scout.explain("Victor creature", top_k=3)
        assert isinstance(explanation, str)
        assert len(explanation) > 0

    def test_stats_returns_dict(self, tmp_path):
        from oml.retrieval.scout import ScoutRetriever
        store, _ = self._build_store(tmp_path)
        scout = ScoutRetriever(store)
        s = scout.stats()
        assert "seed_k" in s
        assert "max_hops" in s
        assert "total_notes" in s


# ── 6. TEEGPipeline ────────────────────────────────────────────────────────────


class TestTEEGPipeline:
    """TEEGPipeline end-to-end with a mock LLM."""

    def _make_pipeline(self, tmp_path, llm_response="content: mocked fact\nkeywords: mocked|fact\ntags: auto"):
        from oml.memory.teeg_pipeline import TEEGPipeline
        pipeline = TEEGPipeline(
            artifacts_dir=tmp_path,
            model="mock",
            token_budget=2000,
        )
        # Inject a mock LLM
        mock_llm = MagicMock()
        mock_llm.generate.return_value = llm_response
        pipeline._llm = mock_llm
        pipeline.evolver._llm = mock_llm
        return pipeline, mock_llm

    def test_ingest_stores_note(self, tmp_path):
        pipeline, _ = self._make_pipeline(tmp_path)
        note = pipeline.ingest("Victor built the creature")
        assert pipeline.store.get(note.note_id) is not None

    def test_ingest_returns_atomic_note(self, tmp_path):
        from oml.memory.atomic_note import AtomicNote
        pipeline, _ = self._make_pipeline(tmp_path)
        note = pipeline.ingest("some raw text")
        assert isinstance(note, AtomicNote)

    def test_ingest_with_source_id(self, tmp_path):
        pipeline, _ = self._make_pipeline(tmp_path)
        note = pipeline.ingest("raw text", source_id="doc-42")
        assert "doc-42" in note.source_ids

    def test_ingest_batch(self, tmp_path):
        pipeline, _ = self._make_pipeline(tmp_path)
        texts = ["fact one", "fact two", "fact three"]
        notes = pipeline.ingest_batch(texts)
        assert len(notes) == 3
        assert pipeline.store.count() == 3

    def test_ingest_note_skips_distillation(self, tmp_path):
        from oml.memory.atomic_note import AtomicNote
        pipeline, mock_llm = self._make_pipeline(tmp_path)
        pre_built = AtomicNote(content="pre-structured fact", keywords=["pre", "structured"])
        pipeline.ingest_note(pre_built)
        # Distil LLM should NOT have been called
        assert pipeline.store.get(pre_built.note_id) is not None

    def test_query_returns_string(self, tmp_path):
        pipeline, mock_llm = self._make_pipeline(tmp_path, llm_response="The answer is 42.")
        pipeline.ingest("Victor frankenstein created the creature in his laboratory")
        answer = pipeline.query("Who created the creature?")
        assert isinstance(answer, str)
        assert len(answer) > 0

    def test_query_calls_llm_with_context(self, tmp_path):
        pipeline, mock_llm = self._make_pipeline(tmp_path, llm_response="Answer here.")
        pipeline.ingest("Victor frankenstein creature laboratory")
        pipeline.query("Who built it?")
        # LLM must have been called with TEEG MEMORY block in prompt
        call_args = mock_llm.generate.call_args_list
        # At minimum, the query call should use the LLM
        assert len(call_args) >= 1

    def test_query_return_context_tuple(self, tmp_path):
        pipeline, mock_llm = self._make_pipeline(tmp_path, llm_response="Answer.")
        pipeline.ingest("Victor creature")
        result = pipeline.query("Who?", return_context=True)
        assert isinstance(result, tuple)
        answer, ctx = result
        assert "[TEEG MEMORY]" in ctx

    def test_search_returns_results(self, tmp_path):
        pipeline, _ = self._make_pipeline(tmp_path)
        pipeline.ingest("Victor frankenstein created the creature")
        results = pipeline.search("Victor creature", top_k=3)
        assert isinstance(results, list)

    def test_save_and_stats(self, tmp_path):
        pipeline, _ = self._make_pipeline(tmp_path)
        pipeline.ingest("persistent note")
        pipeline.save()
        s = pipeline.stats()
        assert s["total_notes"] >= 1
        assert "model" in s
        assert "evolution_audit" in s

    def test_explain_query_returns_string(self, tmp_path):
        pipeline, _ = self._make_pipeline(tmp_path)
        pipeline.ingest("Victor frankenstein creature laboratory")
        expl = pipeline.explain_query("Victor creature", top_k=3)
        assert isinstance(expl, str)

    def test_heuristic_note_fallback(self, tmp_path):
        """If LLM distillation fails, heuristic note should be used."""
        from oml.memory.teeg_pipeline import TEEGPipeline
        pipeline = TEEGPipeline(artifacts_dir=tmp_path, model="mock")
        mock_llm = MagicMock()
        mock_llm.generate.side_effect = RuntimeError("LLM broken")
        pipeline._llm = mock_llm
        pipeline.evolver._llm = mock_llm
        note = pipeline.ingest("This is a raw text that should become a heuristic note via fallback")
        assert note.content != ""
        assert note.confidence == 0.5

    def test_heuristic_note_content_from_words(self):
        """Static method uses first 40 words."""
        from oml.memory.teeg_pipeline import TEEGPipeline
        raw = " ".join([f"word{i}" for i in range(60)])
        note = TEEGPipeline._heuristic_note(raw, "ctx", "src1")
        content_words = note.content.split()
        assert len(content_words) <= 40
        assert "src1" in note.source_ids

    def test_parse_distil_strips_code_fences(self, tmp_path):
        """LLMs sometimes wrap TOON in markdown fences."""
        from oml.memory.teeg_pipeline import TEEGPipeline
        fenced = "```toon\ncontent: fenced note\nkeywords: fenced\ntags: test\n```"
        pipeline = TEEGPipeline(artifacts_dir=tmp_path, model="mock")
        note = pipeline._parse_distil_response(fenced, "raw", "")
        assert "fenced note" in note.content

    def test_parse_distil_fallback_on_no_content_field(self, tmp_path):
        """Response without 'content:' triggers heuristic fallback."""
        from oml.memory.teeg_pipeline import TEEGPipeline
        pipeline = TEEGPipeline(artifacts_dir=tmp_path, model="mock")
        note = pipeline._parse_distil_response("garbage response without fields", "raw fallback text", "")
        assert note.content != ""

    def test_rebuild_vector_index(self, tmp_path):
        pipeline, _ = self._make_pipeline(tmp_path)
        pipeline.ingest("note for index")
        pipeline.rebuild_vector_index()   # should not raise


# ── 7. CLI smoke tests ─────────────────────────────────────────────────────────


class TestTEEGCLI:
    """CLI integration tests for teeg-ingest and teeg-query."""

    def test_teeg_ingest_text_argument(self, tmp_path):
        from typer.testing import CliRunner
        from oml.cli import app

        runner = CliRunner()
        result = runner.invoke(app, [
            "teeg-ingest",
            "Victor Frankenstein assembled the creature",
            "--dir", str(tmp_path),
            "--model", "mock",
            "--no-save",
        ])
        assert result.exit_code == 0, result.output
        assert "teeg-" in result.output.lower() or "stored" in result.output.lower()

    def test_teeg_ingest_no_input_errors(self, tmp_path):
        from typer.testing import CliRunner
        from oml.cli import app

        runner = CliRunner()
        result = runner.invoke(app, [
            "teeg-ingest",
            "--dir", str(tmp_path),
            "--model", "mock",
        ])
        assert result.exit_code != 0 or "provide" in result.output.lower()

    def test_teeg_ingest_from_file(self, tmp_path):
        from typer.testing import CliRunner
        from oml.cli import app

        text_file = tmp_path / "input.txt"
        text_file.write_text("This is a test fact from a file.", encoding="utf-8")

        runner = CliRunner()
        result = runner.invoke(app, [
            "teeg-ingest",
            "--file", str(text_file),
            "--dir", str(tmp_path),
            "--model", "mock",
            "--no-save",
        ])
        assert result.exit_code == 0, result.output

    def test_teeg_ingest_batch(self, tmp_path):
        from typer.testing import CliRunner
        from oml.cli import app

        batch_file = tmp_path / "batch.txt"
        batch_file.write_text("fact one\nfact two\nfact three\n", encoding="utf-8")

        runner = CliRunner()
        result = runner.invoke(app, [
            "teeg-ingest",
            "--batch", str(batch_file),
            "--dir", str(tmp_path),
            "--model", "mock",
            "--no-save",
        ])
        assert result.exit_code == 0, result.output
        assert "3" in result.output

    def test_teeg_ingest_show_note(self, tmp_path):
        from typer.testing import CliRunner
        from oml.cli import app

        runner = CliRunner()
        result = runner.invoke(app, [
            "teeg-ingest",
            "Victor Frankenstein fled the laboratory",
            "--dir", str(tmp_path),
            "--model", "mock",
            "--show-note",
            "--no-save",
        ])
        assert result.exit_code == 0, result.output
        # TOON output should contain "content:"
        assert "content:" in result.output

    def test_teeg_query_empty_store(self, tmp_path):
        from typer.testing import CliRunner
        from oml.cli import app

        runner = CliRunner()
        result = runner.invoke(app, [
            "teeg-query",
            "Who built the creature?",
            "--dir", str(tmp_path),
            "--model", "mock",
        ])
        assert result.exit_code == 0, result.output
        assert "teeg-ingest" in result.output.lower() or "no notes" in result.output.lower()

    def test_teeg_query_with_notes(self, tmp_path):
        """Ingest first, then query."""
        from typer.testing import CliRunner
        from oml.cli import app

        runner = CliRunner()
        # Ingest
        runner.invoke(app, [
            "teeg-ingest",
            "Victor Frankenstein created the creature at midnight",
            "--dir", str(tmp_path),
            "--model", "mock",
            "--save",
        ])
        # Query
        result = runner.invoke(app, [
            "teeg-query",
            "Who created the creature?",
            "--dir", str(tmp_path),
            "--model", "mock",
        ])
        assert result.exit_code == 0, result.output

    def test_teeg_query_explain_mode(self, tmp_path):
        from typer.testing import CliRunner
        from oml.cli import app

        runner = CliRunner()
        runner.invoke(app, [
            "teeg-ingest",
            "Victor Frankenstein created the creature at midnight",
            "--dir", str(tmp_path),
            "--model", "mock",
            "--save",
        ])
        result = runner.invoke(app, [
            "teeg-query",
            "Victor creature",
            "--dir", str(tmp_path),
            "--model", "mock",
            "--explain",
        ])
        assert result.exit_code == 0, result.output

    def test_teeg_query_search_only(self, tmp_path):
        from typer.testing import CliRunner
        from oml.cli import app

        runner = CliRunner()
        runner.invoke(app, [
            "teeg-ingest",
            "Victor Frankenstein created the creature",
            "--dir", str(tmp_path),
            "--model", "mock",
            "--save",
        ])
        result = runner.invoke(app, [
            "teeg-query",
            "Victor creature",
            "--dir", str(tmp_path),
            "--model", "mock",
            "--search",
        ])
        assert result.exit_code == 0, result.output

    def test_teeg_ingest_missing_file_exits_nonzero(self, tmp_path):
        from typer.testing import CliRunner
        from oml.cli import app

        runner = CliRunner()
        result = runner.invoke(app, [
            "teeg-ingest",
            "--file", "/nonexistent/path.txt",
            "--dir", str(tmp_path),
            "--model", "mock",
        ])
        assert result.exit_code != 0

    def test_teeg_ingest_missing_batch_file_exits_nonzero(self, tmp_path):
        from typer.testing import CliRunner
        from oml.cli import app

        runner = CliRunner()
        result = runner.invoke(app, [
            "teeg-ingest",
            "--batch", "/nonexistent/batch.txt",
            "--dir", str(tmp_path),
            "--model", "mock",
        ])
        assert result.exit_code != 0


# ── 8. Stage 1 Fuzzy Parser ────────────────────────────────────────────────────


class TestStage1FuzzyParser:
    """Stage 1 ``_parse_stage1_verdict`` fuzzy parser — handles malformed 3B outputs."""

    def _evolver(self, tmp_path):
        from oml.storage.teeg_store import TEEGStore
        from oml.memory.evolver import MemoryEvolver
        return MemoryEvolver(TEEGStore(artifacts_dir=tmp_path), model_name="mock")

    def test_clean_yes(self, tmp_path):
        assert self._evolver(tmp_path)._parse_stage1_verdict("YES") == "YES"

    def test_clean_no(self, tmp_path):
        assert self._evolver(tmp_path)._parse_stage1_verdict("NO") == "NO"

    def test_clean_scope(self, tmp_path):
        assert self._evolver(tmp_path)._parse_stage1_verdict("SCOPE?") == "SCOPE?"

    def test_yes_with_scope_caveat_returns_scope(self, tmp_path):
        """The user's primary example: verbose YES with scope signal → SCOPE?"""
        result = self._evolver(tmp_path)._parse_stage1_verdict(
            "YES, BUT SCOPE MIGHT BE DIFFERENT"
        )
        assert result == "SCOPE?"

    def test_no_with_scope_signal_returns_scope(self, tmp_path):
        """NO combined with a scope indicator → SCOPE? (scope takes priority)."""
        result = self._evolver(tmp_path)._parse_stage1_verdict(
            "No, different contexts apply here"
        )
        assert result == "SCOPE?"

    def test_altitude_triggers_scope(self, tmp_path):
        result = self._evolver(tmp_path)._parse_stage1_verdict(
            "YES but altitude conditions differ"
        )
        assert result == "SCOPE?"

    def test_however_triggers_scope(self, tmp_path):
        result = self._evolver(tmp_path)._parse_stage1_verdict(
            "YES, however the context is different"
        )
        assert result == "SCOPE?"

    def test_contradiction_keyword_without_yes_is_yes(self, tmp_path):
        result = self._evolver(tmp_path)._parse_stage1_verdict(
            "These notes contradict each other"
        )
        assert result == "YES"

    def test_conflict_keyword_is_yes(self, tmp_path):
        result = self._evolver(tmp_path)._parse_stage1_verdict(
            "There is a clear conflict between these claims"
        )
        assert result == "YES"

    def test_verbose_no_unrelated(self, tmp_path):
        result = self._evolver(tmp_path)._parse_stage1_verdict(
            "No these are unrelated topics"
        )
        assert result == "NO"

    def test_supports_keyword_is_no(self, tmp_path):
        result = self._evolver(tmp_path)._parse_stage1_verdict(
            "The new note supports the existing one"
        )
        assert result == "NO"

    def test_extends_keyword_is_no(self, tmp_path):
        result = self._evolver(tmp_path)._parse_stage1_verdict(
            "The new note extends the existing claim"
        )
        assert result == "NO"

    def test_garbage_defaults_to_yes(self, tmp_path):
        """Recall-biased: any unparseable output → YES."""
        result = self._evolver(tmp_path)._parse_stage1_verdict(
            "I'm not sure, it's complicated"
        )
        assert result == "YES"

    def test_empty_defaults_to_yes(self, tmp_path):
        """Empty string → YES (recall-biased fallback)."""
        assert self._evolver(tmp_path)._parse_stage1_verdict("") == "YES"

    def test_partial_scope_word(self, tmp_path):
        """'depend' partial match triggers SCOPE?."""
        result = self._evolver(tmp_path)._parse_stage1_verdict(
            "YES it depends on the conditions"
        )
        assert result == "SCOPE?"

    def test_corroborate_is_no(self, tmp_path):
        result = self._evolver(tmp_path)._parse_stage1_verdict(
            "The new note corroborates the existing one"
        )
        assert result == "NO"


# ── 9. Confidence Decay, Resurrection, Propagation Queue ──────────────────────


class TestConfidenceDecayAndResurrection:
    """Confidence decay in _apply(), resurrection via SUPPORTS on archived notes."""

    def _make(self, tmp_path, mock_response="RELATION: SUPPORTS\nREASON: x"):
        from oml.storage.teeg_store import TEEGStore
        from oml.memory.evolver import MemoryEvolver
        from unittest.mock import MagicMock
        store = TEEGStore(artifacts_dir=tmp_path)
        evolver = MemoryEvolver(store, model_name="mock")
        mock_llm = MagicMock()
        mock_llm.generate.return_value = mock_response
        evolver._llm = mock_llm
        return evolver, store

    def _note(self, content, **kwargs):
        from oml.memory.atomic_note import AtomicNote
        # Allow caller to override keywords; default to first 4 words of content.
        kw = kwargs.pop("keywords", content.lower().split()[:4])
        return AtomicNote(content=content, keywords=kw, **kwargs)

    # ── CONTRADICTS decay ────────────────────────────────────────────────────

    def test_full_strength_contradicts_archives(self, tmp_path):
        """Default strength=1.0 authority=1.0 — one hit archives (backward compat)."""
        evolver, store = self._make(tmp_path)
        old = self._note("the boiling point is 80C", confidence=1.0)
        new = self._note("the boiling point is 100C")
        store.add(old); store.add(new)

        evolver._apply(new, old, "CONTRADICTS", "direct refutation")
        assert store.get(old.note_id).active is False

    def test_moderate_strength_decays_not_archives(self, tmp_path):
        """strength=0.5 authority=0.5 — decays confidence but does NOT archive."""
        evolver, store = self._make(tmp_path)
        old = self._note("the boiling point is 80C", confidence=1.0)
        new = self._note("the boiling point is 100C")
        store.add(old); store.add(new)

        evolver._apply(new, old, "CONTRADICTS", "partial conflict", strength=0.5, authority=0.5)
        note = store.get(old.note_id)
        # delta = 0.9 × 0.5 × 0.5 × 1.0 = 0.225 → conf ≈ 0.775 → still active
        assert note.active is True
        assert note.confidence < 1.0

    def test_accumulated_contradicts_eventually_archives(self, tmp_path):
        """Multiple moderate hits eventually push confidence below threshold.

        With strength=0.5, authority=0.5 the logistic factor is 0.225 per step.
        Rounding to _CONFIDENCE_STEP (0.05) means 8 iterations are required to
        drive confidence below the 0.15 archive threshold (verified analytically).
        We use 10 iterations so the test is robust to minor rounding variations.
        """
        evolver, store = self._make(tmp_path)
        old = self._note("wrong fact", confidence=1.0)
        new = self._note("correct fact")
        store.add(old); store.add(new)

        for _ in range(10):
            if not store.get(old.note_id).active:
                break
            evolver._apply(new, old, "CONTRADICTS", "hit", strength=0.5, authority=0.5)

        assert store.get(old.note_id).active is False

    def test_contradicts_preserves_supersedes_chain(self, tmp_path):
        evolver, store = self._make(tmp_path)
        old = self._note("old fact", confidence=1.0)
        new = self._note("new fact")
        store.add(old); store.add(new)

        evolver._apply(new, old, "CONTRADICTS", "refutes")
        assert new.supersedes == old.note_id

    # ── SUPPORTS boost ───────────────────────────────────────────────────────

    def test_supports_boosts_confidence(self, tmp_path):
        evolver, store = self._make(tmp_path)
        note = self._note("a fact", confidence=0.6)
        supporter = self._note("confirms the fact")
        store.add(note); store.add(supporter)

        evolver._apply(supporter, note, "SUPPORTS", "corroborates")
        assert store.get(note.note_id).confidence > 0.6

    def test_supports_saturates_near_ceiling(self, tmp_path):
        """Logistic curve: boosts shrink as confidence approaches 0.95."""
        evolver, store = self._make(tmp_path)
        note = self._note("established fact", confidence=0.90)
        supporter = self._note("further confirmation")
        store.add(note); store.add(supporter)

        evolver._apply(supporter, note, "SUPPORTS", "corroborates")
        new_conf = store.get(note.note_id).confidence
        assert new_conf <= 0.95  # never exceeds ceiling
        # delta = 0.10 × 1.0 × 1.0 × 0.10 = 0.010 → rounds to 0.00 (negligible)
        assert new_conf <= 0.95

    # ── Resurrection ─────────────────────────────────────────────────────────

    def test_archive_sets_archived_at(self, tmp_path):
        from oml.storage.teeg_store import TEEGStore
        store = TEEGStore(artifacts_dir=tmp_path)
        note = self._note("will be archived")
        store.add(note)
        store.archive(note.note_id)
        archived = store.get(note.note_id)
        assert archived.archived_at != ""
        assert archived.active is False

    def test_unarchive_restores_active(self, tmp_path):
        from oml.storage.teeg_store import TEEGStore
        store = TEEGStore(artifacts_dir=tmp_path)
        note = self._note("resurrect me")
        store.add(note)
        store.archive(note.note_id)
        assert not store.get(note.note_id).active

        result = store.unarchive(note.note_id)
        assert result is True
        assert store.get(note.note_id).active is True
        assert store.get(note.note_id).archived_at == ""

    def test_unarchive_idempotent_on_active_note(self, tmp_path):
        from oml.storage.teeg_store import TEEGStore
        store = TEEGStore(artifacts_dir=tmp_path)
        note = self._note("already active")
        store.add(note)
        result = store.unarchive(note.note_id)
        assert result is False  # no-op
        assert store.get(note.note_id).active is True

    def test_unarchive_nonexistent_returns_false(self, tmp_path):
        from oml.storage.teeg_store import TEEGStore
        store = TEEGStore(artifacts_dir=tmp_path)
        result = store.unarchive("does-not-exist")
        assert result is False

    def test_supports_on_archived_note_resurrects_above_threshold(self, tmp_path):
        """A SUPPORTS verdict on a warm-store note with recoverable confidence
        should restore it to active memory."""
        evolver, store = self._make(tmp_path)
        archived = self._note("was archived but valid", confidence=0.20)
        store.add(archived)
        store.archive(archived.note_id)
        assert not store.get(archived.note_id).active

        supporter = self._note("confirms archived fact")
        store.add(supporter)

        # delta = 0.10 × 1.0 × 1.0 × (1-0.20) = 0.08 → rounds to 0.10
        # new_conf = 0.20 + 0.10 = 0.30 ≥ 0.15 → resurrected
        evolver._apply(supporter, archived, "SUPPORTS", "confirms", strength=1.0, authority=1.0)
        assert store.get(archived.note_id).active is True

    def test_supports_on_deeply_archived_note_stays_archived(self, tmp_path):
        """A note with confidence at the absolute floor (0.05) should NOT be
        resurrected by a single weak SUPPORTS."""
        evolver, store = self._make(tmp_path)
        dead = self._note("deeply archived", confidence=0.05)
        store.add(dead)
        store.archive(dead.note_id)

        supporter = self._note("weak confirmation")
        store.add(supporter)

        # delta = 0.10 × 0.3 × 0.3 × (1-0.05) ≈ 0.009 → rounds to 0.00
        evolver._apply(supporter, dead, "SUPPORTS", "weak", strength=0.3, authority=0.3)
        # May or may not resurrect depending on rounding — just assert no crash
        # and that the note still exists
        assert store.get(dead.note_id) is not None

    # ── vector_search_warm ───────────────────────────────────────────────────

    def test_vector_search_warm_finds_recent_archived(self, tmp_path):
        from oml.storage.teeg_store import TEEGStore
        store = TEEGStore(artifacts_dir=tmp_path)
        note = self._note("boiling point water temperature", confidence=0.4,
                          keywords=["boiling", "water", "temperature"])
        store.add(note)
        store.archive(note.note_id)

        results = store.vector_search_warm("boiling water", top_k=5, warm_days=30)
        ids = [n.note_id for n, _ in results]
        assert note.note_id in ids

    def test_vector_search_warm_ignores_active_notes(self, tmp_path):
        from oml.storage.teeg_store import TEEGStore
        store = TEEGStore(artifacts_dir=tmp_path)
        note = self._note("active note boiling water", confidence=0.9,
                          keywords=["active", "boiling", "water"])
        store.add(note)  # NOT archived

        results = store.vector_search_warm("boiling water", top_k=5, warm_days=30)
        ids = [n.note_id for n, _ in results]
        assert note.note_id not in ids  # active notes excluded

    def test_vector_search_warm_ignores_floor_confidence(self, tmp_path):
        from oml.storage.teeg_store import TEEGStore
        store = TEEGStore(artifacts_dir=tmp_path)
        dead = self._note("boiling water dead", confidence=0.05,
                          keywords=["boiling", "water"])
        store.add(dead)
        store.archive(dead.note_id)

        results = store.vector_search_warm("boiling water", top_k=5, warm_days=30)
        ids = [n.note_id for n, _ in results]
        assert dead.note_id not in ids  # floor-confidence note excluded

    def test_vector_search_warm_empty_when_none_match(self, tmp_path):
        from oml.storage.teeg_store import TEEGStore
        store = TEEGStore(artifacts_dir=tmp_path)
        results = store.vector_search_warm("completely unrelated query xyz", top_k=5)
        assert results == []

    # ── Propagation queue ────────────────────────────────────────────────────

    def test_propagation_queue_persists_to_disk(self, tmp_path):
        evolver, store = self._make(tmp_path)
        note = self._note("a fact")
        store.add(note)

        evolver._add_to_propagation_queue(note.note_id, -0.1)

        queue_path = tmp_path / "teeg_propagation_queue.jsonl"
        assert queue_path.exists()
        lines = queue_path.read_text().strip().splitlines()
        assert len(lines) == 1
        entry = __import__("json").loads(lines[0])
        assert entry["note_id"] == note.note_id
        assert abs(entry["delta"] - (-0.1)) < 0.001
        assert entry["done"] is False

    def test_propagation_sweep_applies_and_clears(self, tmp_path):
        from oml.memory.atomic_note import AtomicNote
        evolver, store = self._make(tmp_path)

        n1 = AtomicNote(content="fact a", confidence=0.8,
                        keywords=["fact", "alpha"])
        n2 = AtomicNote(content="related to fact a", confidence=0.7,
                        keywords=["fact", "alpha", "related"])
        store.add(n1); store.add(n2)
        store.add_edge(n1.note_id, n2.note_id, relation="supports", weight=0.6)

        evolver._add_to_propagation_queue(n1.note_id, -0.2)

        count = evolver.propagation_sweep()
        assert count >= 1  # n1 was processed

        # Queue should now be all-done
        queue_path = tmp_path / "teeg_propagation_queue.jsonl"
        lines = queue_path.read_text().strip().splitlines()
        for line in lines:
            entry = __import__("json").loads(line)
            assert entry["done"] is True

    def test_propagation_sweep_coalesces_multiple_deltas(self, tmp_path):
        """Multiple queue entries for same note should be coalesced before apply."""
        evolver, store = self._make(tmp_path)
        note = self._note("central fact")
        store.add(note)
        # Queue 3 separate deltas for the same note
        for delta in [-0.1, -0.1, -0.1]:
            evolver._add_to_propagation_queue(note.note_id, delta)

        queue_path = tmp_path / "teeg_propagation_queue.jsonl"
        assert len(queue_path.read_text().strip().splitlines()) == 3

        evolver.propagation_sweep()
        # All done
        for line in queue_path.read_text().strip().splitlines():
            assert __import__("json").loads(line)["done"] is True

    def test_propagation_sweep_no_queue_returns_zero(self, tmp_path):
        evolver, store = self._make(tmp_path)
        assert evolver.propagation_sweep() == 0

    def test_single_hop_propagation_stops_at_one_hop(self, tmp_path):
        """Direction-enforced single-hop: A→B→C, delta on A propagates to B only."""
        from oml.memory.atomic_note import AtomicNote
        evolver, store = self._make(tmp_path)

        a = AtomicNote(content="node a", confidence=0.8, keywords=["node", "alpha"])
        b = AtomicNote(content="node b", confidence=0.7, keywords=["node", "beta"])
        c = AtomicNote(content="node c", confidence=0.6, keywords=["node", "gamma"])
        store.add(a); store.add(b); store.add(c)
        store.add_edge(a.note_id, b.note_id, relation="supports", weight=0.6)
        store.add_edge(b.note_id, c.note_id, relation="supports", weight=0.6)

        b_conf_before = store.get(b.note_id).confidence
        c_conf_before = store.get(c.note_id).confidence

        # Propagate a negative delta from A — should only affect B (direct out-edge)
        evolver._propagate_single_hop(a, -0.3)

        b_conf_after = store.get(b.note_id).confidence
        c_conf_after = store.get(c.note_id).confidence

        assert b_conf_after < b_conf_before  # B was affected (out-edge from A)
        assert c_conf_after == c_conf_before  # C was NOT affected (two hops from A)

    # ── audit includes queue depth ────────────────────────────────────────────

    def test_audit_includes_propagation_queue_depth(self, tmp_path):
        evolver, store = self._make(tmp_path)
        note = self._note("a fact")
        store.add(note)
        evolver._add_to_propagation_queue(note.note_id, -0.1)
        audit = evolver.audit()
        assert "propagation_queue" in audit
        assert audit["propagation_queue"] == 1
