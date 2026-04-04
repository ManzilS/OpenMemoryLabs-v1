"""Tests for the centralized context builder (MemoryNotes -> summaries -> chunks).

These tests use a tiny SQLite DB and monkeypatch the retriever so we don't
need FAISS/LLMs to run.
"""

from pathlib import Path
from typing import List

import pytest

from oml.memory.context import ContextChunk
from oml.models.schema import Document, MemoryNote
from oml.storage.sqlite import (
    init_db,
    upsert_documents,
    upsert_chunks,
    upsert_notes,
    get_all_notes,
    get_document,
)
from oml.retrieval.hybrid import HybridRetriever, SearchResult


@pytest.fixture
def tiny_sqlite_db(tmp_path: Path) -> str:
    """Create a tiny SQLite DB with one doc+summary and one MemoryNote."""
    db_path = tmp_path / "oml_test.db"
    init_db(str(db_path))

    doc = Document(
        doc_id="doc-1",
        source="test",
        raw_text="full raw text of the document",
        clean_text="full raw text of the document",
        summary="short summary of the document",
    )
    upsert_documents(str(db_path), [doc])

    # Also insert a single chunk for this doc so the context builder can
    # resolve doc_ids from chunk_ids returned by the dummy retriever.
    from oml.models.schema import Chunk

    chunk = Chunk(
        chunk_id="chunk-1",
        doc_id="doc-1",
        chunk_text="full raw text of the document",
        start_char=0,
        end_char=len("full raw text of the document"),
    )
    upsert_chunks(str(db_path), [chunk])

    note = MemoryNote(
        note_id="note-1",
        thread_id="thread-1",
        content="summary of the whole thread",
        source_doc_ids=["doc-1"],
    )
    upsert_notes(str(db_path), [note])

    assert get_document(str(db_path), "doc-1") is not None
    assert get_all_notes(str(db_path)), "Expected at least one MemoryNote"

    return str(db_path)


class DummyRetriever(HybridRetriever):
    """HybridRetriever subclass that returns fixed SearchResults.

    We patch HybridRetriever methods so vector/BM25 deps are not needed.
    """

    def __init__(self, artifacts_dir: Path):  # type: ignore[override]
        # Skip Base __init__ to avoid loading indices
        self.artifacts_dir = artifacts_dir
        self._loaded = True

    def load(self):  # type: ignore[override]
        self._loaded = True

    def search_notes(self, query: str, top_k: int = 3) -> List[SearchResult]:  # type: ignore[override]
        return [
            SearchResult(
                chunk_id="note-1",
                score=0.9,
                source="note_vector",
                details={},
            )
        ]

    def search(
        self,
        query: str,
        top_k: int = 10,
        alpha: float = 0.5,
        use_bm25: bool = True,
        use_vector: bool = True,
    ) -> List[SearchResult]:  # type: ignore[override]
        # Always return a single chunk hit corresponding to doc-1
        return [
            SearchResult(
                chunk_id="chunk-1",
                score=0.8,
                source="hybrid",
                details={"doc_id": "doc-1"},
            )
        ]


@pytest.fixture(autouse=True)
def patch_hybrid_retriever(monkeypatch):
    """Replace HybridRetriever with DummyRetriever in context_builder usage."""

    def _dummy_init(self, artifacts_dir):
        DummyRetriever.__init__(self, artifacts_dir)

    monkeypatch.setattr("oml.retrieval.hybrid.HybridRetriever.__init__", _dummy_init)
    monkeypatch.setattr("oml.retrieval.hybrid.HybridRetriever.load", DummyRetriever.load)
    monkeypatch.setattr("oml.retrieval.hybrid.HybridRetriever.search_notes", DummyRetriever.search_notes)
    monkeypatch.setattr("oml.retrieval.hybrid.HybridRetriever.search", DummyRetriever.search)


def test_context_builder_includes_note_and_summary_first(tmp_path: Path, tiny_sqlite_db: str):
    """Context builder should include MemoryNote and DOC SUMMARY, in that order."""
    from oml.context_builder import build_context_chunks_sqlite, RetrievalConfig

    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()

    cfg = RetrievalConfig(
        top_k=1,
        alpha=0.5,
        use_rerank=False,
        storage_path=tiny_sqlite_db,
        min_note_score=0.1,
        note_boost=0.1,
        summary_boost=0.05,
    )

    context_chunks: List[ContextChunk] = build_context_chunks_sqlite(
        query="test query",
        config=cfg,
        artifacts_dir=artifacts_dir,
    )

    texts = [c.text for c in context_chunks]
    assert any("MEMORY NOTE - THREAD SUMMARY" in t for t in texts)
    assert any("DOC SUMMARY" in t for t in texts)

    note_idx = min(i for i, t in enumerate(texts) if "MEMORY NOTE - THREAD SUMMARY" in t)
    summary_idx = min(i for i, t in enumerate(texts) if "DOC SUMMARY" in t)
    assert note_idx < summary_idx
