from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Iterable, Dict, Tuple

from oml.retrieval.hybrid import HybridRetriever, SearchResult
from oml.retrieval.rerank import Reranker
from oml.storage.sqlite import (
    get_chunks_by_ids,
    get_notes_by_ids,
    get_documents_by_ids,
)
from oml.memory.context import ContextChunk


from oml.config import DEFAULT_SQLITE_PATH

@dataclass
class RetrievalConfig:
    top_k: int = 5
    alpha: float = 0.5
    use_rerank: bool = True
    storage_path: str = DEFAULT_SQLITE_PATH
    min_note_score: float = 0.35
    note_boost: float = 0.1
    summary_boost: float = 0.05  # doc summaries > raw chunks


def _dedup_chunks_by_text(
    chunks: Iterable[Tuple[str, str, float]],
) -> List[ContextChunk]:
    """Deduplicate and build ContextChunk list.

    chunks: iterable of (chunk_id, text, score)
    """
    seen_hashes: set[int] = set()
    context: List[ContextChunk] = []

    for cid, text, score in chunks:
        norm = text.strip()
        if not norm:
            continue
        # normalize whitespace to reduce near-duplicates
        norm_ws = " ".join(norm.split())
        h = hash(norm_ws)
        if h in seen_hashes:
            continue
        seen_hashes.add(h)
        context.append(ContextChunk(chunk_id=cid, text=text, score=score))

    return context


def build_context_chunks_sqlite(
    query: str,
    config: RetrievalConfig,
    artifacts_dir: Path | str = Path("artifacts"),
) -> List[ContextChunk]:
    """Shared context builder for SQLite-backed data.

    Hierarchy:
      1. MemoryNotes (thread summaries)
      2. Document summaries (per-doc `summary` field)
      3. Raw chunks from HybridRetriever

    Returns an ordered list of ContextChunk objects.
    """
    artifacts_dir = Path(artifacts_dir)
    retriever = HybridRetriever(artifacts_dir)

    context: List[ContextChunk] = []

    # ---- 1. MemoryNotes (thread summaries) ----
    try:
        note_results: List[SearchResult] = retriever.search_notes(query, top_k=3)
    except Exception:
        note_results = []

    if note_results:
        note_ids = [n.chunk_id for n in note_results if n.score > config.min_note_score]
        if note_ids:
            notes = get_notes_by_ids(config.storage_path, note_ids)
            score_map: Dict[str, float] = {
                n.chunk_id: n.score for n in note_results  # type: ignore[attr-defined]
            }
            note_triplets: List[Tuple[str, str, float]] = []
            for n in notes:
                base = score_map.get(n.note_id, 0.0)
                s = base + config.note_boost
                note_triplets.append(
                    (
                        n.note_id,
                        f"[MEMORY NOTE - THREAD SUMMARY]\n{n.content}",
                        s,
                    )
                )
            context.extend(_dedup_chunks_by_text(note_triplets))

    # ---- 2. Retrieve raw chunks via HybridRetriever ----
    candidate_k = config.top_k * 5 if config.use_rerank else config.top_k
    results: List[SearchResult] = retriever.search(
        query, top_k=candidate_k, alpha=config.alpha
    )

    if not results:
        return context

    chunk_ids = [r.chunk_id for r in results]
    chunks_data = get_chunks_by_ids(config.storage_path, chunk_ids)

    # chunk_id -> (doc_id, text)
    chunk_doc_map: Dict[str, str] = {c.chunk_id: c.doc_id for c in chunks_data}
    chunk_text_map: Dict[str, str] = {c.chunk_id: c.chunk_text for c in chunks_data}

    # ---- 3. Optional reranking ----
    if config.use_rerank:
        try:
            reranker = Reranker()
            doc_texts: List[str] = []
            valid_results: List[SearchResult] = []
            for r in results:
                txt = chunk_text_map.get(r.chunk_id)
                if not txt:
                    continue
                doc_texts.append(txt)
                valid_results.append(r)

            if valid_results:
                results = reranker.rerank(query, doc_texts, valid_results)
                results = results[: config.top_k]
        except Exception:
            results = results[: config.top_k]
    else:
        results = results[: config.top_k]

    # ---- 4. Document summaries (2nd tier) ----
    # Collect unique doc_ids for selected chunks
    doc_ids: List[str] = []
    for r in results:
        d_id = chunk_doc_map.get(r.chunk_id)
        if d_id and d_id not in doc_ids:
            doc_ids.append(d_id)

    doc_summaries_triplets: List[Tuple[str, str, float]] = []
    if doc_ids:
        docs = get_documents_by_ids(config.storage_path, doc_ids)

        # doc_id -> max chunk score for that doc
        doc_score_map: Dict[str, float] = {}
        for r in results:
            d_id = chunk_doc_map.get(r.chunk_id)
            if not d_id:
                continue
            prev = doc_score_map.get(d_id, 0.0)
            doc_score_map[d_id] = max(prev, r.score)

        for d in docs:
            if not d.summary:
                continue
            base_score = doc_score_map.get(d.doc_id, 0.0)
            s = base_score + config.summary_boost
            summary_id = f"summary-{d.doc_id}"
            doc_summaries_triplets.append(
                (
                    summary_id,
                    f"[DOC SUMMARY] {d.summary}",
                    s,
                )
            )

    context.extend(_dedup_chunks_by_text(doc_summaries_triplets))

    # ---- 5. Raw chunks (lowest tier) ----
    chunk_triplets: List[Tuple[str, str, float]] = []
    for r in results:
        txt = chunk_text_map.get(r.chunk_id)
        if not txt:
            continue
        chunk_triplets.append((r.chunk_id, txt, r.score))

    context.extend(_dedup_chunks_by_text(chunk_triplets))

    return context
