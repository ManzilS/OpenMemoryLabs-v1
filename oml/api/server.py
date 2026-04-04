"""oml/api/server.py — OpenMemoryLab REST API
=============================================

FastAPI server that exposes ``QueryPipeline``, ``ChatSession``, and the TEEG
memory system as HTTP endpoints with full OpenAPI documentation.

Usage
-----
Via the OML CLI (recommended)::

    oml api                           # 0.0.0.0:8000
    oml api --host 127.0.0.1 --port 9000
    oml api --reload                  # hot-reload for development

Via uvicorn directly::

    uvicorn oml.api.server:app --host 0.0.0.0 --port 8000 --reload

Endpoints
---------
GET  /health          System status and active configuration
POST /query           Single-turn hybrid RAG query
POST /chat            Multi-turn RAG chat (stateless per-request session)
POST /teeg/ingest     Distil text into an AtomicNote and store in TEEG graph
POST /teeg/query      Query TEEG graph memory with ScoutRetriever BFS traversal
GET  /docs            Interactive Swagger UI
GET  /redoc           ReDoc documentation
"""

from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException

from oml.api.schemas import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    PrismBatchRequest,
    PrismBatchResponse,
    PrismIngestRequest,
    PrismIngestResponse,
    PrismQueryRequest,
    PrismQueryResponse,
    PrismStatsResponse,
    QueryRequest,
    QueryResponse,
    SourceDoc,
    TeegConsolidateRequest,
    TeegConsolidateResponse,
    TeegIngestRequest,
    TeegIngestResponse,
    TeegQueryRequest,
    TeegQueryResponse,
)
from oml.config import DEFAULT_MODEL, DEFAULT_STORAGE
from oml.llm.factory import get_llm_client
from oml.memory.context import ContextBudgeter
from oml.retrieval.pipeline import QueryPipeline


# ── Application-level singletons ─────────────────────────────────────────────
# QueryPipeline is created once at startup (loading indices from disk is
# expensive).  The TEEG pipeline is lightweight and created per-request so
# that notes ingested via /teeg/ingest are immediately visible to /teeg/query.

class _AppState:
    query_pipelines: dict[tuple[str, str], QueryPipeline]


_state = _AppState()
_state.query_pipelines = {}


@asynccontextmanager
async def lifespan(app: FastAPI) -> Any:  # type: ignore[misc]
    """Initialise app state and clean up on shutdown."""
    yield
    _state.query_pipelines.clear()


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="OpenMemoryLab API",
    summary=(
        "REST interface for the OpenMemoryLab RAG & TEEG memory system. "
        "Wraps QueryPipeline, ChatSession, and TEEGPipeline behind typed "
        "HTTP endpoints with auto-generated Swagger documentation."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)


# ── Dependencies ──────────────────────────────────────────────────────────────


def _get_query_pipeline(storage_type: str, artifacts_dir: str) -> QueryPipeline:
    """Return a cached QueryPipeline for the requested experiment configuration."""
    key = (storage_type, artifacts_dir)
    pipeline = _state.query_pipelines.get(key)
    if pipeline is not None:
        return pipeline

    try:
        pipeline = QueryPipeline(storage_type=storage_type, artifacts_dir=artifacts_dir)
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                f"QueryPipeline is not ready for storage='{storage_type}', "
                f"artifacts_dir='{artifacts_dir}'. Run `oml ingest` first. "
                f"Root cause: {exc}"
            ),
        ) from exc

    _state.query_pipelines[key] = pipeline
    return pipeline


# ── /health ───────────────────────────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health() -> HealthResponse:
    """Return system status and active configuration.

    ``teeg_ready`` is ``true`` when a TEEG store is present on disk and notes
    can be queried without running ``oml teeg-ingest`` first.
    """
    teeg_ready = Path("teeg_store/teeg_notes.jsonl").exists()
    return HealthResponse(
        status="ok",
        version="1.0.0",
        storage=DEFAULT_STORAGE,
        llm=DEFAULT_MODEL,
        teeg_ready=teeg_ready,
    )


# ── /query ────────────────────────────────────────────────────────────────────


@app.post("/query", response_model=QueryResponse, tags=["RAG"])
async def query(req: QueryRequest) -> QueryResponse:
    """Run a hybrid RAG query over ingested documents.

    **Retrieval pipeline:**
    1. Hybrid BM25 (α=0) + dense-vector FAISS (α=1) score fusion
    2. Optional Cross-Encoder reranking
    3. Optional HyDE (Hypothetical Document Embeddings) for dense search
    4. ContextBudgeter priority-packs chunks within the token budget
    5. LLM generates a grounded answer from the packed prompt

    Requires at least one document ingested via ``oml ingest`` or the
    ``/ingest`` endpoint.  Returns **404** when the index is empty.
    """
    t0 = time.perf_counter()
    pipeline = _get_query_pipeline(req.storage_type, req.artifacts_dir)

    try:
        # Run the CPU/IO-bound pipeline in a thread so the event loop stays free
        context_chunks, prompt, tokens = await asyncio.to_thread(
            pipeline.run,
            query=req.question,
            top_k=req.top_k,
            alpha=req.alpha,
            budget=req.budget,
            use_rerank=req.use_rerank,
            use_hyde=req.use_hyde,
            use_graph=req.use_graph,
            model_name=req.model,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if not context_chunks:
        raise HTTPException(
            status_code=404,
            detail="No relevant documents found. Run `oml ingest` to build the index first.",
        )

    # Build the packed prompt if the pipeline didn't return one
    if not prompt:
        budgeter = ContextBudgeter()
        prompt, tokens = budgeter.construct_prompt_with_tokens(
            req.question, context_chunks, max_tokens=req.budget
        )

    llm = get_llm_client(req.model)
    # LLM network call — offload to thread so other requests aren't blocked
    answer = await asyncio.to_thread(llm.generate, prompt)

    latency_ms = (time.perf_counter() - t0) * 1000

    # Filter out injected summary/note pseudo-chunks from the sources list
    sources = [
        SourceDoc(chunk_id=c.chunk_id, text=c.text[:300], score=max(0.0, float(c.score)))
        for c in context_chunks[: req.top_k]
        if not c.chunk_id.startswith(("summary_", "note_"))
        and c.chunk_id != "knowledge_graph_context"
    ]

    return QueryResponse(
        answer=answer,
        sources=sources,
        tokens_used=tokens or 0,
        latency_ms=round(latency_ms, 2),
    )


# ── /chat ─────────────────────────────────────────────────────────────────────


@app.post("/chat", response_model=ChatResponse, tags=["RAG"])
async def chat(req: ChatRequest) -> ChatResponse:
    """Send a message to the RAG chat pipeline.

    The **last** message in ``messages`` is used as the retrieval query.
    A new ``ChatSession`` is created per request (stateless design); use the
    returned ``session_id`` to correlate events in the event log.

    Requires at least one document ingested via ``oml ingest``.
    """
    user_query = req.messages[-1].content
    t0 = time.perf_counter()

    from oml.app.chat import ChatSession

    try:
        session = ChatSession(
            model=req.model,
            storage_type=req.storage_type,
            artifacts_dir=req.artifacts_dir,
        )
        response, _, tokens = await asyncio.to_thread(
            session.send_message,
            query=user_query,
            top_k=req.top_k,
            alpha=req.alpha,
            budget=req.budget,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    latency_ms = (time.perf_counter() - t0) * 1000

    return ChatResponse(
        answer=response,
        tokens_used=tokens or 0,
        latency_ms=round(latency_ms, 2),
        session_id=session.session_id,
    )


# ── /teeg/ingest ─────────────────────────────────────────────────────────────


@app.post("/teeg/ingest", response_model=TeegIngestResponse, tags=["TEEG"])
async def teeg_ingest(req: TeegIngestRequest) -> TeegIngestResponse:
    """Distil raw text into an AtomicNote and store it in the TEEG graph.

    **Write-time pipeline:**
    1. LLM extracts a structured ``AtomicNote`` in TOON format (~40% fewer tokens than JSON)
    2. ``MemoryEvolver`` classifies the note's relation to existing memory:
       ``CONTRADICTS`` | ``EXTENDS`` | ``SUPPORTS`` | ``UNRELATED``
    3. The note is persisted to ``teeg_store/`` with a labelled edge in the
       NetworkX DiGraph

    Use ``model=mock`` for offline testing (no LLM call required).
    """
    from oml.memory.factory import get_memory_pipeline

    try:
        pipeline = get_memory_pipeline(
            "teeg",
            artifacts_dir=req.artifacts_dir,
            model=req.model,
        )
        note = await asyncio.to_thread(pipeline.ingest, req.text, context_hint=req.context)
        await asyncio.to_thread(pipeline.save)
        stats = pipeline.stats()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return TeegIngestResponse(
        note_id=note.note_id,
        content=note.content,
        keywords=list(note.keywords),
        message=(
            f"Note stored successfully. "
            f"Store now has {stats['active_notes']} active notes "
            f"and {stats['graph_edges']} graph edges."
        ),
    )


# ── /teeg/query ──────────────────────────────────────────────────────────────


@app.post("/teeg/query", response_model=TeegQueryResponse, tags=["TEEG"])
async def teeg_query(req: TeegQueryRequest) -> TeegQueryResponse:
    """Query the TEEG evolving graph memory and return an LLM-generated answer.

    **Query pipeline:**
    1. ``ScoutRetriever`` seeds from keyword/vector matches
    2. BFS traversal follows relation-labelled graph edges (EXTENDS, SUPPORTS…)
    3. Retrieved notes are serialised to a TOON context block
    4. LLM generates a grounded answer from the TOON context

    Requires notes to have been ingested via ``oml teeg-ingest`` or
    ``POST /teeg/ingest``.  Returns **404** when the store is empty.
    """
    from oml.memory.factory import get_memory_pipeline

    try:
        pipeline = get_memory_pipeline(
            "teeg",
            artifacts_dir=req.artifacts_dir,
            model=req.model,
        )
        stats = pipeline.stats()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if stats["active_notes"] == 0:
        raise HTTPException(
            status_code=404,
            detail="No TEEG notes found. Run `oml teeg-ingest` or POST /teeg/ingest first.",
        )

    t0 = time.perf_counter()

    try:
        # query + search are independent — run them concurrently in threads
        answer_task   = asyncio.to_thread(pipeline.query,  req.question, top_k=req.top_k, return_context=True)
        search_task   = asyncio.to_thread(pipeline.search, req.question, top_k=req.top_k)
        (answer, _), raw_results = await asyncio.gather(answer_task, search_task)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    latency_ms = (time.perf_counter() - t0) * 1000

    notes_used = [
        {
            "note_id": note.note_id,
            "content": note.content[:200],
            "score": round(score, 4),
            "hops": hops,
            "tags": list(note.tags),
        }
        for note, score, hops in raw_results
    ]

    return TeegQueryResponse(
        answer=answer,
        notes_used=notes_used,
        latency_ms=round(latency_ms, 2),
    )


# ── /teeg/consolidate ─────────────────────────────────────────────────────────


@app.post("/teeg/consolidate", response_model=TeegConsolidateResponse, tags=["TEEG"])
async def teeg_consolidate(req: TeegConsolidateRequest) -> TeegConsolidateResponse:
    """Consolidate clusters of related TEEG notes into compressed summaries.

    **How it works:**
    1. Detects note clusters sharing ≥ 2 keywords or with existing graph edges
    2. Creates one summary ``AtomicNote`` per cluster (LLM or heuristic)
    3. Archives original notes (soft-delete; reachable via graph)
    4. Adds ``consolidates`` edges from summary → each archived note

    **Token efficiency:** Each archived note reduces LLM context cost by ~87
    tokens per query (the FULL TOON size of one note).

    Use ``dry_run=true`` to see projected savings before modifying the store.
    """
    from oml.storage.teeg_store import TEEGStore
    from oml.memory.consolidator import MemoryConsolidator

    try:
        store = TEEGStore(artifacts_dir=req.artifacts_dir)
        consolidator = MemoryConsolidator(
            store,
            model_name=req.model,
            min_cluster_size=req.min_cluster_size,
            use_llm_summary=req.use_llm_summary,
        )
        if req.dry_run:
            result = await asyncio.to_thread(consolidator.dry_run)
        else:
            result = await asyncio.to_thread(consolidator.consolidate, max_clusters=req.max_clusters)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return TeegConsolidateResponse(
        clusters_found=result.clusters_found,
        notes_archived=result.notes_archived,
        summaries_created=result.summaries_created,
        token_savings_est=result.token_savings_est,
        skipped_small_clusters=result.skipped_small_clusters,
        dry_run=req.dry_run,
    )


# ── /prism/ingest ─────────────────────────────────────────────────────────────


@app.post("/prism/ingest", response_model=PrismIngestResponse, tags=["PRISM"])
async def prism_ingest(req: PrismIngestRequest) -> PrismIngestResponse:
    """Ingest text via PRISM with write-time near-duplicate detection.

    **PRISM Layer 1 — SketchGate:**
    Before any LLM call, MinHash LSH checks if the incoming text shares ≥
    ``dedup_threshold`` keyword overlap with an existing note.  If so, the
    ingest is skipped and the existing note's ``access_count`` is incremented.
    This prevents redundant notes from proliferating in the knowledge graph
    without wasting any API calls.

    **Pipeline (new notes):**
    1. SketchGate dedup check — O(N × 64) pure-Python MinHash scan
    2. TEEGPipeline.ingest() — distil → evolver → TEEGStore
    3. SketchGate.register() — add note to MinHash + Bloom indices
    """
    from oml.memory.prism_pipeline import PRISMPipeline

    try:
        pipeline = PRISMPipeline(
            artifacts_dir=req.artifacts_dir,
            model=req.model,
            dedup_threshold=req.dedup_threshold,
        )
        result = await asyncio.to_thread(pipeline.ingest, req.text, context_hint=req.context)
        await asyncio.to_thread(pipeline.save)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    store_count = pipeline._store.active_count()
    return PrismIngestResponse(
        note_id=result.note.note_id,
        content=result.note.content,
        keywords=list(result.note.keywords),
        was_deduplicated=result.was_deduplicated,
        merged_into=result.merged_into,
        is_delta=result.is_delta,
        message=(
            f"Near-duplicate of {result.merged_into!r} — skipped."
            if result.was_deduplicated
            else f"Note stored. Store has {store_count} active notes."
        ),
    )


# ── /prism/batch ──────────────────────────────────────────────────────────────


@app.post("/prism/batch", response_model=PrismBatchResponse, tags=["PRISM"])
async def prism_batch(req: PrismBatchRequest) -> PrismBatchResponse:
    """Batch-ingest N texts using a single distillation + evolution LLM call.

    **PRISM Layer 3 — CallBatcher:**
    Reduces API cost from 2N calls (naive) to 2 calls by packing all
    distillation requests into one multi-output prompt and all evolution
    judgments into a second prompt.

    Call efficiency formula: ``1 − 1/N``
    - N=4:  75 % savings
    - N=8:  87.5 % savings
    - N=32: 96.9 % savings

    **Combined with Layer 1 (SketchGate):**
    Near-duplicates are filtered *before* the batch call, so only genuinely
    new texts consume LLM tokens.
    """
    from oml.memory.prism_pipeline import PRISMPipeline

    try:
        pipeline = PRISMPipeline(
            artifacts_dir=req.artifacts_dir,
            model=req.model,
            dedup_threshold=req.dedup_threshold,
            batch_size=req.batch_size,
        )
        result = await asyncio.to_thread(pipeline.batch_ingest, req.texts)
        await asyncio.to_thread(pipeline.save)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return PrismBatchResponse(
        notes_created=len(result.notes) - result.dedup_count,
        dedup_count=result.dedup_count,
        delta_count=result.delta_count,
        llm_calls_made=result.llm_calls_made,
        llm_calls_saved=result.llm_calls_saved,
        call_efficiency=result.call_efficiency,
    )


# ── /prism/query ──────────────────────────────────────────────────────────────


@app.post("/prism/query", response_model=PrismQueryResponse, tags=["PRISM"])
async def prism_query(req: PrismQueryRequest) -> PrismQueryResponse:
    """Query PRISM memory using Scout graph traversal + TieredContextPacker.

    Identical retrieval quality to ``POST /teeg/query`` but runs through the
    full PRISM stack with DeltaStore-aware context assembly.

    Returns **404** when the store is empty.
    """
    from oml.memory.prism_pipeline import PRISMPipeline

    try:
        pipeline = PRISMPipeline(artifacts_dir=req.artifacts_dir, model=req.model)
        if pipeline._store.active_count() == 0:
            raise HTTPException(
                status_code=404,
                detail="No PRISM notes found. POST /prism/ingest or /prism/batch first.",
            )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    t0 = time.perf_counter()
    try:
        # query + search are independent — run concurrently
        answer_task  = asyncio.to_thread(pipeline.query,  req.question, top_k=req.top_k)
        search_task  = asyncio.to_thread(pipeline.search, req.question, top_k=req.top_k)
        (answer, _), raw_results = await asyncio.gather(answer_task, search_task)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    latency_ms = (time.perf_counter() - t0) * 1000

    notes_used = [
        {
            "note_id": note.note_id,
            "content": note.content[:200],
            "score": round(score, 4),
            "hops": hops,
            "tags": list(note.tags),
            "has_patches": pipeline._delta.has_patches(note.note_id),
        }
        for note, score, hops in raw_results
    ]
    return PrismQueryResponse(
        answer=answer,
        notes_used=notes_used,
        latency_ms=round(latency_ms, 2),
    )


# ── /prism/stats ──────────────────────────────────────────────────────────────


@app.get("/prism/stats", response_model=PrismStatsResponse, tags=["PRISM"])
async def prism_stats(
    artifacts_dir: str = "teeg_store",
    model: str = DEFAULT_MODEL,
) -> PrismStatsResponse:
    """Return aggregated efficiency statistics from all three PRISM layers.

    Useful for monitoring dedup effectiveness, token savings, and call
    efficiency over time.  All values are read from persisted state and
    require no LLM calls.
    """
    from oml.memory.prism_pipeline import PRISMPipeline

    try:
        pipeline = PRISMPipeline(artifacts_dir=artifacts_dir, model=model)
        raw = await asyncio.to_thread(pipeline.raw_stats)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return PrismStatsResponse(
        store=raw["store"],
        sketch_gate=raw["sketch_gate"],
        delta_store=raw["delta_store"],
        call_batcher=raw["call_batcher"],
        model=raw["model"],
    )
