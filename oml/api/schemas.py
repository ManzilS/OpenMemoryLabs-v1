"""oml/api/schemas.py — Pydantic request/response models for the OML REST API.

All schemas are documented with field descriptions so FastAPI can generate
accurate OpenAPI / Swagger docs at /docs and /redoc.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from oml.config import DEFAULT_MODEL, DEFAULT_STORAGE


# ── Shared building blocks ────────────────────────────────────────────────────


class SourceDoc(BaseModel):
    """A single retrieved document chunk included in a query response."""

    chunk_id: str = Field(description="Unique chunk identifier from the storage backend.")
    text: str = Field(description="First 300 characters of the chunk text.")
    score: float = Field(ge=0.0, description="Retrieval relevance score (higher is better).")


# ── /health ───────────────────────────────────────────────────────────────────


class HealthResponse(BaseModel):
    """System health and active configuration."""

    status: str = Field(description="'ok' when the server is ready.")
    version: str
    storage: str = Field(description="Active storage backend (sqlite / lancedb / memory).")
    llm: str = Field(description="Default LLM model string.")
    teeg_ready: bool = Field(
        default=False,
        description="True if a TEEG store is present on disk and queryable.",
    )


# ── /query ────────────────────────────────────────────────────────────────────


class QueryRequest(BaseModel):
    """Parameters for a single-turn hybrid RAG query."""

    question: str = Field(min_length=1, description="The question to answer.")
    top_k: int = Field(default=5, ge=1, le=50, description="Number of chunks to retrieve.")
    alpha: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Hybrid weight: 0.0 = BM25 only, 1.0 = vector only.",
    )
    budget: int = Field(
        default=4000,
        ge=256,
        le=32_000,
        description="Maximum token budget for the packed context prompt.",
    )
    model: str = Field(
        default=DEFAULT_MODEL,
        description="LLM model string (mock | ollama:<name> | openai:<name> | gemini:<name>).",
    )
    use_rerank: bool = Field(
        default=True,
        description="Apply Cross-Encoder reranking after initial retrieval.",
    )
    use_hyde: bool = Field(
        default=False,
        description="Use Hypothetical Document Embeddings for the dense search step.",
    )
    use_graph: bool = Field(
        default=False,
        description="Inject 1-hop knowledge-graph context from extracted triples.",
    )
    storage_type: str = Field(
        default=DEFAULT_STORAGE,
        description="Storage backend (sqlite | lancedb | memory).",
    )
    artifacts_dir: str = Field(
        default="artifacts",
        description="Directory containing retrieval artifacts (BM25/FAISS indexes).",
    )


class QueryResponse(BaseModel):
    """RAG query result with provenance and cost information."""

    answer: str
    sources: list[SourceDoc] = Field(description="Retrieved chunks used to generate the answer.")
    tokens_used: int = Field(description="Approximate tokens in the packed prompt.")
    latency_ms: float = Field(description="End-to-end request latency in milliseconds.")


# ── /chat ─────────────────────────────────────────────────────────────────────


class ChatMessage(BaseModel):
    """A single turn in a conversation."""

    role: str = Field(
        pattern="^(user|assistant|system)$",
        description="Message role: 'user', 'assistant', or 'system'.",
    )
    content: str = Field(min_length=1)


class ChatRequest(BaseModel):
    """Multi-turn chat request.

    The last message in ``messages`` must have ``role='user'`` and is treated
    as the current query.  Earlier messages provide conversational context but
    are not currently used for retrieval (stateless per-request sessions).
    """

    messages: list[ChatMessage] = Field(min_length=1)
    model: str = Field(default=DEFAULT_MODEL)
    top_k: int = Field(default=5, ge=1, le=50)
    alpha: float = Field(default=0.5, ge=0.0, le=1.0)
    budget: int = Field(default=4000, ge=256, le=32_000)
    storage_type: str = Field(
        default=DEFAULT_STORAGE,
        description="Storage backend (sqlite | lancedb | memory).",
    )
    artifacts_dir: str = Field(
        default="artifacts",
        description="Directory containing retrieval artifacts (BM25/FAISS indexes).",
    )


class ChatResponse(BaseModel):
    """RAG chat response with session tracking."""

    answer: str
    tokens_used: int
    latency_ms: float
    session_id: str = Field(description="Unique session UUID for event log correlation.")


# ── /teeg/ingest ──────────────────────────────────────────────────────────────


class TeegIngestRequest(BaseModel):
    """Request body for TEEG note ingestion."""

    text: str = Field(min_length=1, description="Raw text to distil into an AtomicNote.")
    context: str = Field(
        default="",
        description="Optional context hint (e.g. source document, chapter, date).",
    )
    model: str = Field(
        default=DEFAULT_MODEL,
        description="LLM used for note distillation and MemoryEvolver classification.",
    )
    artifacts_dir: str = Field(
        default="teeg_store",
        description="Directory for persisted TEEG notes/graph state.",
    )


class TeegIngestResponse(BaseModel):
    """Result of a TEEG ingest operation."""

    note_id: str = Field(description="UUID of the newly created AtomicNote.")
    content: str = Field(description="Distilled one-sentence content of the note.")
    keywords: list[str]
    message: str = Field(description="Human-readable confirmation with store statistics.")


# ── /teeg/query ──────────────────────────────────────────────────────────────


class TeegQueryRequest(BaseModel):
    """Request body for a TEEG graph-memory query."""

    question: str = Field(min_length=1)
    model: str = Field(default=DEFAULT_MODEL)
    top_k: int = Field(
        default=8,
        ge=1,
        le=50,
        description="Maximum notes to retrieve via ScoutRetriever BFS traversal.",
    )
    artifacts_dir: str = Field(
        default="teeg_store",
        description="Directory for persisted TEEG notes/graph state.",
    )


class TeegQueryResponse(BaseModel):
    """TEEG query result with graph provenance."""

    answer: str
    notes_used: list[dict] = Field(
        description="Notes retrieved by ScoutRetriever, including hop distance and score."
    )
    latency_ms: float


# ── /teeg/consolidate ─────────────────────────────────────────────────────────


class TeegConsolidateRequest(BaseModel):
    """Request body for a TEEG memory consolidation pass."""

    min_cluster_size: int = Field(
        default=3,
        ge=2,
        le=50,
        description="Minimum notes per cluster to trigger consolidation.",
    )
    max_clusters: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum clusters to consolidate in this pass.",
    )
    dry_run: bool = Field(
        default=False,
        description="If True, analyse and project savings without modifying the store.",
    )
    use_llm_summary: bool = Field(
        default=False,
        description=(
            "Generate LLM summaries for each cluster.  Requires an LLM endpoint.  "
            "When False (default), heuristic summaries are used — faster and free."
        ),
    )
    model: str = Field(
        default=DEFAULT_MODEL,
        description="LLM model string used when use_llm_summary=True.",
    )
    artifacts_dir: str = Field(
        default="teeg_store",
        description="Directory for persisted TEEG notes/graph state.",
    )


class TeegConsolidateResponse(BaseModel):
    """Result of a TEEG memory consolidation pass."""

    clusters_found: int = Field(description="Number of eligible note clusters detected.")
    notes_archived: int = Field(description="Notes replaced by summary notes.")
    summaries_created: int = Field(description="New summary AtomicNotes created.")
    token_savings_est: int = Field(
        description=(
            "Estimated LLM context tokens saved per query "
            "(= (archived − summaries) × 87 tokens/note)."
        )
    )
    skipped_small_clusters: int = Field(
        default=0,
        description="Clusters below min_cluster_size that were not consolidated.",
    )
    dry_run: bool = Field(description="True if no changes were made to the store.")


# ── /prism/ingest ─────────────────────────────────────────────────────────────


class PrismIngestRequest(BaseModel):
    """Request body for a single PRISM ingest with SketchGate dedup."""

    text: str = Field(min_length=1, description="Raw text to ingest.")
    context: str = Field(default="", description="Optional context hint.")
    model: str = Field(default=DEFAULT_MODEL, description="LLM model string.")
    dedup_threshold: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description=(
            "MinHash Jaccard threshold for near-duplicate detection.  "
            "Notes with keyword overlap ≥ threshold are considered duplicates."
        ),
    )
    artifacts_dir: str = Field(
        default="teeg_store",
        description="Directory for persisted PRISM/TEEG state.",
    )


class PrismIngestResponse(BaseModel):
    """Result of a PRISM ingest operation."""

    note_id: str
    content: str
    keywords: list[str]
    was_deduplicated: bool = Field(
        description="True if SketchGate caught a near-duplicate — no LLM call was made."
    )
    merged_into: str | None = Field(
        default=None,
        description="note_id of the existing note if was_deduplicated is True.",
    )
    is_delta: bool = Field(
        default=False,
        description="True if stored as a SemanticPatch (EXTENDS verdict in batch mode).",
    )
    message: str


# ── /prism/batch ──────────────────────────────────────────────────────────────


class PrismBatchRequest(BaseModel):
    """Request body for batch PRISM ingest with N-to-1 LLM call coalescing."""

    texts: list[str] = Field(
        min_length=1,
        description="List of raw texts to ingest in a single batched LLM call.",
    )
    model: str = Field(default=DEFAULT_MODEL, description="LLM model string.")
    dedup_threshold: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="MinHash Jaccard threshold for near-duplicate detection.",
    )
    batch_size: int = Field(
        default=8,
        ge=1,
        le=64,
        description="Maximum texts per batch LLM call.",
    )
    artifacts_dir: str = Field(
        default="teeg_store",
        description="Directory for persisted PRISM/TEEG state.",
    )


class PrismBatchResponse(BaseModel):
    """Result of a PRISM batch ingest operation."""

    notes_created: int = Field(description="New notes added to the store.")
    dedup_count: int = Field(description="Inputs skipped by SketchGate (near-duplicates).")
    delta_count: int = Field(description="Notes stored as semantic patches (EXTENDS verdicts).")
    llm_calls_made: int = Field(description="Actual LLM calls made.")
    llm_calls_saved: int = Field(description="Calls avoided vs. naive 2-per-text approach.")
    call_efficiency: float = Field(description="llm_calls_saved / total as a fraction [0, 1].")


# ── /prism/query ──────────────────────────────────────────────────────────────


class PrismQueryRequest(BaseModel):
    """Request body for a PRISM memory query."""

    question: str = Field(min_length=1, description="The question to answer.")
    model: str = Field(default=DEFAULT_MODEL, description="LLM model string.")
    top_k: int = Field(default=8, ge=1, le=50, description="Max notes to retrieve.")
    artifacts_dir: str = Field(
        default="teeg_store",
        description="Directory for persisted PRISM/TEEG state.",
    )


class PrismQueryResponse(BaseModel):
    """PRISM query result with provenance."""

    answer: str
    notes_used: list[dict] = Field(description="Notes retrieved by ScoutRetriever.")
    latency_ms: float


# ── /prism/stats ──────────────────────────────────────────────────────────────


class PrismStatsResponse(BaseModel):
    """Aggregated efficiency statistics from all three PRISM layers."""

    store: dict = Field(description="TEEGStore statistics (note counts, graph edges).")
    sketch_gate: dict = Field(description="SketchGate statistics (dedup rate, threshold).")
    delta_store: dict = Field(description="DeltaStore statistics (patches, token savings).")
    call_batcher: dict = Field(description="CallBatcher statistics (calls made/saved, efficiency).")
    model: str
