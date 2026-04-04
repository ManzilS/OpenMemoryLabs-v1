# OpenMemoryLab Experiments (Ollama `qwen3:4b` Edition)

## 1. Hybrid Retrieval α (Alpha) Sweep

**Objective:**
Determine the optimal weighting parameter (`α`) between sparse keyword retrieval (BM25) and dense semantic retrieval (Vector Search).

- `α = 0.0` (100% BM25)
- `α = 1.0` (100% Vector)
- `α = 0.5` (50/50 Hybrid)

**Methodology:**
The Mary Shelley's Frankenstein test corpus was ingested. We queried the system using 10 queries ranging from exact keywords ("Justine Moritz", "Elizabeth") to semantic queries ("goal of the expedition", "monster's request") while routing completion generation through `ollama:qwen3:4b`.

**Findings:**
1. **`α = 0.0` (BM25 Only):** Extremely fast and highly accurate when the query possessed exact unique keywords (e.g., “Justine Moritz”). However, it struggled on purely semantic queries such as "goal of the expedition".
2. **`α = 1.0` (Vector Only):** Excelled at conceptual matching (e.g., retrieving the correct chunk where Frankenstein discusses his materials, even if the phrasing didn't perfectly match). On the flip side, `qwen3:4b` generation sometimes drifted if the retrieved chunk missed the exact names.
3. **`α = 0.5` (Hybrid):** The clear winner. By normalizing both scores before combining them, the hybrid approach consistently ranked the most relevant document #1, catching both exact jargon for names and semantic intent. This provided the cleanest context for `qwen3:4b` to synthesize an accurate answer.

---

## 2. Cross-Encoder Reranking: On vs Off

**Objective:**
Measure the qualitative impact of applying a Cross-Encoder Reranker (`sentence-transformers/all-MiniLM-L6-v2` cross-encoder) on the top-K retrieved candidates before sending to `qwen3:4b`.

**Methodology:**
We retrieved candidates (`K=5`) with `α=0.5` across 5 complex queries. We then compared the results before and after Reranking.

**Findings:**
- **Reranker Off:** The initial Bi-Encoder retrieval occasionally struggled with relational queries. For example, "What was the monster's request to Victor in the mountains?" returned slightly tangential narrative passages rather than the explicit demand for a companion.
- **Reranker On:** The Cross-Encoder, which computes attention between the query and the chunk simultaneously, correctly pushed the explicit dialogue of the monster's demand to #1. This resulted in a much more faithful generation from `qwen3:4b`.
- **Trade-off:** Reranking adds latency per query but provides a noticeable boost to precision. *Note: Local testing on Windows environments showed dependency issues with `torch` which can necessitate falling back to Bi-Encoder scores.*

---

## 3. Storage Backends: SQLite vs LanceDB

**Objective:**
Assess the trade-offs of using `SQLite + FAISS` vs `LanceDB` natively.

**Findings:**
- **LanceDB** provides superior developer experience by unifying metadata and vector storage. It effortlessly scales without requiring a secondary FAISS index built in memory.
- **SQLite + FAISS** remains the best fallback for purely CPU-bound, zero-dependency ingestion where we only rely on built-in libraries and `rank_bm25`. 
