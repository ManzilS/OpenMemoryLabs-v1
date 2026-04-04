# OpenMemoryLab Roadmap

OpenMemoryLab is a modular retrieval and memory lab focused on measurable AI
engineering tradeoffs: quality, latency, and context efficiency.

## v1.0 (Current Baseline)

Core capabilities already implemented:

- Modular ingestion pipeline (`txt`, `eml`, `pdf`) with parser registry.
- Storage abstraction (`sqlite`, `memory`, `lancedb`).
- Retrieval stack (BM25, dense vector, hybrid fusion, rerank, HyDE, graph).
- Memory systems (TEEG and PRISM).
- FastAPI service and Streamlit UI.
- Evaluation tasks and benchmark reporting.
- Automated tests and CI on Python 3.12.

## v1.1 (Polish and Expand)

High-impact incremental work:

- Add Markdown parser support (`.md`) in ingest parser registry.
- Add Recall@K metric alongside Precision@K.
- Add rubric-based LLM-as-judge scoring in `faithfulness`.
- Add streaming API responses (SSE) for chat/query endpoints.
- Add Codecov upload for public coverage trend visibility.

## v2.x (Separate Deeper Labs)

Longer-horizon ideas best shipped as focused projects:

- Graph database retrieval lab (Neo4j-backed retrieval experiments).
- Event-sourced memory replay lab.
- RDF/SPARQL fact-checking lab.
- Advanced multi-document entity-linking retrieval.

## Usage Guidance

- v1.0 is the stable baseline.
- v1.1 items are scoped PR-sized enhancements.
- v2.x items are standalone project candidates.
