# OpenMemoryLab

Experimental retrieval and agent-memory lab for testing RAG strategies, context efficiency, and memory architecture ideas.

![Python](https://img.shields.io/badge/python-3.12-blue)
![License](https://img.shields.io/badge/license-Apache%202.0-green)
![Type%20Hints](https://img.shields.io/badge/typing-mypy-blue)
![Style](https://img.shields.io/badge/lint-ruff-orange)
![Tests](https://img.shields.io/badge/tests-432%20passing-brightgreen)

OpenMemoryLab is built for one core purpose: make retrieval and memory design choices measurable.

Instead of shipping one fixed pipeline, it gives you modular components you can swap and benchmark:
- Retrieval strategy (`bm25`, vector, hybrid, rerank, HyDE, graph).
- Storage backend (`sqlite`, `lancedb`, `memory`).
- LLM provider (`mock`, `ollama:*`, `openai:*`, `gemini:*`, `lmstudio:*`, `openrouter:*`).
- Memory pipeline (`TEEG`, `PRISM`) for long-horizon context quality and efficiency.
- Visual Technique Composer for drag-and-drop pipeline building and execution.

## Architecture

```text
Ingest -> Store -> Index -> Retrieve -> Context Budget -> Generate -> Evaluate

oml/ingest      Parse + chunk + optional summarization/graph extraction
oml/storage     Pluggable backends (sqlite/lancedb/memory)
oml/retrieval   Hybrid retrieval, rerank, HyDE, graph augment
oml/memory      TEEG + PRISM memory systems
oml/api         FastAPI server and schemas
oml/app         CLI chat + Streamlit UI + Technique Composer
oml/eval        Benchmarks and task framework
oml/techniques  Decomposed technique modules (distillers, judges, etc.)
```

## Main Modules

### 1. RAG Core
- Hybrid BM25 + dense vector retrieval.
- Optional Cross-Encoder reranking.
- HyDE support for hard semantic queries.
- Token-aware context packing.

### 2. TEEG Memory
- Atomic notes with graph relations.
- Write-time evolution/consistency checks.
- Graph traversal retrieval for relation-first grounding.

### 3. PRISM Efficiency Layer
- SketchGate near-duplicate detection.
- DeltaStore semantic patch storage.
- CallBatcher N-to-1 LLM call coalescing.

### 4. Technique Composer
- Visual drag-and-drop pipeline builder in the Streamlit UI.
- 11 composable block types across 5 categories (data, ingest, evolve, retrieve, generate).
- Per-block settings via gear icon (custom prompts, temperature, thresholds).
- Real-time execution with per-node status, I/O inspection, and error display.
- Recipe quick-start and pipeline save/load.

### 5. Evaluation Framework
- `lost-in-middle`
- `faithfulness`
- `retrieval_precision`
- `cost_latency`
- `oml_vs_rag`
- `ablations`

## Quickstart

### Requirements
- Python 3.12.x
- Optional: local Ollama or API keys for provider-backed models

### Install

```bash
py -3.12 -m venv .venv  # Windows
.venv\Scripts\activate  # Windows
pip install -e .
pip install -e ".[dev]"
```

### Ingest Demo Corpus

```bash
oml ingest --demo
```

### Query

```bash
oml query "What is TEEG?" --top-k 5 --alpha 0.5 --show-tokens
```

### Chat

```bash
oml chat --model mock --top-k 5 --budget 4000
```

### Run API

```bash
oml api --host 0.0.0.0 --port 8000
# Swagger: http://localhost:8000/docs
```

### Run UI

```bash
oml ui
```

## TEEG / PRISM Examples

```bash
# TEEG
oml teeg-ingest "Victor created the creature in 1797."
oml teeg-query "Who created the creature?"

# PRISM
oml prism-ingest "Victor created the creature in 1797."
oml prism-stats
```

## Evaluation

```bash
oml eval lost-in-middle --model mock
oml eval faithfulness --model mock
```

Reports are generated under `reports/`.

## Testing

```bash
python -m pytest -q
```

## Validation

```bash
python scripts/validate.py
```

## Repo Layout

```text
oml/                Core package
scripts/            Experiment and utility scripts
tests/              Test suite
reports/            Experiment output
docs/               Project docs (roadmap, changelog, architecture)
```

## Documentation

- Docs index: [docs/README.md](docs/README.md)
- Architecture: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- Roadmap: [docs/ROADMAP.md](docs/ROADMAP.md)
- TODO: [docs/TODO.md](docs/TODO.md)
- Changelog: [docs/CHANGELOG.md](docs/CHANGELOG.md)
- Demo walkthrough: [docs/DEMO_SCRIPT.md](docs/DEMO_SCRIPT.md)
- Contribution guide: [CONTRIBUTING.md](CONTRIBUTING.md)
- Scripts guide: [scripts/README.md](scripts/README.md)

## License

Apache-2.0. See [LICENSE](LICENSE).
