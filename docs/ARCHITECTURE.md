# Architecture Overview

OpenMemoryLab is built as a modular retrieval and memory lab. Core components
can be swapped without rewriting the rest of the pipeline.

## High-Level Flow

```text
Ingest -> Parse -> Chunk -> Store -> Index -> Retrieve -> Budget -> Generate -> Evaluate
```

## Modules

- `oml/ingest`: file parsing, chunking, optional summarization and extraction
- `oml/storage`: backend abstraction (`memory`, `sqlite`, `lancedb`)
- `oml/retrieval`: BM25, vector, hybrid, rerank, graph and HyDE support
- `oml/memory`: TEEG and PRISM memory pipelines
- `oml/llm`: provider abstraction (`mock`, `ollama`, `openai`, `gemini`, etc.)
- `oml/api`: FastAPI service layer
- `oml/app`: CLI, Streamlit UI, and visual Technique Composer
- `oml/eval`: benchmark tasks and run harness
- `oml/memory/techniques`: decomposed technique modules (distillers, judges, confidence, propagation)
- `oml/techniques`: technique registry and composability protocols

## Modularity Principles

- Providers are selected through factories and config, not hardcoded imports.
- Retrieval and memory strategies are composable feature toggles.
- Storage implementations share common interfaces.
- Evaluation tasks are registry-based, so new tasks can be added independently.

## Technique Composer

The Technique Composer (`oml/app/composer.py` + `oml/app/composer_component/`) is
a visual pipeline builder implemented as a custom Streamlit component using
`declare_component` with bidirectional JS-Python communication via `postMessage`.

- 11 block types across 5 categories (data, ingest, evolve, retrieve, generate).
- Per-block configurable settings (prompts, temperature, thresholds) via gear icon.
- Execution engine runs the pipeline sequentially, capturing per-node I/O and timing.
- Race-condition-safe state sync via `_skipRenders` counter pattern.

## Extension Points

- Add a new storage backend: `oml/storage/` + `oml/storage/factory.py`
- Add a new retrieval strategy: `oml/retrieval/` + pipeline wiring
- Add a new LLM provider: `oml/llm/` + `oml/llm/factory.py`
- Add a new evaluation task: `oml/eval/tasks/` + task registration
- Add a new technique block: `NODE_TYPES` in `composer.py` + handler in `_exec_node`
