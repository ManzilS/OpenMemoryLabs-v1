# Demo Script

## Goal

Walk through the core capabilities of OpenMemoryLab — modular retrieval,
memory systems, and visual pipeline composition.

## 1. Setup

```bash
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -e .
pip install -e ".[dev]"
```

## 2. Ingest

```bash
oml ingest --demo
```

Explain:
- Parse, chunk, store, and index flow.
- Artifacts are generated for retrieval.

## 3. Query Baseline

```bash
oml query "What is TEEG?" --top-k 5 --alpha 0.5 --show-tokens
```

Explain:
- Hybrid retrieval behavior.
- Token-budgeted context assembly.

## 4. Compare Retrieval Modes

```bash
oml query "What is TEEG?" --top-k 5 --alpha 0.0
oml query "What is TEEG?" --top-k 5 --alpha 1.0
oml query "What is TEEG?" --top-k 5 --alpha 0.5 --rerank
```

Explain:
- Sparse vs dense vs hybrid.
- Precision and latency tradeoff with reranking.

## 5. TEEG Memory Demo

```bash
oml teeg-ingest "Victor created the creature in 1797."
oml teeg-ingest "Victor fled after the creature awoke."
oml teeg-query "What did Victor do after creation?"
```

Explain:
- Atomic note distillation.
- Relation-aware retrieval over evolving memory.

## 6. PRISM Efficiency Demo

```bash
oml prism-ingest "Victor created the creature in 1797."
oml prism-ingest "Victor created the creature in 1797."
oml prism-stats
```

Explain:
- Near-duplicate gating.
- Storage and call-efficiency gains.

## 7. Evaluation

```bash
oml eval lost-in-middle --model mock
oml eval faithfulness --model mock
```

Explain:
- Task-based quality checks.
- Consistent comparison across experiments.

## 8. Technique Composer (Visual)

```bash
oml ui
```

In the UI, navigate to the **Techniques** tab and open the **Composer** sub-tab.

Show:
- Load the "Evolution Chain" recipe (one click).
- Drag-and-drop a block from the palette to reorder.
- Click the gear icon on the LLM Distiller to show per-block settings.
- Click Run to execute the pipeline and show per-node results.
- Click a block to inspect I/O in the inspector panel.

## 9. API (Optional)

```bash
oml api
```

Show:
- FastAPI Swagger docs at `http://localhost:8000/docs`.
