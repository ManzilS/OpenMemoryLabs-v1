# OpenMemoryLabs-v1

**An experimental lab for evaluating RAG strategies and long-term agent-memory architectures.**

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)]()
[![Status](https://img.shields.io/badge/status-research-purple.svg)]()

---

## What it is

OpenMemoryLabs is a controlled environment for answering one question:

> **Given a fixed agent task, which memory architecture actually improves reasoning quality — and at what cost?**

It benchmarks competing designs side-by-side on:
- Retrieval accuracy (did we fetch the right context?)
- Context efficiency (how many tokens did we burn?)
- Latency (how slow is it in practice?)
- Downstream answer quality

## Why

"Give your agent memory" has become a line in every LLM demo, but the field conflates a dozen different architectures under that label. Flat vector stores, hierarchical summarization, entity-graph memory, structured recall, sliding-window attention, RAG-over-history — all of these are called "memory." They don't cost the same. They don't work the same. And they don't behave the same way as sessions get long.

This repo is how I pull them apart.

## Architectures under evaluation

| Architecture | One-line summary |
|---|---|
| Flat vector store | All messages embedded, top-k over cosine similarity |
| Hierarchical summarization | Recursive summary ladder — raw → per-session → per-topic |
| Structured recall | Extract facts into a typed store; query by relation |
| Episodic + semantic split | Time-indexed events vs. distilled knowledge |
| Sliding window + long-term index | Recent full context + RAG over older archive |

## How it works

```
                  ┌─────────────────────────┐
                  │   task harness          │
                  │   • fixed task set      │
                  │   • fixed seed          │
                  │   • deterministic runs  │
                  └───────────┬─────────────┘
                              │
         ┌────────────────────┼────────────────────┐
         ▼                    ▼                    ▼
   ┌──────────┐        ┌──────────────┐      ┌──────────────┐
   │  arch A  │        │   arch B     │      │   arch C     │
   │ (flat)   │        │ (hierarchy)  │      │ (structured) │
   └────┬─────┘        └──────┬───────┘      └──────┬───────┘
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              ▼
                  ┌─────────────────────────┐
                  │   evaluator             │
                  │   • retrieval P/R       │
                  │   • token usage         │
                  │   • answer quality      │
                  │   • latency percentiles │
                  └─────────────────────────┘
```

## Quickstart

```bash
git clone https://github.com/ManzilS/OpenMemoryLabs-v1
cd OpenMemoryLabs-v1
pip install -r requirements.txt

# run a comparison
python -m openmemorylabs run --task long_project_recall --archs flat,hierarchy,structured

# view results
python -m openmemorylabs report
```

## Current findings (WIP)

*Benchmarks in progress — this section updates as runs complete. Check back or watch the repo for updates.*

- **Flat vector store** is the cheapest baseline — low ingestion overhead, predictable latency, degrades gracefully on short sessions.
- **Hierarchical summarization** shows stronger retrieval on long sessions (50+ turns); ingestion latency is noticeably higher at scale.
- **Structured recall** wins on narrow factual Q&A but is brittle whenever a query falls outside the extraction schema.
- The "best" architecture depends heavily on whether the task is recall vs. synthesis — there is no single winner.

*(Quantitative breakdowns — turn thresholds, latency multipliers, retrieval P/R — will be published here as the benchmark dataset stabilizes.)*

## What's implemented

- [x] Task harness with deterministic, seeded runs
- [x] Flat vector store baseline
- [x] Hierarchical summarization architecture
- [x] Structured recall architecture
- [x] Evaluator with retrieval and latency metrics
- [ ] Episodic + semantic split architecture
- [ ] Sliding window + long-term index architecture
- [ ] Published benchmark dataset

## License

MIT

## About

Built by [Manzil "Nick" Sapkota](https://github.com/ManzilS) — open to AI/ML Engineer roles. [Email](mailto:manzilsapkota@gmail.com) · [LinkedIn](https://www.linkedin.com/in/manzilsapkota/).
