# Scripts Guide

This directory contains non-library entry points for experiments, benchmarks,
and operational checks.

## Main Scripts

- `benchmark_models.py`: run multi-model benchmark harness
- `eval_combinations.py`: compare retrieval configuration presets
- `eval_lmstudio.py`: model evals against LM Studio targets
- `api_demo.py`: quick API workflow examples
- `validate.py`: one-command Python 3.12 + lint + test verification
- `teeg.py`: end-to-end TEEG pipeline demo (ingest, evolve, query)
- `prism.py`: PRISM efficiency layer demo
- `prepare_wiki_docs.py`: download and prepare Wikipedia corpus for benchmarks
- `run_wiki_benchmark.py`: full retrieval benchmark against wiki corpus

## Experiments

- `run_*_experiment.py`: focused experiment runners for retrieval components

## Utilities

- `utility/verify_*.py`: environment/index/storage verification helpers
- `utility/debug_*.py`: backend debugging helpers

## Recommendation

Keep new one-off exploratory scripts in `dev/scratch/` and only promote stable,
repeatable scripts into this folder.
