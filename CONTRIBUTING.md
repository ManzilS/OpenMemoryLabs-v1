# Contributing to OpenMemoryLab

## Environment

- Python: `3.12.x` only
- Recommended setup (Windows):

```bash
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -e .
pip install -e ".[dev]"
```

## Quality Gates

Run these before opening a PR:

```bash
python -m ruff check oml --select F
python -m pytest -q
```

Optional type-check pass:

```bash
python -m mypy oml/
```

## Project Structure

- `oml/`: core package code
- `tests/`: test suite
- `scripts/`: experiments and utility scripts
- `docs/`: architecture, roadmap, and changelog docs
- `reports/`: generated benchmark/evaluation outputs

## Notes

- Keep modules composable (storage/retrieval/LLM/memory should remain swappable).
- Avoid coupling UI/CLI logic into core retrieval and memory modules.
- Do not commit local scratch files; use `dev/scratch/` for temporary work.
