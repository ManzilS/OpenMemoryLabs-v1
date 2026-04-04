# Contributing to OpenMemoryLab

Thanks for your interest in contributing! Bug reports, feature requests, and pull requests are all welcome.

---

## Development Setup

```bash
git clone https://github.com/ManzilS/openmemorylab-main2
cd openmemorylab-main2

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -e ".[dev]"
```

This installs the package in editable mode along with `pytest`, `ruff`, and `rdflib` (used by the RDF fact-checker tests).

---

## Running Tests

```bash
pytest                          # full suite — 144 tests (1 skipped if rdflib absent)
pytest tests/test_teeg.py -v    # TEEG module only (97 tests)
pytest tests/test_eval_tasks.py # eval framework only
```

No API keys are required — all tests use the deterministic `mock` LLM backend.

---

## Code Style

- **Linter / formatter:** `ruff` (configured in `pyproject.toml`)
- **Line length:** 100 characters
- **Python:** 3.11+

Run lint before pushing:

```bash
ruff check .
```

CI will fail on lint errors.

---

## Contribution Workflow

1. **Open an issue** to discuss larger features or breaking changes before writing code.
2. **Fork and branch** with a descriptive name:
   - `feat/markdown-parser`
   - `fix/scout-bfs-edge-case`
3. **Write tests** for new behavior. Keep the `mock` LLM backend pattern — no live API calls in tests.
4. **Keep PRs small and focused** — one feature or fix per PR.
5. **Fill out the PR template** when submitting.

---

## Project Layout

The key entry points for contributors:

| Area | Path |
|------|------|
| CLI commands | `oml/cli.py` |
| Parser registry | `oml/ingest/parsers/__init__.py` |
| Retrieval strategies | `oml/retrieval/` |
| Memory / TEEG | `oml/memory/` |
| Storage backends | `oml/storage/factory.py` |
| LLM providers | `oml/llm/factory.py` |
| Eval tasks | `oml/eval/tasks/` |

---

## Reporting Issues

Use the GitHub issue templates:

- **Bug report** — unexpected behavior, errors, or test failures
- **Feature request** — new parsers, retrieval strategies, or eval tasks

---

## License

By contributing you agree that your contributions will be licensed under the [Apache 2.0 License](../LICENSE).
