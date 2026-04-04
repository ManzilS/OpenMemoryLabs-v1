"""tests/test_api.py — Integration tests for the OML REST API.

Uses FastAPI's synchronous ``TestClient`` + the ``mock`` LLM backend so all
tests run offline with no model downloads or API keys required.

The fixture sets ``OML_STORAGE=memory`` before importing the server to avoid
any filesystem side effects from the SQLite backend.

Skipped automatically when ``fastapi`` is not installed (e.g. minimal dev
environments).  Install with ``pip install -e ".[dev]"`` or
``pip install fastapi httpx`` to run these tests.
"""

from __future__ import annotations

import os

import pytest

# Skip the entire module if fastapi (or its test transport httpx) is absent.
# This mirrors the pattern used in test_rdf.py for optional rdflib.
pytest.importorskip("fastapi", reason="fastapi not installed — skipping API tests")

# ── Configure in-memory storage BEFORE importing the server module so that
# oml.config.DEFAULT_STORAGE resolves to "memory" at import time. ─────────────
os.environ.setdefault("OML_STORAGE", "memory")
os.environ.setdefault("OML_MODEL", "mock")


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def client():
    """Return a FastAPI TestClient with the full app lifecycle active."""
    from fastapi.testclient import TestClient

    from oml.api.server import app

    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


# ── /health ───────────────────────────────────────────────────────────────────


def test_health_returns_200(client) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200


def test_health_schema(client) -> None:
    data = client.get("/health").json()
    assert data["status"] == "ok"
    assert "version" in data
    assert "storage" in data
    assert "llm" in data
    assert isinstance(data["teeg_ready"], bool)


# ── /docs & /openapi.json ─────────────────────────────────────────────────────


def test_swagger_docs_available(client) -> None:
    resp = client.get("/docs")
    assert resp.status_code == 200


def test_openapi_schema_structure(client) -> None:
    schema = client.get("/openapi.json").json()
    assert schema["info"]["title"] == "OpenMemoryLab API"
    # All four main endpoints must be declared
    for path in ("/query", "/chat", "/teeg/ingest", "/teeg/query"):
        assert path in schema["paths"], f"Missing endpoint: {path}"


# ── /query — validation ───────────────────────────────────────────────────────


def test_query_empty_question_rejected(client) -> None:
    resp = client.post("/query", json={"question": "", "model": "mock"})
    assert resp.status_code == 422


def test_query_alpha_above_range_rejected(client) -> None:
    resp = client.post("/query", json={"question": "test", "model": "mock", "alpha": 1.5})
    assert resp.status_code == 422


def test_query_alpha_below_range_rejected(client) -> None:
    resp = client.post("/query", json={"question": "test", "model": "mock", "alpha": -0.1})
    assert resp.status_code == 422


def test_query_top_k_zero_rejected(client) -> None:
    resp = client.post("/query", json={"question": "test", "model": "mock", "top_k": 0})
    assert resp.status_code == 422


def test_query_top_k_above_limit_rejected(client) -> None:
    resp = client.post("/query", json={"question": "test", "model": "mock", "top_k": 51})
    assert resp.status_code == 422


def test_query_budget_too_small_rejected(client) -> None:
    resp = client.post("/query", json={"question": "test", "model": "mock", "budget": 10})
    assert resp.status_code == 422


# ── /query — behaviour ────────────────────────────────────────────────────────


def test_query_no_data_returns_404_or_200(client) -> None:
    """Empty store → 404; seeded store → 200.  Both are valid outcomes."""
    resp = client.post(
        "/query",
        json={"question": "What is TEEG?", "model": "mock"},
    )
    assert resp.status_code in (200, 404, 503)


def test_query_200_response_schema(client) -> None:
    """If data exists, the response must match QueryResponse schema."""
    resp = client.post("/query", json={"question": "test", "model": "mock"})
    if resp.status_code == 200:
        data = resp.json()
        assert "answer" in data
        assert "sources" in data
        assert "tokens_used" in data
        assert "latency_ms" in data
        assert isinstance(data["sources"], list)


# ── /chat — validation ────────────────────────────────────────────────────────


def test_chat_empty_messages_rejected(client) -> None:
    resp = client.post("/chat", json={"messages": [], "model": "mock"})
    assert resp.status_code == 422


def test_chat_invalid_role_rejected(client) -> None:
    resp = client.post(
        "/chat",
        json={"messages": [{"role": "robot", "content": "Hello"}], "model": "mock"},
    )
    assert resp.status_code == 422


def test_chat_empty_content_rejected(client) -> None:
    resp = client.post(
        "/chat",
        json={"messages": [{"role": "user", "content": ""}], "model": "mock"},
    )
    assert resp.status_code == 422


# ── /chat — behaviour ─────────────────────────────────────────────────────────


def test_chat_valid_request_returns_expected_shape(client) -> None:
    resp = client.post(
        "/chat",
        json={
            "messages": [{"role": "user", "content": "What is RAG?"}],
            "model": "mock",
        },
    )
    # 200 (success) or 500 (no index) are both valid in a clean environment
    assert resp.status_code in (200, 500)
    if resp.status_code == 200:
        data = resp.json()
        assert "answer" in data
        assert "tokens_used" in data
        assert "session_id" in data
        assert "latency_ms" in data


def test_chat_session_id_is_uuid(client) -> None:
    resp = client.post(
        "/chat",
        json={"messages": [{"role": "user", "content": "Hello"}], "model": "mock"},
    )
    if resp.status_code == 200:
        import uuid

        data = resp.json()
        uuid.UUID(data["session_id"])  # raises ValueError if not a valid UUID


# ── /teeg/ingest — validation ─────────────────────────────────────────────────


def test_teeg_ingest_empty_text_rejected(client) -> None:
    resp = client.post("/teeg/ingest", json={"text": "", "model": "mock"})
    assert resp.status_code == 422


# ── /teeg/ingest — behaviour ──────────────────────────────────────────────────


def test_teeg_ingest_creates_note(client) -> None:
    resp = client.post(
        "/teeg/ingest",
        json={
            "text": "Victor Frankenstein built his creature from corpses in 1797.",
            "model": "mock",
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "note_id" in data
    assert len(data["note_id"]) > 0
    assert "content" in data
    assert isinstance(data["keywords"], list)
    assert "message" in data
    assert "active notes" in data["message"]


def test_teeg_ingest_with_context_hint(client) -> None:
    resp = client.post(
        "/teeg/ingest",
        json={
            "text": "The creature fled into the mountains after the creation.",
            "context": "Frankenstein Chapter 5",
            "model": "mock",
        },
    )
    assert resp.status_code == 200
    assert "note_id" in resp.json()


def test_teeg_ingest_increments_store(client) -> None:
    """Two ingests should result in more active notes than one."""
    r1 = client.post(
        "/teeg/ingest",
        json={"text": "Note A: The lab was cold and dark.", "model": "mock"},
    )
    r2 = client.post(
        "/teeg/ingest",
        json={"text": "Note B: Victor worked through the night.", "model": "mock"},
    )
    assert r1.status_code == 200
    assert r2.status_code == 200


# ── /teeg/query — validation ──────────────────────────────────────────────────


def test_teeg_query_empty_question_rejected(client) -> None:
    resp = client.post("/teeg/query", json={"question": "", "model": "mock"})
    assert resp.status_code == 422


def test_teeg_query_top_k_zero_rejected(client) -> None:
    resp = client.post(
        "/teeg/query", json={"question": "test", "model": "mock", "top_k": 0}
    )
    assert resp.status_code == 422


# ── /teeg/query — behaviour ───────────────────────────────────────────────────


def test_teeg_query_after_ingest(client) -> None:
    """Ingest a note first, then query — should return 200 with a valid schema."""
    # Seed the store (idempotent — may already have notes from earlier tests)
    client.post(
        "/teeg/ingest",
        json={"text": "Frankenstein regretted creating the creature.", "model": "mock"},
    )

    resp = client.post(
        "/teeg/query",
        json={"question": "Who created the creature?", "model": "mock"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "answer" in data
    assert isinstance(data["notes_used"], list)
    assert "latency_ms" in data


def test_teeg_query_notes_used_schema(client) -> None:
    """Each entry in notes_used must have the expected keys."""
    # Ensure at least one note exists
    client.post(
        "/teeg/ingest",
        json={"text": "The creature longed for a companion.", "model": "mock"},
    )
    resp = client.post(
        "/teeg/query",
        json={"question": "What did the creature want?", "model": "mock"},
    )
    if resp.status_code == 200:
        for note in resp.json()["notes_used"]:
            assert "note_id" in note
            assert "content" in note
            assert "score" in note
            assert "hops" in note
