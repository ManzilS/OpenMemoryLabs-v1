"""
OpenMemoryLab REST API demo.
=============================

Demonstrates every endpoint of the OML FastAPI server using ``httpx``:

  GET  /health          — server status and active configuration
  POST /query           — single-turn hybrid RAG query
  POST /chat            — multi-turn chat (stateless per-request session)
  POST /teeg/ingest     — distil text into an AtomicNote and store in graph
  POST /teeg/query      — query the TEEG evolving graph memory

Prerequisites
-------------
1. Start the API server in a separate terminal::

       oml api
       # or: uvicorn oml.api.server:app --reload

2. (Optional) Ingest some documents for the /query and /chat endpoints::

       oml ingest data/frankenstein.txt

3. Run this script::

       python scripts/api_demo.py

The script expects the server at http://localhost:8000 by default.
Set the OML_API_BASE environment variable to override.
"""

import json
import os
import sys
import textwrap
from pathlib import Path
from typing import Any

# Allow running directly from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import httpx
except ImportError:
    print("ERROR: httpx is required — install with:  pip install httpx")
    sys.exit(1)

# ── Configuration ─────────────────────────────────────────────────────────────

BASE_URL = os.getenv("OML_API_BASE", "http://localhost:8000")
TIMEOUT = 30.0  # seconds

# ── Helpers ───────────────────────────────────────────────────────────────────

_WIDTH = 72


def _banner(title: str) -> None:
    print()
    print("=" * _WIDTH)
    print(f"  {title}")
    print("=" * _WIDTH)


def _section(label: str) -> None:
    print(f"\n── {label} {'─' * max(0, _WIDTH - len(label) - 4)}")


def _pp(data: Any) -> None:
    """Pretty-print a dict/list as indented JSON."""
    print(textwrap.indent(json.dumps(data, indent=2, ensure_ascii=False), "    "))


def _request(
    client: httpx.Client,
    method: str,
    path: str,
    body: dict | None = None,
) -> dict:
    """Make a request and return the parsed JSON body, or print an error."""
    url = f"{BASE_URL}{path}"
    try:
        if method.upper() == "GET":
            resp = client.get(url, timeout=TIMEOUT)
        else:
            resp = client.post(url, json=body, timeout=TIMEOUT)

        if resp.status_code >= 400:
            print(f"  ✗ HTTP {resp.status_code} — {resp.text[:200]}")
            return {}

        return resp.json()

    except httpx.ConnectError:
        print(
            f"\n  ✗ Cannot connect to {BASE_URL}\n"
            f"    Start the server with:  oml api\n"
        )
        sys.exit(1)
    except Exception as exc:
        print(f"  ✗ Unexpected error: {exc}")
        return {}


# ── Demo sections ─────────────────────────────────────────────────────────────


def demo_health(client: httpx.Client) -> None:
    _section("GET /health — system status and configuration")
    data = _request(client, "GET", "/health")
    if data:
        print(f"  status  : {data.get('status')}")
        print(f"  version : {data.get('version')}")
        print(f"  storage : {data.get('storage')}")
        print(f"  llm     : {data.get('llm')}")
        print(f"  teeg    : {'ready ✓' if data.get('teeg_ready') else 'no notes yet'}")


def demo_query(client: httpx.Client) -> None:
    _section("POST /query — single-turn hybrid RAG query")

    payload = {
        "question": "Who created the creature, and how was it brought to life?",
        "top_k": 3,
        "alpha": 0.5,        # balanced BM25 + vector
        "budget": 2000,
        "use_rerank": True,
        "use_hyde": False,
    }
    print(f"  question : {payload['question']!r}")
    print(f"  alpha    : {payload['alpha']}  (BM25 ↔ vector balance)")

    data = _request(client, "POST", "/query", payload)
    if data:
        print(f"\n  Answer:\n")
        answer = data.get("answer", "")
        print(textwrap.indent(textwrap.fill(answer, 68), "    "))
        print(f"\n  Sources retrieved: {len(data.get('sources', []))}")
        for src in data.get("sources", [])[:2]:
            snippet = src["text"][:80].replace("\n", " ")
            print(f"    [{src['chunk_id']}  score={src['score']:.3f}]  {snippet}…")
        print(f"\n  tokens_used : {data.get('tokens_used')}")
        print(f"  latency_ms  : {data.get('latency_ms')}")


def demo_chat(client: httpx.Client) -> None:
    _section("POST /chat — multi-turn RAG chat (stateless session)")

    payload = {
        "messages": [
            {"role": "user", "content": "Tell me about Victor Frankenstein."},
        ],
        "top_k": 3,
        "alpha": 0.5,
        "budget": 1500,
    }
    print(f"  message : {payload['messages'][-1]['content']!r}")

    data = _request(client, "POST", "/chat", payload)
    if data:
        answer = data.get("answer", "")
        print(f"\n  Answer:\n")
        print(textwrap.indent(textwrap.fill(answer, 68), "    "))
        print(f"\n  session_id  : {data.get('session_id')}")
        print(f"  tokens_used : {data.get('tokens_used')}")
        print(f"  latency_ms  : {data.get('latency_ms')}")


def demo_teeg(client: httpx.Client) -> None:
    """Demonstrate the TEEG evolving graph memory endpoints."""

    _section("POST /teeg/ingest — distil text into AtomicNote")

    facts = [
        {
            "text": "Victor Frankenstein assembled the creature from corpse parts over two years.",
            "context": "Frankenstein, Chapter 4",
        },
        {
            "text": "The creature was brought to life on a dark, stormy November night.",
            "context": "Frankenstein, Chapter 5",
        },
        {
            "text": "The creature learned to speak and read by secretly observing the De Lacey family.",
            "context": "Frankenstein, Chapter 15",
        },
    ]

    note_ids = []
    for fact in facts:
        data = _request(client, "POST", "/teeg/ingest", fact)
        if data:
            note_ids.append(data.get("note_id", ""))
            kws = ", ".join(data.get("keywords", [])[:4])
            print(f"  ✓ note_id : {data.get('note_id')}")
            print(f"    content : {data.get('content', '')[:80]}")
            print(f"    keywords: {kws}")
            print(f"    {data.get('message', '')}")
            print()

    _section("POST /teeg/query — BFS graph traversal + LLM answer")

    questions = [
        "Who created the creature and how?",
        "How did the creature acquire language and knowledge?",
    ]

    for q in questions:
        payload = {"question": q, "top_k": 5}
        print(f"  Q: {q!r}")
        data = _request(client, "POST", "/teeg/query", payload)
        if data:
            print(f"\n  Answer:\n")
            print(textwrap.indent(textwrap.fill(data.get("answer", ""), 68), "    "))

            notes = data.get("notes_used", [])
            print(f"\n  Notes retrieved: {len(notes)}")
            for n in notes[:3]:
                label = "seed" if n.get("hops", 0) == 0 else f"hop-{n['hops']}"
                snippet = n.get("content", "")[:70]
                print(f"    [{label}  score={n.get('score', 0):.3f}]  {snippet}")
            print(f"\n  latency_ms : {data.get('latency_ms')}")
            print()


def demo_openapi(client: httpx.Client) -> None:
    _section("GET /docs — Swagger UI (open in your browser)")
    print(f"  {BASE_URL}/docs")
    print(f"  {BASE_URL}/redoc")

    # Verify the OpenAPI schema is served
    try:
        resp = client.get(f"{BASE_URL}/openapi.json", timeout=5.0)
        if resp.status_code == 200:
            schema = resp.json()
            paths = list(schema.get("paths", {}).keys())
            print(f"\n  OpenAPI schema OK — {len(paths)} paths: {', '.join(paths)}")
    except Exception:
        pass


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    _banner("OpenMemoryLab REST API Demo")
    print(f"  Server: {BASE_URL}")

    with httpx.Client() as client:
        demo_health(client)
        demo_query(client)
        demo_chat(client)
        demo_teeg(client)
        demo_openapi(client)

    print()
    print("=" * _WIDTH)
    print("  Demo complete.")
    print(f"  Full docs → {BASE_URL}/docs")
    print("=" * _WIDTH)
    print()


if __name__ == "__main__":
    main()
