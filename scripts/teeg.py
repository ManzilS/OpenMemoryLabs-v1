"""
TEEG — TOON-Encoded Evolving Graph memory demo.
================================================

Runs all TEEG steps in order:

  1. Ingest raw facts → LLM distils each into an AtomicNote (TOON format)
  2. MemoryEvolver resolves contradictions at write time, links related notes
  3. TEEGStore persists notes (JSON-Lines) + relation graph (NetworkX DiGraph)
  4. ScoutRetriever traverses the graph to answer queries

Run:
    python scripts/teeg.py

Or via the CLI (same underlying classes):
    oml teeg-ingest "Victor Frankenstein created the creature."
    oml teeg-query  "Who built the monster?"
"""

import sys
from pathlib import Path

# Ensure the project root is on the path when run directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from oml.memory.atomic_note import AtomicNote
from oml.memory.evolver import MemoryEvolver
from oml.storage.teeg_store import TEEGStore
from oml.retrieval.scout import ScoutRetriever

# ── configuration ─────────────────────────────────────────────────────────────

STORE_DIR = Path("teeg_store_demo")
MODEL = "mock"   # swap to "ollama:qwen3:4b" for a real LLM

# Sample facts from Frankenstein — demonstrates contradiction resolution
# (note how facts 3 & 4 contradict each other; evolver resolves this)
FACTS = [
    "Victor Frankenstein assembled the creature from corpse parts over two years.",
    "The creature was brought to life on a stormy November night.",
    "Victor fled his laboratory immediately after the creature first opened its eyes.",
    "Victor calmly observed the creature for several minutes before leaving.",  # contradicts #3
    "The creature learned to speak and read by secretly observing the De Lacey family.",
    "Walton encountered Victor adrift on the Arctic ice, near death.",
]

QUERIES = [
    "Who created the creature and how?",
    "What did Victor do right after the creature came to life?",
    "How did the creature learn language?",
]


def run():
    print("=" * 60)
    print("  TEEG — TOON-Encoded Evolving Graph Memory Demo")
    print("=" * 60)

    # ── Step 1: set up store + evolver ────────────────────────────────────
    print(f"\n[1/4] Initialising TEEGStore at '{STORE_DIR}' …")
    store = TEEGStore(artifacts_dir=STORE_DIR)
    evolver = MemoryEvolver(store, model_name=MODEL)

    # ── Step 2: ingest facts via MemoryEvolver ────────────────────────────
    print(f"\n[2/4] Ingesting {len(FACTS)} facts (model='{MODEL}') …")
    notes = []
    for i, fact in enumerate(FACTS, 1):
        # Build a heuristic AtomicNote (mock LLM returns plain text, not TOON)
        words = fact.split()
        note = AtomicNote(
            content=" ".join(words[:40]),
            context="Frankenstein demo",
            keywords=list({w.lower().strip(".,!?;:") for w in words if len(w) > 3})[:6],
            tags=["frankenstein", "demo"],
            confidence=0.9,
        )
        evolver.evolve(note)
        notes.append(note)
        status = "active" if note.active else "archived"
        print(f"  [{i}] {note.note_id}  ({status})  {fact[:60]}…")

    # ── Step 3: show store stats + audit ──────────────────────────────────
    print("\n[3/4] Store stats after ingestion:")
    stats = store.stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")

    audit = evolver.audit()
    print("\nEvolution audit:")
    for k, v in audit.items():
        print(f"  {k}: {v}")

    # ── Step 4: query with ScoutRetriever ─────────────────────────────────
    print("\n[4/4] Querying with ScoutRetriever …")
    scout = ScoutRetriever(store, seed_k=2, max_hops=2)

    for query in QUERIES:
        print(f"\n  Q: {query}")
        results = scout.search(query, top_k=3)
        if not results:
            print("  (no results)")
            continue
        for note, score, hops in results:
            label = "seed" if hops == 0 else f"hop-{hops}"
            print(f"    [{label}  score={score:.3f}]  {note.content[:80]}")

    # Show traversal explanation for the first query
    print(f"\nScout explain for: {QUERIES[0]!r}")
    print(scout.explain(QUERIES[0], top_k=3))

    # ── persist ──────────────────────────────────────────────────────────
    store.save()
    print(f"\nStore saved to '{STORE_DIR}/'")
    print("  teeg_notes.jsonl  — one AtomicNote per line (human-readable)")
    print("  teeg_graph.pkl    — NetworkX DiGraph with relation-labelled edges")
    print("\nDone.")


if __name__ == "__main__":
    run()
