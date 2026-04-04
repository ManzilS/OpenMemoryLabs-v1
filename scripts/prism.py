"""scripts/prism.py — End-to-end PRISM demo script.

Demonstrates all three PRISM efficiency layers running on the Frankenstein
corpus (same data set as scripts/teeg.py for easy comparison):

  Layer 1 — SketchGate  (write-time near-duplicate deduplication via MinHash LSH)
  Layer 2 — DeltaStore  (semantic patch storage for EXTENDS notes)
  Layer 3 — CallBatcher (N-to-1 LLM call coalescing for bulk ingestion)

Run with:
    python scripts/prism.py

No API key required — uses the mock LLM by default.
Set OML_MODEL=ollama:qwen3:4b (or any other provider) for a real LLM.
"""

import os
import shutil
import sys
import time
from pathlib import Path

# ── ensure project root on sys.path ──────────────────────────────────────────
ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from oml.memory.prism_pipeline import PRISMPipeline

# ── config ────────────────────────────────────────────────────────────────────

MODEL = os.getenv("OML_MODEL", "mock")
ARTIFACTS_DIR = Path("prism_demo_store")

# ── corpus: five Frankenstein passages + two near-duplicates ──────────────────
TEXTS = [
    "Victor Frankenstein spent two years assembling the creature from charnel-house remains.",
    "The creature was brought to life on a stormy November night using electrical apparatus.",
    "Victor fled his laboratory immediately after the creature opened its watery eyes.",
    "The creature, abandoned and alone, wandered into the forest to survive by itself.",
    "Victor's friend Henry Clerval arrived in Ingolstadt to help nurse Victor back to health.",
]

# Two near-duplicates of TEXTS[0] and TEXTS[1] — SketchGate should catch these
NEAR_DUPLICATES = [
    "Victor Frankenstein spent two years building the creature from charnel-house remains.",  # dup of [0]
    "The creature was brought to life on a stormy November night using electrical equipment.",  # dup of [1]
]

# Three additional texts for batch ingestion demo
BATCH_TEXTS = [
    "The creature taught himself to read by secretly observing a family in a forest cottage.",
    "Victor agreed to create a female companion for the creature but later destroyed his work.",
    "The creature pursued Victor across Europe and the Arctic, swearing eternal revenge.",
]


def hr(char: str = "─", width: int = 60) -> str:
    return char * width


def section(title: str) -> None:
    print(f"\n{hr('═')}")
    print(f"  {title}")
    print(hr("═"))


def main() -> None:
    # ── clean previous run ────────────────────────────────────────────────────
    if ARTIFACTS_DIR.exists():
        shutil.rmtree(ARTIFACTS_DIR)

    print("\n" + hr("▓"))
    print("  PRISM Demo — Probabilistic Retrieval with Intelligent Sparse Memory")
    print(hr("▓"))
    print(f"  Model: {MODEL}   |   Store: {ARTIFACTS_DIR}")

    pipeline = PRISMPipeline(
        artifacts_dir=ARTIFACTS_DIR,
        model=MODEL,
        dedup_threshold=0.5,  # lower threshold for demo (mock LLM keywords are generic)
    )

    # ═══════════════════════════════════════════════════════════════════════
    # Layer 1 demo: single ingest + near-duplicate detection
    # ═══════════════════════════════════════════════════════════════════════
    section("LAYER 1 — SketchGate: Write-time Near-Duplicate Detection")

    print("\n  Ingesting 5 unique Frankenstein passages…")
    t0 = time.perf_counter()
    for text in TEXTS:
        result = pipeline.ingest(text)
        status = "STORED" if not result.was_deduplicated else f"DEDUPED → {result.merged_into}"
        print(f"  [{status}]  {text[:60]}…")
    elapsed = time.perf_counter() - t0

    print(f"\n  ✓  {pipeline._store.active_count()} unique notes stored in {elapsed:.2f}s")

    print("\n  Now ingesting 2 near-duplicate passages (should be caught by SketchGate)…")
    for text in NEAR_DUPLICATES:
        result = pipeline.ingest(text)
        status = (
            f"DEDUPED → {result.merged_into}"
            if result.was_deduplicated
            else "STORED  (below threshold)"
        )
        print(f"  [{status}]  {text[:60]}…")

    s = pipeline._sketch.stats()
    print(f"\n  SketchGate stats:")
    print(f"    Checks:         {s['checks_total']}")
    print(f"    Skips:          {s['skips_total']}")
    print(f"    Dedup rate:     {s['dedup_rate']:.1%}")
    print(f"    Registered:     {s['registered_notes']} notes")

    # ═══════════════════════════════════════════════════════════════════════
    # Layer 3 demo: batch ingestion (1 LLM call instead of 2N)
    # ═══════════════════════════════════════════════════════════════════════
    section("LAYER 3 — CallBatcher: N-to-1 LLM Call Coalescing")

    print(f"\n  Batch-ingesting {len(BATCH_TEXTS)} texts…")
    print(f"  Naive cost:   {2 * len(BATCH_TEXTS)} LLM calls")
    print(f"  PRISM cost:   2 LLM calls (1 distil batch + 1 evolve batch)")

    t0 = time.perf_counter()
    batch = pipeline.batch_ingest(BATCH_TEXTS)
    elapsed = time.perf_counter() - t0

    print(f"\n  Batch result:")
    print(f"    Notes created:   {len(batch.notes)}")
    print(f"    LLM calls made:  {batch.llm_calls_made}")
    print(f"    LLM calls saved: {batch.llm_calls_saved}")
    print(f"    Call efficiency: {batch.call_efficiency:.1%}")
    print(f"    Delta notes:     {batch.delta_count}")
    print(f"    Dedup skips:     {batch.dedup_count}")
    print(f"    Elapsed:         {elapsed:.2f}s")

    # ═══════════════════════════════════════════════════════════════════════
    # Layer 2 demo: DeltaStore patches
    # ═══════════════════════════════════════════════════════════════════════
    section("LAYER 2 — DeltaStore: Semantic Patch Storage")

    delta_stats = pipeline._delta.stats()
    print(f"\n  DeltaStore stats:")
    print(f"    Base notes with patches: {delta_stats['bases_with_patches']}")
    print(f"    Total patches stored:    {delta_stats['total_patches']}")
    print(f"    Token savings (est.):    ~{delta_stats['token_savings_est']} tokens/query")

    if pipeline._delta.count() > 0:
        patched_ids = pipeline._delta.get_all_patched_note_ids()
        base_id = patched_ids[0]
        base_note = pipeline._store.get(base_id)
        if base_note:
            reconstructed = pipeline._delta.reconstruct(base_id, base_note)
            print(f"\n  Example reconstruction:")
            print(f"    Base:            {base_note.content[:70]}")
            print(f"    Reconstructed:   {reconstructed[:120]}")
    else:
        print("  (No delta patches in this run — try with a real LLM for EXTENDS verdicts)")

    # ═══════════════════════════════════════════════════════════════════════
    # Query demo
    # ═══════════════════════════════════════════════════════════════════════
    section("QUERY — Scout + TieredContextPacker")

    pipeline.save()
    question = "Who created the creature and what happened afterwards?"
    print(f"\n  Question: {question!r}")
    print(f"  Active notes in store: {pipeline._store.active_count()}")

    t0 = time.perf_counter()
    answer, context = pipeline.query(question, top_k=5)
    elapsed = time.perf_counter() - t0

    print(f"\n  Answer:\n  {answer[:300]}")
    print(f"\n  Context tokens (est.): {len(context) // 4}")
    print(f"  Query latency:         {elapsed * 1000:.0f}ms")

    # ═══════════════════════════════════════════════════════════════════════
    # Final aggregate stats
    # ═══════════════════════════════════════════════════════════════════════
    section("AGGREGATE PRISM STATS")

    stats = pipeline.stats()
    print(f"\n  Store:")
    print(f"    Total notes:        {stats.total_notes}")
    print(f"    Active notes:       {stats.active_notes}")
    print(f"    Delta patches:      {stats.delta_notes}")
    print(f"\n  Efficiency:")
    print(f"    Dedup rate:         {stats.dedup_rate:.1%}")
    print(f"    Avg call efficiency:{stats.avg_call_efficiency:.1%}")
    print(f"    Token savings est.: ~{stats.token_savings_est} tokens/query")
    print(f"\n  Configuration:")
    print(f"    Bloom FP rate:      {stats.bloom_fp_rate:.1%}")
    print(f"    MinHash threshold:  {stats.minhash_threshold}")

    print(f"\n{hr('▓')}")
    print("  PRISM demo complete.")
    print(hr("▓") + "\n")

    # Clean up demo store
    shutil.rmtree(ARTIFACTS_DIR, ignore_errors=True)


if __name__ == "__main__":
    main()
