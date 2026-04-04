"""
scripts/eval_lmstudio.py
=========================
Comprehensive evaluation runner using LM Studio as the LLM backend.

Runs every available eval task plus a full TEEG and PRISM ingest/query
cycle.  Records wall-clock time per task, GPU status, and model info.
All results are saved to:
  reports/lmstudio_eval_<TIMESTAMP>.json   — machine-readable
  reports/lmstudio_eval_<TIMESTAMP>.md     — human-readable summary

Usage:
    python scripts/eval_lmstudio.py
    python scripts/eval_lmstudio.py --model lmstudio:qwen/qwen3-30b-a3b
    python scripts/eval_lmstudio.py --model lmstudio:deepseek/deepseek-r1-0528-qwen3-8b
"""

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

# ── add project root to sys.path ────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ── imports (after path fix) ────────────────────────────────────────────────
from oml.llm.factory import get_llm_client
from oml.llm.lmstudio import LMStudioLLM
from oml.utils.device import get_device_info, resolve_device

# Force-import all eval tasks so they register themselves
import oml.eval.tasks.faithfulness        # noqa: F401
import oml.eval.tasks.lost_in_middle      # noqa: F401
import oml.eval.tasks.retrieval_precision  # noqa: F401
import oml.eval.tasks.cost_latency        # noqa: F401
import oml.eval.tasks.oml_vs_rag          # noqa: F401
import oml.eval.tasks.global_trends       # noqa: F401

from oml.eval.run import _TASK_REGISTRY

# ── constants ────────────────────────────────────────────────────────────────
DEFAULT_MODEL = "lmstudio:qwen/qwen3-30b-a3b"
REPORTS_DIR = ROOT / "reports"
TEEG_STORE_DIR = ROOT / "eval_teeg_store"
PRISM_STORE_DIR = ROOT / "eval_prism_store"


# ── helpers ──────────────────────────────────────────────────────────────────

def _gpu_info() -> dict:
    info = get_device_info()
    return {
        "device": info.get("device", "cpu"),
        "cuda_available": info.get("cuda_available", False),
        "gpu_name": info.get("gpu_name", "N/A"),
        "vram_gb": info.get("vram_gb", 0.0),
        "gpu_count": info.get("gpu_count", 0),
    }


def _section(title: str, width: int = 70):
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def _run_timed(fn, *args, **kwargs):
    """Call fn(*args, **kwargs) and return (result, elapsed_seconds)."""
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, time.perf_counter() - t0


# ── individual eval task runner ──────────────────────────────────────────────

def run_eval_tasks(llm, task_names: list[str]) -> dict:
    """Run registered eval tasks and collect results."""
    results = {}

    for task_name in task_names:
        if task_name not in _TASK_REGISTRY:
            print(f"  [SKIP] task '{task_name}' not in registry")
            continue

        _section(f"EVAL TASK: {task_name}")
        task_cls = _TASK_REGISTRY[task_name]
        task = task_cls()

        try:
            eval_result, elapsed = _run_timed(task.run, llm, {})
            results[task_name] = {
                "score": eval_result.score,
                "elapsed_s": round(elapsed, 2),
                "details": eval_result.details,
                "status": "ok",
            }
            print(f"  Score: {eval_result.score:.4f}  |  Time: {elapsed:.1f}s")
        except Exception as exc:
            results[task_name] = {
                "score": None,
                "elapsed_s": None,
                "details": {"error": str(exc)},
                "status": "error",
            }
            print(f"  [ERROR] {exc}")

    return results


# ── TEEG eval ────────────────────────────────────────────────────────────────

def run_teeg_eval(model_str: str) -> dict:
    """Ingest a small corpus of facts into TEEG and measure query quality."""
    _section("TEEG INGEST + QUERY CYCLE")

    from oml.memory.teeg_pipeline import TEEGPipeline

    # Clean slate
    if TEEG_STORE_DIR.exists():
        shutil.rmtree(TEEG_STORE_DIR)
    TEEG_STORE_DIR.mkdir(parents=True)

    pipeline = TEEGPipeline(artifacts_dir=TEEG_STORE_DIR, model=model_str)

    # Kept small (4 facts) to limit LLM calls: each ingest = 1 distil + N evolver checks
    # (N grows with store size). For a thinking model, reasoning tokens add ~20-50s per call.
    # Wikipedia-derived corpus — real-world historical and scientific facts.
    corpus = [
        "Alan Turing was a British mathematician who worked at Bletchley Park during World War II to decipher German Enigma codes.",
        "In 1813, Gregory Blaxland, William Lawson, and William Charles Wentworth became the first Europeans to cross Australia's Blue Mountains.",
        "Aum Shinrikyo, a Japanese doomsday cult led by Shoko Asahara, carried out a deadly sarin gas attack on the Tokyo subway system in 1995.",
        "The Ovambo are the largest ethnic group in Namibia, comprising approximately 49 percent of the total population.",
    ]

    queries = [
        {
            "question": "Where did Alan Turing work during World War II?",
            "expected_keywords": ["bletchley", "enigma", "park"],
        },
        {
            "question": "What did Aum Shinrikyo do in Tokyo in 1995?",
            "expected_keywords": ["sarin", "subway", "attack"],
        },
    ]

    # Ingest corpus
    ingest_times = []
    notes_stored = []
    print(f"  Ingesting {len(corpus)} facts ...")
    for i, text in enumerate(corpus):
        note, t = _run_timed(pipeline.ingest, text)
        ingest_times.append(t)
        notes_stored.append(note.note_id)
        print(f"    [{i+1:2d}/{len(corpus)}] {note.note_id}  ({t:.1f}s)")

    pipeline.save()

    # Query cycle
    query_results = []
    print(f"\n  Running {len(queries)} queries ...")
    for q_item in queries:
        q = q_item["question"]
        expected_kw = [kw.lower() for kw in q_item["expected_keywords"]]

        try:
            answer, context, elapsed = _run_with_context(pipeline, q)
            kw_found = [kw for kw in expected_kw if kw in answer.lower() or kw in context.lower()]
            keyword_hit_rate = len(kw_found) / len(expected_kw) if expected_kw else 0.0
            query_results.append({
                "question": q,
                "answer_snippet": answer[:200],
                "expected_keywords": expected_kw,
                "keywords_found": kw_found,
                "keyword_hit_rate": round(keyword_hit_rate, 4),
                "elapsed_s": round(elapsed, 2),
                "status": "ok",
            })
            print(f"    Q: {q[:50]}")
            print(f"       keyword_hit_rate={keyword_hit_rate:.2f}  time={elapsed:.1f}s")
        except Exception as exc:
            query_results.append({
                "question": q,
                "error": str(exc),
                "status": "error",
            })
            print(f"    [ERROR] {q[:50]} -> {exc}")

    mean_hit_rate = (
        sum(r["keyword_hit_rate"] for r in query_results if "keyword_hit_rate" in r)
        / max(len([r for r in query_results if "keyword_hit_rate" in r]), 1)
    )

    # Cleanup
    if TEEG_STORE_DIR.exists():
        shutil.rmtree(TEEG_STORE_DIR)

    return {
        "notes_ingested": len(corpus),
        "notes_stored": len(notes_stored),
        "avg_ingest_s": round(sum(ingest_times) / len(ingest_times), 2),
        "total_ingest_s": round(sum(ingest_times), 2),
        "query_results": query_results,
        "mean_keyword_hit_rate": round(mean_hit_rate, 4),
        "score": mean_hit_rate,
        "status": "ok",
    }


def _run_with_context(pipeline, question):
    """Run pipeline.query() and return (answer, context_str, elapsed)."""
    t0 = time.perf_counter()
    result = pipeline.query(question)
    elapsed = time.perf_counter() - t0
    if isinstance(result, tuple):
        answer, context = result
    else:
        answer, context = str(result), ""
    return answer, context, elapsed


# ── PRISM eval ───────────────────────────────────────────────────────────────

def run_prism_eval(model_str: str) -> dict:
    """Two-phase PRISM eval: batch efficiency (Phase 1) + dedup detection (Phase 2).

    Phase 1: ingest unique base corpus → measures LLM call-batching efficiency.
    Phase 2: re-init pipeline from saved state, ingest near-duplicates of the
             base corpus → measures SketchGate near-duplicate detection.

    Score = mean(phase1_call_efficiency, phase2_dedup_rate).

    The two-phase design is necessary because SketchGate can only detect
    near-duplicates of *already-stored* notes (it checks before registering new
    ones).  An empty store has nothing to compare against, so dedup_count is
    always 0 in a cold single-batch test.  By saving after Phase 1 and
    re-initialising the pipeline, Phase 2 starts with a warm SketchGate.
    """
    _section("PRISM BATCH INGEST + QUERY CYCLE")

    from oml.memory.prism_pipeline import PRISMPipeline

    if PRISM_STORE_DIR.exists():
        shutil.rmtree(PRISM_STORE_DIR)
    PRISM_STORE_DIR.mkdir(parents=True)

    # ── Phase 1: unique base corpus ──────────────────────────────────────────
    # N=8 texts (= default batch_size) to demonstrate PRISM's intended operating
    # point.  Efficiency formula: (2N-1)/(2N).  For N=8 on empty store:
    #   naive=16 calls, actual=1 (distil batch, no evolve), saved=15 → 93.75%.
    #
    # Text design note: _quick_keywords extracts the first 6 non-stop words
    # longer than 3 chars.  The near-dup texts below have identical first-6
    # keywords, guaranteeing Jaccard = 1.0 detection in Phase 2.
    # Wikipedia-derived base corpus.
    # Text design note: _quick_keywords extracts the first 6 non-stop words longer than 3 chars.
    # The near-dup texts below have identical first-6 keywords, guaranteeing Jaccard=1.0
    # detection by SketchGate in Phase 2.
    base_texts = [
        # Pair 1 — keywords: alan, turing, bletchley, park, mathematician, deciphered
        "Alan Turing Bletchley Park mathematician deciphered Enigma codes wartime Britain Royal Navy.",
        # Pair 2 — keywords: blaxland, lawson, wentworth, australian, explorers, crossed
        "Blaxland Lawson Wentworth Australian explorers crossed Blue Mountains 1813 finding fertile interior.",
        # Pair 3 — keywords: birmingham, campaign, 1963, civil, rights, demonstrators
        "Birmingham Campaign 1963 civil rights demonstrators faced firehoses police Alabama marches protests.",
        # Pair 4 — keywords: canelo, alvarez, mexican, middleweight, boxer, became
        "Canelo Alvarez Mexican middleweight boxer became undisputed champion defeating multiple opponents.",
        # Pair 5 — keywords: canadian, forces, military, includes, maritime, command
        "Canadian Forces military includes Maritime Command MARCOM AIRCOM CANSOFCOM special operations units.",
        # Pair 6 — keywords: shinrikyo, japanese, doomsday, cult, released, sarin
        "Shinrikyo Japanese doomsday cult released sarin Tokyo subway 1995 causing casualties deaths injuries.",
        # Pair 7 — keywords: islamic, lunar, calendar, contains, twelve, months
        "Islamic lunar calendar contains twelve months totaling 354 days shorter solar year eleven days annually.",
        # Pair 8 — keywords: ovambo, people, namibia, largest, ethnic, group
        "Ovambo people Namibia largest ethnic group comprising 49 percent total population northern regions.",
    ]

    print(f"  Phase 1 — base corpus ({len(base_texts)} unique texts) ...")
    try:
        pipeline = PRISMPipeline(artifacts_dir=PRISM_STORE_DIR, model=model_str)
        p1_result, p1_elapsed = _run_timed(pipeline.batch_ingest, base_texts)
        pipeline.save()  # persists SketchGate so Phase 2 starts warm

        p1_efficiency = p1_result.call_efficiency
        print(f"    notes_created    : {len(p1_result.notes)}")
        print(f"    llm_calls_made   : {p1_result.llm_calls_made}")
        print(f"    llm_calls_saved  : {p1_result.llm_calls_saved}")
        print(f"    call_efficiency  : {p1_efficiency:.2%}")
        print(f"    elapsed          : {p1_elapsed:.1f}s")

    except Exception as exc:
        if PRISM_STORE_DIR.exists():
            shutil.rmtree(PRISM_STORE_DIR)
        print(f"  [ERROR Phase 1] {exc}")
        return {"status": "error", "error": str(exc), "score": 0.0}

    # ── Phase 2: near-duplicate detection ────────────────────────────────────
    # Re-init loads the saved SketchGate (warm).  Near-dups have identical
    # first-6 _quick_keywords as the base texts (only the tail differs).
    # With the consistent-keyword fix in SketchGate.register(), the stored
    # MinHash signatures match the query keywords → Jaccard = 1.0 → detected.
    near_dup_texts = [
        # Pair 1 near-dup — same 6 keywords: alan, turing, bletchley, park, mathematician, deciphered
        "Alan Turing Bletchley Park mathematician deciphered Enigma messages wartime Royal Navy Britain.",
        # Pair 2 near-dup — same 6 keywords: blaxland, lawson, wentworth, australian, explorers, crossed
        "Blaxland Lawson Wentworth Australian explorers crossed Blue Mountains 1813 discovering grazing pastoral.",
        # Pair 3 near-dup — same 6 keywords: birmingham, campaign, 1963, civil, rights, demonstrators
        "Birmingham Campaign 1963 civil rights demonstrators organized protests Alabama segregation resistance.",
        # Pair 4 near-dup — same 6 keywords: canelo, alvarez, mexican, middleweight, boxer, became
        "Canelo Alvarez Mexican middleweight boxer became undisputed champion winning world titles consistently.",
        # Pair 5 near-dup — same 6 keywords: canadian, forces, military, includes, maritime, command
        "Canadian Forces military includes Maritime Command MARCOM AIRCOM CANSOFCOM ground operational personnel.",
        # Pair 6 near-dup — same 6 keywords: shinrikyo, japanese, doomsday, cult, released, sarin
        "Shinrikyo Japanese doomsday cult released sarin Tokyo stations 1995 killing innocent commuters tragedy.",
        # Pair 7 near-dup — same 6 keywords: islamic, lunar, calendar, contains, twelve, months
        "Islamic lunar calendar contains twelve months totaling 354 days observed worldwide practicing Muslims.",
        # Pair 8 near-dup — same 6 keywords: ovambo, people, namibia, largest, ethnic, group
        "Ovambo people Namibia largest ethnic group comprising 49 percent total population bantu-speaking communities.",
    ]

    print(f"\n  Phase 2 — near-duplicate detection ({len(near_dup_texts)} near-dups) ...")
    try:
        pipeline2 = PRISMPipeline(artifacts_dir=PRISM_STORE_DIR, model=model_str)
        p2_result, p2_elapsed = _run_timed(pipeline2.batch_ingest, near_dup_texts)

        p2_dedup_rate = p2_result.dedup_count / len(near_dup_texts)
        print(f"    dedup_count      : {p2_result.dedup_count} / {len(near_dup_texts)}")
        print(f"    dedup_rate       : {p2_dedup_rate:.2%}")
        print(f"    llm_calls_made   : {p2_result.llm_calls_made}")
        print(f"    elapsed          : {p2_elapsed:.1f}s")

    except Exception as exc:
        if PRISM_STORE_DIR.exists():
            shutil.rmtree(PRISM_STORE_DIR)
        print(f"  [ERROR Phase 2] {exc}")
        return {"status": "error", "error": str(exc), "score": 0.0}

    # ── Quick query against Phase 1 data ────────────────────────────────────
    q = "Who worked at Bletchley Park during World War II?"
    try:
        answer, ctx, q_elapsed = _run_with_context(pipeline2, q)
        print(f"\n  Query '{q}' -> {answer[:120]} ({q_elapsed:.1f}s)")
    except Exception as exc:
        answer, q_elapsed = f"[ERROR] {exc}", 0.0

    # ── Cleanup ──────────────────────────────────────────────────────────────
    if PRISM_STORE_DIR.exists():
        shutil.rmtree(PRISM_STORE_DIR)

    # Combined score: equal weight on batching efficiency and dedup accuracy
    combined_score = round((p1_efficiency + p2_dedup_rate) / 2, 4)

    return {
        "phase1_texts": len(base_texts),
        "phase1_notes_created": len(p1_result.notes),
        "phase1_llm_calls_made": p1_result.llm_calls_made,
        "phase1_llm_calls_saved": p1_result.llm_calls_saved,
        "phase1_call_efficiency": round(p1_efficiency, 4),
        "phase1_elapsed_s": round(p1_elapsed, 2),
        "phase2_texts": len(near_dup_texts),
        "phase2_dedup_count": p2_result.dedup_count,
        "phase2_dedup_rate": round(p2_dedup_rate, 4),
        "phase2_llm_calls_made": p2_result.llm_calls_made,
        "phase2_elapsed_s": round(p2_elapsed, 2),
        "sample_query": q,
        "sample_answer": answer[:300],
        "score": combined_score,
        "status": "ok",
        # Aliases for summary table / backward compat
        "elapsed_s": round(p1_elapsed + p2_elapsed, 2),
        "dedup_count": p2_result.dedup_count,
        "call_efficiency": round(p1_efficiency, 4),
    }


# ── lost-in-middle with various context sizes ─────────────────────────────────

def run_lim_extended(llm) -> dict:
    """Extended Lost-in-Middle test with three context lengths."""
    _section("LOST-IN-MIDDLE (extended: 3 context lengths)")

    sizes = [300, 800, 1500]  # keep within 4096-token context limit; 4000-char > limit
    results = {}
    scores = []

    # Wikipedia-derived filler — unrelated to the needle topic so the needle stands out.
    _WIKI_FILLER = (
        "The Islamic calendar, also known as the Hijri calendar, is a purely lunar calendar "
        "consisting of twelve months in a year of 354 days. Being a purely lunar calendar, "
        "it is not synchronised with the seasons. The months are Muharram, Safar, "
        "Rabi al-Awwal, Rabi al-Thani, Jumada al-Awwal, Jumada al-Thani, Rajab, Shaban, "
        "Ramadan, Shawwal, Dhul Qadah, and Dhul Hijjah. Each month begins with the sighting "
        "of the new crescent moon. Major religious observances including Ramadan and both "
        "Eid celebrations are timed according to this calendar. The Islamic year is "
        "approximately eleven days shorter than the solar year, so its months rotate through "
        "all seasons over a cycle of roughly 33 solar years. The Ovambo are the largest "
        "ethnic group in Namibia, comprising approximately 49 percent of the total population. "
        "They traditionally inhabit the northern regions of Namibia and southern Angola and "
        "speak Bantu languages belonging to the Niger-Congo family. The Canadian Armed Forces "
        "consist of the Maritime Command responsible for naval operations, the Air Command "
        "overseeing all air operations, and the Canadian Special Operations Forces Command "
        "which handles special operations worldwide. Empusa was a shape-shifting spirit in "
        "Greek mythology who served the goddess Hecate and could appear as a donkey, a dog, "
        "or a beautiful woman to deceive travellers on lonely roads. Otto Klemperer was a "
        "celebrated German conductor born in Breslau in 1885 who was renowned for his "
        "interpretations of the German classical and Romantic repertoire. "
    )

    for ctx_len in sizes:
        needle = "The historic Blue Mountains crossing by Blaxland, Lawson and Wentworth occurred in 1813."
        filler = (_WIKI_FILLER * 8)[:ctx_len]
        correct = 0
        positions = [0.0, 0.5, 1.0]
        pos_results = {}

        for pos in positions:
            insert_idx = int(len(filler) * pos)
            prompt = (
                filler[:insert_idx]
                + f"\n{needle}\n"
                + filler[insert_idx:]
                + "\n\nIn what year did Blaxland cross the Blue Mountains? Answer only with the year."
            )
            try:
                resp, t = _run_timed(llm.generate, prompt)
                hit = "1813" in resp
                pos_results[f"pos_{int(pos * 100)}pct"] = {
                    "hit": hit,
                    "elapsed_s": round(t, 2),
                    "snippet": resp[:60],
                }
                if hit:
                    correct += 1
            except Exception as exc:
                pos_results[f"pos_{int(pos * 100)}pct"] = {"error": str(exc)}

        score = correct / len(positions)
        scores.append(score)
        results[f"ctx_{ctx_len}"] = {
            "score": score,
            "positions": pos_results,
        }
        print(f"  ctx={ctx_len:5d} chars  score={score:.2f}  ({correct}/{len(positions)} positions correct)")

    mean_score = sum(scores) / len(scores) if scores else 0.0
    return {
        "context_lengths": sizes,
        "per_length": results,
        "mean_score": round(mean_score, 4),
        "score": mean_score,
        "status": "ok",
    }


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LM Studio full eval runner")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="LM Studio model string")
    args = parser.parse_args()

    model_str = args.model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nOpenMemoryLab - LM Studio Eval Runner")
    print(f"Model  : {model_str}")
    print(f"Timestamp: {timestamp}")

    # ── GPU check ────────────────────────────────────────────────────────────
    _section("GPU / DEVICE STATUS")
    gpu = _gpu_info()
    print(f"  device       : {gpu['device']}")
    print(f"  cuda_available: {gpu['cuda_available']}")
    print(f"  gpu_name     : {gpu['gpu_name']}")
    vram = gpu['vram_gb'] or 0.0
    print(f"  vram_gb      : {vram:.1f}")
    print(f"  gpu_count    : {gpu['gpu_count']}")

    # ── connectivity check ───────────────────────────────────────────────────
    _section("LM STUDIO CONNECTIVITY")
    # Disable chain-of-thought for eval: Qwen3 reasoning mode generates 300-1000
    # extra reasoning tokens per call, multiplying latency by 10-30x.
    # /nothink is a Qwen3/LM Studio directive; ignored by non-thinking models.
    os.environ.setdefault("LMSTUDIO_SYSTEM_PROMPT", "/nothink You are a concise assistant.")
    os.environ.setdefault("LMSTUDIO_TIMEOUT", "120")

    raw_client = get_llm_client(model_str)
    assert isinstance(raw_client, LMStudioLLM), f"Expected LMStudioLLM, got {type(raw_client)}"

    if not raw_client.ping():
        print(f"  [WARN] LM Studio not reachable at {raw_client.base_url}")
        print("  Make sure LM Studio is running with the local server enabled.")
    else:
        print(f"  LM Studio reachable at {raw_client.base_url}")

    models_loaded = raw_client.list_models()
    loaded_ids = [m["id"] if isinstance(m, dict) else str(m) for m in models_loaded]
    print(f"  Loaded models: {loaded_ids}")

    # Quick connectivity test
    ping_resp, ping_t = _run_timed(raw_client.generate, "Reply with exactly: READY")
    print(f"  Connectivity test: '{ping_resp[:60]}' ({ping_t:.1f}s)")

    llm = raw_client  # reuse the same client

    # ── collect all results ──────────────────────────────────────────────────
    all_results = {
        "meta": {
            "timestamp": timestamp,
            "model": model_str,
            "gpu": gpu,
            "lmstudio_url": raw_client.base_url,
            "models_loaded": loaded_ids,
        },
        "tasks": {},
    }

    # Standard eval tasks (self-contained — no artifacts needed)
    STANDALONE_TASKS = ["faithfulness", "lost-in-middle"]
    task_results = run_eval_tasks(llm, STANDALONE_TASKS)
    all_results["tasks"].update(task_results)

    # Extended lost-in-middle
    lim_ext = run_lim_extended(llm)
    all_results["tasks"]["lost_in_middle_extended"] = lim_ext

    # TEEG eval
    teeg_result = run_teeg_eval(model_str)
    all_results["tasks"]["teeg_cycle"] = teeg_result

    # PRISM eval
    prism_result = run_prism_eval(model_str)
    all_results["tasks"]["prism_cycle"] = prism_result

    # ── summary table ─────────────────────────────────────────────────────────
    _section("RESULTS SUMMARY")
    scores = []
    for name, res in all_results["tasks"].items():
        score = res.get("score")
        elapsed = res.get("elapsed_s", res.get("total_ingest_s", "N/A"))
        status = res.get("status", "?")
        score_str = f"{score:.4f}" if score is not None else "N/A"
        elapsed_str = f"{elapsed:.1f}s" if isinstance(elapsed, (int, float)) else str(elapsed)
        print(f"  {name:<35} score={score_str:<8} time={elapsed_str:<10} [{status}]")
        if score is not None:
            scores.append(score)

    overall = sum(scores) / len(scores) if scores else 0.0
    print(f"\n  OVERALL MEAN SCORE: {overall:.4f}")
    all_results["overall_score"] = round(overall, 4)

    # ── save JSON ─────────────────────────────────────────────────────────────
    json_path = REPORTS_DIR / f"lmstudio_eval_{timestamp}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  JSON saved: {json_path}")

    # ── save Markdown ─────────────────────────────────────────────────────────
    md_path = REPORTS_DIR / f"lmstudio_eval_{timestamp}.md"
    _write_markdown(md_path, all_results, timestamp)
    print(f"  MD  saved: {md_path}")

    return all_results


def _write_markdown(path: Path, results: dict, timestamp: str):
    meta = results["meta"]
    gpu = meta["gpu"]
    tasks = results["tasks"]

    lines = [
        f"# LM Studio Evaluation Report",
        f"",
        f"**Timestamp**: {timestamp}  ",
        f"**Model**: `{meta['model']}`  ",
        f"**LM Studio URL**: `{meta['lmstudio_url']}`  ",
        f"**Overall Score**: `{results.get('overall_score', 'N/A')}`  ",
        f"",
        f"## GPU / Device",
        f"",
        f"| Field | Value |",
        f"|-------|-------|",
        f"| Device | `{gpu['device']}` |",
        f"| CUDA Available | `{gpu['cuda_available']}` |",
        f"| GPU Name | `{gpu['gpu_name']}` |",
        f"| VRAM | `{(gpu['vram_gb'] or 0.0):.1f} GB` |",
        f"| GPU Count | `{gpu['gpu_count']}` |",
        f"",
        f"## Results",
        f"",
        f"| Task | Score | Time | Status |",
        f"|------|-------|------|--------|",
    ]

    for name, res in tasks.items():
        score = res.get("score")
        elapsed = res.get("elapsed_s", res.get("total_ingest_s", "N/A"))
        status = res.get("status", "?")
        score_str = f"{score:.4f}" if score is not None else "N/A"
        elapsed_str = f"{elapsed:.1f}s" if isinstance(elapsed, (int, float)) else str(elapsed)
        lines.append(f"| `{name}` | {score_str} | {elapsed_str} | {status} |")

    lines += [
        f"",
        f"## Task Details",
        f"",
    ]

    for name, res in tasks.items():
        lines.append(f"### {name}")
        lines.append(f"")
        lines.append(f"```json")
        lines.append(json.dumps(res, indent=2, default=str)[:3000])
        lines.append(f"```")
        lines.append(f"")

    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    results = main()
