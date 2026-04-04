"""
scripts/benchmark_models.py
============================
Runs the full eval suite against ALL loaded LM Studio models and produces a
consolidated benchmark report for the Streamlit Benchmarks dashboard.

Usage
-----
    # Run all discovered models (full suite)
    python scripts/benchmark_models.py

    # Quick mode — faithfulness + lost-in-middle only, skip TEEG/PRISM
    python scripts/benchmark_models.py --quick

    # One specific model
    python scripts/benchmark_models.py --model lmstudio:qwen3-0.6b

    # Multiple specific models
    python scripts/benchmark_models.py --model lmstudio:qwen3-0.6b --model lmstudio:google/gemma-3-4b

Output
------
    reports/benchmark_results.json  — consolidated results (read by the dashboard)
    reports/benchmark_TIMESTAMP.md  — human-readable comparison table

The script is designed to be re-run safely: it MERGES new results into the
existing benchmark_results.json rather than overwriting previous runs, so you
can add one model at a time without losing earlier data.
"""

import argparse
import importlib.util
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Force UTF-8 output on Windows to avoid cp1252 UnicodeEncodeError
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ── project root on sys.path ─────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ── env defaults (disable CoT for eval speed) ────────────────────────────────
os.environ.setdefault("LMSTUDIO_SYSTEM_PROMPT", "/nothink You are a concise assistant.")
os.environ.setdefault("LMSTUDIO_TIMEOUT", "180")

# ── load eval_lmstudio helpers without running its main() ────────────────────
_eval_path = ROOT / "scripts" / "eval_lmstudio.py"
_spec = importlib.util.spec_from_file_location("_eval_lmstudio", _eval_path)
_eval_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_eval_mod)

# Re-export helpers
_run_timed       = _eval_mod._run_timed
run_eval_tasks   = _eval_mod.run_eval_tasks
run_lim_extended = _eval_mod.run_lim_extended
run_teeg_eval    = _eval_mod.run_teeg_eval
run_prism_eval   = _eval_mod.run_prism_eval

from oml.llm.lmstudio import LMStudioLLM
from oml.llm.factory   import get_llm_client

# ── constants ─────────────────────────────────────────────────────────────────
REPORTS_DIR          = ROOT / "reports"
BENCHMARK_RESULTS    = REPORTS_DIR / "benchmark_results.json"
PREDICTIONS_FILE     = REPORTS_DIR / "model_predictions.json"

# Model IDs that are embeddings and should never be benchmarked
EMBEDDING_SKIP = {
    "text-embedding-nomic-embed-text-v1.5",
    "nomic-embed-text-v1.5",
}

# Human-readable short names for known model IDs
DISPLAY_NAMES = {
    "qwen3-0.6b":                         "Qwen3 0.6B",
    "google/gemma-3-4b":                  "Gemma 3 4B",
    "deepseek/deepseek-r1-0528-qwen3-8b": "DeepSeek-R1 Qwen3 8B",
    "openai/gpt-oss-20b":                 "GPT-OSS 20B",
    "qwen/qwen3-30b-a3b":                 "Qwen3 30B-A3B",
    "qwen/qwen3-coder-30b":               "Qwen3 Coder 30B",
}


# ── helpers ───────────────────────────────────────────────────────────────────

def _discover_models() -> list[str]:
    """Query LM Studio for all loaded chat models (skipping embeddings)."""
    try:
        probe = LMStudioLLM(model_name="dummy")
        raw = probe.list_models()
    except Exception as exc:
        print(f"  [WARN] Could not discover models: {exc}")
        return []

    models = []
    for m in raw:
        mid = m["id"] if isinstance(m, dict) else str(m)
        if any(skip in mid.lower() for skip in ("embed", "embedding")):
            continue
        if mid in EMBEDDING_SKIP:
            continue
        models.append(f"lmstudio:{mid}")
    return models


def _measure_latency(llm, n_samples: int = 3) -> dict:
    """Time n_samples calls to a short prompt, return stats."""
    prompt = "Reply with exactly: READY"
    times = []
    for _ in range(n_samples):
        _, t = _run_timed(llm.generate, prompt)
        times.append(round(t, 3))
    times_sorted = sorted(times)
    return {
        "samples":    n_samples,
        "times_s":    times,
        "min_s":      times_sorted[0],
        "max_s":      times_sorted[-1],
        "mean_s":     round(sum(times) / len(times), 3),
        "median_s":   times_sorted[len(times_sorted) // 2],
    }


def _section(title: str, width: int = 70):
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def _load_existing() -> dict:
    """Load existing benchmark_results.json or return empty structure."""
    if BENCHMARK_RESULTS.exists():
        try:
            with open(BENCHMARK_RESULTS, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"generated_at": "", "models": {}}


def _save_results(data: dict):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    data["updated_at"] = datetime.now().isoformat()
    with open(BENCHMARK_RESULTS, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"\n  [OK] benchmark_results.json updated -> {BENCHMARK_RESULTS}")


# ── per-model benchmark ───────────────────────────────────────────────────────

def benchmark_model(model_str: str, quick: bool = False) -> dict:
    """Run the full eval suite for *model_str*, return result dict."""
    model_id = model_str.removeprefix("lmstudio:")
    display_name = DISPLAY_NAMES.get(model_id, model_id.split("/")[-1])

    _section(f"BENCHMARKING: {display_name}  ({model_str})")

    # Build LLM client
    try:
        llm = get_llm_client(model_str)
    except Exception as exc:
        print(f"  [ERROR] Could not create client: {exc}")
        return {
            "model_str":    model_str,
            "model_id":     model_id,
            "display_name": display_name,
            "status":       "error",
            "error":        str(exc),
        }

    # Connectivity check
    if not llm.ping():
        print(f"  [ERROR] LM Studio not reachable at {llm.base_url}")
        return {
            "model_str":    model_str,
            "model_id":     model_id,
            "display_name": display_name,
            "status":       "unreachable",
        }

    wall_start = time.perf_counter()
    result = {
        "model_str":    model_str,
        "model_id":     model_id,
        "display_name": display_name,
        "benchmarked_at": datetime.now().isoformat(),
        "tasks": {},
    }

    # ── 1. Latency warmup (3 pings) ──────────────────────────────────────────
    print("  Measuring latency (3 pings)…")
    try:
        lat = _measure_latency(llm, n_samples=3)
        result["latency"] = lat
        print(f"    mean={lat['mean_s']:.2f}s  min={lat['min_s']:.2f}s  max={lat['max_s']:.2f}s")
    except Exception as exc:
        result["latency"] = {"error": str(exc)}
        print(f"  [WARN] Latency measurement failed: {exc}")

    # ── 2. Self-contained registered eval tasks (always run) ─────────────────
    # These tasks require no artifacts directory and work out-of-the-box.
    SELF_CONTAINED_TASKS = ["faithfulness", "lost-in-middle"]
    print(f"\n  Running self-contained tasks: {SELF_CONTAINED_TASKS}")
    try:
        task_results = run_eval_tasks(llm, SELF_CONTAINED_TASKS)
        for task_key, task_data in task_results.items():
            # Normalise key: "lost-in-middle" -> "lost_in_middle" for JSON
            norm_key = task_key.replace("-", "_")
            result["tasks"][norm_key] = task_data
            print(f"    {task_key}: score={task_data.get('score', '?')}")
    except Exception as exc:
        for t in SELF_CONTAINED_TASKS:
            result["tasks"][t.replace("-", "_")] = {"status": "error", "error": str(exc)}
        print(f"  [ERROR] self-contained tasks: {exc}")

    # ── 3. Artifacts-dependent tasks (full mode only) ─────────────────────────
    # These tasks require an ingested corpus in the artifacts/ directory.
    # They fail gracefully with a clear error message if no data has been ingested.
    if not quick:
        ARTIFACT_TASKS = ["retrieval_precision", "cost_latency", "oml_vs_rag", "global_trends"]
        print(f"\n  Running artifact-dependent tasks: {ARTIFACT_TASKS}")
        try:
            # Force-import so they register themselves in _TASK_REGISTRY
            import oml.eval.tasks.retrieval_precision  # noqa: F401
            import oml.eval.tasks.cost_latency         # noqa: F401
            import oml.eval.tasks.oml_vs_rag           # noqa: F401
            import oml.eval.tasks.global_trends        # noqa: F401
            art_results = run_eval_tasks(llm, ARTIFACT_TASKS)
            for task_key, task_data in art_results.items():
                # Reclassify graceful "no artifacts" failures as "skipped" so they
                # are excluded from the overall score (score=0.0 here is not a model
                # quality signal — it just means no data has been ingested yet).
                detail_err = str(task_data.get("details", {}).get("error", "")).lower()
                if task_data.get("score") == 0.0 and (
                    "artifact" in detail_err or "ingest" in detail_err or "no artifact" in detail_err
                ):
                    task_data["status"] = "skipped"
                    task_data["skip_reason"] = "No ingested corpus — run `oml ingest` first"
                result["tasks"][task_key] = task_data
                print(f"    {task_key}: score={task_data.get('score', '?')}  "
                      f"[{task_data.get('status', '?')}]")
        except Exception as exc:
            for t in ARTIFACT_TASKS:
                result["tasks"][t] = {"status": "error", "error": str(exc)}
            print(f"  [ERROR] artifact tasks: {exc}")

    # ── 4. Lost-in-Middle extended (optional unless quick) ────────────────────
    if not quick:
        print("\n  Running lost-in-middle extended (3 context lengths)…")
        try:
            lim_ext = run_lim_extended(llm)
            result["tasks"]["lost_in_middle_extended"] = lim_ext
            print(f"    mean score = {lim_ext.get('mean_score', '?')}")
        except Exception as exc:
            result["tasks"]["lost_in_middle_extended"] = {"status": "error", "error": str(exc)}
            print(f"  [ERROR] lost-in-middle-extended: {exc}")

    # ── 5. TEEG memory recall (optional unless quick) ─────────────────────────
    if not quick:
        print("\n  Running TEEG ingest+query cycle…")
        try:
            teeg_result = run_teeg_eval(model_str)
            result["tasks"]["teeg_cycle"] = teeg_result
            print(f"    TEEG score (keyword hit rate) = {teeg_result.get('score', '?')}")
        except Exception as exc:
            result["tasks"]["teeg_cycle"] = {"status": "error", "error": str(exc)}
            print(f"  [ERROR] TEEG: {exc}")

    # ── 6. PRISM batch efficiency + dedup (optional unless quick) ─────────────
    if not quick:
        print("\n  Running PRISM batch+dedup cycle…")
        try:
            prism_result = run_prism_eval(model_str)
            result["tasks"]["prism_cycle"] = prism_result
            print(f"    PRISM score = {prism_result.get('score', '?')}")
        except Exception as exc:
            result["tasks"]["prism_cycle"] = {"status": "error", "error": str(exc)}
            print(f"  [ERROR] PRISM: {exc}")

    # ── Overall score (only count tasks that actually succeeded) ───────────────
    # Errored tasks (status != "ok") are excluded — including them as 0.0 would
    # unfairly penalise models for infrastructure issues (missing artifacts, DLL errors).
    ok_scores = [
        v.get("score")
        for v in result["tasks"].values()
        if v.get("status") == "ok" and isinstance(v.get("score"), (int, float))
    ]
    result["overall_score"] = round(sum(ok_scores) / len(ok_scores), 4) if ok_scores else None
    result["tasks_scored"]  = len(ok_scores)
    result["tasks_errored"] = sum(1 for v in result["tasks"].values() if v.get("status") == "error")
    result["total_wall_s"]  = round(time.perf_counter() - wall_start, 1)
    result["status"]        = "ok"

    print(f"\n  -- {display_name} done in {result['total_wall_s']:.1f}s "
          f"| overall={result['overall_score']} --")
    return result


# ── summary table ─────────────────────────────────────────────────────────────

def _print_summary(data: dict):
    _section("BENCHMARK SUMMARY")
    header = f"{'Model':<35} {'Faith':>6} {'LiM':>6} {'TEEG':>6} {'PRISM':>6} {'Overall':>8} {'Latency':>8}"
    print(header)
    print("-" * len(header))
    for mid, m in data["models"].items():
        tasks = m.get("tasks", {})
        faith  = tasks.get("faithfulness", {}).get("score")
        lim    = tasks.get("lost_in_middle", {}).get("score")
        teeg   = tasks.get("teeg_cycle", {}).get("score")
        prism  = tasks.get("prism_cycle", {}).get("score")
        overall = m.get("overall_score")
        lat    = m.get("latency", {}).get("mean_s")
        name   = m.get("display_name", mid)[:33]

        def _fmt(v):
            return f"{v:.2f}" if isinstance(v, (int, float)) else " — "

        print(f"{name:<35} {_fmt(faith):>6} {_fmt(lim):>6} {_fmt(teeg):>6} {_fmt(prism):>6} {_fmt(overall):>8} {_fmt(lat):>7}s")


def _write_markdown(data: dict, timestamp: str):
    lines = [
        "# OpenMemoryLab — Model Benchmark Report",
        "",
        f"**Generated**: {timestamp}  ",
        f"**Models tested**: {len(data['models'])}  ",
        "",
        "## Summary Table",
        "",
        "| Model | Faithfulness | Lost-in-Middle | TEEG Recall | PRISM Score | Overall | Avg Latency |",
        "|-------|-------------|----------------|-------------|-------------|---------|-------------|",
    ]
    for mid, m in data["models"].items():
        tasks = m.get("tasks", {})
        def _f(v): return f"{v:.3f}" if isinstance(v, (int, float)) else "—"
        faith   = _f(tasks.get("faithfulness", {}).get("score"))
        lim     = _f(tasks.get("lost_in_middle", {}).get("score"))
        teeg    = _f(tasks.get("teeg_cycle", {}).get("score"))
        prism   = _f(tasks.get("prism_cycle", {}).get("score"))
        overall = _f(m.get("overall_score"))
        lat     = _f(m.get("latency", {}).get("mean_s"))
        name    = m.get("display_name", mid)
        lines.append(f"| {name} | {faith} | {lim} | {teeg} | {prism} | {overall} | {lat}s |")

    lines += ["", "## Per-Model Details", ""]
    for mid, m in data["models"].items():
        name = m.get("display_name", mid)
        lines += [
            f"### {name}",
            "",
            f"- **Model ID**: `{m.get('model_str', mid)}`",
            f"- **Benchmarked at**: {m.get('benchmarked_at', '—')}",
            f"- **Overall score**: {m.get('overall_score', '—')}",
            f"- **Total wall time**: {m.get('total_wall_s', '—')}s",
            "",
            "```json",
            json.dumps(m.get("tasks", {}), indent=2, default=str)[:4000],
            "```",
            "",
        ]

    md_path = REPORTS_DIR / f"benchmark_{timestamp}.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  [OK] Markdown saved -> {md_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark all (or selected) LM Studio models and update reports/benchmark_results.json"
    )
    parser.add_argument(
        "--model", dest="models", action="append", default=[],
        metavar="MODEL_STR",
        help="lmstudio:<model_id> to benchmark (repeat for multiple; default: all discovered)"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Only run faithfulness + lost-in-middle (skip TEEG/PRISM ingest cycles)"
    )
    parser.add_argument(
        "--no-merge", action="store_true",
        help="Start fresh instead of merging into existing benchmark_results.json"
    )
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── determine which models to test ────────────────────────────────────────
    if args.models:
        target_models = args.models
    else:
        print("  Discovering loaded models from LM Studio…")
        target_models = _discover_models()
        if not target_models:
            print("  [ERROR] No models found — is LM Studio running?")
            sys.exit(1)

    print(f"\nOpenMemoryLab — Model Benchmark Runner")
    print(f"Timestamp : {timestamp}")
    print(f"Quick mode: {args.quick}")
    print(f"Models    : {target_models}")

    # ── load (or create) the consolidated results file ────────────────────────
    if args.no_merge:
        data = {"generated_at": timestamp, "models": {}}
    else:
        data = _load_existing()
        if not data.get("generated_at"):
            data["generated_at"] = timestamp

    # ── run each model ────────────────────────────────────────────────────────
    for model_str in target_models:
        model_id = model_str.removeprefix("lmstudio:")
        result = benchmark_model(model_str, quick=args.quick)
        data["models"][model_id] = result
        # Save after every model so partial results are preserved on crash
        _save_results(data)

    # ── final summary ─────────────────────────────────────────────────────────
    _print_summary(data)
    _write_markdown(data, timestamp)

    print(f"\n  Done — {len(target_models)} model(s) benchmarked.")
    return data


if __name__ == "__main__":
    main()
