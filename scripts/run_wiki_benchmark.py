"""
run_wiki_benchmark.py
=====================
All-in-one script that:
  1. Cleans existing data stores (artifacts/, data/oml.db)
  2. Copies ALL Wikipedia chunk files into data/docs/ for ingestion
  3. Ingests them into OML (builds BM25 + vector indices)
  4. Runs the full benchmark across LM Studio models (one at a time)
  5. Writes reports/benchmark_results.json + a timestamped .md report

Model management:
  - Before each model, unloads all LLM models from LM Studio
  - Loads only the model being tested
  - Waits for the model to be ready before running tasks
  - Keeps the embedding model (nomic) untouched

Usage:
    python scripts/run_wiki_benchmark.py
    python scripts/run_wiki_benchmark.py --quick                  # skip TEEG/PRISM
    python scripts/run_wiki_benchmark.py --model lmstudio:qwen3-0.6b
    python scripts/run_wiki_benchmark.py --concurrent-prompts 2   # parallel prompts
"""

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ── Force UTF-8 output on Windows ─────────────────────────────────────────────
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("LMSTUDIO_SYSTEM_PROMPT", "/nothink You are a concise assistant.")
os.environ.setdefault("LMSTUDIO_TIMEOUT", "180")

# ── Paths ──────────────────────────────────────────────────────────────────────
WIKI_DIRS = [
    ROOT / "oml" / "eval" / "datasets" / "wiki" / "1of2",
    ROOT / "oml" / "eval" / "datasets" / "wiki" / "2of2",
]
DOCS_DIR        = ROOT / "data" / "docs"
ARTIFACTS_DIR   = ROOT / "artifacts"
DB_PATH         = ROOT / "data" / "oml.db"
REPORTS_DIR     = ROOT / "reports"
BENCHMARK_JSON  = REPORTS_DIR / "benchmark_results.json"

# Stores created by TEEG/PRISM eval cycles
TEMP_STORES = [
    ROOT / "teeg_store",
    ROOT / "eval_teeg_store",
    ROOT / "eval_prism_store",
    ROOT / "test_teeg_tmp",
]

# ── LM Studio CLI ─────────────────────────────────────────────────────────────
LMS_CLI = Path.home() / ".lmstudio" / "bin" / "lms"
if sys.platform == "win32" and not LMS_CLI.suffix:
    LMS_EXE = LMS_CLI.with_suffix(".exe")
    if LMS_EXE.exists():
        LMS_CLI = LMS_EXE


# ══════════════════════════════════════════════════════════════════════════════
# LM STUDIO MODEL MANAGEMENT
# ══════════════════════════════════════════════════════════════════════════════

def _run_lms(*args, timeout: int = 120) -> str:
    """Run an lms CLI command and return stdout."""
    cmd = [str(LMS_CLI)] + list(args)
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip()
    except FileNotFoundError:
        print(f"  [ERROR] lms CLI not found at {LMS_CLI}")
        return ""
    except subprocess.TimeoutExpired:
        print(f"  [WARN] lms command timed out: {' '.join(args)}")
        return ""


def lms_list_loaded() -> list[str]:
    """Return list of currently loaded model identifiers (excluding embedding models)."""
    try:
        import requests
        resp = requests.get("http://localhost:1234/api/v0/models", timeout=10)
        models = resp.json().get("data", [])
        loaded = []
        for m in models:
            if m.get("state") == "loaded" and m.get("type") != "embeddings":
                loaded.append(m["id"])
        return loaded
    except Exception:
        return []


def lms_unload_all_llms():
    """Unload all LLM models (keep embedding models)."""
    loaded = lms_list_loaded()
    if not loaded:
        print("  No LLM models loaded.")
        return

    for model_id in loaded:
        print(f"  Unloading: {model_id} …")
        _run_lms("unload", model_id, timeout=60)
        time.sleep(2)

    # Verify
    still_loaded = lms_list_loaded()
    if still_loaded:
        print(f"  [WARN] Models still loaded after unload: {still_loaded}")
    else:
        print("  All LLM models unloaded.")


def lms_load_model(model_id: str, max_wait: int = 120):
    """Load a single model and wait for it to be ready."""
    print(f"  Loading model: {model_id} …")
    _run_lms("load", model_id, "-y", timeout=max_wait)

    # Wait for it to appear as loaded
    deadline = time.time() + max_wait
    while time.time() < deadline:
        loaded = lms_list_loaded()
        if model_id in loaded:
            print(f"  [OK] Model loaded: {model_id}")
            return True
        time.sleep(3)

    print(f"  [ERROR] Model did not load within {max_wait}s: {model_id}")
    return False


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1: Clean + Copy All Wiki + Ingest
# ══════════════════════════════════════════════════════════════════════════════

def _banner(title: str, char: str = "=", width: int = 70):
    print(f"\n{char * width}")
    print(f"  {title}")
    print(f"{char * width}")


def phase1_clean():
    """Remove all existing stores so we start completely fresh."""
    _banner("PHASE 1a: CLEANING EXISTING DATA")

    targets = [ARTIFACTS_DIR, DB_PATH, DOCS_DIR] + TEMP_STORES
    for p in targets:
        if p.exists():
            try:
                if p.is_dir():
                    shutil.rmtree(p)
                    print(f"  Removed directory: {p.relative_to(ROOT)}")
                else:
                    p.unlink()
                    print(f"  Removed file:      {p.relative_to(ROOT)}")
            except PermissionError:
                print(f"  [WARN] Could not remove (locked): {p.relative_to(ROOT)}")
                # Try renaming as workaround on Windows
                try:
                    tmp = p.with_suffix(".old")
                    p.rename(tmp)
                    tmp.unlink(missing_ok=True)
                    print(f"  Removed via rename: {p.relative_to(ROOT)}")
                except Exception:
                    print(f"  [ERROR] File locked, cannot remove: {p}")
    print("  Clean slate ready.")


def phase1_copy_all_wiki():
    """Copy ALL Wikipedia chunk files into data/docs/ for full ingestion."""
    _banner("PHASE 1b: COPYING ALL WIKIPEDIA DATA")

    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    total_files = 0
    total_bytes = 0

    for dir_idx, wiki_dir in enumerate(WIKI_DIRS):
        if not wiki_dir.exists():
            print(f"  [WARN] Wiki dir missing: {wiki_dir}")
            continue

        files = sorted(f for f in wiki_dir.iterdir() if f.is_file())
        print(f"  {wiki_dir.name}/: {len(files)} files")

        # Prefix with directory index to avoid filename collisions
        # (1of2 and 2of2 both have wiki_00, wiki_01, etc.)
        prefix = f"p{dir_idx}_"
        for src in files:
            dst = DOCS_DIR / f"{prefix}{src.name}"
            shutil.copy2(src, dst)
            total_files += 1
            total_bytes += src.stat().st_size

    size_mb = total_bytes / (1024 * 1024)
    print(f"  Copied {total_files} files ({size_mb:.1f} MB) → {DOCS_DIR.relative_to(ROOT)}")
    return total_files


def phase1_ingest():
    """Run OML ingestion pipeline on data/docs/."""
    _banner("PHASE 1c: INGESTING INTO OML")

    from oml.ingest.pipeline import IngestionPipeline

    pipeline = IngestionPipeline(storage_type="sqlite", device="auto")
    pipeline.run(path=str(DOCS_DIR), rebuild_indices=True)

    # Verify artefacts
    expected = [ARTIFACTS_DIR / f for f in ("bm25.pkl", "vector.index", "vector_map.json")]
    ok = all(f.exists() for f in expected)
    if ok:
        print(f"  [OK] Artifacts created: {[f.name for f in expected]}")
    else:
        missing = [f.name for f in expected if not f.exists()]
        print(f"  [ERROR] Missing artifacts: {missing}")
        sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2: Benchmark with model load/unload
# ══════════════════════════════════════════════════════════════════════════════

def phase2_benchmark(target_models: list[str], quick: bool = False,
                     concurrent_prompts: int = 1) -> dict:
    """Run benchmark across all target models, loading one at a time."""
    _banner("PHASE 2: RUNNING FULL BENCHMARK")

    # Import benchmark machinery
    _eval_path = ROOT / "scripts" / "eval_lmstudio.py"
    _spec = importlib.util.spec_from_file_location("_eval_lmstudio", _eval_path)
    _eval_mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_eval_mod)

    _run_timed       = _eval_mod._run_timed
    run_eval_tasks   = _eval_mod.run_eval_tasks
    run_lim_extended = _eval_mod.run_lim_extended
    run_teeg_eval    = _eval_mod.run_teeg_eval
    run_prism_eval   = _eval_mod.run_prism_eval

    from oml.llm.lmstudio import LMStudioLLM
    from oml.llm.factory   import get_llm_client

    DISPLAY_NAMES = {
        "qwen3-0.6b":                         "Qwen3 0.6B",
        "google/gemma-3-4b":                  "Gemma 3 4B",
        "deepseek/deepseek-r1-0528-qwen3-8b": "DeepSeek-R1 Qwen3 8B",
        "openai/gpt-oss-20b":                 "GPT-OSS 20B",
        "qwen/qwen3-30b-a3b":                 "Qwen3 30B-A3B",
        "qwen/qwen3-coder-30b":               "Qwen3 Coder 30B",
    }

    # ── Discover models if none specified ──────────────────────────────────────
    if not target_models:
        print("  Discovering models from LM Studio …")
        try:
            import requests
            resp = requests.get("http://localhost:1234/api/v0/models", timeout=10)
            all_models = resp.json().get("data", [])
            for m in all_models:
                mid = m["id"]
                if m.get("type") == "embeddings":
                    continue
                target_models.append(f"lmstudio:{mid}")
        except Exception as exc:
            print(f"  [ERROR] Cannot reach LM Studio: {exc}")
            sys.exit(1)

    if not target_models:
        print("  [ERROR] No models found — is LM Studio running?")
        sys.exit(1)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    n_models = len(target_models)
    n_tasks = 2 if quick else 9  # faithfulness + LiM, or all 9
    # Rough estimate: ~30s load/unload, ~20s warmup, ~30s per task (varies hugely by model size)
    est_per_model_s = 30 + 20 + (n_tasks * 30)
    est_total_s = est_per_model_s * n_models
    est_total_min = est_total_s / 60

    print(f"  Timestamp          : {timestamp}")
    print(f"  Models ({n_models})       : {target_models}")
    print(f"  Tasks per model    : {n_tasks}")
    print(f"  Quick mode         : {quick}")
    print(f"  Concurrent prompts : {concurrent_prompts}")
    print(f"  Estimated time     : ~{est_total_min:.0f} min "
          f"({est_per_model_s}s/model x {n_models} models)")
    print(f"  (Estimate updates after each model with actual timing)")

    data = {"generated_at": timestamp, "settings": {
        "concurrent_prompts": concurrent_prompts,
        "quick": quick,
    }, "models": {}}
    model_times: list[float] = []  # wall seconds per completed model

    for idx, model_str in enumerate(target_models, 1):
        model_id = model_str.removeprefix("lmstudio:")
        display  = DISPLAY_NAMES.get(model_id, model_id.split("/")[-1])

        _banner(f"MODEL {idx}/{len(target_models)}: {display}  ({model_str})", char="-")

        # ── Unload all LLMs, then load just this one ───────────────────────────
        print("  Swapping model …")
        lms_unload_all_llms()
        time.sleep(3)

        if not lms_load_model(model_id, max_wait=180):
            data["models"][model_id] = {"status": "load_failed", "display_name": display}
            continue

        # Give the model a moment to fully initialize
        time.sleep(5)

        try:
            llm = get_llm_client(model_str)
        except Exception as exc:
            print(f"  [ERROR] Client creation failed: {exc}")
            data["models"][model_id] = {"status": "error", "error": str(exc), "display_name": display}
            continue

        if not llm.ping():
            print(f"  [ERROR] LM Studio unreachable after loading model")
            data["models"][model_id] = {"status": "unreachable", "display_name": display}
            continue

        wall_start = time.perf_counter()
        result = {
            "model_str":     model_str,
            "model_id":      model_id,
            "display_name":  display,
            "benchmarked_at": datetime.now().isoformat(),
            "tasks": {},
        }

        # Latency warmup
        print("  Latency warmup (3 pings)…")
        try:
            times = []
            for _ in range(3):
                _, t = _run_timed(llm.generate, "Reply with exactly: READY")
                times.append(round(t, 3))
            result["latency"] = {
                "mean_s": round(sum(times) / len(times), 3),
                "min_s":  min(times),
                "max_s":  max(times),
            }
            print(f"    mean={result['latency']['mean_s']:.2f}s")
        except Exception as exc:
            result["latency"] = {"error": str(exc)}

        # Self-contained tasks (faithfulness, lost-in-middle) — 1 prompt at a time
        print("  Running self-contained tasks …")
        try:
            sc_results = run_eval_tasks(llm, ["faithfulness", "lost-in-middle"])
            for k, v in sc_results.items():
                norm = k.replace("-", "_")
                result["tasks"][norm] = v
                print(f"    {k}: {v.get('score', '?')}")
        except Exception as exc:
            print(f"  [ERROR] {exc}")

        if not quick:
            # Artifact-dependent tasks — run sequentially (1 prompt at a time)
            print("  Running artifact-dependent tasks …")
            for task_name in ["retrieval_precision", "cost_latency", "oml_vs_rag", "global_trends"]:
                try:
                    art = run_eval_tasks(llm, [task_name])
                    for k, v in art.items():
                        detail_err = str(v.get("details", {}).get("error", "")).lower()
                        if v.get("score") == 0.0 and ("artifact" in detail_err or "ingest" in detail_err):
                            v["status"] = "skipped"
                        result["tasks"][k] = v
                        print(f"    {k}: {v.get('score', '?')}  [{v.get('status', '?')}]")
                except Exception as exc:
                    print(f"  [ERROR] {task_name}: {exc}")
                    result["tasks"][task_name] = {"status": "error", "error": str(exc)}

            # LiM extended
            print("  Running lost-in-middle extended …")
            try:
                lim = run_lim_extended(llm)
                result["tasks"]["lost_in_middle_extended"] = lim
                print(f"    LiM-ext mean: {lim.get('mean_score', '?')}")
            except Exception as exc:
                result["tasks"]["lost_in_middle_extended"] = {"status": "error", "error": str(exc)}

            # TEEG
            print("  Running TEEG ingest+query …")
            try:
                teeg = run_teeg_eval(model_str)
                result["tasks"]["teeg_cycle"] = teeg
                print(f"    TEEG score: {teeg.get('score', '?')}")
            except Exception as exc:
                result["tasks"]["teeg_cycle"] = {"status": "error", "error": str(exc)}

            # PRISM
            print("  Running PRISM batch+dedup …")
            try:
                prism = run_prism_eval(model_str)
                result["tasks"]["prism_cycle"] = prism
                print(f"    PRISM score: {prism.get('score', '?')}")
            except Exception as exc:
                result["tasks"]["prism_cycle"] = {"status": "error", "error": str(exc)}

        # Overall score (only count status="ok")
        ok_scores = [
            v["score"] for v in result["tasks"].values()
            if v.get("status") == "ok" and isinstance(v.get("score"), (int, float))
        ]
        result["overall_score"] = round(sum(ok_scores) / len(ok_scores), 4) if ok_scores else None
        result["tasks_scored"]  = len(ok_scores)
        result["tasks_errored"] = sum(1 for v in result["tasks"].values() if v.get("status") == "error")
        result["total_wall_s"]  = round(time.perf_counter() - wall_start, 1)
        result["status"]        = "ok"

        print(f"\n  -- {display} done in {result['total_wall_s']:.1f}s | overall={result['overall_score']} --")

        data["models"][model_id] = result
        model_times.append(result["total_wall_s"])

        # ── Time estimation ────────────────────────────────────────────────────
        models_remaining = n_models - idx
        if models_remaining > 0:
            avg_model_s = sum(model_times) / len(model_times)
            eta_s = avg_model_s * models_remaining
            eta_min = eta_s / 60
            finish_time = datetime.now().timestamp() + eta_s
            finish_str = datetime.fromtimestamp(finish_time).strftime("%H:%M:%S")
            print(f"  >> ETA: ~{eta_min:.0f} min remaining "
                  f"({models_remaining} models x {avg_model_s:.0f}s avg) "
                  f"— est. finish at {finish_str}")
        else:
            print(f"  >> All models complete!")

        # Save after each model (crash-safe)
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        data["updated_at"] = datetime.now().isoformat()
        with open(BENCHMARK_JSON, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

    return data, timestamp


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3: Summary + Dashboard report
# ══════════════════════════════════════════════════════════════════════════════

def phase3_report(data: dict, timestamp: str):
    """Print summary table and write markdown report."""
    _banner("PHASE 3: RESULTS SUMMARY")

    header = f"{'Model':<35} {'Faith':>6} {'LiM':>6} {'LiM-X':>6} {'RetP':>6} {'TEEG':>6} {'PRISM':>6} {'Overall':>8} {'Time':>7}"
    print(header)
    print("-" * len(header))

    def _f(v):
        return f"{v:.2f}" if isinstance(v, (int, float)) else "  — "

    for mid, m in data["models"].items():
        t = m.get("tasks", {})
        name = m.get("display_name", mid)[:33]
        print(
            f"{name:<35} "
            f"{_f(t.get('faithfulness', {}).get('score')):>6} "
            f"{_f(t.get('lost_in_middle', {}).get('score')):>6} "
            f"{_f(t.get('lost_in_middle_extended', {}).get('score', t.get('lost_in_middle_extended', {}).get('mean_score'))):>6} "
            f"{_f(t.get('retrieval_precision', {}).get('score')):>6} "
            f"{_f(t.get('teeg_cycle', {}).get('score')):>6} "
            f"{_f(t.get('prism_cycle', {}).get('score')):>6} "
            f"{_f(m.get('overall_score')):>8} "
            f"{_f(m.get('total_wall_s')):>6}s"
        )

    # ── Markdown report ────────────────────────────────────────────────────────
    n_docs = len(list(DOCS_DIR.glob("*"))) if DOCS_DIR.exists() else 0
    lines = [
        "# OpenMemoryLab — Wikipedia Benchmark Report",
        "",
        f"**Generated**: {timestamp}  ",
        f"**Corpus**: Full Wikipedia dump ({n_docs} files, ingested fresh)  ",
        f"**Models tested**: {len(data['models'])}  ",
        f"**Settings**: concurrent_prompts={data.get('settings', {}).get('concurrent_prompts', 1)}, quick={data.get('settings', {}).get('quick', False)}  ",
        "",
        "## Summary Table",
        "",
        "| Model | Faithfulness | Lost-in-Middle | LiM Extended | Retrieval P@K | Cost/Latency | OML vs RAG | Global Trends | TEEG | PRISM | Overall | Wall Time |",
        "|-------|-------------|----------------|-------------|---------------|-------------|------------|---------------|------|-------|---------|-----------|",
    ]

    def _mf(v):
        return f"{v:.3f}" if isinstance(v, (int, float)) else "—"

    for mid, m in data["models"].items():
        t = m.get("tasks", {})
        name = m.get("display_name", mid)
        lines.append(
            f"| {name} "
            f"| {_mf(t.get('faithfulness', {}).get('score'))} "
            f"| {_mf(t.get('lost_in_middle', {}).get('score'))} "
            f"| {_mf(t.get('lost_in_middle_extended', {}).get('score', t.get('lost_in_middle_extended', {}).get('mean_score')))} "
            f"| {_mf(t.get('retrieval_precision', {}).get('score'))} "
            f"| {_mf(t.get('cost_latency', {}).get('score'))} "
            f"| {_mf(t.get('oml_vs_rag', {}).get('score'))} "
            f"| {_mf(t.get('global_trends', {}).get('score'))} "
            f"| {_mf(t.get('teeg_cycle', {}).get('score'))} "
            f"| {_mf(t.get('prism_cycle', {}).get('score'))} "
            f"| {_mf(m.get('overall_score'))} "
            f"| {m.get('total_wall_s', '—')}s |"
        )

    lines += ["", "## Per-Model Details", ""]
    for mid, m in data["models"].items():
        name = m.get("display_name", mid)
        lines += [
            f"### {name}",
            "",
            f"- **Model ID**: `{m.get('model_str', mid)}`",
            f"- **Benchmarked at**: {m.get('benchmarked_at', '—')}",
            f"- **Overall score**: {m.get('overall_score', '—')}",
            f"- **Tasks scored**: {m.get('tasks_scored', 0)}  |  Errored: {m.get('tasks_errored', 0)}",
            f"- **Wall time**: {m.get('total_wall_s', '—')}s",
            "",
            "```json",
            json.dumps(m.get("tasks", {}), indent=2, default=str)[:5000],
            "```",
            "",
        ]

    md_path = REPORTS_DIR / f"wiki_benchmark_{timestamp}.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n  [OK] Markdown report → {md_path.relative_to(ROOT)}")
    print(f"  [OK] JSON results   → {BENCHMARK_JSON.relative_to(ROOT)}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Wikipedia-based OML benchmark (ingest + test + report)")
    parser.add_argument("--model", dest="models", action="append", default=[], help="Specific model(s)")
    parser.add_argument("--quick", action="store_true", help="Skip TEEG/PRISM/artifact tasks")
    parser.add_argument("--concurrent-prompts", type=int, default=1,
                        help="Max concurrent prompts per model (default: 1)")
    parser.add_argument("--concurrent-models", type=int, default=1,
                        help="Max models loaded at once (default: 1, strongly recommended)")
    args = parser.parse_args()

    if args.concurrent_models > 1:
        print("  [WARN] concurrent-models > 1 will load multiple models and may exhaust RAM!")

    total_start = time.perf_counter()

    # Phase 1: Clean → Copy all wiki data → Ingest
    phase1_clean()
    n_files = phase1_copy_all_wiki()

    ingest_start = time.perf_counter()
    print(f"\n  Ingesting {n_files} files — this may take several minutes …")
    phase1_ingest()
    ingest_elapsed = time.perf_counter() - ingest_start
    print(f"  Ingestion took {ingest_elapsed:.1f}s ({ingest_elapsed/60:.1f} min)")

    # Phase 2: Benchmark (one model at a time with load/unload)
    data, timestamp = phase2_benchmark(
        args.models, quick=args.quick,
        concurrent_prompts=args.concurrent_prompts,
    )

    # Phase 3: Report
    phase3_report(data, timestamp)

    elapsed = time.perf_counter() - total_start
    print(f"\n  Total elapsed: {elapsed / 60:.1f} minutes")
    print("  Done!")


if __name__ == "__main__":
    main()
