"""
Massively Batched RAG Evaluation Script
=======================================
Evaluates all configurations against the dataset using minimal LLM calls.
It first gathers all retrieval contexts offline (caching HyDE/Graph calls per unique query),
then batches Generation and Judging into large JSON LLM requests.

Usage:
    uv run --python 3.12 python scripts/eval_batched.py
"""

import sys
import time
import json
import re
import argparse
from pathlib import Path
from datetime import datetime

# Ensure 'oml' module can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
import os
load_dotenv()

from oml.llm import get_llm_client
from oml.retrieval.hybrid import HybridRetriever
from oml.retrieval.graph_retriever import GraphRetriever
from oml.retrieval.hyde import generate_hypothetical_document
from oml.retrieval.rerank import Reranker
from oml.retrieval.gtcc import GTCCRetriever
from oml.memory.context import ContextBudgeter, ContextChunk
from oml.storage.sqlite import get_chunks_by_ids

from rich.console import Console
from rich.table import Table
from rich.progress import track

console = Console()

# ---------------------------------------------------------------------------
# Test Dataset
# ---------------------------------------------------------------------------
DATASET = [
    {"q": "Where is the narrator travelling to?", "a": "The North Pole."},
    {"q": "How was the European stranger rescued?", "a": "He was found on a sledge drifting on ice and brought aboard the ship."},
    {"q": "What is the narrator's name?", "a": "Robert Walton."},
    {"q": "Why does Robert Walton feel he needs a friend?", "a": "To share his joy, sympathize with him, and amend his plans."},
    {"q": "Who is Margaret?", "a": "Margaret is Robert Walton's sister."},
]

# ---------------------------------------------------------------------------
# All 18 Combinations
# ---------------------------------------------------------------------------
CONFIGS = [
    # Original 10
    {"name": "1. BM25 Only",           "bm25": True,  "vector": False, "hyde": False, "graph": False, "rerank": False, "summary": False, "gtcc": False},
    {"name": "2. Vector Only",          "bm25": False, "vector": True,  "hyde": False, "graph": False, "rerank": False, "summary": False, "gtcc": False},
    {"name": "3. Hybrid (Baseline)",    "bm25": True,  "vector": True,  "hyde": False, "graph": False, "rerank": False, "summary": False, "gtcc": False},
    {"name": "4. Hybrid + Rerank",      "bm25": True,  "vector": True,  "hyde": False, "graph": False, "rerank": True,  "summary": False, "gtcc": False},
    {"name": "5. Hybrid + HyDE",        "bm25": True,  "vector": True,  "hyde": True,  "graph": False, "rerank": False, "summary": False, "gtcc": False},
    {"name": "6. Hybrid + Graph",       "bm25": True,  "vector": True,  "hyde": False, "graph": True,  "rerank": False, "summary": False, "gtcc": False},
    {"name": "7. HyDE + Rerank",        "bm25": True,  "vector": True,  "hyde": True,  "graph": False, "rerank": True,  "summary": False, "gtcc": False},
    {"name": "8. HyDE + Graph",         "bm25": True,  "vector": True,  "hyde": True,  "graph": True,  "rerank": False, "summary": False, "gtcc": False},
    {"name": "9. Graph + Rerank",       "bm25": True,  "vector": True,  "hyde": False, "graph": True,  "rerank": True,  "summary": False, "gtcc": False},
    {"name": "10. Everything",          "bm25": True,  "vector": True,  "hyde": True,  "graph": True,  "rerank": True,  "summary": False, "gtcc": False},
    # T5 Summary experiments
    {"name": "11. Hybrid + T5 Summary",        "bm25": True,  "vector": True,  "hyde": False, "graph": False, "rerank": False, "summary": True,  "gtcc": False},
    {"name": "12. Hybrid + Graph + T5",         "bm25": True,  "vector": True,  "hyde": False, "graph": True,  "rerank": False, "summary": True,  "gtcc": False},
    {"name": "13. Hybrid + Graph + Rerank + T5","bm25": True,  "vector": True,  "hyde": False, "graph": True,  "rerank": True,  "summary": True,  "gtcc": False},
    {"name": "14. Everything + T5",             "bm25": True,  "vector": True,  "hyde": True,  "graph": True,  "rerank": True,  "summary": True,  "gtcc": False},
    # GTCC experiments
    {"name": "15. Hybrid + GTCC",               "bm25": True,  "vector": True,  "hyde": False, "graph": False, "rerank": False, "summary": False, "gtcc": True},
    {"name": "16. Hybrid + Graph + GTCC",        "bm25": True,  "vector": True,  "hyde": False, "graph": True,  "rerank": False, "summary": False, "gtcc": True},
    {"name": "17. Hybrid + GTCC + T5",           "bm25": True,  "vector": True,  "hyde": False, "graph": False, "rerank": False, "summary": True,  "gtcc": True},
    {"name": "18. Full Stack",                   "bm25": True,  "vector": True,  "hyde": False, "graph": True,  "rerank": False, "summary": True,  "gtcc": True},
]

# ---------------------------------------------------------------------------
# Caches
# ---------------------------------------------------------------------------
HYDE_CACHE = {}
GRAPH_CACHE = {}

def prep_caches(model_str: str, artifacts_dir: Path):
    """Pre-computes HyDE documents and Graph contexts for all queries to avoid redundant LLM calls."""
    console.print("[cyan]Pre-computing HyDE & Graph queries...[/cyan]")
    g_retriever = GraphRetriever(artifacts_dir)
    for q_idx, item in enumerate(DATASET):
        q = item["q"]
        console.print(f"  Caching query {q_idx+1}/{len(DATASET)}: {q[:40]}...")
        # Cache HyDE
        HYDE_CACHE[q] = generate_hypothetical_document(q, model_str)
        # Cache Graph
        GRAPH_CACHE[q] = g_retriever.search_graph(q, model_str)
    console.print("[green]Caching complete.[/green]")

# ---------------------------------------------------------------------------
# Pipeline Local Retrieval
# ---------------------------------------------------------------------------
def gather_context_for_task(
    query: str,
    retriever: HybridRetriever,
    cfg: dict,
    artifacts_dir: Path,
    db_path: str,
) -> dict:
    """Runs retrieval steps (no LLM generation) and returns the combined context string."""
    use_bm25 = cfg["bm25"]
    use_vector = cfg["vector"]
    use_hyde = cfg["hyde"]
    use_graph = cfg["graph"]
    use_rerank = cfg["rerank"]
    use_summary = cfg.get("summary", False)
    use_gtcc = cfg.get("gtcc", False)

    vector_query = HYDE_CACHE.get(query) if use_hyde else None

    # Hybrid Retrieval
    candidate_k = 25 if use_rerank else 5
    results = retriever.search(
        query, top_k=candidate_k, use_bm25=use_bm25, use_vector=use_vector, vector_query=vector_query
    )

    chunk_ids = [r.chunk_id for r in results]
    chunks_data = get_chunks_by_ids(db_path, chunk_ids)
    chunk_map = {c.chunk_id: c.chunk_text for c in chunks_data}

    # Rerank
    if use_rerank and results:
        try:
            reranker = Reranker()
            doc_texts = [chunk_map.get(r.chunk_id) for r in results if chunk_map.get(r.chunk_id)]
            valid_results = [r for r in results if chunk_map.get(r.chunk_id)]
            if valid_results:
                results = reranker.rerank(query, doc_texts, valid_results)[:5]
        except Exception as e:
            results = results[:5]
    else:
        results = results[:5]

    context_chunks = []
    for res in results:
        text = chunk_map.get(res.chunk_id, "")
        if text:
            context_chunks.append(ContextChunk(chunk_id=res.chunk_id, text=text, score=res.score))

    # GTCC
    if use_gtcc:
        try:
            gtcc = GTCCRetriever(artifacts_dir)
            retrieved_ids = [r.chunk_id for r in results]
            expanded = gtcc.expand_results(retrieved_ids, max_bridges=3)
            bridge_ids = [cid for cid, score, source in expanded if source == "bridge"]
            if bridge_ids:
                bridge_data = get_chunks_by_ids(db_path, bridge_ids)
                bridge_map = {c.chunk_id: c.chunk_text for c in bridge_data}
                for cid, score, source in expanded:
                    if source == "bridge" and cid in bridge_map:
                        context_chunks.append(ContextChunk(chunk_id=cid, text=f"[BRIDGE CONTEXT]\n{bridge_map[cid]}", score=score))
        except Exception:
            pass

    # Graph
    if use_graph:
        graph_ctx = GRAPH_CACHE.get(query)
        if graph_ctx:
            context_chunks.append(ContextChunk(chunk_id="knowledge_graph_context", text=graph_ctx, score=1.0))

    # Summary
    if use_summary:
        try:
            import sqlite3
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT summary FROM documents WHERE summary IS NOT NULL AND summary != '' LIMIT 1")
            row = cursor.fetchone()
            conn.close()
            if row and row[0]:
                context_chunks.append(ContextChunk(chunk_id="doc_summary", text=f"[DOCUMENT SUMMARY]\n{row[0]}", score=0.9))
        except Exception:
            pass

    full_context = "\n\n".join(c.text for c in context_chunks)
    return {
        "context": full_context,
        "chunk_count": len(context_chunks)
    }

# ---------------------------------------------------------------------------
# Batch LLM Operations
# ---------------------------------------------------------------------------
def extract_json_array(text: str):
    if not text or not text.strip():
        return []
    matches = re.findall(r"```json\s*\n(.*?)\n```", text, re.DOTALL)
    if matches:
        text = matches[0]
    else:
        # try to find first [ and last ]
        start = text.find('[')
        end = text.rfind(']')
        if start != -1 and end != -1:
            text = text[start:end+1]
    return json.loads(text)


def batch_generate(llm, tasks: list, log_file: Path) -> dict:
    """Takes a list of tasks (has task_id, context, question) and asks LLM to answer all."""
    if not tasks:
        return {}
    
    prompt = "You are a highly efficient RAG QA system. You will be provided with a JSON array of tasks.\n"
    prompt += "For each task, read the 'context' and answer the 'question' using ONLY that context. Keep your 'answer' concise (1-2 sentences).\n"
    prompt += "If the context doesn't have the answer, reply 'I don't know'.\n\n"
    prompt += "You MUST return a pure JSON array of objects with keys 'task_id' and 'answer'. Do not include any other text.\n\n"
    
    # Send minimal data to save tokens
    mini_tasks = [{"task_id": t["task_id"], "question": t["question"], "context": t["context"]} for t in tasks]
    prompt += "TASKS_JSON:\n" + json.dumps(mini_tasks, indent=2) + "\n"

    res = llm.generate(prompt)
    
    # Save to log
    with open(log_file, "a", encoding="utf-8") as f:
        f.write("\n" + "="*80 + "\n")
        f.write(f"TIMESTAMP: {datetime.now().isoformat()}\n")
        f.write("TYPE: GENERATION BATCH\n")
        f.write("-" * 40 + " PROMPT " + "-" * 40 + "\n")
        f.write(prompt + "\n")
        f.write("-" * 39 + " RESPONSE " + "-" * 39 + "\n")
        f.write(res + "\n")
        f.write("="*80 + "\n")

    try:
        parsed = extract_json_array(res)
        return {item["task_id"]: item["answer"] for item in parsed if "task_id" in item}
    except Exception as e:
        console.print(f"[red]Batch Generation JSON parse failed: {e}[/red]")
        console.print(f"Raw Output start: {res[:200]}")
        return {}

def batch_judge(llm, tasks: list, log_file: Path) -> dict:
    """Takes a list of tasks (task_id, question, context, answer) and asks LLM to judge faithfulness."""
    if not tasks:
        return {}

    prompt = "You are an expert AI evaluator calculating the Faithfulness metric.\n"
    prompt += "For each task below, you are given a QUESTION, a REFERENCE CONTEXT, and the AI's ANSWER.\n"
    prompt += "Determine if the ANSWER is fully supported by the CONTEXT. If it contains hallucinated facts not in the context, output verdict NO.\n"
    prompt += "If it says 'I don't know' and it's not in the context, output verdict YES.\n\n"
    prompt += "You MUST return a pure JSON array of objects with keys 'task_id', 'verdict' ('YES' or 'NO'), and 'reasoning'.\n\n"
    
    mini_tasks = [{"task_id": t["task_id"], "question": t["question"], "context": t["context"], "answer": t["answer"]} for t in tasks]
    prompt += "TASKS_JSON:\n" + json.dumps(mini_tasks, indent=2) + "\n"

    res = llm.generate(prompt)
    
    # Save to log
    with open(log_file, "a", encoding="utf-8") as f:
        f.write("\n" + "="*80 + "\n")
        f.write(f"TIMESTAMP: {datetime.now().isoformat()}\n")
        f.write("TYPE: JUDGMENT BATCH\n")
        f.write("-" * 40 + " PROMPT " + "-" * 40 + "\n")
        f.write(prompt + "\n")
        f.write("-" * 39 + " RESPONSE " + "-" * 39 + "\n")
        f.write(res + "\n")
        f.write("="*80 + "\n")

    try:
        parsed = extract_json_array(res)
        return {item["task_id"]: item.get("verdict", "NO").upper() for item in parsed if "task_id" in item}
    except Exception as e:
        console.print(f"[red]Batch Judgment JSON parse failed: {e}[/red]")
        return {}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Batched RAG Evaluation")
    parser.add_argument("--dump-only", action="store_true", help="Dump the contexts to JSON and exit without calling LLM")
    parser.add_argument("--export-for-ai", action="store_true", help="Export contexts and ground-truth answers to a clean JSON file for external AIs")
    args = parser.parse_args()

    model_name = os.environ.get("OML_MODEL", "ollama:gemma3:270m")
    from oml.config import DEFAULT_SQLITE_PATH
    db_path = DEFAULT_SQLITE_PATH
    artifacts_dir = Path("artifacts")
    report_path = Path("reports/BATCHED_COMBINATIONS_REPORT.md")
    log_path = Path("reports/llm_interactions.log")
    
    # clear log file
    if not args.dump_only:
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(f"--- LLM BATCH LOG {datetime.now().isoformat()} ---\n")

    console.rule(f"[bold magenta]Batched RAG Evaluation — frank.txt + {model_name}[/bold magenta]")
    
    llm = get_llm_client(model_name)
    retriever = HybridRetriever(artifacts_dir)

    # 1. Prep Cache
    start_total = time.time()
    prep_caches(model_name, artifacts_dir)

    # 2. Gather Local Contexts
    console.print("[cyan]Gathering contexts for all configs locally...[/cyan]")
    all_tasks = []
    
    start_local = time.time()
    for cfg_idx, cfg in enumerate(CONFIGS):
        for q_idx, item in enumerate(DATASET):
            task_id = f"cfg{cfg_idx}_q{q_idx}"
            ctx_data = gather_context_for_task(item["q"], retriever, cfg, artifacts_dir, db_path)
            
            all_tasks.append({
                "task_id": task_id,
                "config_name": cfg["name"],
                "cfg_idx": cfg_idx,
                "question": item["q"],
                "context": ctx_data["context"],
                "chunk_count": ctx_data["chunk_count"]
            })
    console.print(f"[green]Context gathering complete in {time.time() - start_local:.2f}s.[/green]")

    if args.dump_only or args.export_for_ai:
        if args.dump_only:
            dump_path = Path("reports/batched_prompts_dump.json")
            with open(dump_path, "w", encoding="utf-8") as f:
                json.dump(all_tasks, f, indent=2)
            console.print(f"[bold green]Dumped all contexts to {dump_path}.[/bold green]")
        
        if args.export_for_ai:
            # Create a simplified version for external AI judgment
            export_tasks = []
            for t in all_tasks:
                # Find the original ground-truth answer from DATASET
                truth_ans = next((d["a"] for d in DATASET if d["q"] == t["question"]), "Unknown")
                
                export_tasks.append({
                    "task_id": t["task_id"],
                    "configuration_tested": t["config_name"],
                    "question": t["question"],
                    "ground_truth_answer": truth_ans,
                    "retrieved_context": t["context"]
                })
                
            export_path = Path("reports/ai_judge_export.json")
            
            # Add a system prompt wrapper to make it easier to paste into ChatGPT/Claude
            wrapper = {
                "instructions": "You are an expert AI Judge. Please evaluate the following RAG contexts. For each item, decide if the 'retrieved_context' contains enough information to correctly answer the 'question' based on the 'ground_truth_answer'.",
                "required_output_format": "Return a JSON array where each object has 'task_id', 'verdict' (YES/NO), and 'reasoning'.",
                "tasks": export_tasks
            }
            
            with open(export_path, "w", encoding="utf-8") as f:
                json.dump(wrapper, f, indent=2)
            console.print(f"[bold green]Exported AI Evaluation Prompt to {export_path}.[/bold green]")
            
        console.print("[bold cyan]Exiting without calling LLM.[/bold cyan]")
        return

    # 3. Batch Generation
    # Let's chunk the tasks to prevent huge API timeouts. 18 configs * 5 qs = 90 tasks. 
    # CHUNK_SIZE = 10 tasks per API call (2 configs)
    CHUNK_SIZE = 10
    console.print(f"[cyan]Dispatching batched Generations... (Chunk Size: {CHUNK_SIZE})[/cyan]")
    
    answers_map = {}
    for i in range(0, len(all_tasks), CHUNK_SIZE):
        chunk = all_tasks[i:i+CHUNK_SIZE]
        console.print(f"  > Sending Generation batch {i//CHUNK_SIZE + 1}...")
        start_chunk = time.time()
        chunk_answers = batch_generate(llm, chunk, log_path)
        answers_map.update(chunk_answers)
        console.print(f"    Batch received in {time.time() - start_chunk:.2f}s.")

    # Apply answers to tasks
    for t in all_tasks:
        t["answer"] = answers_map.get(t["task_id"], "Error or Missing JSON")

    # 4. Batch Judging
    console.print(f"[cyan]Dispatching batched Judges... (Chunk Size: {CHUNK_SIZE})[/cyan]")
    verdicts_map = {}
    for i in range(0, len(all_tasks), CHUNK_SIZE):
        chunk = all_tasks[i:i+CHUNK_SIZE]
        console.print(f"  > Sending Judgment batch {i//CHUNK_SIZE + 1}...")
        start_chunk = time.time()
        chunk_verdicts = batch_judge(llm, chunk, log_path)
        verdicts_map.update(chunk_verdicts)
        console.print(f"    Batch received in {time.time() - start_chunk:.2f}s.")

    # Apply verdicts
    for t in all_tasks:
        val = verdicts_map.get(t["task_id"], "NO")
        t["faith"] = 1.0 if val == "YES" else 0.0

    total_time = time.time() - start_total
    
    # 5. Compile Results
    results_by_cfg = {}
    for t in all_tasks:
        cname = t["config_name"]
        if cname not in results_by_cfg:
            results_by_cfg[cname] = {"faiths": [], "chunk_counts": [], "details": []}
        results_by_cfg[cname]["faiths"].append(t["faith"])
        results_by_cfg[cname]["chunk_counts"].append(t["chunk_count"])
    
    # Format for Report
    report_rows = []
    for cname, data in results_by_cfg.items():
        avg_f = sum(data["faiths"]) / len(data["faiths"])
        avg_c = sum(data["chunk_counts"]) / len(data["chunk_counts"])
        report_rows.append({
            "name": cname,
            "avg_faith": avg_f,
            "avg_chunks": avg_c,
            "avg_lat": total_time / len(CONFIGS) # rough amortized
        })
    
    report_rows.sort(key=lambda r: (-r["avg_faith"], r["avg_chunks"]))

    console.print("\n[bold]--- Final Batched Results ---[/bold]")
    table = Table()
    table.add_column("Rank")
    table.add_column("Config Name")
    table.add_column("Avg Faith")
    table.add_column("Avg Chunks")
    
    for i, r in enumerate(report_rows, 1):
        style = "green" if r["avg_faith"] >= 0.8 else "yellow" if r["avg_faith"] >= 0.5 else "red"
        table.add_row(str(i), r["name"], f"[{style}]{r['avg_faith']:.2f}[/{style}]", f"{r['avg_chunks']:.1f}")
    
    console.print(table)
    console.print(f"\n[bold green]Total batched eval completed in {total_time/60:.2f} minutes.[/bold green]")

if __name__ == "__main__":
    main()
