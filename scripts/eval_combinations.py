"""
Comprehensive RAG Strategy Combination Evaluation
==================================================
Tests all meaningful combinations of retrieval strategies against the
Frankenstein letters dataset using ollama:gemma3:270m.

Usage:
    uv run --python 3.11 python scripts/eval_combinations.py
"""

import sys
import time
import json
from pathlib import Path
from datetime import datetime

# Ensure 'oml' module can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Load .env
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
from oml.eval.tasks.faithfulness import FaithfulnessTask

from rich.console import Console
from rich.table import Table

console = Console()

# ---------------------------------------------------------------------------
# Test Dataset (based on frank.txt — Robert Walton's letters)
# ---------------------------------------------------------------------------
DATASET = [
    {
        "q": "Where is the narrator travelling to?",
        "a": "The North Pole.",
    },
    {
        "q": "How was the European stranger rescued?",
        "a": "He was found on a sledge drifting on ice and brought aboard the ship.",
    },
    {
        "q": "What is the narrator's name?",
        "a": "Robert Walton.",
    },
    {
        "q": "Why does Robert Walton feel he needs a friend?",
        "a": "To share his joy, sympathize with him, and amend his plans.",
    },
    {
        "q": "Who is Margaret?",
        "a": "Margaret is Robert Walton's sister.",
    },
]

# ---------------------------------------------------------------------------
# All 10 Combinations
# ---------------------------------------------------------------------------
CONFIGS = [
    # --- Original 10 ---
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
    # --- T5 Summary experiments ---
    {"name": "11. Hybrid + T5 Summary",        "bm25": True,  "vector": True,  "hyde": False, "graph": False, "rerank": False, "summary": True,  "gtcc": False},
    {"name": "12. Hybrid + Graph + T5",         "bm25": True,  "vector": True,  "hyde": False, "graph": True,  "rerank": False, "summary": True,  "gtcc": False},
    {"name": "13. Hybrid + Graph + Rerank + T5","bm25": True,  "vector": True,  "hyde": False, "graph": True,  "rerank": True,  "summary": True,  "gtcc": False},
    {"name": "14. Everything + T5",             "bm25": True,  "vector": True,  "hyde": True,  "graph": True,  "rerank": True,  "summary": True,  "gtcc": False},
    # --- GTCC experiments (novel) ---
    {"name": "15. Hybrid + GTCC",               "bm25": True,  "vector": True,  "hyde": False, "graph": False, "rerank": False, "summary": False, "gtcc": True},
    {"name": "16. Hybrid + Graph + GTCC",        "bm25": True,  "vector": True,  "hyde": False, "graph": True,  "rerank": False, "summary": False, "gtcc": True},
    {"name": "17. Hybrid + GTCC + T5",           "bm25": True,  "vector": True,  "hyde": False, "graph": False, "rerank": False, "summary": True,  "gtcc": True},
    {"name": "18. Full Stack (Graph+GTCC+T5)",   "bm25": True,  "vector": True,  "hyde": False, "graph": True,  "rerank": False, "summary": True,  "gtcc": True},
]

# ---------------------------------------------------------------------------
# Pipeline runner (same logic as eval_frank.py)
# ---------------------------------------------------------------------------
def run_pipeline(
    query: str,
    retriever: HybridRetriever,
    llm,
    cfg: dict,
    artifacts_dir: Path,
    db_path: str,
    model_str: str = "ollama:gemma3:270m",
):
    """Runs a single query through the configured pipeline and returns metrics."""
    start_time = time.time()

    use_bm25  = cfg["bm25"]
    use_vector = cfg["vector"]
    use_hyde   = cfg["hyde"]
    use_graph  = cfg["graph"]
    use_rerank = cfg["rerank"]
    use_summary = cfg.get("summary", False)
    use_gtcc = cfg.get("gtcc", False)

    # 1. HyDE — generate hypothetical document for vector search
    vector_query = None
    if use_hyde:
        vector_query = generate_hypothetical_document(query, model_str)

    # 2. Hybrid retrieval
    candidate_k = 25 if use_rerank else 5
    results = retriever.search(
        query,
        top_k=candidate_k,
        use_bm25=use_bm25,
        use_vector=use_vector,
        vector_query=vector_query,
    )

    # 3. Fetch chunk text from DB
    chunk_ids = [r.chunk_id for r in results]
    chunks_data = get_chunks_by_ids(db_path, chunk_ids)
    chunk_map = {c.chunk_id: c.chunk_text for c in chunks_data}

    # 4. Rerank
    if use_rerank and results:
        try:
            reranker = Reranker()
            doc_texts = []
            valid_results = []
            for r in results:
                txt = chunk_map.get(r.chunk_id)
                if txt:
                    doc_texts.append(txt)
                    valid_results.append(r)
            if valid_results:
                results = reranker.rerank(query, doc_texts, valid_results)[:5]
        except Exception as e:
            console.print(f"[yellow]  Rerank fallback: {e}[/yellow]")
            results = results[:5]
    else:
        results = results[:5]

    # 5. Build context chunks
    context_chunks = []
    for res in results:
        text = chunk_map.get(res.chunk_id, "")
        if text:
            context_chunks.append(ContextChunk(chunk_id=res.chunk_id, text=text, score=res.score))

    # 5b. GTCC — expand with bridge chunks
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
                        context_chunks.append(
                            ContextChunk(chunk_id=cid, text=f"[BRIDGE CONTEXT]\n{bridge_map[cid]}", score=score)
                        )
        except Exception as e:
            console.print(f"[yellow]  GTCC fallback: {e}[/yellow]")

    # 6. Knowledge Graph context injection
    if use_graph:
        g_retriever = GraphRetriever(artifacts_dir)
        graph_context = g_retriever.search_graph(query, model_str)
        if graph_context:
            context_chunks.append(
                ContextChunk(chunk_id="knowledge_graph_context", text=graph_context, score=1.0)
            )

    # 6b. Document summary injection
    if use_summary:
        try:
            import sqlite3
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT summary FROM documents WHERE summary IS NOT NULL AND summary != '' LIMIT 1")
            row = cursor.fetchone()
            conn.close()
            if row and row[0]:
                context_chunks.append(
                    ContextChunk(chunk_id="doc_summary", text=f"[DOCUMENT SUMMARY]\n{row[0]}", score=0.9)
                )
        except Exception as e:
            console.print(f"[yellow]  Summary injection failed: {e}[/yellow]")

    # 7. Generate answer
    budgeter = ContextBudgeter()
    prompt = budgeter.construct_prompt(query, context_chunks)
    answer = llm.generate(prompt)

    total_time = time.time() - start_time
    full_context = "\n\n".join(c.text for c in context_chunks)

    return {
        "answer": answer,
        "context": full_context,
        "latency": total_time,
        "chunk_count": len(context_chunks),
    }


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------
def write_report(results_rows: list, report_path: Path):
    """Writes the final markdown report with rankings."""
    # Sort by faithfulness desc, then latency asc
    ranked = sorted(results_rows, key=lambda r: (-r["avg_faith"], r["avg_lat"]))

    lines = [
        f"# Combination Testing Report (Frankenstein + gemma3:270m)",
        f"",
        f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Model**: `ollama:gemma3:270m`",
        f"**Dataset**: `data/books/frank.txt` (Walton's letters, 39 chunks)",
        f"**Questions**: {len(DATASET)}",
        f"",
        f"## Results (Ranked by Faithfulness)",
        f"",
        f"| Rank | Configuration | Faithfulness | Avg Latency (s) | Avg Chunks |",
        f"|------|--------------|-------------|-----------------|------------|",
    ]

    for i, row in enumerate(ranked, 1):
        medal = ""
        if i == 1:
            medal = " 🥇"
        elif i == 2:
            medal = " 🥈"
        elif i == 3:
            medal = " 🥉"
        lines.append(
            f"| {i} | **{row['name']}**{medal} | {row['avg_faith']:.2f} | {row['avg_lat']:.2f} | {row['avg_chunks']:.1f} |"
        )

    # Winner analysis
    winner = ranked[0]
    lines += [
        "",
        "---",
        "",
        "## Winner Analysis",
        "",
        f"### 🏆 Best Combination: **{winner['name']}**",
        "",
        f"- **Faithfulness Score**: {winner['avg_faith']:.2f}",
        f"- **Average Latency**: {winner['avg_lat']:.2f}s",
        f"- **Average Chunks**: {winner['avg_chunks']:.1f}",
        "",
    ]

    # Per-question detail for winner
    if winner.get("details"):
        lines += [
            "### Per-Question Breakdown (Winner)",
            "",
            "| Question | Faithful? | Latency |",
            "|----------|-----------|---------|",
        ]
        for d in winner["details"]:
            status = "✅" if d["faith"] == 1.0 else "❌"
            lines.append(f"| {d['q'][:50]}... | {status} | {d['lat']:.2f}s |")

    # Comparison notes
    lines += [
        "",
        "---",
        "",
        "## Strategy Notes",
        "",
        "- **BM25-Only**: Fast keyword matching, no semantic understanding.",
        "- **Vector-Only**: Dense semantic search using `all-MiniLM-L6-v2` embeddings via FAISS.",
        "- **HyDE**: Adds LLM pre-generation step to expand query semantics; adds latency but can improve recall on conceptual questions.",
        "- **Graph**: Injects structured entity-relationship facts directly into the context, bypassing statistical retrieval for known entities.",
        "- **Rerank**: Cross-encoder attention pass on candidates to boost precision; requires PyTorch.",
        "",
    ]

    report_path.write_text("\n".join(lines), encoding="utf-8")
    console.print(f"\n[green]Report written to {report_path}[/green]")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    model_name = os.environ.get("OML_MODEL", "ollama:gemma3:270m")
    model_display = model_name.split(":")[0] if ":" in model_name else model_name
    from oml.config import DEFAULT_SQLITE_PATH
    db_path = DEFAULT_SQLITE_PATH
    artifacts_dir = Path("artifacts")
    report_path = Path("reports/COMBINATIONS_REPORT.md")

    console.rule(f"[bold magenta]RAG Combination Evaluation — frank.txt + {model_name}[/bold magenta]")
    console.print(f"Model: {model_name}")
    console.print(f"Configurations: {len(CONFIGS)}")
    console.print(f"Questions per config: {len(DATASET)}")
    console.print(f"Total LLM calls: ~{len(CONFIGS) * len(DATASET) * 2} (generation + judge)")
    console.print()

    llm = get_llm_client(model_name)
    retriever = HybridRetriever(artifacts_dir)
    judge = FaithfulnessTask()

    # Summary table
    summary_table = Table(title="Frank.txt — All Combination Results")
    summary_table.add_column("Config", style="cyan", min_width=22)
    summary_table.add_column("Faithfulness", justify="center")
    summary_table.add_column("Avg Latency (s)", justify="center")
    summary_table.add_column("Avg Chunks", justify="center")

    all_results = []

    for cfg in CONFIGS:
        console.rule(f"[bold blue]{cfg['name']}[/bold blue]")
        scores = []
        latencies = []
        counts = []
        details = []

        for item in DATASET:
            try:
                res = run_pipeline(item["q"], retriever, llm, cfg, artifacts_dir, db_path, model_str=model_name)

                # Judge faithfulness
                verdict, reasoning = judge._evaluate_single(llm, item["q"], res["answer"], res["context"])
                faith = 1.0 if verdict == "YES" else 0.0

                scores.append(faith)
                latencies.append(res["latency"])
                counts.append(res["chunk_count"])
                details.append({"q": item["q"], "faith": faith, "lat": res["latency"]})

                icon = "✅" if faith == 1.0 else "❌"
                console.print(
                    f"  {icon} Q: {item['q'][:45]}  |  Faith: {faith}  |  Lat: {res['latency']:.2f}s  |  Chunks: {res['chunk_count']}"
                )
            except Exception as e:
                console.print(f"  [red]ERROR: {item['q'][:40]} — {e}[/red]")

        avg_faith = sum(scores) / len(scores) if scores else 0
        avg_lat = sum(latencies) / len(latencies) if latencies else 0
        avg_chunks = sum(counts) / len(counts) if counts else 0

        style = "bold green" if avg_faith >= 0.8 else ("yellow" if avg_faith >= 0.5 else "red")
        summary_table.add_row(cfg["name"], f"[{style}]{avg_faith:.2f}[/{style}]", f"{avg_lat:.2f}", f"{avg_chunks:.1f}")

        all_results.append({
            "name": cfg["name"],
            "avg_faith": avg_faith,
            "avg_lat": avg_lat,
            "avg_chunks": avg_chunks,
            "details": details,
        })

    console.print("\n")
    console.print(summary_table)

    # Write report
    write_report(all_results, report_path)

    # Print winner
    ranked = sorted(all_results, key=lambda r: (-r["avg_faith"], r["avg_lat"]))
    console.print(f"\n[bold green]🏆 WINNER: {ranked[0]['name']} (Faithfulness: {ranked[0]['avg_faith']:.2f}, Latency: {ranked[0]['avg_lat']:.2f}s)[/bold green]")


if __name__ == "__main__":
    main()
