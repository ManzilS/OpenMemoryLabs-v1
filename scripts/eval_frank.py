import sys
from pathlib import Path

# Ensure 'oml' module can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from oml.eval.ablations import run_pipeline
from oml.eval.base import EvalResult
from oml.llm import get_llm_client
from oml.retrieval.hybrid import HybridRetriever
from oml.retrieval.graph_retriever import GraphRetriever
from oml.memory.context import ContextBudgeter, ContextChunk
from oml.storage.sqlite import get_chunks_by_ids, get_notes_by_ids
from oml.retrieval.hyde import generate_hypothetical_document
from oml.retrieval.rerank import Reranker
from oml.eval.tasks.faithfulness import FaithfulnessTask
import time
from rich.console import Console
from rich.table import Table

console = Console()

# Test dataset based on frank.txt
DATASET = [
    {"q": "Where is the narrator travelling to?", "a": "The North Pole."},
    {"q": "How was the European stranger rescued?", "a": "He was found on a sledge drifting on ice and brought aboard the ship."},
    {"q": "What is the narrator's name?", "a": "Robert Walton."},
    {"q": "Why does Robert Walton feel he needs a friend?", "a": "To share his joy, sympathize with him, and amend his plans."},
    {"q": "Who is Margaret?", "a": "Margaret is Robert Walton's sister."}
]

def run_frank_pipeline(
    query: str,
    retriever: HybridRetriever,
    llm,
    use_bm25: bool,
    use_vector: bool,
    use_hyde: bool,
    use_graph: bool,
    use_rerank: bool
):
    start_time = time.time()
    artifacts_dir = Path("artifacts")
    from oml.config import DEFAULT_SQLITE_PATH
    db_path = DEFAULT_SQLITE_PATH
    
    # 1. HyDE
    vector_query = None
    if use_hyde:
        vector_query = generate_hypothetical_document(query, llm.model_name)
    
    # 2. Retrieve Hybrid
    candidate_k = 25 if use_rerank else 5
    results = retriever.search(query, top_k=candidate_k, use_bm25=use_bm25, use_vector=use_vector, vector_query=vector_query)
    
    # 3. Fetch current chunk content
    context_chunks = []
    chunk_ids = [r.chunk_id for r in results]
    chunks_data = get_chunks_by_ids(db_path, chunk_ids)
    
    # 4. Rerank
    if use_rerank and results:
        reranker = Reranker()
        doc_texts = []
        valid_results = []
        chunk_map = {c.chunk_id: c.chunk_text for c in chunks_data}
        for r in results:
            if chunk_map.get(r.chunk_id):
                doc_texts.append(chunk_map[r.chunk_id])
                valid_results.append(r)
        
        try:
            results = reranker.rerank(query, doc_texts, valid_results)[:5]
        except Exception as e:
            results = results[:5]
    else:
        results = results[:5]

    # Convert results to ContextChunks
    chunk_map = {c.chunk_id: c.chunk_text for c in chunks_data}
    for res in results:
        text = chunk_map.get(res.chunk_id, "")
        context_chunks.append(ContextChunk(chunk_id=res.chunk_id, text=text, score=res.score))

    # 5. Graph
    if use_graph:
        g_retriever = GraphRetriever(artifacts_dir)
        graph_context = g_retriever.search_graph(query, llm.model_name)
        if graph_context:
            context_chunks.append(
                ContextChunk(
                    chunk_id="knowledge_graph_context",
                    text=graph_context,
                    score=1.0 # High priority
                )
            )

    # 6. Generate
    budgeter = ContextBudgeter()
    prompt = budgeter.construct_prompt(query, context_chunks)
    answer = llm.generate(prompt)
    
    total_time = time.time() - start_time
    # Map back contexts
    full_context_text = "\n\n".join([c.text for c in context_chunks])
    
    return {
        "answer": answer,
        "context": full_context_text,
        "latency": total_time,
        "retrieve_count": len(context_chunks)
    }

def main():
    configs = [
        {"name": "Hybrid + Reranking", "bm25": True, "vector": True, "hyde": False, "graph": False, "rerank": True},
        {"name": "HyDE + Hybrid", "bm25": True, "vector": True, "hyde": True, "graph": False, "rerank": False},
        {"name": "Graph + Hybrid", "bm25": True, "vector": True, "hyde": False, "graph": True, "rerank": False},
        {"name": "The Everything Combo", "bm25": True, "vector": True, "hyde": True, "graph": True, "rerank": True},
    ]

    model_name = "ollama:gemma3:270m"  # Local test
    llm = get_llm_client(model_name)
    artifacts_dir = Path("artifacts")
    retriever = HybridRetriever(artifacts_dir)
    judge = FaithfulnessTask()

    summary_table = Table(title="Frank.txt Combination Evaluation")
    summary_table.add_column("Config", style="cyan")
    summary_table.add_column("Faithfulness Score")
    summary_table.add_column("Avg Latency (s)")
    summary_table.add_column("Avg Context Size")

    for cfg in configs:
        console.rule(f"[bold blue]Running Config: {cfg['name']}")
        scores = []
        latencies = []
        counts = []

        for item in DATASET:
            try:
                res = run_frank_pipeline(
                    item["q"],
                    retriever,
                    llm,
                    use_bm25=cfg["bm25"],
                    use_vector=cfg["vector"],
                    use_hyde=cfg["hyde"],
                    use_graph=cfg["graph"],
                    use_rerank=cfg["rerank"]
                )
                
                # Check faithfulness
                verdict, _ = judge._evaluate_single(llm, item["q"], res["answer"], res["context"])
                score = 1.0 if verdict == "YES" else 0.0
                
                scores.append(score)
                latencies.append(res["latency"])
                counts.append(res["retrieve_count"])
                
                console.print(f"  Q: {item['q'][:40]}... | F-Score: {score} | Latency: {res['latency']:.2f}s")
            except Exception as e:
                console.print(f"[red]Error on {item['q']}: {e}[/red]")

        avg_score = sum(scores) / len(scores) if scores else 0
        avg_lat = sum(latencies) / len(latencies) if latencies else 0
        avg_count = sum(counts) / len(counts) if counts else 0
        
        summary_table.add_row(
            cfg["name"],
            f"{avg_score:.2f}",
            f"{avg_lat:.2f}",
            f"{avg_count:.1f}"
        )

    console.print("\n")
    console.print(summary_table)

if __name__ == "__main__":
    main()
