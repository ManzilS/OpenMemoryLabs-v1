import time
from pathlib import Path
from rich.console import Console
from rich.table import Table
from typing import Dict, Any

from oml.retrieval.hybrid import HybridRetriever
from oml.memory.context import ContextBudgeter, ContextChunk
from oml.storage.sqlite import get_chunks_by_ids, get_notes_by_ids
from oml.eval.tasks.faithfulness import FaithfulnessTask
from oml.eval.base import EvalTask, EvalResult, ModelInterface
from oml.eval.run import register_task

console = Console()

# Synthetic Dataset for Enron
DATASET = [
    {"q": "Who is the CEO of Enron?", "a": ["Skilling", "Lay"]},
    {"q": "What happened with the Raptor vehicles?", "a": ["hedging", "debt", "SPE"]},
    {"q": "Who sent the email about the Westgate Proforma?", "a": ["Phillip", "Allen"]},
    {"q": "Is there a parking issue at Enron?", "a": ["parking", "transportation"]},
    {"q": "What is the California energy crisis impact?", "a": ["California", "prices"]}
]

def run_pipeline(
    query: str, 
    retriever: HybridRetriever, 
    llm, 
    use_bm25: bool, 
    use_vector: bool, 
    use_sleep: bool
) -> Dict:
    """Runs Retrieval -> Generation and returns context + answer."""
    
    start_time = time.time()
    
    # 1. Retrieve Chunks
    # Only search if at least one method is active
    if not use_bm25 and not use_vector:
        results = []
    else:
        results = retriever.search(query, top_k=5, use_bm25=use_bm25, use_vector=use_vector)
    
    # 2. Retrieve Notes (Sleep)
    note_results = []
    if use_sleep:
        note_results = retriever.search_notes(query, top_k=2)
        
    # 3. Context Construction
    context_chunks = []
    from oml.config import DEFAULT_SQLITE_PATH
    db_path = DEFAULT_SQLITE_PATH
    
    # Inject Notes
    full_context_text = ""
    
    if note_results:
        note_ids = [n.chunk_id for n in note_results if n.score > 0.35]
        if note_ids:
            notes_data = get_notes_by_ids(db_path, note_ids)
            for n_data in notes_data:
                text = f"[MEMORY NOTE]\n{n_data.content}"
                full_context_text += text + "\n\n"
                context_chunks.append(ContextChunk(chunk_id=n_data.note_id, text=text, score=1.0))

    # Add Chunks
    if results:
         chunk_ids = [r.chunk_id for r in results]
         chunks_data = get_chunks_by_ids(db_path, chunk_ids)
         for res in results:
             c_data = next((c for c in chunks_data if c.chunk_id == res.chunk_id), None)
             if c_data:
                 full_context_text += c_data.chunk_text + "\n\n"
                 context_chunks.append(ContextChunk(chunk_id=res.chunk_id, text=c_data.chunk_text, score=res.score))
                 
    # 4. Generate
    budgeter = ContextBudgeter()
    prompt = budgeter.construct_prompt(query, context_chunks)
    
    # Use the passed LLM generator (or we could use model from args)
    # The 'llm' argument here is expected to be a ModelInterface-like object
    answer = llm.generate(prompt)
    
    total_time = time.time() - start_time
    
    return {
        "answer": answer,
        "context": full_context_text,
        "latency": total_time,
        "retrieve_count": len(results) + len(note_results)
    }

@register_task("ablations")
class AblationsTask(EvalTask):
    name = "ablations"

    def run(self, model: ModelInterface, config: dict[str, Any]) -> EvalResult:
        """Orchestrates the ablation study."""
        limit = config.get("limit", 100)
        
        configs = [
            {"name": "BM25 Only", "bm25": True, "vector": False, "sleep": False},
            {"name": "Vector Only", "bm25": False, "vector": True, "sleep": False},
            {"name": "Hybrid", "bm25": True, "vector": True, "sleep": False},
            {"name": "Hybrid + Sleep", "bm25": True, "vector": True, "sleep": True},
        ]
        
        artifacts_dir = Path("artifacts")
        if not artifacts_dir.exists():
             return EvalResult(self.name, 0.0, {"error": "No artifacts found."})

        retriever = HybridRetriever(artifacts_dir)
        # We use the passed 'model' as the generator
        
        # We need a judge. We can re-use the passed model as judge or instantiate a new one.
        # For simplicity, reuse the same model interface (which wraps an LLM).
        # But FaithfulnessEvaluator used get_llm_client internally.
        # Let's adapt FaithfulnessTask to be used here.
        judge_task = FaithfulnessTask() 
        # But judge_task.run signature is different.
        # The original code instantiated FaithfulnessEvaluator. 
        # Since I refactored FaithfulnessEvaluator -> FaithfulnessTask, 
        # I should probably use `_evaluate_single` logic or instantiate a helper.
        # I'll add a helper method to FaithfulnessTask or just use logic inline.
        # Better: use the FaithfulnessTask just for its logic.
        
        summary_table = Table(title="Ablation Results")
        summary_table.add_column("Config")
        summary_table.add_column("Faithfulness")
        summary_table.add_column("Latency (s)")
        summary_table.add_column("Retrieved")
        
        dataset = DATASET[:limit]
        
        details = {}
        
        for cfg in configs:
            console.rule(f"[bold blue]Running Config: {cfg['name']}")
            
            scores = []
            latencies = []
            counts = []
            
            for item in dataset:
                try:
                    res = run_pipeline(
                        item["q"], 
                        retriever, 
                        model, # Use passed model
                        use_bm25=cfg["bm25"], 
                        use_vector=cfg["vector"], 
                        use_sleep=cfg["sleep"]
                    )
                    
                    # Eval Faithfulness
                    # We can use the judge task's logic if we exposure it.
                    # Or just instantiate a judge client:
                    # But 'model' is our model interface.
                    verdict, _ = judge_task._evaluate_single(model, item["q"], res["answer"], res["context"])
                    
                    score = 1.0 if verdict == "YES" else 0.0
                    
                    scores.append(score)
                    latencies.append(res["latency"])
                    counts.append(res["retrieve_count"])
                    
                    print(f"  Q: {item['q'][:30]}... | F: {score} | T: {res['latency']:.2f}s")
                    
                except Exception as e:
                    console.print(f"[red]Error processing {item['q']}: {e}[/red]")
            
            avg_score = sum(scores) / len(scores) if scores else 0
            avg_lat = sum(latencies) / len(latencies) if latencies else 0
            avg_count = sum(counts) / len(counts) if counts else 0
            
            summary_table.add_row(
                cfg["name"], 
                f"{avg_score:.2f}", 
                f"{avg_lat:.2f}", 
                f"{avg_count:.1f}"
            )
            
            details[cfg["name"]] = {
                "avg_score": avg_score,
                "avg_latency": avg_lat,
                "avg_count": avg_count
            }
            
        console.print("\n")
        console.print(summary_table)
        
        # return result of the *last* config or an aggregate? 
        # EvalResult expects one score. Let's return the Hybrid score as the "main" score.
        hybrid_score = details.get("Hybrid", {}).get("avg_score", 0.0)
        
        return EvalResult(self.name, hybrid_score, details)
