import time
from typing import Any
from pathlib import Path

from oml.eval.base import EvalTask, EvalResult, ModelInterface
from oml.eval.run import register_task

@register_task("cost_latency")
class CostLatencyTask(EvalTask):
    name = "cost_latency"

    def run(self, model: ModelInterface, config: dict[str, Any]) -> EvalResult:
        """
        Measures latency of key pipeline components: Retrieval, Prompting, Generation.
        """
        queries = config.get("queries", [
            "Who were the key figures in breaking the Enigma code at Bletchley Park?",
            "What were the major consequences of the Birmingham Campaign in 1963?",
            "Describe the structure and commands of the Canadian Armed Forces.",
        ])
        
        try:
            from oml.retrieval.hybrid import HybridRetriever
            from oml.memory.context import ContextBudgeter, ContextChunk
            from oml.storage.sqlite import get_chunks_by_ids
            
            artifacts_dir = Path("artifacts")
            if not artifacts_dir.exists():
                 return EvalResult(self.name, 0.0, {"error": "No artifacts found."})
                 
            retriever = HybridRetriever(artifacts_dir)
            budgeter = ContextBudgeter()
            from oml.config import DEFAULT_SQLITE_PATH
            db_path = DEFAULT_SQLITE_PATH

        except Exception as e:
            return EvalResult(self.name, 0.0, {"error": f"Setup failed: {e}"})
            
        metrics = {
            "retrieve_time": [],
            "db_fetch_time": [],
            "prompt_time": [],
            "generate_time": [],
            "total_time": []
        }
        
        for q in queries:
            t0 = time.time()
            
            # 1. Retrieve
            t_ret_start = time.time()
            search_res = retriever.search(q, top_k=10)
            t_ret_end = time.time()
            metrics["retrieve_time"].append(t_ret_end - t_ret_start)
            
            # 2. DB Fetch
            t_db_start = time.time()
            context_chunks = []
            if search_res:
                ids = [r.chunk_id for r in search_res]
                data = get_chunks_by_ids(db_path, ids)
                for r in search_res:
                    d = next((x for x in data if x.chunk_id == r.chunk_id), None)
                    if d:
                        context_chunks.append(ContextChunk(r.chunk_id, d.chunk_text, r.score))
            t_db_end = time.time()
            metrics["db_fetch_time"].append(t_db_end - t_db_start)
            
            # 3. Prompt Construction
            t_prompt_start = time.time()
            prompt = budgeter.construct_prompt(q, context_chunks)
            t_prompt_end = time.time()
            metrics["prompt_time"].append(t_prompt_end - t_prompt_start)
            
            # 4. Generate
            t_gen_start = time.time()
            _ = model.generate(prompt)
            t_gen_end = time.time()
            metrics["generate_time"].append(t_gen_end - t_gen_start)
            
            t_total_end = time.time()
            metrics["total_time"].append(t_total_end - t0)
            
        # Aggregate
        agg_metrics = {k: sum(v)/len(v) if v else 0.0 for k, v in metrics.items()}
        
        # Score is just 1.0 (pass) if we completed without error. 
        # Real eval might score based on SLA (e.g. latency < 2s).
        score = 1.0
        
        return EvalResult(
            task_name=self.name,
            score=score,
            details=agg_metrics
        )
