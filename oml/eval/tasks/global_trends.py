from typing import Any
from oml.eval.base import EvalTask, EvalResult, ModelInterface
from oml.eval.run import register_task

@register_task("global_trends")
class GlobalTrendsTask(EvalTask):
    name = "global_trends"

    def run(self, model: ModelInterface, config: dict[str, Any]) -> EvalResult:
        """
        Evaluates the system's ability to detect global trends over time.
        """
        # Questions that surface temporal change across multiple Wikipedia corpus documents
        questions = [
            "How has the role of military special operations forces evolved in the Canadian Armed Forces?",
            "What changes in civil rights demonstration tactics occurred during the Birmingham Campaign?",
            "How did codebreaking methods develop and change during World War II at Bletchley Park?",
        ]
        
        results = {}
        passed_count = 0
        
        # Ideally, we would use the full 'culture drift' pipeline here.
        # Since that might not be fully ready, we will attempt a standard RAG query
        # and check if the answer contains temporal markers (e.g., "initially", "later", "increased").
        
        # We use the passed 'model' to generate the answer.
        # But wait, the 'model' is just an LLM. We need context!
        # We will assume for this Eval Task that we just want to check if the LLM *can* 
        # structure a trend answer if we give it a mocked context or if we just ask it 
        # (simulating it reading the whole training data which it hasn't).
        
        # BETTER APPROACH for this Phase: 
        # Instantiate the Retriever here, fetch relevant chunks, and ask the model.
        
        try:
            from pathlib import Path
            from oml.retrieval.hybrid import HybridRetriever
            from oml.memory.context import ContextBudgeter, ContextChunk
            from oml.storage.sqlite import get_chunks_by_ids
            
            # Setup Pipeline components
            artifacts_dir = Path("artifacts")
            if not artifacts_dir.exists():
                return EvalResult(self.name, 0.0, {"error": "No artifacts found. Run ingest first."})
                
            retriever = HybridRetriever(artifacts_dir)
            budgeter = ContextBudgeter()
            from oml.config import DEFAULT_SQLITE_PATH
            db_path = DEFAULT_SQLITE_PATH
            
        except ImportError:
            return EvalResult(self.name, 0.0, {"error": "Could not import core modules."})
        except Exception as e:
            return EvalResult(self.name, 0.0, {"error": f"Setup failed: {e}"})

        for q in questions:
            # 1. Retrieve
            # We want broad retrieval for trends
            search_res = retriever.search(q, top_k=10)
            
            context_chunks = []
            if search_res:
                ids = [r.chunk_id for r in search_res]
                data = get_chunks_by_ids(db_path, ids)
                for r in search_res:
                    d = next((x for x in data if x.chunk_id == r.chunk_id), None)
                    if d:
                        context_chunks.append(ContextChunk(r.chunk_id, d.chunk_text, r.score))
            
            # 2. Construct Prompt
            # We ask the model to answer specifically focusing on change over time
            prompt = budgeter.construct_prompt(q, context_chunks)
            prompt += "\nIMPORTANT: Focus on describing the change or trend over time."
            
            # 3. Generate
            answer = model.generate(prompt)
            
            # 4. Score (Heuristic: detection of change words)
            change_words = ["changed", "increased", "decreased", "shifted", "initially", "later", "became", "trend"]
            has_change = any(w in answer.lower() for w in change_words)
            
            results[q] = {
                "answer_snippet": answer[:100],
                "has_trend_words": has_change
            }
            
            if has_change:
                passed_count += 1
                
        score = passed_count / len(questions) if questions else 0.0
        
        return EvalResult(
            task_name=self.name, 
            score=score, 
            details=results
        )
