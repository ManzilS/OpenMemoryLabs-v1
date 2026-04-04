import time
from typing import Any
from pathlib import Path

from oml.eval.base import EvalTask, EvalResult, ModelInterface
from oml.eval.run import register_task

@register_task("oml_vs_rag")
class OmlVsRagTask(EvalTask):
    name = "oml_vs_rag"

    def run(self, model: ModelInterface, config: dict[str, Any]) -> EvalResult:
        """
        Compares Standard RAG (Vector-only, top-k) vs OpenMemoryLab (Hybrid + Sleep + Budget).
        """
        questions = [
            # Requires specific Wikipedia details about Turing / Bletchley Park
            "What specific mathematical and mechanical techniques did Alan Turing develop at Bletchley Park to break the Enigma cipher, and why were those techniques effective against the German encryption system?",
            # Requires specific details about the 1813 Blue Mountains crossing
            "Who were the three European explorers who first crossed Australia's Blue Mountains in 1813, what route did they take, and what did they find on the other side that was significant for the colony?",
            # Requires specific Aum Shinrikyo / Tokyo attack details
            "What was the name of the Japanese doomsday cult that carried out the 1995 Tokyo subway attack, who led it, what agent did they release, and how many stations were affected?",
            # Requires specific Ovambo / Namibia demographic details
            "What percentage of Namibia's total population do the Ovambo people comprise, in which geographic regions of Namibia do they traditionally live, and to which African language family do their languages belong?",
            # Requires specific Empusa / Greek mythology details
            "In Greek mythology, what were the three physical forms that Empusa could assume, which goddess did she serve, and what was her primary purpose according to the myths?",
        ]
        
        try:
            from oml.retrieval.hybrid import HybridRetriever
            from oml.retrieval.rerank import Reranker
            from oml.memory.context import ContextBudgeter, ContextChunk
            from oml.storage.sqlite import get_chunks_by_ids, get_notes_by_ids
            
            artifacts_dir = Path("artifacts")
            if not artifacts_dir.exists():
                 return EvalResult(self.name, 0.0, {"error": "No artifacts found."})
                 
            retriever = HybridRetriever(artifacts_dir)
            try:
                reranker = Reranker()
            except ImportError:
                print("Warning: Reranker not available. Install sentence-transformers.")
                reranker = None
            budgeter = ContextBudgeter()
            from oml.config import DEFAULT_SQLITE_PATH
            db_path = DEFAULT_SQLITE_PATH
            
        except Exception as e:
            return EvalResult(self.name, 0.0, {"error": f"Setup failed: {e}"})
            
        # We need a judge model to score A vs B
        # Using the same model as judge for simplicity, could be separate
        judge = model 

        results = {}
        oml_wins = 0
        rag_wins = 0
        ties = 0
        
        total_oml_tokens = 0
        total_rag_tokens = 0
        
        for q in questions:
            # --- 1. Standard RAG Pipeline ---
            # Strategy: Vector search only, simple concatenation
            rag_start = time.time()
            rag_docs = retriever.search(q, top_k=5, use_bm25=False, use_vector=True)
            rag_context = ""
            if rag_docs:
                rag_ids = [r.chunk_id for r in rag_docs]
                rag_data = get_chunks_by_ids(db_path, rag_ids)
                for r in rag_docs:
                    d = next((x for x in rag_data if x.chunk_id == r.chunk_id), None)
                    if d:
                        rag_context += d.chunk_text + "\n\n"
            
            rag_prompt = f"Context:\n{rag_context}\n\nQuestion: {q}\nAnswer:"
            rag_answer = model.generate(rag_prompt)
            rag_time = time.time() - rag_start
            rag_tokens = len(rag_prompt) // 4 # heuristic
            total_rag_tokens += rag_tokens

            # --- 2. OML Pipeline ---
            # Strategy: Hybrid + Sleep + Budgeting
            oml_start = time.time()
            # Retrieve
            oml_docs = retriever.search(q, top_k=10, use_bm25=True, use_vector=True)
            oml_notes = retriever.search_notes(q, top_k=2)
            
            # Construct Context Items
            context_chunks = []
            
            # Add Notes (High Priority)
            if oml_notes:
                 note_ids = [n.chunk_id for n in oml_notes if n.score > 0.35]
                 if note_ids:
                     notes_data = get_notes_by_ids(db_path, note_ids)
                     for n in notes_data:
                         text = f"[MEMORY NOTE] {n.content}"
                         context_chunks.append(ContextChunk(n.note_id, text, 1.0))
            
            # Add Chunks
            # Add Chunks with Reranking
            if oml_docs:
                ids = [r.chunk_id for r in oml_docs]
                data = get_chunks_by_ids(db_path, ids)
                
                # Pair content with results
                doc_texts = []
                valid_results = []
                chunk_map = {}
                
                for r in oml_docs:
                    d = next((x for x in data if x.chunk_id == r.chunk_id), None)
                    if d:
                        doc_texts.append(d.chunk_text)
                        valid_results.append(r)
                        chunk_map[r.chunk_id] = d.chunk_text

                # Rerank if available
                final_rankings = valid_results
                if reranker and doc_texts:
                    final_rankings = reranker.rerank(q, doc_texts, valid_results)
                    # Take top 10 after reranking
                    final_rankings = final_rankings[:10]

                for r in final_rankings:
                    text = chunk_map.get(r.chunk_id, "")
                    if text:
                        context_chunks.append(ContextChunk(r.chunk_id, text, r.score))
            
            # Budget
            oml_prompt = budgeter.construct_prompt(q, context_chunks) 
            oml_answer = model.generate(oml_prompt)
            oml_time = time.time() - oml_start
            oml_tokens = len(oml_prompt) // 4
            total_oml_tokens += oml_tokens
            
            # --- 3. Judgment ---
            judge_prompt = f"""
            Compare two answers to the following question.
            
            QUESTION: {q}
            
            ANSWER A (Standard RAG):
            {rag_answer}
            
            ANSWER B (OpenMemoryLab):
            {oml_answer}
            
            Which answer is better? Consider:
            1. Completeness
            2. Grounding (use of context)
            3. Relevance
            
            Output strictly one of: "WIN: A", "WIN: B", "TIE".
            Then explain why in one sentence.
            """
            
            judgment = judge.generate(judge_prompt)
            
            winner = "TIE"
            if "WIN: A" in judgment:
                winner = "RAG"
                rag_wins += 1
            elif "WIN: B" in judgment:
                winner = "OML"
                oml_wins += 1
            else:
                ties += 1

            results[q] = {
                "winner": winner,
                "judgment": judgment.strip(),
                "rag": {"time": rag_time, "tokens": rag_tokens},
                "oml": {"time": oml_time, "tokens": oml_tokens}
            }
            
        # Overall Score: OML Win Rate
        score = oml_wins / len(questions) if questions else 0.0
        
        # Summary details
        details = {
            "individual_results": results,
            "metrics": {
                "oml_wins": oml_wins,
                "rag_wins": rag_wins,
                "ties": ties,
                "avg_oml_tokens": total_oml_tokens / len(questions) if questions else 0,
                "avg_rag_tokens": total_rag_tokens / len(questions) if questions else 0
            }
        }
        
        return EvalResult(self.name, score, details)
