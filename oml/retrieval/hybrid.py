from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Tuple, Dict

from oml.retrieval.base import BaseRetriever, SearchResult
from oml.retrieval.bm25 import BM25Index
from oml.retrieval.vector import VectorIndex

class HybridRetriever(BaseRetriever):
    """
    Combines BM25 and Vector Search with Score Normalization.
    Strategy: (Alpha * Norm_Vector) + ((1-Alpha) * Norm_BM25)
    """
    
    def __init__(self, artifacts_dir: Path):
        self.artifacts_dir = artifacts_dir
        self.bm25_path = artifacts_dir / "bm25.pkl"
        self.vector_index_path = artifacts_dir / "vector.index"
        self.vector_map_path = artifacts_dir / "vector_map.json"
        
        self.note_index_path = artifacts_dir / "notes_vector.index"
        self.note_map_path = artifacts_dir / "notes_vector_map.json"
        
        self.bm25 = BM25Index(self.bm25_path)
        self.vector = VectorIndex(self.vector_index_path, self.vector_map_path)
        self.note_vector = VectorIndex(self.note_index_path, self.note_map_path)
        
        self._loaded = False

    def load(self):
        """Loads indices."""
        b_ok = self.bm25.load()
        v_ok = self.vector.load()
        self.note_vector.load()
        
        if b_ok:
            self._loaded = True
            if not v_ok:
                print(f"Warning: Main indices not fully loaded. BM25: {b_ok}, Vector: {v_ok}")
        else:
            self._loaded = False
            print(f"Warning: Main indices not fully loaded. BM25: {b_ok}, Vector: {v_ok}")

    def _normalize_scores(self, results: List[Tuple[str, float]]) -> Dict[str, float]:
        """Min-Max Normalization -> [0, 1]"""
        if not results:
            return {}
            
        scores = [r[1] for r in results]
        min_s = min(scores)
        max_s = max(scores)
        
        normalized = {}
        
        if max_s == min_s:
            return {r[0]: 1.0 for r in results}
            
        for cid, score in results:
            norm_score = (score - min_s) / (max_s - min_s)
            normalized[cid] = norm_score
            
        return normalized

    def search_notes(self, query: str, top_k: int = 3) -> List[SearchResult]:
        """Search specifically for MemoryNotes using vector search."""
        if not self._loaded:
            self.load()
            
        if not self.note_vector.index:
            return []
            
        vector_res = self.note_vector.search(query, top_k=top_k)
        
        results = []
        for nid, score in vector_res:
            results.append(SearchResult(
                chunk_id=nid,
                score=score,
                source="note_vector",
                details={"raw_score": score}
            ))
        return results

    def search(self, query: str, top_k: int = 10, alpha: float = 0.5, use_bm25: bool = True, use_vector: bool = True, vector_query: str = None) -> List[SearchResult]:
        """
        Performs hybrid search.
        If vector_query is provided, it is used for the dense vector search (e.g. for HyDE),
        while the original 'query' is used for BM25.
        """
        if not self._loaded:
            self.load()

        if not use_bm25 and not use_vector:
            return []

        # Retrieve more candidates to ensure overlap
        candidate_k = top_k * 2
        
        bm25_res = []
        vector_res = []
        v_query = vector_query if vector_query else query

        if use_bm25 and use_vector:
            # Run BM25 and vector search concurrently — they are independent
            with ThreadPoolExecutor(max_workers=2) as _pool:
                _bm25_fut = _pool.submit(self.bm25.search, query, candidate_k)
                _vec_fut  = _pool.submit(self.vector.search, v_query, candidate_k)
                bm25_res   = _bm25_fut.result()
                vector_res = _vec_fut.result()
        elif use_bm25:
            bm25_res = self.bm25.search(query, top_k=candidate_k)
        elif use_vector:
            vector_res = self.vector.search(v_query, top_k=candidate_k)
        
        norm_bm25 = self._normalize_scores(bm25_res)
        norm_vector = self._normalize_scores(vector_res)
        
        all_ids = set(norm_bm25.keys()) | set(norm_vector.keys())
        
        final_scores = []
        
        for cid in all_ids:
            s_bm25 = norm_bm25.get(cid, 0.0)
            s_vector = norm_vector.get(cid, 0.0)
            
            if use_bm25 and use_vector:
                final_score = (alpha * s_vector) + ((1 - alpha) * s_bm25)
            elif use_bm25:
                final_score = s_bm25
            elif use_vector:
                final_score = s_vector
            else:
                final_score = 0.0
            
            details = {
                "bm25_raw": next((s for c,s in bm25_res if c==cid), 0.0),
                "vector_raw": next((s for c,s in vector_res if c==cid), 0.0),
                "bm25_norm": s_bm25,
                "vector_norm": s_vector
            }
            
            final_scores.append(SearchResult(
                chunk_id=cid,
                score=final_score,
                source="hybrid",
                details=details
            ))
            
        final_scores.sort(key=lambda x: x.score, reverse=True)
        
        return final_scores[:top_k]
