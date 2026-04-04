import logging
from typing import List
from oml.retrieval.base import SearchResult
from oml.utils.device import resolve_device

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import CrossEncoder
except Exception:
    CrossEncoder = None

class Reranker:
    """
    Reranks search results using a Cross-Encoder model.
    Automatically uses the GPU when available (device="auto").
    """
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", device: str = "auto"):
        self.model_name = model_name
        self.device = resolve_device(device)
        self._model = None

    @property
    def model(self):
        if self._model is None:
            if CrossEncoder is None:
                raise ImportError("sentence-transformers not installed. Install it with `pip install sentence-transformers`.")
            logger.info(f"Loading reranker: {self.model_name} on {self.device}...")
            self._model = CrossEncoder(self.model_name, device=self.device)
        return self._model

    def rerank(self, query: str, texts: List[str], original_results: List[SearchResult]) -> List[SearchResult]:
        """
        Reranks results based on their content text.
        returns: Re-ordered list of SearchResults with updated scores.
        """
        if not texts or not original_results:
            return []
            
        if len(texts) != len(original_results):
            raise ValueError("mismatch between texts and results count")

        pairs = [[query, t] for t in texts]
        scores = self.model.predict(pairs)
        
        # Merge back
        reranked = []
        for i, score in enumerate(scores):
            original = original_results[i]
            reranked.append(SearchResult(
                chunk_id=original.chunk_id,
                score=float(score),
                source="reranker",
                details={**original.details, "pre_rerank_score": original.score}
            ))
            
        # Sort desc
        reranked.sort(key=lambda x: x.score, reverse=True)
        return reranked
