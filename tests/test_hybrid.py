"""Tests focused on the HybridRetriever scoring behavior.

These tests mock out underlying BM25/vector scores to verify that the
alpha-weighted combination works as intended.
"""
from oml.retrieval.hybrid import HybridRetriever, SearchResult


class DummyHybrid(HybridRetriever):
    """HybridRetriever subclass with injected score tables for testing.

    Instead of calling real BM25/vector indices, we override the `load`
    method and `search` behavior by pre-populating score lists.
    """

    def __init__(self, artifacts_dir, bm25_scores, vector_scores):
        super().__init__(artifacts_dir)
        # bm25_scores / vector_scores: list of (chunk_id, score)
        self._bm25_scores = bm25_scores
        self._vector_scores = vector_scores
        self._loaded = True

    def load(self):
        # Skip loading real indices
        self._loaded = True

    def search(self, query, top_k=10, alpha=0.5, use_bm25=True, use_vector=True):  # type: ignore[override]
        # Bypass base class retrieval, reuse its normalization logic
        candidate_k = top_k * 2

        bm25_res = self._bm25_scores[:candidate_k] if use_bm25 else []
        vector_res = self._vector_scores[:candidate_k] if use_vector else []

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
                "bm25_norm": s_bm25,
                "vector_norm": s_vector,
            }

            final_scores.append(
                SearchResult(chunk_id=cid, score=final_score, source="hybrid", details=details)
            )

        final_scores.sort(key=lambda x: x.score, reverse=True)
        return final_scores[:top_k]


def test_alpha_zero_uses_bm25(tmp_path):
    bm25_scores = [("c1", 1.0), ("c2", 0.5)]
    vector_scores = [("c1", 0.1), ("c2", 10.0)]

    retriever = DummyHybrid(tmp_path, bm25_scores, vector_scores)
    results = retriever.search("query", top_k=2, alpha=0.0, use_bm25=True, use_vector=True)

    # With alpha=0, normalized BM25 should dominate regardless of vector scores
    assert results[0].chunk_id == "c1"


def test_alpha_one_uses_vector(tmp_path):
    bm25_scores = [("c1", 1.0), ("c2", 0.5)]
    vector_scores = [("c1", 0.1), ("c2", 10.0)]

    retriever = DummyHybrid(tmp_path, bm25_scores, vector_scores)
    results = retriever.search("query", top_k=2, alpha=1.0, use_bm25=True, use_vector=True)

    # With alpha=1, normalized vector scores should dominate
    assert results[0].chunk_id == "c2"
