from pathlib import Path
from typing import List, Optional
import re

from oml.retrieval.base import BaseRetriever, SearchResult
from oml.retrieval.hybrid import HybridRetriever
from oml.storage.factory import get_storage

class UnifiedRetriever(BaseRetriever):
    """
    A unified interface for retrieval that abstracts the underlying storage 
    and indexing strategy (e.g. LanceDB vs SQLite/FAISS/BM25).
    """

    def __init__(self, storage_type: str, artifacts_dir: Path | str):
        self.storage_type = storage_type
        self.artifacts_dir = Path(artifacts_dir)
        self.storage = get_storage(storage_type)
        
        if self.storage_type != "lancedb":
            self.hybrid_retriever = HybridRetriever(self.artifacts_dir)
            bm25_exists = self.hybrid_retriever.bm25_path.exists()
            vector_exists = self.hybrid_retriever.vector_index_path.exists()

            # BM25 is the minimal index required for query to function.
            # Vector artifacts are optional and should gracefully degrade to lexical-only search.
            if not bm25_exists:
                raise FileNotFoundError("BM25 index not found! Run 'oml ingest' first.")
            if not vector_exists:
                print("Warning: Vector index not found. Falling back to BM25-only retrieval.")
            self.hybrid_retriever.load()
        else:
            self.hybrid_retriever = None

    def search(
        self, 
        query: str, 
        top_k: int = 10, 
        alpha: float = 0.5, 
        use_bm25: bool = True, 
        use_vector: bool = True, 
        vector_query: Optional[str] = None
    ) -> List[SearchResult]:
        """
        Executes a search against the configured backend.
        Returns a uniform list of SearchResult objects.
        """
        if self.storage_type == "lancedb":
            # LanceDB handles dense vector search internally.
            chunks = self.storage.query(query, top_k=top_k)
            # Normalize to SearchResult, setting a default score for now.
            return [
                SearchResult(
                    chunk_id=c.chunk_id, 
                    score=0.0, 
                    source="lancedb", 
                    details={}
                ) for c in chunks
            ]
        else:
            return self.hybrid_retriever.search(
                query=query,
                top_k=top_k,
                alpha=alpha,
                use_bm25=use_bm25,
                use_vector=use_vector,
                vector_query=vector_query
            )

    def search_notes(self, query: str, top_k: int = 3) -> List[SearchResult]:
        if self.storage_type == "lancedb":
            notes = self.storage.get_all_notes()
            if not notes:
                return []

            # Lightweight lexical scorer to keep note retrieval available across backends.
            query_terms = set(re.findall(r"\w+", query.lower()))
            if not query_terms:
                return []

            scored: list[SearchResult] = []
            for note in notes:
                note_text = (note.content or note.summary or "").lower()
                note_terms = set(re.findall(r"\w+", note_text))
                if not note_terms:
                    continue
                overlap = len(query_terms.intersection(note_terms))
                if overlap == 0:
                    continue
                score = overlap / max(len(query_terms), 1)
                scored.append(
                    SearchResult(
                        chunk_id=note.note_id,
                        score=float(score),
                        source="lancedb-notes",
                        details={"overlap_terms": overlap},
                    )
                )

            scored.sort(key=lambda r: r.score, reverse=True)
            return scored[:top_k]
        else:
            return self.hybrid_retriever.search_notes(query, top_k=top_k)
