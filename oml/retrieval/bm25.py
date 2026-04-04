import pickle
import re
from pathlib import Path
from typing import List, Tuple

from rank_bm25 import BM25Okapi
from oml.retrieval.base import BaseIndex

class BM25Index(BaseIndex):
    """Wrapper around rank_bm25 with save/load persistence."""
    
    def __init__(self, storage_path: Path):
        self.storage_path = storage_path
        self.bm25 = None
        self.chunk_ids = []

    def tokenize(self, text: str) -> List[str]:
        """Simple regex-based tokenization."""
        return re.findall(r"\w+", text.lower())

    def build(self, chunk_ids: List[str], texts: List[str]):
        """Builds the BM25 index from valid texts."""
        self.chunk_ids = chunk_ids
        tokenized_corpus = [self.tokenize(doc) for doc in texts]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def save(self):
        """Saves the index and chunk IDs to disk."""
        if self.bm25 is None:
            raise ValueError("Index not built, cannot save.")
        
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        # Note: pickle is simple but not secure for untrusted data.
        data = {
            "bm25": self.bm25,
            "chunk_ids": self.chunk_ids
        }
        
        with open(self.storage_path, "wb") as f:
            pickle.dump(data, f)
        print(f"BM25 index saved to {self.storage_path}")

    def load(self) -> bool:
        """Loads the index from disk. Returns True if successful."""
        if not self.storage_path.exists():
            return False
            
        try:
            with open(self.storage_path, "rb") as f:
                data = pickle.load(f)
            self.bm25 = data["bm25"]
            self.chunk_ids = data["chunk_ids"]
            return True
        except Exception as e:
            print(f"Failed to load BM25 index: {e}")
            return False

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Returns list of (chunk_id, score).
        """
        if self.bm25 is None:
            # print("Warning: BM25 index not loaded or built.")
            return []

        tokenized_query = self.tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        # Zip scores with IDs and sort
        results = []
        for cid, score in zip(self.chunk_ids, scores):
            if score > 0:
                results.append((cid, float(score)))
        
        # Sort descending by score
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
