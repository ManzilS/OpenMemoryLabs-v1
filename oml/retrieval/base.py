from abc import ABC, abstractmethod
from typing import List, Tuple
from dataclasses import dataclass

@dataclass
class SearchResult:
    chunk_id: str
    score: float
    source: str # 'hybrid', 'bm25', 'vector'
    details: dict = None

class BaseRetriever(ABC):
    """Abstract base class for retrievers."""
    
    @abstractmethod
    def search(self, query: str, top_k: int) -> List[SearchResult]:
        pass

class BaseIndex(ABC):
    """Abstract base class for indexers."""
    
    @abstractmethod
    def build(self, chunk_ids: List[str], texts: List[str]):
        pass
        
    @abstractmethod
    def search(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        pass
        
    @abstractmethod
    def save(self):
        pass
        
    @abstractmethod
    def load(self) -> bool:
        pass
