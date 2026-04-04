from abc import ABC, abstractmethod
from typing import List, Optional
from oml.models.schema import Document, Chunk, MemoryNote

class BaseStorage(ABC):
    """Abstract base class for storage adapters."""
    
    @abstractmethod
    def init_db(self) -> None:
        pass
        
    @abstractmethod
    def upsert_documents(self, docs: List[Document]) -> None:
        pass
        
    @abstractmethod
    def get_document(self, doc_id: str) -> Optional[Document]:
        pass
        
    @abstractmethod
    def search_documents(self, **filters) -> List[Document]:
        pass
        
    @abstractmethod
    def upsert_chunks(self, chunks: List[Chunk]) -> None:
        pass
        
    @abstractmethod
    def get_chunks_by_ids(self, chunk_ids: List[str]) -> List[Chunk]:
        pass

    @abstractmethod
    def get_all_chunks(self) -> List[Chunk]:
        pass

    @abstractmethod
    def upsert_notes(self, notes: List[MemoryNote]) -> None:
        pass
        
    @abstractmethod
    def get_notes_by_ids(self, note_ids: List[str]) -> List[MemoryNote]:
        pass
    
    @abstractmethod
    def get_all_notes(self) -> List[MemoryNote]:
        pass
