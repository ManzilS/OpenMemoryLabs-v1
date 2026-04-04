from typing import List, Optional
from oml.models.schema import Document, Chunk, MemoryNote
from oml.storage.base import BaseStorage

class MemoryStorage(BaseStorage):
    """In-memory storage for testing."""
    
    def __init__(self):
        self.documents = {}
        self.chunks = {}
        self.notes = {}
        
    def init_db(self) -> None:
        pass

    def upsert_documents(self, docs: List[Document]) -> None:
        for doc in docs:
            self.documents[doc.doc_id] = doc

    def get_document(self, doc_id: str) -> Optional[Document]:
        return self.documents.get(doc_id)

    def search_documents(self, **filters) -> List[Document]:
        results = []
        for doc in self.documents.values():
            match = True
            for k, v in filters.items():
                if getattr(doc, k, None) != v:
                    match = False
                    break
            if match:
                results.append(doc)
        return results

    def upsert_chunks(self, chunks: List[Chunk]) -> None:
        for chunk in chunks:
            self.chunks[chunk.chunk_id] = chunk

    def get_chunks_by_ids(self, chunk_ids: List[str]) -> List[Chunk]:
        return [self.chunks[cid] for cid in chunk_ids if cid in self.chunks]

    def get_all_chunks(self) -> List[Chunk]:
        return list(self.chunks.values())

    def upsert_notes(self, notes: List[MemoryNote]) -> None:
        for note in notes:
            self.notes[note.note_id] = note

    def get_notes_by_ids(self, note_ids: List[str]) -> List[MemoryNote]:
        return [self.notes[nid] for nid in note_ids if nid in self.notes]

    def get_all_notes(self) -> List[MemoryNote]:
        return list(self.notes.values())

    def get_documents_by_ids(self, doc_ids: List[str]) -> List[Document]:
        return [self.documents[did] for did in doc_ids if did in self.documents]
