from typing import List, Optional
import chromadb
from oml.models.schema import Document, Chunk, MemoryNote
from oml.storage.base import BaseStorage
import json

class ChromaStorage(BaseStorage):
    """ChromaDB implementation of BaseStorage."""
    
    def __init__(self, persist_path: str = "data/chroma"):
        self.client = chromadb.PersistentClient(path=persist_path)
        
        # Collections
        self.docs_col = self.client.get_or_create_collection("documents")
        self.chunks_col = self.client.get_or_create_collection("chunks")
        self.notes_col = self.client.get_or_create_collection("notes")

    def init_db(self) -> None:
        # Chroma handles collection creation in __init__
        pass

    def upsert_documents(self, docs: List[Document]) -> None:
        if not docs:
            return
            
        ids = [d.doc_id for d in docs]
        texts = [d.clean_text or "" for d in docs]
        metadatas = []
        
        for d in docs:
            meta = {
                "source": d.source,
                "author": d.author or "",
                "subject": d.subject or "",
                "timestamp": str(d.timestamp) if d.timestamp else "",
                "recipients": json.dumps(d.recipients),
                "summary": d.summary or "",
                "doc_type": d.doc_type
            }
            metadatas.append(meta)
            
        self.docs_col.upsert(
            ids=ids,
            documents=texts,
            metadatas=metadatas
        )

    def get_document(self, doc_id: str) -> Optional[Document]:
        result = self.docs_col.get(ids=[doc_id], include=["metadatas", "documents"])
        if not result["ids"]:
            return None
            
        meta = result["metadatas"][0]
        text = result["documents"][0]
        
        return Document(
            doc_id=doc_id,
            source=meta.get("source", ""),
            author=meta.get("author"),
            subject=meta.get("subject"),
            # timestamp restore todo
            recipients=json.loads(meta.get("recipients", "[]")),
            raw_text=text, # Storing raw text in 'documents' field for now
            clean_text=text,
            summary=meta.get("summary"),
            doc_type=meta.get("doc_type", "text")
        )

    def search_documents(self, **filters) -> List[Document]:
        # Chroma 'where' filter
        if not filters:
            return []
            
        result = self.docs_col.get(where=filters, include=["metadatas", "documents"])
        docs = []
        for i, doc_id in enumerate(result["ids"]):
            meta = result["metadatas"][i]
            text = result["documents"][i]
            docs.append(Document(
                doc_id=doc_id,
                source=meta.get("source", ""),
                author=meta.get("author"),
                subject=meta.get("subject"),
                recipients=json.loads(meta.get("recipients", "[]")),
                raw_text=text,
                clean_text=text,
                summary=meta.get("summary"),
                doc_type=meta.get("doc_type", "text")
            ))
        return docs

    def upsert_chunks(self, chunks: List[Chunk]) -> None:
        if not chunks:
            return
            
        ids = [c.chunk_id for c in chunks]
        texts = [c.chunk_text for c in chunks]
        metadatas = []
        
        for c in chunks:
            meta = {
                "doc_id": c.doc_id,
                "start_char": c.start_char,
                "end_char": c.end_char
            }
            metadatas.append(meta)
            
        self.chunks_col.upsert(
            ids=ids,
            documents=texts,
            metadatas=metadatas
        )

    def get_chunks_by_ids(self, chunk_ids: List[str]) -> List[Chunk]:
        if not chunk_ids:
            return []
            
        result = self.chunks_col.get(ids=chunk_ids, include=["metadatas", "documents"])
        chunks = []
        for i, cid in enumerate(result["ids"]):
            meta = result["metadatas"][i]
            text = result["documents"][i]
            chunks.append(Chunk(
                chunk_id=cid,
                doc_id=meta.get("doc_id", ""),
                chunk_text=text,
                start_char=meta.get("start_char", 0),
                end_char=meta.get("end_char", 0)
            ))
        return chunks

    def query(self, text: str, top_k: int = 5) -> List[Chunk]:
        """Vector search for chunks."""
        results = self.chunks_col.query(
            query_texts=[text],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        
        chunks = []
        if not results["ids"]:
            return []
            
        for i, cid in enumerate(results["ids"][0]):
            meta = results["metadatas"][0][i]
            doc_text = results["documents"][0][i]
            # score = results["distances"][0][i] # Optional: Return score if needed
            
            chunks.append(Chunk(
                chunk_id=cid,
                doc_id=meta.get("doc_id", ""),
                chunk_text=doc_text,
                start_char=meta.get("start_char", 0),
                end_char=meta.get("end_char", 0)
            ))
        return chunks

    def upsert_notes(self, notes: List[MemoryNote]) -> None:
        if not notes:
            return
        
        ids = [n.note_id for n in notes]
        texts = [n.content for n in notes]
        metadatas = []
        
        for n in notes:
            meta = {
                "thread_id": n.thread_id,
                "source_doc_ids": json.dumps(n.source_doc_ids),
                "timestamp": str(n.timestamp) if n.timestamp else ""
            }
            metadatas.append(meta)

        self.notes_col.upsert(ids=ids, documents=texts, metadatas=metadatas)

    def get_notes_by_ids(self, note_ids: List[str]) -> List[MemoryNote]:
        # Minimal impl
        return []

    def get_all_notes(self) -> List[MemoryNote]:
        return []
