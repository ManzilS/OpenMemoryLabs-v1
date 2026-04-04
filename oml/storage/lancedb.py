import json
import logging
from datetime import datetime
from typing import List, Optional

import lancedb

from oml.models.schema import Document, Chunk, MemoryNote
from oml.storage.base import BaseStorage
from oml.utils.device import resolve_device

logger = logging.getLogger(__name__)

# Resolve once at module import so all calls in this file share the same device
_DEVICE = resolve_device("auto")
_EMBEDDER = None


def _get_embedder():
    global _EMBEDDER
    if _EMBEDDER is None:
        from sentence_transformers import SentenceTransformer
        _EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2", device=_DEVICE)
    return _EMBEDDER

def _row_to_document(row) -> Document:
    return Document(
        doc_id=row["doc_id"],
        source=row["source"],
        author=row["author"] if row["author"] else None,
        subject=row["subject"] if row["subject"] else None,
        recipients=json.loads(row["recipients"]) if row["recipients"] else [],
        raw_text=row["raw_text"],
        clean_text=row["clean_text"],
        summary=row["summary"],
        doc_type=row["doc_type"]
    )

def _row_to_chunk(row) -> Chunk:
    return Chunk(
        chunk_id=row["chunk_id"],
        doc_id=row["doc_id"],
        chunk_text=row["chunk_text"],
        start_char=row["start_char"],
        end_char=row["end_char"]
    )

class LanceDBStorage(BaseStorage):
    """LanceDB implementation of BaseStorage."""
    
    def __init__(self, persist_path: str = "data/lancedb"):
        self.persist_path = persist_path
        self.db = lancedb.connect(persist_path)
        
        # LanceDB infers schema from the first insert for each table.
        pass

    def init_db(self) -> None:
        # LanceDB creates tables on first write.
        return None

    def upsert_documents(self, docs: List[Document]) -> None:
        if not docs:
            return
            
        data = []
        for d in docs:
            item = {
                "doc_id": d.doc_id,
                "clean_text": d.clean_text or "",
                "raw_text": d.raw_text or "",
                "source": d.source,
                "author": d.author or "",
                "subject": d.subject or "",
                "timestamp": str(d.timestamp) if d.timestamp else "",
                "recipients": json.dumps(d.recipients),
                "summary": d.summary or "",
                "doc_type": d.doc_type,
            }
            data.append(item)
            
        try:
            tbl = self.db.open_table("documents")
            tbl.add(data)
        except Exception:
            # Table doesn't exist yet — create it
            self.db.create_table("documents", data)

    def get_document(self, doc_id: str) -> Optional[Document]:
        try:
            tbl = self.db.open_table("documents")
            # LanceDB filtering is via SQL-like string or pyarrow
            res = tbl.search().where(f"doc_id = '{doc_id}'").limit(1).to_pandas()
            if res.empty:
                return None
            
            row = res.iloc[0]
            return _row_to_document(row)
        except Exception:
            return None

    def search_documents(self, **filters) -> List[Document]:
        # Minimal filter support
        return []

    def upsert_chunks(self, chunks: List[Chunk]) -> None:
        if not chunks:
            return
            
        # Compute embeddings inline so LanceDB can do vector search on chunks.
        model = _get_embedder()
        texts = [c.chunk_text for c in chunks]
        embeddings = model.encode(texts)
        
        data = []
        for i, c in enumerate(chunks):
            item = {
                "chunk_id": c.chunk_id,
                "doc_id": c.doc_id,
                "chunk_text": c.chunk_text,
                "start_char": c.start_char,
                "end_char": c.end_char,
                "vector": embeddings[i]
            }
            data.append(item)
            
        try:
            tbl = self.db.open_table("chunks")
            tbl.add(data)
        except Exception:
            self.db.create_table("chunks", data)

    def get_chunks_by_ids(self, chunk_ids: List[str]) -> List[Chunk]:
        if not chunk_ids:
            return []
        try:
            tbl = self.db.open_table("chunks")
            # Construct where clause
            ids_str = ", ".join([f"'{cid}'" for cid in chunk_ids])
            res = tbl.search().where(f"chunk_id IN ({ids_str})").to_pandas()
            
            chunks = []
            for _, row in res.iterrows():
                chunks.append(_row_to_chunk(row))
            return chunks
        except Exception:
            return []

    def get_all_chunks(self) -> List[Chunk]:
        try:
            tbl = self.db.open_table("chunks")
            res = tbl.search().to_pandas()
            
            chunks = []
            for _, row in res.iterrows():
                chunks.append(Chunk(
                    chunk_id=row["chunk_id"],
                    doc_id=row["doc_id"],
                    chunk_text=row["chunk_text"],
                    start_char=row["start_char"],
                    end_char=row["end_char"]
                ))
            return chunks
        except Exception:
            return []

    def query(self, text: str, top_k: int = 5) -> List[Chunk]:
        try:
            tbl = self.db.open_table("chunks")
            
            # Embed query
            model = _get_embedder()
            query_vec = model.encode([text])[0]
            
            res = tbl.search(query_vec).limit(top_k).to_pandas()
            
            chunks = []
            for _, row in res.iterrows():
                chunks.append(Chunk(
                    chunk_id=row["chunk_id"],
                    doc_id=row["doc_id"],
                    chunk_text=row["chunk_text"],
                    start_char=row["start_char"],
                    end_char=row["end_char"]
                ))
            return chunks
        except Exception as e:
            logger.warning("LanceDB query error: %s", e)
            return []

    def upsert_notes(self, notes: List[MemoryNote]) -> None:
        if not notes:
            return

        model = _get_embedder()
        texts = [n.content or n.summary or "" for n in notes]
        embeddings = model.encode(texts)

        data = []
        for idx, n in enumerate(notes):
            created = n.created_at if isinstance(n.created_at, datetime) else None
            data.append(
                {
                    "note_id": n.note_id,
                    "thread_id": n.thread_id or "",
                    "content": n.content or n.summary or "",
                    "summary": n.summary or "",
                    "source_doc_ids": json.dumps(n.source_doc_ids),
                    "created_at": created.isoformat() if created else "",
                    "vector": embeddings[idx],
                }
            )

        try:
            tbl = self.db.open_table("notes")
            tbl.add(data)
        except Exception:
            self.db.create_table("notes", data)

    def get_notes_by_ids(self, note_ids: List[str]) -> List[MemoryNote]:
        if not note_ids:
            return []
        try:
            tbl = self.db.open_table("notes")
            ids_str = ", ".join([f"'{nid}'" for nid in note_ids])
            res = tbl.search().where(f"note_id IN ({ids_str})").to_pandas()
        except Exception:
            return []

        notes: List[MemoryNote] = []
        for _, row in res.iterrows():
            ts = row.get("created_at") or ""
            created_at = None
            if ts:
                try:
                    created_at = datetime.fromisoformat(ts)
                except Exception:
                    created_at = None

            notes.append(
                MemoryNote(
                    note_id=row["note_id"],
                    thread_id=row.get("thread_id", ""),
                    content=row.get("content", ""),
                    summary=row.get("summary", ""),
                    source_doc_ids=json.loads(row.get("source_doc_ids", "[]") or "[]"),
                    created_at=created_at or datetime.utcnow(),
                )
            )
        return notes

    def get_all_notes(self) -> List[MemoryNote]:
        try:
            tbl = self.db.open_table("notes")
            res = tbl.search().to_pandas()
        except Exception:
            return []

        notes: List[MemoryNote] = []
        for _, row in res.iterrows():
            ts = row.get("created_at") or ""
            created_at = None
            if ts:
                try:
                    created_at = datetime.fromisoformat(ts)
                except Exception:
                    created_at = None

            notes.append(
                MemoryNote(
                    note_id=row["note_id"],
                    thread_id=row.get("thread_id", ""),
                    content=row.get("content", ""),
                    summary=row.get("summary", ""),
                    source_doc_ids=json.loads(row.get("source_doc_ids", "[]") or "[]"),
                    created_at=created_at or datetime.utcnow(),
                )
            )
        return notes

    def get_documents_by_ids(self, doc_ids: List[str]) -> List[Document]:
        if not doc_ids:
            return []
        try:
            tbl = self.db.open_table("documents")
            ids_str = ", ".join([f"'{did}'" for did in doc_ids])
            res = tbl.search().where(f"doc_id IN ({ids_str})").to_pandas()
            
            docs = []
            for _, row in res.iterrows():
                docs.append(_row_to_document(row))
            return docs
        except Exception:
            return []
