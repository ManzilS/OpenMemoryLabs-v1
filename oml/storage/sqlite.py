import json
from pathlib import Path
from typing import List, Optional

from sqlalchemy import (
    create_engine,
    Table,
    Column,
    String,
    Integer,
    MetaData,
    Text,
    DateTime,
    select,
)
from sqlalchemy.pool import NullPool
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from oml.models.schema import Document, Chunk, MemoryNote
from oml.storage.base import BaseStorage

metadata_obj = MetaData()

documents_table = Table(
    "documents",
    metadata_obj,
    Column("doc_id", String, primary_key=True),
    Column("source", String),
    Column("timestamp", DateTime, nullable=True),
    Column("author", String, nullable=True),
    Column("recipients", Text, default="[]"),  # JSON serialized list
    Column("subject", String, nullable=True),
    Column("thread_id", String, nullable=True),
    Column("raw_text", Text),
    Column("clean_text", Text),
    Column("summary", Text, nullable=True),
)

chunks_table = Table(
    "chunks",
    metadata_obj,
    Column("chunk_id", String, primary_key=True),
    Column("doc_id", String, nullable=False),  # logical FK, strict constraint optional for speed
    Column("chunk_text", Text),
    Column("start_char", Integer),
    Column("end_char", Integer),
)

memory_notes_table = Table(
    "memory_notes",
    metadata_obj,
    Column("note_id", String, primary_key=True),
    Column("thread_id", String, index=True),
    Column("content", Text),
    Column("source_doc_ids", Text), # JSON list
    Column("timestamp", DateTime, nullable=True),
)


def _row_to_document(row) -> Document:
    recipients = json.loads(row.recipients) if row.recipients else []
    return Document(
        doc_id=row.doc_id,
        source=row.source or "",
        timestamp=row.timestamp,
        author=row.author,
        recipients=recipients,
        subject=row.subject,
        thread_id=row.thread_id,
        raw_text=row.raw_text or "",
        clean_text=row.clean_text or "",
        summary=row.summary,
    )

def _row_to_chunk(row) -> Chunk:
    return Chunk(
        chunk_id=row.chunk_id,
        doc_id=row.doc_id,
        chunk_text=row.chunk_text or "",
        start_char=row.start_char,
        end_char=row.end_char,
    )

def _row_to_note(row) -> MemoryNote:
    return MemoryNote(
        note_id=row.note_id,
        thread_id=row.thread_id,
        content=row.content or "",
        source_doc_ids=json.loads(row.source_doc_ids) if row.source_doc_ids else [],
        created_at=row.timestamp,
    )


class SQLiteStorage(BaseStorage):
    """SQLite implementation using SQLAlchemy."""
    
    def __init__(self, db_path: str):
        if "://" not in db_path:
            # Ensure the parent directory exists so SQLite can create the file
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
            self.conn_str = f"sqlite:///{db_path}"
        else:
            self.conn_str = db_path

        # Use NullPool to avoid file locking issues on Windows/tests
        self.engine = create_engine(self.conn_str, future=True, poolclass=NullPool)

    def init_db(self) -> None:
        """Initialize the database tables."""
        metadata_obj.create_all(self.engine)

    def upsert_documents(self, docs: List[Document]) -> None:
        """Insert or update documents."""
        if not docs:
            return

        records = []
        for d in docs:
            records.append(
                {
                    "doc_id": d.doc_id,
                    "source": d.source,
                    "timestamp": d.timestamp,
                    "author": d.author,
                    "recipients": json.dumps(d.recipients),
                    "subject": d.subject,
                    "thread_id": d.thread_id,
                    "raw_text": d.raw_text,
                    "clean_text": d.clean_text,
                    "summary": d.summary,
                }
            )

        stmt = sqlite_insert(documents_table).values(records)
        # Exclude doc_id from update
        update_dict = {c.name: c for c in stmt.excluded if c.name != "doc_id"}
        upsert_stmt = stmt.on_conflict_do_update(index_elements=["doc_id"], set_=update_dict)

        with self.engine.begin() as conn:
            conn.execute(upsert_stmt)

    def get_document(self, doc_id: str) -> Optional[Document]:
        """Retrieve a document by ID."""
        stmt = select(documents_table).where(documents_table.c.doc_id == doc_id)

        with self.engine.connect() as conn:
            row = conn.execute(stmt).first()
            if not row:
                return None

            return _row_to_document(row)

    def upsert_chunks(self, chunks: List[Chunk]) -> None:
        """Insert or update chunks (batched to stay under SQLite variable limit)."""
        if not chunks:
            return

        records = []
        for c in chunks:
            records.append(
                {
                    "chunk_id": c.chunk_id,
                    "doc_id": c.doc_id,
                    "chunk_text": c.chunk_text,
                    "start_char": c.start_char,
                    "end_char": c.end_char,
                }
            )

        # SQLite has a default limit of 999 variables per statement.
        # Each record uses 5 columns, so batch at most 199 rows per INSERT.
        batch_size = 199
        with self.engine.begin() as conn:
            for i in range(0, len(records), batch_size):
                batch = records[i : i + batch_size]
                stmt = sqlite_insert(chunks_table).values(batch)
                update_dict = {c.name: c for c in stmt.excluded if c.name != "chunk_id"}
                upsert_stmt = stmt.on_conflict_do_update(
                    index_elements=["chunk_id"], set_=update_dict
                )
                conn.execute(upsert_stmt)

    def get_chunks_by_ids(self, chunk_ids: List[str]) -> List[Chunk]:
        """Retrieve chunks by a list of IDs."""
        if not chunk_ids:
            return []

        stmt = select(chunks_table).where(chunks_table.c.chunk_id.in_(chunk_ids))
        results = []
        with self.engine.connect() as conn:
            rows = conn.execute(stmt).all()
            for row in rows:
                results.append(_row_to_chunk(row))
        return results

    def get_all_chunks(self) -> List[Chunk]:
        """Retrieve all chunks from the database."""
        stmt = select(chunks_table)
        results = []
        with self.engine.connect() as conn:
            rows = conn.execute(stmt).all()
            for row in rows:
                results.append(
                    Chunk(
                        chunk_id=row.chunk_id,
                        doc_id=row.doc_id,
                        chunk_text=row.chunk_text or "",
                        start_char=row.start_char,
                        end_char=row.end_char,
                    )
                )
        return results

    def search_documents(self, **filters) -> List[Document]:
        """Search documents by metadata fields (exact match)."""
        stmt = select(documents_table)
        valid_cols = documents_table.c.keys()

        for key, val in filters.items():
            if key in valid_cols:
                stmt = stmt.where(documents_table.c[key] == val)

        results = []
        with self.engine.connect() as conn:
            rows = conn.execute(stmt).all()
            for row in rows:
                results.append(_row_to_document(row))
        return results

    def upsert_notes(self, notes: List[MemoryNote]) -> None:
        """Insert or update memory notes."""
        if not notes:
            return

        records = []
        for n in notes:
            records.append(
                {
                    "note_id": n.note_id,
                    "thread_id": n.thread_id,
                    "content": n.content,
                    "source_doc_ids": json.dumps(n.source_doc_ids),
                    "timestamp": n.created_at,
                }
            )

        stmt = sqlite_insert(memory_notes_table).values(records)
        update_dict = {c.name: c for c in stmt.excluded if c.name != "note_id"}
        upsert_stmt = stmt.on_conflict_do_update(index_elements=["note_id"], set_=update_dict)

        with self.engine.begin() as conn:
            conn.execute(upsert_stmt)

    def get_notes_by_ids(self, note_ids: List[str]) -> List[MemoryNote]:
        """Retrieve notes by a list of IDs."""
        if not note_ids:
            return []

        stmt = select(memory_notes_table).where(memory_notes_table.c.note_id.in_(note_ids))
        results = []
        with self.engine.connect() as conn:
            rows = conn.execute(stmt).all()
            for row in rows:
                results.append(_row_to_note(row))
        return results
    
    def get_all_notes(self) -> List[MemoryNote]:
        """Retrieve all memory notes."""
        stmt = select(memory_notes_table)
        results = []
        with self.engine.connect() as conn:
            rows = conn.execute(stmt).all()
            for row in rows:
                results.append(
                    MemoryNote(
                        note_id=row.note_id,
                        thread_id=row.thread_id,
                        content=row.content or "",
                        source_doc_ids=json.loads(row.source_doc_ids) if row.source_doc_ids else [],
                        created_at=row.timestamp,
                    )
                )
        return results

    def get_documents_by_ids(self, doc_ids: List[str]) -> List[Document]:
        """Retrieve documents by a list of IDs."""
        if not doc_ids:
            return []

        stmt = select(documents_table).where(documents_table.c.doc_id.in_(doc_ids))
        results: List[Document] = []
        with self.engine.connect() as conn:
            rows = conn.execute(stmt).all()
            for row in rows:
                recipients = json.loads(row.recipients) if row.recipients else []
                results.append(
                    Document(
                        doc_id=row.doc_id,
                        source=row.source or "",
                        timestamp=row.timestamp,
                        author=row.author,
                        recipients=recipients,
                        subject=row.subject,
                        thread_id=row.thread_id,
                        raw_text=row.raw_text or "",
                        clean_text=row.clean_text or "",
                        summary=row.summary,
                    )
                )
        return results

# Standalone helpers for backward compatibility and ease of use

def _get_engine(db_path: str):
    """Helper to get engine from path."""
    return SQLiteStorage(db_path).engine

def init_db(db_path: str):
    storage = SQLiteStorage(db_path)
    storage.init_db()

def upsert_documents(db_path: str, docs: List[Document]):
    storage = SQLiteStorage(db_path)
    storage.upsert_documents(docs)

def get_document(db_path: str, doc_id: str) -> Optional[Document]:
    storage = SQLiteStorage(db_path)
    return storage.get_document(doc_id)

def upsert_chunks(db_path: str, chunks: List[Chunk]):
    storage = SQLiteStorage(db_path)
    storage.upsert_chunks(chunks)

def get_chunks_by_ids(db_path: str, chunk_ids: List[str]) -> List[Chunk]:
    storage = SQLiteStorage(db_path)
    return storage.get_chunks_by_ids(chunk_ids)

def search_documents(db_path: str, **filters) -> List[Document]:
    storage = SQLiteStorage(db_path)
    return storage.search_documents(**filters)

def upsert_notes(db_path: str, notes: List[MemoryNote]):
    storage = SQLiteStorage(db_path)
    storage.upsert_notes(notes)

def get_notes_by_ids(db_path: str, note_ids: List[str]) -> List[MemoryNote]:
    storage = SQLiteStorage(db_path)
    return storage.get_notes_by_ids(note_ids)

def get_all_notes(db_path: str) -> List[MemoryNote]:
    storage = SQLiteStorage(db_path)
    return storage.get_all_notes()

def get_documents_by_ids(db_path: str, doc_ids: List[str]) -> List[Document]:
    storage = SQLiteStorage(db_path)
    return storage.get_documents_by_ids(doc_ids)
