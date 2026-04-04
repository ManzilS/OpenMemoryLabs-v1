"""Core data models for OpenMemoryLab.

Document, Chunk, Citation, and MemoryNote are lightweight models used by the
"spine" to represent ingested documents, derived chunks, and later-memory notes.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional, Dict
from uuid import uuid4
from pydantic import BaseModel, Field, model_validator


def _make_id(prefix: Optional[str] = None) -> str:
    raw = uuid4().hex
    return f"{prefix}-{raw}" if prefix else raw


class Document(BaseModel):
    """Represents an ingested document.

    Fields:
      - doc_id: unique string id for the document
      - source: origin (filename/url/service)
      - timestamp: when the document was created/received (optional)
      - author: optional author name
      - recipients: optional list of recipients
      - subject: optional subject
      - thread_id: optional thread/conversation id
      - raw_text: original text
      - clean_text: cleaned/normalized text used for chunking/search
    """

    doc_id: str = Field(default_factory=lambda: _make_id("doc"))
    source: str = ""
    timestamp: Optional[datetime] = None
    author: Optional[str] = None
    recipients: List[str] = Field(default_factory=list)
    subject: Optional[str] = None
    thread_id: Optional[str] = None
    raw_text: str = ""
    clean_text: str = ""
    summary: Optional[str] = None
    doc_type: str = "text" # 'text', 'code', 'email'


class Chunk(BaseModel):
    """A chunk derived from a document for retrieval and indexing."""

    chunk_id: str = Field(default_factory=lambda: _make_id("chunk"))
    doc_id: str = ""
    chunk_text: str = ""
    start_char: int = 0
    end_char: int = 0

    @model_validator(mode='after')
    def validate_chars(self) -> 'Chunk':
        if self.start_char < 0 or self.end_char < 0:
            raise ValueError("start_char and end_char must be non-negative")
        if self.end_char < self.start_char:
            raise ValueError("end_char must be >= start_char")
        return self


class Citation(BaseModel):
    """A quoted span inside a chunk that cites original content."""
    doc_id: str
    chunk_id: str
    quote: str
    start_char: int
    end_char: int


class MemoryNote(BaseModel):
    """A short note representing a 'memory' entry to be stored/slept on later."""

    note_id: str = Field(default_factory=lambda: _make_id("note"))
    doc_id: Optional[str] = None
    chunk_ids: List[str] = Field(default_factory=list)
    summary: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, str] = Field(default_factory=dict)
    
    # Optional fields from other definition
    thread_id: str = ""
    content: str = "" # Alias for summary basically
    source_doc_ids: List[str] = Field(default_factory=list)


__all__ = ["Document", "Chunk", "Citation", "MemoryNote"]
