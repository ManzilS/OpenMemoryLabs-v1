"""
AtomicNote — The fundamental memory unit of the TEEG system.
=============================================================

Inspired by the Zettelkasten method: every piece of knowledge is stored as a
single, self-describing, atomic note rather than as a raw transcript chunk.

An AtomicNote contains:
  - ``note_id``    — unique identifier
  - ``content``    — the distilled fact / observation (one sentence preferred)
  - ``context``    — where/when this was observed (source, chapter, date, …)
  - ``keywords``   — searchable terms extracted from content
  - ``tags``       — high-level semantic categories
  - ``created_at`` — ISO-8601 timestamp
  - ``supersedes`` — note_id of the note this one replaces (evolution chain)
  - ``confidence`` — 0.0–1.0 self-reported confidence in this fact
  - ``source_ids`` — IDs of the original documents/chunks that produced this note

Serialization
-------------
AtomicNote ↔ TOON is the canonical on-disk / in-context format.
AtomicNote ↔ dict is used for LanceDB storage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List
from uuid import uuid4

from oml.memory import toon


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _new_note_id() -> str:
    return f"teeg-{uuid4().hex[:12]}"


@dataclass
class AtomicNote:
    """A single, self-describing memory note in TOON-serializable form."""

    # ── identity ────────────────────────────────────────────────────────────
    note_id: str = field(default_factory=_new_note_id)

    # ── core payload ────────────────────────────────────────────────────────
    content: str = ""
    context: str = ""

    # ── indexing signals ────────────────────────────────────────────────────
    keywords: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)

    # ── provenance ──────────────────────────────────────────────────────────
    created_at: str = field(default_factory=_utcnow_iso)
    supersedes: str = ""          # note_id of the note this replaces; "" = new note
    confidence: float = 1.0       # [0, 1]
    source_ids: List[str] = field(default_factory=list)

    # ── active / archived flag ──────────────────────────────────────────────
    active: bool = True           # False = superseded / soft-deleted
    archived_at: str = ""         # ISO-8601 timestamp when archived; "" = active
                                  # Used by TEEGStore.vector_search_warm() to bound
                                  # the resurrection search window (_WARM_STORE_DAYS)

    # ── usage tracking (for ImportanceScorer) ───────────────────────────────
    access_count: int = 0         # incremented each time this note is retrieved
    last_accessed: str = ""       # ISO-8601 timestamp of most recent retrieval

    # ── TOON serialization ──────────────────────────────────────────────────

    def to_toon(self) -> str:
        """Serialize this note to TOON format for LLM context injection."""
        d = {
            "note_id": self.note_id,
            "content": self.content,
            "context": self.context,
            "keywords": self.keywords,
            "tags": self.tags,
            "created_at": self.created_at,
            "supersedes": self.supersedes,
            "confidence": str(self.confidence),
            "source_ids": self.source_ids,
            "active": str(self.active),
        }
        return toon.dumps(d)

    @classmethod
    def from_toon(cls, text: str) -> "AtomicNote":
        """Deserialize an AtomicNote from TOON format."""
        d = toon.loads(text)
        return cls(
            note_id=d.get("note_id", _new_note_id()),
            content=d.get("content", ""),
            context=d.get("context", ""),
            keywords=d.get("keywords", []),
            tags=d.get("tags", []),
            created_at=d.get("created_at", _utcnow_iso()),
            supersedes=d.get("supersedes", ""),
            confidence=float(d.get("confidence", "1.0") or "1.0"),
            source_ids=d.get("source_ids", []),
            active=d.get("active", "True") not in ("False", "false", "0"),
        )

    # ── dict I/O (for LanceDB / SQLite storage) ─────────────────────────────

    def to_dict(self) -> dict:
        """Return a flat dict suitable for LanceDB or JSON storage."""
        return {
            "note_id": self.note_id,
            "content": self.content,
            "context": self.context,
            "keywords": "|".join(self.keywords),
            "tags": "|".join(self.tags),
            "created_at": self.created_at,
            "supersedes": self.supersedes,
            "confidence": self.confidence,
            "source_ids": "|".join(self.source_ids),
            "active": self.active,
            "archived_at": self.archived_at,
            # usage tracking — optional fields (absent in legacy stores)
            "access_count": self.access_count,
            "last_accessed": self.last_accessed,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AtomicNote":
        """Reconstruct an AtomicNote from a flat storage dict."""
        def split_pipe(v) -> List[str]:
            if isinstance(v, list):
                return v
            return [x for x in str(v).split("|") if x] if v else []

        return cls(
            note_id=d.get("note_id", _new_note_id()),
            content=d.get("content", ""),
            context=d.get("context", ""),
            keywords=split_pipe(d.get("keywords", "")),
            tags=split_pipe(d.get("tags", "")),
            created_at=d.get("created_at", _utcnow_iso()),
            supersedes=d.get("supersedes", "") or "",
            confidence=float(d.get("confidence", 1.0) or 1.0),
            source_ids=split_pipe(d.get("source_ids", "")),
            active=bool(d.get("active", True)),
            archived_at=d.get("archived_at", "") or "",
            # usage tracking — default to 0/"" for notes persisted before this field existed
            access_count=int(d.get("access_count", 0) or 0),
            last_accessed=d.get("last_accessed", "") or "",
        )

    # ── text for embedding ───────────────────────────────────────────────────

    def embedding_text(self) -> str:
        """Compact text used to build the vector embedding for this note.

        Combines content + context + keywords so the embedding captures both
        the fact and its domain signals.
        """
        parts = [self.content]
        if self.context:
            parts.append(f"[context: {self.context}]")
        if self.keywords:
            parts.append(f"[keywords: {', '.join(self.keywords)}]")
        return " ".join(parts)

    # ── helpers ──────────────────────────────────────────────────────────────

    def token_cost(self) -> int:
        """Estimated tokens when injected into LLM context via TOON."""
        return toon.token_count_estimate(self.to_toon())

    def __repr__(self) -> str:
        snippet = self.content[:60] + ("…" if len(self.content) > 60 else "")
        return f"AtomicNote(id={self.note_id!r}, content={snippet!r})"
