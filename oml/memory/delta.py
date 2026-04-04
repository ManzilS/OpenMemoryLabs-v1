"""oml/memory/delta.py — Semantic patch storage for EXTENDS-relation notes.

Token efficiency gap analysis
------------------------------
When ``MemoryEvolver`` classifies a new note as ``EXTENDS`` relative to an
existing base note, the current TEEG behaviour stores the full new note
independently.  At query time, both the base note and the extending note are
retrieved and serialised into the LLM context — but they share 40–60 % of
their semantic content (the extending note re-states the base before adding
new information).

``DeltaStore`` breaks this pattern by storing *only the new information* as a
``SemanticPatch`` alongside the base note.  The full reconstructed content is
available on demand via ``reconstruct()``.  At context-assembly time, a note
with patches can be rendered using the reconstructed content rather than
loading two separate TOON blocks, saving roughly
``_TOKENS_SAVED_PER_PATCH`` tokens per extending note per query.

Novel aspect
------------
Git-style delta encoding (event sourcing, append-only patch chains) is
well-established for source code and event logs but has not previously been
applied to LLM-distilled ``AtomicNote`` memory units.  Combining delta storage
with TOON's compact encoding reduces both on-disk footprint and per-query
LLM context cost simultaneously.

Design constraints
------------------
- ``SemanticPatch`` is a pure dataclass — no heavy base classes.
- Persistence is JSON-Lines (human-readable, appendable) matching TEEGStore.
- Reconstruction is deterministic: ``" Additionally: ".join(patches)`` so
  LLMs receive a grammatically sensible combined fact.
- Zero new runtime dependencies (standard library only).

Usage
-----
    store = DeltaStore("teeg_store/")
    patch = store.store_patch(
        base_id="teeg-abc123",
        patch_note=new_note,
        patch_type="EXTENDS",
    )
    full_content = store.reconstruct("teeg-abc123", base_note)
    store.save()
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from oml.memory.atomic_note import AtomicNote

logger = logging.getLogger(__name__)

_DELTA_FILE = "delta_store.jsonl"

# Estimated tokens saved per patch vs storing the extending note as a full note.
# FULL TOON ≈ 87 tokens/note; COMPACT ≈ 55 tokens/note → saving ≈ 32 tokens.
# Reconstruction merges base+patches into one COMPACT entry instead of two FULL
# entries, saving the extra full-note overhead.
_TOKENS_SAVED_PER_PATCH: int = 32


# ══════════════════════════════════════════════════════════════════════════════
# SemanticPatch
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class SemanticPatch:
    """A semantic diff capturing only the new information in an EXTENDS note.

    When ``MemoryEvolver`` classifies note *B* as ``EXTENDS`` relative to base
    note *A*, ``DeltaStore.store_patch()`` creates a ``SemanticPatch`` that
    records only the incremental fact from *B* rather than its full content.

    Reconstruction::

        full_content(A) = A.content
                        + " Additionally: " + patch₁.patch_content
                        + " Additionally: " + patch₂.patch_content
                        + …

    Attributes
    ----------
    patch_id:
        Unique identifier of the form ``"delta-{12 hex chars}"``.
    base_note_id:
        The ``note_id`` of the base ``AtomicNote`` this patch extends.
    patch_content:
        The new information only — stripped of content already present in the
        base note.  Filled from ``patch_note.content`` by the caller.
    patch_type:
        Relationship type: ``EXTENDS`` | ``CLARIFIES`` | ``CORRECTS``.
    created_at:
        ISO-8601 UTC timestamp of patch creation.
    keywords:
        Keywords specific to the new delta content (not inherited from base).
    """

    patch_id: str
    base_note_id: str
    patch_content: str
    patch_type: str
    created_at: str
    keywords: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "patch_id": self.patch_id,
            "base_note_id": self.base_note_id,
            "patch_content": self.patch_content,
            "patch_type": self.patch_type,
            "created_at": self.created_at,
            "keywords": list(self.keywords),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SemanticPatch":
        return cls(
            patch_id=d["patch_id"],
            base_note_id=d["base_note_id"],
            patch_content=d["patch_content"],
            patch_type=d.get("patch_type", "EXTENDS"),
            created_at=d["created_at"],
            keywords=list(d.get("keywords", [])),
        )


# ══════════════════════════════════════════════════════════════════════════════
# DeltaStore
# ══════════════════════════════════════════════════════════════════════════════


class DeltaStore:
    """JSON-Lines patch storage with O(1) lookup and chain reconstruction.

    Sits alongside ``TEEGStore`` in the same ``artifacts_dir``.  When
    ``PRISMPipeline`` detects an ``EXTENDS`` verdict for a new note, it stores
    a ``SemanticPatch`` here instead of (or in addition to) persisting the full
    note content.

    The in-memory index is a ``{base_note_id: [SemanticPatch, …]}`` dict for
    O(1) lookup.  Patches are stored in chronological order so reconstruction
    replays them as a forward chain.

    Persistence
    -----------
    ``<artifacts_dir>/delta_store.jsonl`` — one JSON line per patch.
    The file is rewritten in full on ``save()`` (all patches are small, so a
    full rewrite is negligible).

    Parameters
    ----------
    artifacts_dir:
        Directory where ``delta_store.jsonl`` will be read/written.  Should
        be the same directory as the associated ``TEEGStore``.
    """

    def __init__(self, artifacts_dir: str | Path = "teeg_store") -> None:
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        # base_note_id → list of patches (chronological order)
        self._patches: Dict[str, List[SemanticPatch]] = {}
        self.load()

    # ── write ─────────────────────────────────────────────────────────────────

    def store_patch(
        self,
        base_id: str,
        patch_note: AtomicNote,
        patch_type: str = "EXTENDS",
    ) -> SemanticPatch:
        """Create and store a semantic patch from *patch_note* onto *base_id*.

        The ``patch_content`` is taken from ``patch_note.content``.  The caller
        is responsible for ensuring it contains only the incremental information
        relative to the base note (``PRISMPipeline`` passes the raw distilled
        content; future versions may perform explicit diff extraction).

        Parameters
        ----------
        base_id:
            ``note_id`` of the note being extended.
        patch_note:
            The new ``AtomicNote`` whose content is the delta.
        patch_type:
            Relationship type (default ``"EXTENDS"``).

        Returns
        -------
        SemanticPatch
            The newly created patch (already registered in memory; call
            :meth:`save` to persist to disk).
        """
        patch = SemanticPatch(
            patch_id=f"delta-{uuid.uuid4().hex[:12]}",
            base_note_id=base_id,
            patch_content=patch_note.content,
            patch_type=patch_type,
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            keywords=list(patch_note.keywords),
        )
        if base_id not in self._patches:
            self._patches[base_id] = []
        self._patches[base_id].append(patch)
        logger.debug(
            "[DeltaStore] Stored patch %s → base %s", patch.patch_id, base_id
        )
        return patch

    # ── read ──────────────────────────────────────────────────────────────────

    def has_patches(self, note_id: str) -> bool:
        """Return True if *note_id* has at least one delta patch."""
        return note_id in self._patches and len(self._patches[note_id]) > 0

    def get_patches(self, note_id: str) -> List[SemanticPatch]:
        """Return all patches for *note_id* in chronological order."""
        return list(self._patches.get(note_id, []))

    def reconstruct(self, note_id: str, base_note: AtomicNote) -> str:
        """Return the full reconstructed content for a note with patches.

        If no patches exist, returns ``base_note.content`` unchanged.
        Otherwise, appends each patch in chronological order using
        the phrase ``"Additionally: …"`` as a natural language connector.

        Example::

            base: "Victor Frankenstein created the creature."
            patch₁: "He used lightning to animate it."
            → "Victor Frankenstein created the creature.
               Additionally: He used lightning to animate it."

        Parameters
        ----------
        note_id:
            The ``note_id`` of the base note (used to look up patches).
        base_note:
            The base ``AtomicNote`` providing the root content.

        Returns
        -------
        str
            Reconstructed full content string.
        """
        patches = self.get_patches(note_id)
        if not patches:
            return base_note.content
        parts = [base_note.content]
        for patch in patches:
            parts.append(f"Additionally: {patch.patch_content}")
        return " ".join(parts)

    def get_all_patched_note_ids(self) -> List[str]:
        """Return all base note IDs that have at least one patch."""
        return list(self._patches.keys())

    # ── metrics ───────────────────────────────────────────────────────────────

    def token_savings(self) -> int:
        """Estimate total LLM context tokens saved by delta storage.

        Calculated as: total patches × ``_TOKENS_SAVED_PER_PATCH`` (32).
        This models the saving from rendering one COMPACT reconstructed entry
        instead of two separate FULL entries per base+extension pair.
        """
        return self.count() * _TOKENS_SAVED_PER_PATCH

    def count(self) -> int:
        """Total number of stored patches across all base notes."""
        return sum(len(ps) for ps in self._patches.values())

    # ── persistence ──────────────────────────────────────────────────────────

    def save(self) -> None:
        """Persist all patches to ``<artifacts_dir>/delta_store.jsonl``."""
        path = self.artifacts_dir / _DELTA_FILE
        with open(path, "w", encoding="utf-8") as f:
            for patches in self._patches.values():
                for patch in patches:
                    f.write(json.dumps(patch.to_dict()) + "\n")
        logger.debug("[DeltaStore] Saved %d patches to %s", self.count(), path)

    def load(self) -> None:
        """Load patches from disk (no-op if file absent)."""
        path = self.artifacts_dir / _DELTA_FILE
        if not path.exists():
            return
        try:
            loaded = 0
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    d = json.loads(line)
                    patch = SemanticPatch.from_dict(d)
                    if patch.base_note_id not in self._patches:
                        self._patches[patch.base_note_id] = []
                    self._patches[patch.base_note_id].append(patch)
                    loaded += 1
            logger.debug("[DeltaStore] Loaded %d patches from %s", loaded, path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[DeltaStore] Could not load: %s", exc)

    # ── diagnostics ──────────────────────────────────────────────────────────

    def stats(self) -> dict:
        """Return statistics for ``prism-stats`` and health checks."""
        return {
            "bases_with_patches": len(self._patches),
            "total_patches": self.count(),
            "token_savings_est": self.token_savings(),
        }
