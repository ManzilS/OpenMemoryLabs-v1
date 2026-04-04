"""oml/memory/compressor.py — Tiered TOON compression for LLM context assembly.

The key insight: when you have 8 notes and a 1 000-token budget, you should
show the *most important* notes in full detail and progressively compress the
rest — rather than dropping whole notes the moment the budget is exceeded.

Three compression tiers
-----------------------
FULL (~87 tok/note)
    Standard TOON with all fields. Used for the most important notes.

COMPACT (~55 tok/note)
    Abbreviated keys, empty fields omitted, note_id truncated to 8 chars.
    ~37% token savings vs FULL. Used for mid-importance notes.

MINIMAL (~28 tok/note)
    Single-line format: ``[id] content (kw: k1|k2)``.
    ~68% token savings vs FULL. Used for low-importance background context.

Example packing for 8 notes with a 500-token budget:
  FULL     top 2 notes  → 2 × 87  = 174 tokens
  COMPACT  next 4 notes → 4 × 55  = 220 tokens
  MINIMAL  last 2 notes → 2 × 28  =  56 tokens
  Total                            450 tokens   (vs 696 if all FULL)

Usage
-----
    from oml.memory.compressor import TieredContextPacker, Tier

    packer = TieredContextPacker(budget=800)
    context_str = packer.pack(scout_results)   # List[ScoutResult]

    # Manual tier encoding (for inspection / testing)
    from oml.memory.compressor import encode_compact, encode_minimal
    compact_str = encode_compact(note)
    minimal_str = encode_minimal(note)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple, TYPE_CHECKING

from oml.memory.atomic_note import AtomicNote

if TYPE_CHECKING:
    from oml.retrieval.scout import ScoutResult


# ── tier definitions ─────────────────────────────────────────────────────────


class Tier(str, Enum):
    """Compression tier for a single note in the LLM context block."""

    FULL = "full"
    """Standard TOON — all fields, maximum detail."""

    COMPACT = "compact"
    """Abbreviated keys, empty fields omitted, truncated note_id."""

    MINIMAL = "minimal"
    """Single-line overview — content + top keywords only."""


# Default thresholds for tier assignment based on ranked position.
# Notes in the top ``FULL_FRACTION`` by importance get FULL encoding;
# the next ``COMPACT_FRACTION`` get COMPACT; the rest get MINIMAL.
FULL_FRACTION: float = 0.25     # top 25%  → FULL
COMPACT_FRACTION: float = 0.50  # next 50% → COMPACT
# remaining 25% → MINIMAL


# ── compact / minimal encoders ───────────────────────────────────────────────


def encode_compact(note: AtomicNote) -> str:
    """Return a COMPACT TOON representation of *note*.

    Uses single-character or abbreviated keys and omits fields that are empty
    or that the LLM does not need (timestamps, active flag, source_ids).

    Approximate token cost: ~55 tokens per note (vs ~87 for FULL TOON).
    """
    lines: list[str] = []

    # Truncate note_id to 13 chars ("teeg-" + 8 hex) — still unique enough
    short_id = note.note_id[:13]
    lines.append(f"id: {short_id}")
    lines.append(f"c: {note.content}")

    if note.context:
        lines.append(f"ctx: {note.context}")
    if note.keywords:
        lines.append(f"kw: {'|'.join(note.keywords)}")
    if note.tags:
        lines.append(f"tags: {'|'.join(note.tags)}")
    lines.append(f"conf: {note.confidence:.2f}")
    if note.supersedes:
        lines.append(f"supersedes: {note.supersedes[:13]}")

    return "\n".join(lines)


def encode_minimal(note: AtomicNote) -> str:
    """Return a MINIMAL single-line representation of *note*.

    Format: ``[<short_id>] <content> (kw: <kw1>|<kw2>)``

    Approximate token cost: ~28 tokens per note (vs ~87 for FULL TOON).
    """
    short_id = note.note_id[:13]
    kws = "|".join(note.keywords[:3]) if note.keywords else ""
    kw_part = f" (kw: {kws})" if kws else ""
    return f"[{short_id}] {note.content}{kw_part}"


# ── tier estimation ───────────────────────────────────────────────────────────


def estimate_tokens(text: str) -> int:
    """Rough token estimate using the 3.5-chars/token heuristic."""
    return math.ceil(len(text) / 3.5)


def tier_token_cost(note: AtomicNote, tier: Tier) -> int:
    """Return estimated token cost for *note* at the given *tier*."""
    if tier is Tier.FULL:
        # Header line + full TOON
        return estimate_tokens(f"--- [seed  score=1.000] ---\n{note.to_toon()}")
    if tier is Tier.COMPACT:
        return estimate_tokens(f"--- [seed  score=1.000] ---\n{encode_compact(note)}")
    # Tier.MINIMAL
    return estimate_tokens(encode_minimal(note))


# ── packer ───────────────────────────────────────────────────────────────────


@dataclass
class PackerStats:
    """Statistics about a single packing run."""

    notes_packed: int
    full_count: int
    compact_count: int
    minimal_count: int
    tokens_used: int
    tokens_saved_vs_all_full: int


class TieredContextPacker:
    """Pack a list of scored notes into a token-budget context string.

    Assigns each note a compression tier (FULL / COMPACT / MINIMAL) based on
    its rank within the result set, then serialises them in descending
    importance order within the ``[TEEG MEMORY]`` block.

    Parameters
    ----------
    budget:
        Maximum token budget for the entire context block (including headers).
        Notes are dropped from the bottom if the budget would be exceeded even
        at MINIMAL tier.
    full_fraction:
        Fraction of the result set to encode at FULL tier (default: 0.25).
    compact_fraction:
        Fraction of the result set to encode at COMPACT tier (default: 0.50).
        The remaining ``1 - full - compact`` fraction is encoded at MINIMAL.
    """

    def __init__(
        self,
        budget: int = 2000,
        full_fraction: float = FULL_FRACTION,
        compact_fraction: float = COMPACT_FRACTION,
    ) -> None:
        self.budget = budget
        self.full_fraction = full_fraction
        self.compact_fraction = compact_fraction

    # ── public API ────────────────────────────────────────────────────────────

    def pack(
        self,
        results: "List[ScoutResult]",
        importance_scores: Optional[dict[str, float]] = None,
    ) -> str:
        """Serialise *results* into a tiered TOON context block.

        Parameters
        ----------
        results:
            List of ``(AtomicNote, score, hops)`` tuples from ScoutRetriever.
        importance_scores:
            Optional pre-computed importance scores keyed by ``note_id``.  If
            provided, tier assignment uses importance scores instead of raw
            position.  This is the recommended mode when ``ImportanceScorer``
            is available.

        Returns
        -------
        str
            A multiline string wrapped in ``[TEEG MEMORY]`` / ``[/TEEG MEMORY]``
            tags, ready to inject into an LLM prompt.
        """
        if not results:
            return "[TEEG MEMORY]\n(no relevant memory found)\n[/TEEG MEMORY]"

        # Assign tiers
        tiered = self._assign_tiers(results, importance_scores)

        # Serialise with budget enforcement
        lines = ["[TEEG MEMORY]"]
        tokens_used = estimate_tokens("[TEEG MEMORY]\n[/TEEG MEMORY]")
        full_c = compact_c = minimal_c = 0

        for note, scout_score, hops, tier in tiered:
            label = "seed" if hops == 0 else f"hop-{hops}"

            if tier is Tier.FULL:
                body = note.to_toon()
                header = f"--- [{label}  score={scout_score:.3f}] ---"
            elif tier is Tier.COMPACT:
                body = encode_compact(note)
                header = f"--- [{label}  score={scout_score:.3f}  compact] ---"
            else:  # Tier.MINIMAL
                # No header block for minimal — inline
                entry = encode_minimal(note)
                cost = estimate_tokens(entry)
                if tokens_used + cost > self.budget:
                    break
                lines.append(entry)
                tokens_used += cost
                minimal_c += 1
                continue

            entry = f"{header}\n{body}"
            cost = estimate_tokens(entry)
            if tokens_used + cost > self.budget:
                # Downgrade to MINIMAL before giving up
                minimal_entry = encode_minimal(note)
                minimal_cost = estimate_tokens(minimal_entry)
                if tokens_used + minimal_cost <= self.budget:
                    lines.append(minimal_entry)
                    tokens_used += minimal_cost
                    minimal_c += 1
                break

            lines.append(entry)
            tokens_used += cost
            if tier is Tier.FULL:
                full_c += 1
            else:
                compact_c += 1

        lines.append("[/TEEG MEMORY]")
        return "\n".join(lines)

    def stats(
        self,
        results: "List[ScoutResult]",
        importance_scores: Optional[dict[str, float]] = None,
    ) -> PackerStats:
        """Return packing statistics *without* building the full context string.

        Useful for benchmarking token savings before deploying a new budget.
        """
        tiered = self._assign_tiers(results, importance_scores)
        all_full_tokens = sum(tier_token_cost(n, Tier.FULL) for n, *_ in tiered)

        tokens_used = estimate_tokens("[TEEG MEMORY]\n[/TEEG MEMORY]")
        full_c = compact_c = minimal_c = packed = 0

        for note, _, __, tier in tiered:
            cost = tier_token_cost(note, tier)
            if tokens_used + cost > self.budget:
                break
            tokens_used += cost
            packed += 1
            if tier is Tier.FULL:
                full_c += 1
            elif tier is Tier.COMPACT:
                compact_c += 1
            else:
                minimal_c += 1

        return PackerStats(
            notes_packed=packed,
            full_count=full_c,
            compact_count=compact_c,
            minimal_count=minimal_c,
            tokens_used=tokens_used,
            tokens_saved_vs_all_full=all_full_tokens - tokens_used,
        )

    # ── internals ────────────────────────────────────────────────────────────

    def _assign_tiers(
        self,
        results: "List[ScoutResult]",
        importance_scores: Optional[dict[str, float]],
    ) -> List[Tuple[AtomicNote, float, int, Tier]]:
        """Return ``(note, scout_score, hops, tier)`` in importance order."""
        n = len(results)
        full_cutoff = max(1, math.ceil(n * self.full_fraction))
        compact_cutoff = max(full_cutoff + 1, math.ceil(n * (self.full_fraction + self.compact_fraction)))

        # Sort by composite priority: importance (if available) then scout_score
        if importance_scores:
            ordered = sorted(
                results,
                key=lambda r: (importance_scores.get(r[0].note_id, 0.0), r[1]),
                reverse=True,
            )
        else:
            # Fallback: seeds first, then by score
            ordered = sorted(results, key=lambda r: (r[2], -r[1]))

        tiered = []
        for i, (note, score, hops) in enumerate(ordered):
            if i < full_cutoff:
                tier = Tier.FULL
            elif i < compact_cutoff:
                tier = Tier.COMPACT
            else:
                tier = Tier.MINIMAL
            tiered.append((note, score, hops, tier))

        return tiered
