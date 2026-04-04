"""
HeuristicDistiller — Rule-based distillation without LLM.
==========================================================

Creates a minimal :class:`AtomicNote` from raw text using simple
heuristics (first N words, keyword extraction).  Used as the fallback
when LLM distillation is unavailable or fails.
"""

from __future__ import annotations

from oml.memory.atomic_note import AtomicNote


class HeuristicDistiller:
    """
    Distil raw text into an :class:`AtomicNote` without any LLM call.

    All methods are static — the class exists purely to provide a
    uniform interface alongside :class:`LLMDistiller`.
    """

    @staticmethod
    def distil(
        raw_text: str,
        context_hint: str = "",
        source_id: str = "",
    ) -> AtomicNote:
        """Create a minimal AtomicNote from raw text without LLM help."""
        words = raw_text.split()
        content = " ".join(words[:40])
        keywords = list(
            {w.lower().strip(".,!?;:") for w in words[:20] if len(w) > 3}
        )[:6]
        return AtomicNote(
            content=content,
            context=context_hint or "auto-generated",
            keywords=keywords,
            tags=["auto"],
            confidence=0.5,
            source_ids=[source_id] if source_id else [],
        )
