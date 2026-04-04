"""
LLMDistiller — LLM-based distillation of raw text into AtomicNotes.
====================================================================

Extracts the distillation logic from TEEGPipeline into a reusable
technique module.  The LLM is prompted to produce a structured
AtomicNote in TOON format; if parsing fails the caller can fall back
to :class:`HeuristicDistiller`.
"""

from __future__ import annotations

import logging

from oml.memory.atomic_note import AtomicNote

logger = logging.getLogger(__name__)


class LLMDistiller:
    """
    Distil raw text into an :class:`AtomicNote` via an LLM call.

    Parameters
    ----------
    llm_client:
        Any object exposing a ``generate(prompt: str) -> str`` method.
    """

    def __init__(self, llm_client):
        self._llm = llm_client

    # ── public API ────────────────────────────────────────────────────────────

    def distil(
        self,
        raw_text: str,
        context_hint: str = "",
        source_id: str = "",
    ) -> AtomicNote:
        """
        Ask the LLM to distil *raw_text* into a structured AtomicNote.

        Returns the parsed note on success.  Raises on LLM or parse failure
        so the caller can decide on fallback behaviour.
        """
        prompt = self._build_distil_prompt(raw_text, context_hint)
        response = self._llm.generate(prompt)
        note = self._parse_distil_response(response, raw_text, source_id)
        if source_id:
            note.source_ids = [source_id]
        return note

    # ── prompt construction ───────────────────────────────────────────────────

    @staticmethod
    def _build_distil_prompt(raw_text: str, context_hint: str) -> str:
        hint_line = f"Context: {context_hint}\n" if context_hint else ""
        return (
            f"TOON note: content (<=30 words), context, "
            f"keywords (3-6), tags (2-4), confidence.\n"
            f"{hint_line}Text: {raw_text[:2000]}\n"
            f"TOON fields:\n"
            f"content: Victor built the creature.\n"
            f"context: novel\n"
            f"keywords: victor|creature|built|laboratory\n"
            f"tags: fiction|science\n"
            f"confidence: 0.9"
        )

    # ── response parsing ──────────────────────────────────────────────────────

    @staticmethod
    def _parse_distil_response(
        response: str, raw_text: str, source_id: str
    ) -> AtomicNote:
        """Parse a TOON-formatted LLM response into an AtomicNote."""
        from oml.memory.techniques.heuristic_distiller import HeuristicDistiller

        text = response.strip()
        for fence in ("```toon", "```yaml", "```"):
            if text.startswith(fence):
                text = text[len(fence):]
            if text.endswith("```"):
                text = text[:-3]
        text = text.strip()

        if not text or "content:" not in text.lower():
            return HeuristicDistiller.distil(raw_text, "", source_id)

        note = AtomicNote.from_toon(text)
        if not note.content:
            note.content = raw_text[:200]
        return note
