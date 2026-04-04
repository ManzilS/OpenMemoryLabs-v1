"""
Stage 1 Pre-Screen — fast cheap-model contradiction detector.
=============================================================

Runs a lightweight 3B model prompt on each (new, existing) pair and
classifies the response as YES / SCOPE? / NO using fuzzy keyword matching.

Priority order (handles malformed outputs):
  1. SCOPE signals (scope, context, condition, altitude, ...) -> SCOPE?
  2. Contradiction keywords (contradict, conflict, oppose, ...) OR "yes" -> YES
  3. No-signals (unrelated, support, elaborate, ...) without yes -> NO
  4. Everything else -> YES (recall-biased fallback)
"""

from __future__ import annotations

import logging
import re
from typing import List

from oml.memory.atomic_note import AtomicNote

logger = logging.getLogger(__name__)

# ── Stage 1 fuzzy parser patterns ────────────────────────────────────────────
# Priority: SCOPE? > YES > NO > YES (fallback)

_SCOPE_PATTERNS: List[str] = [
    r'\bscope\b', r'\bcontext\b', r'\bcondition\b', r'\baltitude\b',
    r'\bhowever\b', r'\bcaveat\b', r'\bdepend', r'\bqualif',
    r'\bspecific\b', r'\bvary\b', r'\bvaries\b', r'\bdifferent\b',
    r'\bbut\b',
]
_CONTRADICTION_PATTERNS: List[str] = [
    r'\bcontradict', r'\bconflict', r'\boppos', r'\brefut',
    r'\bdisagree\b', r'\binconsist', r'\bincompat',
]
_NO_PATTERNS: List[str] = [
    r'\bunrelated\b', r'\bsupport', r'\belaborat', r'\bextend',
    r'\bcorroborat', r'\bconsistent\b', r'\bagree\b',
    r'\bsame\s+(fact|claim|idea|thing)\b',
]


class Stage1PreScreen:
    """Fast pre-screen that filters candidate pairs before the full Stage 2 judge.

    Parameters
    ----------
    llm_client:
        The LLM client to use for Stage 1 calls.  Must support
        ``generate(prompt)`` and ``generate_many(prompts)``.
    """

    def __init__(self, llm_client=None):
        self._llm = llm_client

    @property
    def llm(self):
        return self._llm

    @llm.setter
    def llm(self, value):
        self._llm = value

    def parse_stage1_verdict(self, response: str) -> str:
        """Fuzzy-parse Stage 1 output -> YES / SCOPE? / NO.

        Priority (handles malformed outputs like "YES, BUT SCOPE MIGHT BE DIFFERENT"):
          1. Any scope signal present -> SCOPE?  (overrides YES/NO)
          2. "yes" word or contradiction keyword -> YES
          3. Clear no-signal with no contradiction -> NO
          4. Everything else -> YES  (recall-biased fallback; false-positive is safe)

        Parameters
        ----------
        response:
            Raw string from the Stage 1 LLM.

        Returns
        -------
        str
            One of ``"YES"``, ``"SCOPE?"``, ``"NO"``.
        """
        # Normalise: lowercase, strip punctuation noise
        text = re.sub(r"[^\w\s]", " ", response.strip().lower())

        has_scope        = any(re.search(p, text) for p in _SCOPE_PATTERNS)
        has_yes_word     = bool(re.search(r"\byes\b", text))
        has_contradiction = any(re.search(p, text) for p in _CONTRADICTION_PATTERNS)
        has_no_signal    = any(re.search(p, text) for p in _NO_PATTERNS)
        # "no" only at start of response or as a standalone word in short replies
        has_no_word      = bool(re.search(r"^\s*no\b", text) or
                                (len(text.split()) <= 4 and re.search(r"\bno\b", text)))

        # 1. Scope signals take highest priority (even if YES is also present)
        if has_scope:
            logger.debug("[Stage1] SCOPE? detected in %r", response[:60])
            return "SCOPE?"

        # 2. Yes word or contradiction keyword
        if has_yes_word or has_contradiction:
            return "YES"

        # 3. Clean no-signal with no contradiction override
        if (has_no_word or has_no_signal) and not has_contradiction:
            return "NO"

        # 4. Recall-biased fallback
        logger.debug("[Stage1] Unparseable %r -> defaulting to YES", response[:60])
        return "YES"

    def build_stage1_prompt(
        self, new_note: AtomicNote, existing_note: AtomicNote
    ) -> str:
        """Ultra-light Stage 1 prompt -- optimised for speed over precision."""
        def _slim(note: AtomicNote) -> str:
            parts = [f"content: {note.content}"]
            if note.keywords:
                parts.append(f"keywords: {'|'.join(note.keywords[:4])}")
            return " | ".join(parts)

        return (
            "Contradiction pre-check. Answer only:\n"
            "YES (possible conflict) / SCOPE? (possible conflict but different "
            "conditions or context) / NO.\n"
            "If unsure: YES.\n\n"
            f"EXISTING: {_slim(existing_note)}\n"
            f"NEW: {_slim(new_note)}"
        )

    def screen(
        self,
        new_notes: List[AtomicNote],
        existing_notes: List[AtomicNote],
    ) -> List[str]:
        """Run Stage 1 fast pre-screen on all pairs concurrently.

        Returns a list of ``"YES"`` / ``"SCOPE?"`` / ``"NO"`` per pair.
        On any failure defaults to ``"YES"`` (recall-biased).

        Prompts are built sequentially then dispatched via
        ``llm.generate_many()`` so that network I/O for all pairs overlaps.
        """
        llm = self._llm
        pairs = list(zip(new_notes, existing_notes))

        prompts: List[str] = [
            self.build_stage1_prompt(new_n, existing_n)
            for new_n, existing_n in pairs
        ]

        try:
            responses = llm.generate_many(prompts)
        except Exception as exc:
            logger.warning("[Stage1] generate_many failed (%s) -> falling back to sequential", exc)
            responses = []
            for prompt in prompts:
                try:
                    responses.append(llm.generate(prompt))
                except Exception as inner_exc:
                    logger.warning("[Stage1] generate failed (%s) -> defaulting to YES", inner_exc)
                    responses.append("YES")

        results: List[str] = []
        for response in responses:
            try:
                verdict = self.parse_stage1_verdict(response)
            except Exception as exc:
                logger.warning("[Stage1] parse failed (%s) -> defaulting to YES", exc)
                verdict = "YES"
            results.append(verdict)
        return results
