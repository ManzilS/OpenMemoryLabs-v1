"""
Stage 2 Judge — full LLM-driven relationship classification.
=============================================================

Classifies (new_note, existing_note) pairs into one of:
  CONTRADICTS / EXTENDS / SUPPORTS / UNRELATED

Also extracts STRENGTH, AUTHORITY, and SCOPE_MATCH from the response.
"""

from __future__ import annotations

import logging
from typing import List, Tuple

from oml.memory.atomic_note import AtomicNote

logger = logging.getLogger(__name__)

# ── Verdict constants ─────────────────────────────────────────────────────────
_CONTRADICTS = "CONTRADICTS"
_EXTENDS     = "EXTENDS"
_SUPPORTS    = "SUPPORTS"
_UNRELATED   = "UNRELATED"
_VALID_RELATIONS = {_CONTRADICTS, _EXTENDS, _SUPPORTS, _UNRELATED}


class Stage2Judge:
    """Full LLM judge that classifies the relationship between note pairs.

    Parameters
    ----------
    llm_client:
        The LLM client for Stage 2 calls.  Must support ``generate(prompt)``.
    """

    def __init__(self, llm_client=None):
        self._llm = llm_client

    @property
    def llm(self):
        return self._llm

    @llm.setter
    def llm(self, value):
        self._llm = value

    def build_judge_prompt(
        self, new_note: AtomicNote, existing_note: AtomicNote
    ) -> str:
        """Build the Stage 2 TOON-encoded judge prompt with authority/strength fields."""
        from oml.memory.batcher import _judge_toon
        new_toon      = _judge_toon(new_note.to_toon())
        existing_toon = _judge_toon(existing_note.to_toon())

        return f"""You are a memory consistency judge for an AI knowledge base.

Compare the NEW memory note against the EXISTING memory note.

NEW NOTE (TOON format):
{new_toon}

EXISTING NOTE (TOON format):
{existing_toon}

Classify the relationship. Choose exactly one:
- CONTRADICTS: The new note directly conflicts with or refutes the existing note.
- EXTENDS: The new note adds detail to or builds upon the existing note.
- SUPPORTS: The new note independently corroborates the existing note.
- UNRELATED: The notes describe entirely different facts or entities.

Also rate:
- STRENGTH: 0.0-1.0 (how strongly does this relation hold?)
- AUTHORITY: 0.0-1.0 (how credible is the new note's source? High=expert/study Low=hearsay/anecdote)
- SCOPE_MATCH: YES or NO (are both notes making claims about the same scope/conditions?)

Respond with EXACTLY this format (nothing else):
RELATION: <CONTRADICTS|EXTENDS|SUPPORTS|UNRELATED>
STRENGTH: <0.0-1.0>
AUTHORITY: <0.0-1.0>
SCOPE_MATCH: <YES|NO>
REASON: <one sentence>"""

    def parse_verdict(self, response: str) -> Tuple[str, str]:
        """Parse Stage 2 response -> (relation, reason).  Backward-compat alias."""
        rel, reason, _s, _a = self.parse_verdict_full(response)
        return rel, reason

    def parse_verdict_full(
        self, response: str
    ) -> Tuple[str, str, float, float]:
        """Parse Stage 2 response -> (relation, reason, strength, authority).

        Defaults:
          - relation  -> SUPPORTS (safe conservative)
          - strength  -> 1.0 (full weight for backward-compat single-judge path)
          - authority -> 0.8 (moderate default; explicit low-authority sources
                             must be flagged by the judge to take effect)
        """
        relation  = _SUPPORTS
        reason    = ""
        strength  = 1.0
        authority = 0.8

        for line in response.strip().splitlines():
            line  = line.strip()
            upper = line.upper()
            if upper.startswith("RELATION:"):
                raw = line.split(":", 1)[1].strip().upper()
                for valid in _VALID_RELATIONS:
                    if valid in raw:
                        relation = valid
                        break
            elif upper.startswith("STRENGTH:"):
                try:
                    strength = max(0.0, min(1.0, float(line.split(":", 1)[1].strip())))
                except (ValueError, IndexError):
                    pass
            elif upper.startswith("AUTHORITY:"):
                try:
                    authority = max(0.0, min(1.0, float(line.split(":", 1)[1].strip())))
                except (ValueError, IndexError):
                    pass
            elif upper.startswith("REASON:"):
                reason = line.split(":", 1)[1].strip()
            # SCOPE_MATCH is parsed but not returned -- used to log warnings
            elif upper.startswith("SCOPE_MATCH:") and "NO" in upper:
                logger.debug(
                    "[MemoryEvolver] Judge flagged SCOPE_MATCH: NO -- "
                    "consider using EXTENDS instead of CONTRADICTS"
                )

        return relation, reason, strength, authority

    def judge(
        self, new_note: AtomicNote, existing_note: AtomicNote
    ) -> Tuple[str, str]:
        """Single-pair fallback judge (no Stage 1).  Returns (relation, reason).

        Used as a last-resort fallback path.
        """
        prompt = self.build_judge_prompt(new_note, existing_note)
        try:
            llm = self._llm
            response = llm.generate(prompt)
            rel, reason, _s, _a = self.parse_verdict_full(response)
            return rel, reason
        except Exception as exc:
            logger.warning("[MemoryEvolver] Judge LLM failed (%s); defaulting to SUPPORTS", exc)
            return _SUPPORTS, "judge unavailable"

    def judge_full(
        self, new_note: AtomicNote, existing_note: AtomicNote
    ) -> Tuple[str, str, float, float]:
        """Single-pair full judge returning (relation, reason, strength, authority)."""
        prompt = self.build_judge_prompt(new_note, existing_note)
        try:
            llm = self._llm
            response = llm.generate(prompt)
            return self.parse_verdict_full(response)
        except Exception as exc:
            logger.warning("[MemoryEvolver] Full judge failed (%s); defaulting to SUPPORTS", exc)
            return _SUPPORTS, "judge unavailable", 1.0, 1.0

    def judge_batch(
        self,
        new_notes: List[AtomicNote],
        existing_notes: List[AtomicNote],
        stage1_prescreen,
        similarity_top_k: int = 5,
    ) -> List[Tuple[str, str, float, float]]:
        """Two-stage batch judge.

        Stage 1: cheap pre-screen on all pairs -> YES / SCOPE? / NO
        Stage 2: full judge only on escalated (YES / SCOPE?) pairs

        Returns
        -------
        List[Tuple[str, str, float, float]]
            One ``(relation, reason, strength, authority)`` per pair.
            Fast-path (Stage 1 NO) pairs return ``("SUPPORTS", "stage1-no", 1.0, 1.0)``.
        """
        # Default: treat all as SUPPORTS (conservative, no archiving)
        results: List[Tuple[str, str, float, float]] = [
            (_SUPPORTS, "stage1-no", 1.0, 1.0)
        ] * len(new_notes)

        stage1_verdicts = stage1_prescreen.screen(new_notes, existing_notes)

        escalated = [i for i, v in enumerate(stage1_verdicts) if v != "NO"]
        fast_path  = [i for i, v in enumerate(stage1_verdicts) if v == "NO"]

        for i in fast_path:
            logger.debug("[MemoryEvolver] Stage1 NO for pair %d -> fast-path SUPPORTS", i)

        if not escalated:
            return results

        # Stage 2 -- batch escalated pairs through the full judge
        try:
            from oml.memory.batcher import CallBatcher
            llm = self._llm
            batcher = CallBatcher(llm_client=llm, max_batch_size=similarity_top_k)
            esc_new      = [new_notes[i]      for i in escalated]
            esc_existing = [existing_notes[i] for i in escalated]
            verdict_result = batcher.evolve_batch(esc_new, esc_existing)

            for local_i, global_i in enumerate(escalated):
                verdict = verdict_result.verdicts[local_i]
                # evolve_batch returns bare relation strings; default strength/authority
                results[global_i] = (verdict, "stage2-batch", 0.8, 0.8)

        except Exception as exc:
            logger.warning(
                "[MemoryEvolver] Stage 2 batch failed (%s); falling back to individual calls",
                exc,
            )
            for i in escalated:
                rel, reason, strength, authority = self.judge_full(
                    new_notes[i], existing_notes[i]
                )
                results[i] = (rel, reason, strength, authority)

        return results
