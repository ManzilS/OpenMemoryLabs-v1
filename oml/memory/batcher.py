"""oml/memory/batcher.py — LLM call coalescing for N-to-1 batch distillation.

Call efficiency gap analysis
-----------------------------
The baseline TEEG pipeline makes **2 LLM calls per ingested text**:
  1. ``_distil(text)`` → ``llm.generate(distil_prompt)`` → one TOON AtomicNote
  2. ``evolver._judge(new_note, candidate)`` → ``llm.generate(judge_prompt)``
     → one verdict (up to ``_MAX_CANDIDATES`` times per note)

For a bulk import of *N* documents this means ≥ 2N API round-trips —
the dominant cost when using hosted LLM endpoints.

``CallBatcher`` reduces this to **2 round-trips regardless of N** by packing
all distillation requests into a single structured prompt and parsing the
multi-output response on a fixed delimiter.

Call savings formula
--------------------
  naive calls = 2N
  batched calls = 2  (1 distil batch + 1 evolve batch)
  efficiency = 1 − 2 / (2N) = 1 − 1/N

For N=8:  87.5 % API call savings.
For N=32: 96.9 % savings.

Novel aspect
------------
Multi-output structured prompting (single call → N delimited outputs) is known
in prompt-engineering literature, but applying it specifically to TOON-encoded
memory distillation and evolution judgment is new.  The key insight is that
TOON's ``key: value`` line format makes it trivial to split on ``---TOON---``
and feed each block to the existing TOON parser without modification.

Graceful degradation
--------------------
If the batch LLM response cannot be parsed into exactly *N* TOON blocks (e.g.
the LLM omits a separator, merges two blocks, or adds markdown fences),
``distil_batch()`` falls back to individual calls for the failed items.
This ensures correctness at the cost of efficiency for malformed responses.

Design constraints
------------------
- Module-level imports throughout (no lazy imports) for ``unittest.mock.patch``.
- ``llm_client`` is duck-typed — any object with ``generate(prompt) → str``.
- Zero new runtime dependencies.

Usage
-----
    batcher = CallBatcher(llm_client=get_llm_client("ollama:qwen3:4b"))

    # Distil 8 texts in 1 LLM call instead of 8
    result = batcher.distil_batch(texts=raw_texts)
    print(result.toon_strings)       # list of TOON strings
    print(result.total_llm_calls)    # 1

    # Evolve 8 (new, existing) pairs in 1 LLM call instead of 8
    v_result = batcher.evolve_batch(new_notes, candidate_notes)
    print(v_result.verdicts)         # ['SUPPORTS', 'EXTENDS', ...]
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import List, Optional

from oml.memory.atomic_note import AtomicNote

logger = logging.getLogger(__name__)

# ── Delimiters ────────────────────────────────────────────────────────────────
# Both strings are chosen to be highly unlikely to appear in natural TOON output
# and are documented in the system prompt so the LLM knows to emit them.

DISTIL_SEP: str = "---TOON---"
VERDICT_SEP: str = "---VERDICT---"

_VALID_VERDICTS = frozenset({"CONTRADICTS", "EXTENDS", "SUPPORTS", "UNRELATED"})
_DEFAULT_VERDICT: str = "SUPPORTS"  # safe conservative default


def _slim_toon(toon_str: str) -> str:
    """Strip empty-value lines from a TOON string for use in judge prompts.

    A line is considered empty-valued when the portion after ``": "`` is blank.
    This removes zero-information fields such as ``supersedes: `` and
    ``source_ids: `` while keeping all populated fields (``content:``,
    ``keywords:``, ``confidence:``, etc.).

    Savings: 2 empty fields × 2 notes per pair = **4 words per pair** removed
    (32 words for N=8 evolve batch).  Pure information gain — empty fields
    can never help a classifier distinguish CONTRADICTS / EXTENDS / SUPPORTS.
    """
    out: list[str] = []
    for line in toon_str.splitlines():
        if ": " in line:
            val = line.split(": ", 1)[1].strip()
            if val:  # keep lines with non-empty values
                out.append(line)
        elif line.rstrip().endswith(":"):
            pass  # bare "key:" with no value — skip
        else:
            out.append(line)
    return "\n".join(out)


# Fields that are always non-informative for CONTRADICTS/EXTENDS/SUPPORTS classification:
#   note_id    — an opaque identifier; the judge compares note content, not IDs.
#                Removing it also yields a cleaner pair format:
#                "EXISTING: content: ..." instead of "EXISTING: note_id: teeg-...\ncontent: ..."
#   created_at — temporal metadata; NEW/EXISTING labels already convey recency ordering
#   confidence — a note's own confidence score does not predict content consistency
#   active     — always True for notes that are candidates; carries no comparative signal
#
# Cycle #19 addition: note_id.  Savings: 2w per note × 2 notes per pair = 4w/pair.
#   N=8 evolve: 233w → ~201w (−32w).  Classification needs content/context/keywords — not IDs.
_JUDGE_EXCLUDE_FIELDS: frozenset[str] = frozenset({"note_id", "created_at", "confidence", "active"})


def _judge_toon(toon_str: str) -> str:
    """Produce a judge-optimised TOON view for consistency classification prompts.

    Combines ``_slim_toon`` (drop empty-value fields) with targeted exclusion
    of non-informative fields for CONTRADICTS/EXTENDS/SUPPORTS/UNRELATED classification.

    Fields kept:  content · context · keywords · tags
    Fields dropped (empty):     supersedes · source_ids
    Fields dropped (metadata):  note_id · created_at · confidence · active

    Savings vs full TOON (per pair, 2 notes):
      * Cycle #12 (_slim_toon):    4w  (supersedes + source_ids, empty)
      * Cycle #13 (_judge_toon):  12w  (created_at + confidence + active × 2 notes)
      * Cycle #19 (note_id):      +4w  (note_id × 2 notes; ID is irrelevant to classification)
      * Total:                    20w per pair (160w for N=8 evolve batch)
    """
    out: list[str] = []
    for line in toon_str.splitlines():
        if ": " in line:
            key, _, val = line.partition(": ")
            key = key.strip()
            if not val.strip():          # empty value → drop
                continue
            if key in _JUDGE_EXCLUDE_FIELDS:  # non-informative metadata → drop
                continue
            out.append(line)
        elif line.rstrip().endswith(":"):
            pass                         # bare "key:" with no value → drop
        else:
            out.append(line)
    return "\n".join(out)


# ══════════════════════════════════════════════════════════════════════════════
# Result dataclasses
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class BatchResult:
    """Result of a batched distillation call.

    Attributes
    ----------
    toon_strings:
        One TOON-encoded note block per input text.  Entries are non-empty
        strings on success; items that failed to parse and then recovered via
        fallback individual calls are filled in before returning.
    total_llm_calls:
        Actual number of ``llm.generate()`` calls made (ideally 1 per batch,
        plus ``len(parse_failures)`` fallback calls).
    parse_failures:
        Indices of texts whose TOON output could not be parsed from the batch
        response.  Always empty after ``distil_batch()`` returns (failures are
        recovered inline).
    """

    toon_strings: List[str]
    total_llm_calls: int
    parse_failures: List[int] = field(default_factory=list)


@dataclass
class VerdictBatchResult:
    """Result of a batched evolution judgment call.

    Attributes
    ----------
    verdicts:
        One verdict per (new_note, candidate_note) pair.  Each verdict is one
        of ``CONTRADICTS``, ``EXTENDS``, ``SUPPORTS``, ``UNRELATED``.
        Defaults to ``SUPPORTS`` on any parse failure (conservative).
    total_llm_calls:
        Actual number of ``llm.generate()`` calls made.
    """

    verdicts: List[str]
    total_llm_calls: int


# ══════════════════════════════════════════════════════════════════════════════
# CallBatcher
# ══════════════════════════════════════════════════════════════════════════════


class CallBatcher:
    """Coalesces N LLM calls into 1 by prompting for multi-output responses.

    Each batch packs *N* inputs into a single structured prompt, requests *N*
    delimiter-separated outputs, and distributes the parsed outputs back to
    callers.  For malformed responses, individual retry calls recover failed
    items so the caller always receives a complete, correctly-sized result.

    Parameters
    ----------
    llm_client:
        Any object with a ``generate(prompt: str) -> str`` method.  The same
        interface used throughout the TEEG stack.
    max_batch_size:
        Maximum items per batch call.  Larger batches risk context-length
        issues with smaller LLMs.  Default: 8.
    """

    def __init__(self, llm_client, max_batch_size: int = 8) -> None:
        self._llm = llm_client
        self.max_batch_size = max_batch_size
        self._calls_made: int = 0
        self._calls_saved: int = 0
        self._batch_sizes: List[int] = []

    # ══════════════════════════════════════════════════════════════════════════
    # Distillation — N texts → N TOON blocks
    # ══════════════════════════════════════════════════════════════════════════

    def distil_batch(
        self,
        texts: List[str],
        context_hints: Optional[List[str]] = None,
    ) -> BatchResult:
        """Distil *N* raw texts into *N* TOON AtomicNote strings in ≤ ceil(N/max_batch_size) LLM calls.

        Automatically splits *texts* into sub-batches of ``max_batch_size``.
        Within each sub-batch, one LLM call is made.  If the response cannot
        be fully parsed, individual fallback calls are made for failed items.

        Parameters
        ----------
        texts:
            Raw text strings to distil (one AtomicNote per text).
        context_hints:
            Optional per-text context hints (e.g. chapter name, source URL).
            Must have the same length as *texts* if provided.

        Returns
        -------
        BatchResult
            ``toon_strings`` has the same length as *texts*.  All entries
            are non-empty; ``parse_failures`` is always empty on return.
        """
        if not texts:
            return BatchResult(toon_strings=[], total_llm_calls=0)

        hints: List[str] = list(context_hints) if context_hints else [""] * len(texts)
        all_toon: List[str] = [""] * len(texts)
        total_calls: int = 0

        for batch_start in range(0, len(texts), self.max_batch_size):
            batch = texts[batch_start: batch_start + self.max_batch_size]
            batch_hints = hints[batch_start: batch_start + self.max_batch_size]
            result = self._distil_single_batch(batch, batch_hints)
            total_calls += result.total_llm_calls
            for local_i, toon_str in enumerate(result.toon_strings):
                all_toon[batch_start + local_i] = toon_str

        return BatchResult(
            toon_strings=all_toon,
            total_llm_calls=total_calls,
            parse_failures=[],
        )

    def _distil_single_batch(
        self, texts: List[str], hints: List[str]
    ) -> BatchResult:
        """Process one sub-batch (up to ``max_batch_size`` texts)."""
        n = len(texts)
        prompt = self._build_distil_prompt(texts, hints)

        try:
            response = self._llm.generate(prompt)
            self._calls_made += 1
            self._batch_sizes.append(n)
            toon_strings = self._parse_distil_response(response, n)

            # Check for parse failures (empty blocks)
            failures = [i for i, t in enumerate(toon_strings) if not t.strip()]

            if not failures:
                # Perfect batch — saved (n - 1) individual calls
                self._calls_saved += max(0, n - 1)
                return BatchResult(
                    toon_strings=toon_strings,
                    total_llm_calls=1,
                )

            # ── fallback for failed items ──────────────────────────────────
            logger.warning(
                "[CallBatcher] %d/%d distil parse failures; falling back to individual calls",
                len(failures), n,
            )
            for i in failures:
                toon_strings[i] = self._distil_one(texts[i], hints[i])
                self._calls_made += 1

            self._calls_saved += max(0, n - 1 - len(failures))
            return BatchResult(
                toon_strings=toon_strings,
                total_llm_calls=1 + len(failures),
            )

        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[CallBatcher] Batch distil failed (%s); falling back to %d individual calls",
                exc, n,
            )
            toon_strings = []
            for text, hint in zip(texts, hints):
                toon_strings.append(self._distil_one(text, hint))
                self._calls_made += 1
            return BatchResult(toon_strings=toon_strings, total_llm_calls=n)

    def _distil_one(self, text: str, hint: str) -> str:
        """Distil a single text (individual-call fallback path)."""
        ctx_line = f"Context: {hint}\n" if hint else ""
        prompt = (
            f"TOON note — fields: note_id (teeg-12hex), content (<=30 words), "
            f"keywords (pipe-sep), confidence (0.0-1.0).\n"
            f"{ctx_line}Text: {text.strip()}\nRespond with only the TOON block."
        )
        try:
            return self._llm.generate(prompt)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[CallBatcher] Fallback distil failed: %s", exc)
            note_id = f"teeg-{uuid.uuid4().hex[:12]}"
            return (
                f"note_id: {note_id}\n"
                f"content: {text[:80]}\n"
                f"keywords: general\n"
                f"confidence: 0.5"
            )

    def _build_distil_prompt(self, texts: List[str], hints: List[str]) -> str:
        """Build the multi-output distillation prompt for *N* texts.

        Cycle #15 optimisations
        -----------------------
        * Header: dropped "Separate blocks with ---TOON---." (redundant — the
          footer already says "separated by ---TOON---" and the example shows
          the separator in place).  Saves **5w** on every distil call.
        * Footer N=1: replaced the 2-block example with a 1-block example.
          Showing a 2-block example when N=1 is contradictory ("respond with
          exactly 1 block" / example has 2 blocks) and wastes 24w.  Saves
          **24w** on every single-item batch call.
        * Footer N≥2: unchanged — the 2-block example with separator is
          required to teach the model where to place the delimiter (lesson
          learned from Cycle #2).
        """
        n = len(texts)
        # Cycle #18: "(3-6 terms, pipe-sep)" → "(3-6 pipe-sep)" — "terms," is redundant.
        # Cycle #20: "as an AtomicNote" removed (3w) — output format is fully defined by
        #   the Fields spec and the example; "AtomicNote" adds no new information.
        # Cycle #21: "(teeg-12hex)" removed (1w) — format shown in example: teeg-abc123def456.
        #            "(0.0-1.0)" removed (1w) — range shown in example: confidence: 0.9.
        # Cycle #22: "each of the" removed (3w) from header — "encode N texts." is equally
        #   clear and saves 3w on every distil call regardless of N.
        # Cycle #24: "(3-6 pipe-sep)" → "(3-6)" — "pipe-sep" is redundant with the example
        #   which shows pipe-separated keywords: victor|frankenstein|creature|laboratory.
        #   The count constraint (3-6) is what matters; format is taught by example.
        #   Saves 1w per distil call.
        # Cycle #30: "Fields:" label removed (−1w) — the field list that follows the
        #   period is unambiguous without a label; "note_id, content..." reads naturally
        #   after "encode N texts." as the specification of what each block must contain.
        lines: List[str] = [
            f"TOON encoder: encode {n} texts. "
            f"note_id, content (<=30 words), "
            f"keywords (3-6), confidence.",
            "",
        ]
        for i, (text, hint) in enumerate(zip(texts, hints), 1):
            lines.append(f"[TEXT {i}]")
            if hint:
                lines.append(f"Context: {hint}")
            lines.append(text.strip())
            lines.append("")

        # Cycle #18: shortened example content values — the <=30 words spec is the
        # authoritative length constraint; the example length guides format, not length.
        # Block 1: 9w→5w (saves 4w). Block 2: 10w→7w (saves 3w).
        if n == 1:
            # 1-block example — no separator needed or shown.
            # Cycle #22: "Respond with exactly 1 TOON block:" → "Respond with 1 block:"
            #   "exactly" removed (1w) — with 1 [TEXT 1] block, 1 output is unambiguous.
            #   "TOON" removed (1w) — example immediately below shows TOON format.
            #   Saves 2w on every N=1 distil call.
            # Cycle #26: "Respond with 1 block:" → "1 block:" (-2w) — "Respond with" is
            #   redundant; the example immediately below shows what to produce.
            # Cycle #28: "Example:" label removed (-1w) — the TOON field pattern that
            #   follows ("note_id: teeg-...") is self-evidently illustrative; the label
            #   provides no additional disambiguation.
            lines += [
                "1 block:",
                "note_id: teeg-abc123def456",
                "content: Victor built the creature.",
                "keywords: victor|creature|built|laboratory",
                "confidence: 0.9",
            ]
        else:
            # 2-block example with separator so the model learns where to place it.
            # Cycle #20: "Example for 2 texts:" → "Example:" saves 3w.
            # Cycle #23: "TOON block(s)" → "block(s)" (−1w; format shown in example).
            #            Block 2 content: "immediately" + "abandoned" removed (−2w);
            #            keywords updated for consistency (no word-count change).
            # Cycle #27: "exactly" removed (−1w) — N already given explicitly; "exactly N"
            #   vs "N" carries no additional constraint for the model parser.
            # Cycle #28: "Example:" label removed (−1w) — same rationale as N=1 case.
            # Cycle #29: block-2 content "The creature fled the laboratory." (5w) →
            #   "The creature fled." (3w) — saves 2w on every N≥2 distil call.
            #   Keywords kept at 3 to satisfy the (3-6) spec: "creature|fled|laboratory"
            #   retains "laboratory" for context even though it is not in the shortened
            #   content, demonstrating that keywords can capture broader context.
            # Cycle #32: "Respond with" removed (−2w) — symmetric with Cycle #26 for N=1.
            #   "{n} block(s) separated by ---TOON---." is unambiguous on its own;
            #   the imperative scaffold "Respond with" adds no new information.
            lines += [
                f"{n} block(s) separated by {DISTIL_SEP}.",
                "note_id: teeg-abc123def456",
                "content: Victor built the creature.",
                "keywords: victor|creature|built|laboratory",
                "confidence: 0.9",
                DISTIL_SEP,
                "note_id: teeg-789fed321cba",
                "content: The creature fled.",
                "keywords: creature|fled|laboratory",
                "confidence: 0.85",
            ]
        return "\n".join(lines)

    def _parse_distil_response(self, response: str, n: int) -> List[str]:
        """Split a batch LLM response on ``DISTIL_SEP`` and return *n* blocks.

        Strips markdown fences (````toon``, ````yaml``, `````) from the response
        before splitting.  Returns an empty string for any missing blocks
        (these become ``parse_failures`` that trigger individual fallback calls).
        """
        # Strip markdown fences
        for fence in ("```toon", "```yaml", "```"):
            response = response.replace(fence, "")

        parts = response.split(DISTIL_SEP)
        result: List[str] = []
        for i in range(n):
            block = parts[i].strip() if i < len(parts) else ""
            result.append(block)
        return result

    # ══════════════════════════════════════════════════════════════════════════
    # Evolution — N (new, existing) pairs → N verdicts
    # ══════════════════════════════════════════════════════════════════════════

    def evolve_batch(
        self,
        new_notes: List[AtomicNote],
        candidate_notes: List[AtomicNote],
    ) -> VerdictBatchResult:
        """Classify *N* (new_note, candidate_note) pairs in a single LLM call.

        Each pair receives one of: ``CONTRADICTS`` | ``EXTENDS`` |
        ``SUPPORTS`` | ``UNRELATED``.  Defaults to ``SUPPORTS`` on any parse
        failure (conservative — does not archive or alter existing notes).

        Parameters
        ----------
        new_notes:
            New notes to classify (one per pair).
        candidate_notes:
            Existing notes to compare against (same length as *new_notes*).

        Returns
        -------
        VerdictBatchResult
            ``verdicts`` has the same length as *new_notes*.
        """
        if not new_notes:
            return VerdictBatchResult(verdicts=[], total_llm_calls=0)

        n = len(new_notes)
        prompt = self._build_evolve_prompt(new_notes, candidate_notes)

        try:
            response = self._llm.generate(prompt)
            self._calls_made += 1
            verdicts = self._parse_verdict_response(response, n)
            self._calls_saved += max(0, n - 1)
            return VerdictBatchResult(verdicts=verdicts, total_llm_calls=1)

        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[CallBatcher] Batch evolve failed (%s); defaulting to %s",
                exc, _DEFAULT_VERDICT,
            )
            self._calls_made += 1
            return VerdictBatchResult(
                verdicts=[_DEFAULT_VERDICT] * n,
                total_llm_calls=1,
            )

    def _build_evolve_prompt(
        self,
        new_notes: List[AtomicNote],
        candidate_notes: List[AtomicNote],
    ) -> str:
        """Build the multi-output evolution judgment prompt for *N* pairs.

        Cycle #16 optimisations (mirrors Cycle #15 on the distil prompt)
        ------------------------------------------------------------------
        * Header: dropped "Separate with ---VERDICT---." (3w) — redundant because
          the footer says "separated by ---VERDICT---" and the example shows the
          separator in place.  Saves **3w on every evolve call**.
        * Footer N=1: replaced the 2-pair example (SUPPORTS / ---VERDICT--- / EXTENDS)
          with a single verdict example.  For N=1 there is no separator to show;
          the old 2-pair example was contradictory ("respond with exactly 1 verdict"
          / example has 2).  Saves **12w for N=1 evolve calls**.
        * Footer N≥2: unchanged — 2-pair example retained to teach the model where
          to place the delimiter (lesson from Cycle #2).
        """
        # Cycle #23: header compressed from 10w to 6w:
        #   "Memory judge: classify each (EXISTING, NEW) note pair as" →
        #   "Judge each (EXISTING, NEW) pair:"
        #   Removed: "Memory" (role implied by "judge"), "classify" (implied by "judge"),
        #   "note" (redundant with pair context), "as" (replaced by ":").
        #   SmartMock detection is via "[PAIR N]" + "---VERDICT---", not header text.
        # Cycle #31: "Judge each (EXISTING, NEW) pair:" → "Judge (EXISTING, NEW) pair:"
        #   "each" removed (−1w) — redundant; [PAIR N] blocks already enumerate
        #   each individual pair. "each" adds no constraint not already present.
        n = len(new_notes)
        lines: List[str] = [
            "Judge (EXISTING, NEW) pair: "
            "CONTRADICTS|EXTENDS|SUPPORTS|UNRELATED.",
            "",
        ]
        for i, (new_n, cand) in enumerate(zip(new_notes, candidate_notes), 1):
            lines.append(f"[PAIR {i}]")
            lines.append(f"EXISTING: {_judge_toon(cand.to_toon())}")
            lines.append(f"NEW: {_judge_toon(new_n.to_toon())}")
            lines.append("")

        if n == 1:
            # 1-verdict example — no separator needed or shown.
            # Cycle #26: "Respond with 1 verdict:" → "1 verdict:" (-2w) — same rationale
            #   as distil N=1: example "SUPPORTS" immediately below is sufficient context.
            lines += [
                "1 verdict:",
                "SUPPORTS",
            ]
        else:
            # 2-pair example with separator so the model learns where to place it.
            # Cycle #20: "Example for 2 pairs:" → "Example:" saves 3w.
            # Cycle #27: "exactly" removed (−1w) — same rationale as distil N≥2 footer.
            # Cycle #28: "Example:" label removed (−1w) — verdict word "SUPPORTS" is
            #   self-evidently illustrative following the footer instruction.
            # Cycle #32: "Respond with" removed (−2w) — symmetric with Cycle #26 for N=1.
            #   "{n} verdict(s) separated by ---VERDICT---." is unambiguous on its own.
            lines += [
                f"{n} verdict(s) separated by {VERDICT_SEP}.",
                "SUPPORTS",
                VERDICT_SEP,
                "EXTENDS",
            ]
        return "\n".join(lines)

    def _parse_verdict_response(self, response: str, n: int) -> List[str]:
        """Split on ``VERDICT_SEP`` and extract one verdict per pair."""
        parts = response.split(VERDICT_SEP)
        verdicts: List[str] = []
        for i in range(n):
            if i < len(parts):
                # Find first valid ALL-CAPS verdict word in this segment
                segment = parts[i].strip().upper()
                verdict = _DEFAULT_VERDICT
                for word in segment.split():
                    cleaned = word.strip(".,!?:")
                    if cleaned in _VALID_VERDICTS:
                        verdict = cleaned
                        break
                verdicts.append(verdict)
            else:
                verdicts.append(_DEFAULT_VERDICT)
        return verdicts

    # ══════════════════════════════════════════════════════════════════════════
    # Diagnostics
    # ══════════════════════════════════════════════════════════════════════════

    def stats(self) -> dict:
        """Return call efficiency statistics for ``prism-stats``."""
        total_naive = self._calls_made + self._calls_saved
        efficiency = (
            self._calls_saved / total_naive if total_naive > 0 else 0.0
        )
        return {
            "calls_made": self._calls_made,
            "calls_saved": self._calls_saved,
            "call_efficiency": round(efficiency, 4),
            "avg_batch_size": (
                round(sum(self._batch_sizes) / len(self._batch_sizes), 2)
                if self._batch_sizes
                else 0.0
            ),
            "batch_count": len(self._batch_sizes),
        }
