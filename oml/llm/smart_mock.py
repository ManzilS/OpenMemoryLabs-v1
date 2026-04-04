"""oml/llm/smart_mock.py — Prompt-aware mock LLM (zero API calls).

``SmartMockLLM`` detects the *type* of prompt and returns a structurally
correct response — valid TOON blocks, properly delimited verdict lists, or a
plausible-sounding QA answer — without touching any external API.

Use this as the default backend for:
  * Dev-time iteration (instant, $0 cost)
  * CI smoke tests that need realistic structured output
  * Pipeline wiring checks where answer quality doesn't matter

Prompt type detection (checked top-to-bottom):

  1. Distil batch  — contains ``"TOON memory encoder"`` + ``"[TEXT 1]"``
                     → N TOON blocks separated by ``---TOON---``
  2. Distil single — contains ``"TOON memory encoder"``
                     → 1 TOON block
  3. Evolve batch  — contains ``"---VERDICT---"`` + ``"[PAIR 1]"``
                     → N verdicts separated by ``---VERDICT---``
  4. Evolve single — contains ``"memory consistency judge"``
                     → ``RELATION: SUPPORTS\\nREASON: …``
  5. Generic QA    — anything else
                     → answer string referencing extracted keywords

Registered in ``oml/llm/factory.py`` as model string ``"smart-mock"``.
"""

from __future__ import annotations

import hashlib
import re
from typing import List

from oml.llm.base import BaseLLM

# ---------------------------------------------------------------------------
# Stop-word filter for keyword extraction
# ---------------------------------------------------------------------------
_STOP_WORDS = frozenset(
    "the a an is was to of in and or that it he she for on with his her at "
    "by be as from this which but are were had have has do does did not no "
    "so if then when where how what who all been will would could should may "
    "might its we they them their i me my you your".split()
)


def _extract_keywords(text: str, max_kw: int = 3) -> List[str]:
    """Return up to ``max_kw`` non-stop-words from ``text`` (lowercase)."""
    tokens = re.split(r"\W+", text.lower())
    seen: List[str] = []
    for tok in tokens:
        if len(tok) >= 3 and tok not in _STOP_WORDS and tok not in seen:
            seen.append(tok)
        if len(seen) >= max_kw:
            break
    return seen if seen else ["mock", "note", "test"]


def _short_id(text: str) -> str:
    """8-character hex ID deterministic for a given text snippet."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]


# ---------------------------------------------------------------------------
# TOON block builder
# ---------------------------------------------------------------------------
_TOON_TEMPLATE = """\
note_id: teeg-{note_id}
content: {content}
context: smart-mock
keywords: {keywords}
tags: mock
confidence: 0.85
active: True"""


def _build_toon(text_snippet: str) -> str:
    """Build a single TOON block from a raw text snippet."""
    # Truncate to ~25 words for the content field
    words = text_snippet.strip().split()
    content = " ".join(words[:25])
    if not content:
        content = "Smart Mock note with no content."
    keywords = "|".join(_extract_keywords(content))
    return _TOON_TEMPLATE.format(
        note_id=_short_id(content[:40]),
        content=content,
        keywords=keywords,
    )


# ---------------------------------------------------------------------------
# Prompt parsers
# ---------------------------------------------------------------------------

def _extract_text_blocks(prompt: str) -> List[str]:
    """Extract text bodies from ``[TEXT N]`` blocks in a distil-batch prompt."""
    # Split on [TEXT N] markers; first element is the preamble (discarded)
    parts = re.split(r"\[TEXT\s+\d+\]", prompt)
    blocks = []
    for part in parts[1:]:  # skip preamble
        # The text body ends at the next [TEXT …] which we already split on,
        # OR at the end of the prompt.  Strip surrounding whitespace.
        # Also strip the context hint line if present (starts with "Context:")
        lines = []
        for line in part.strip().splitlines():
            stripped = line.strip()
            if stripped.lower().startswith("context:") or not stripped:
                continue
            lines.append(stripped)
        blocks.append(" ".join(lines) if lines else part.strip()[:80])
    return blocks


def _count_pairs(prompt: str) -> int:
    """Count ``[PAIR N]`` groups in an evolve-batch prompt."""
    return len(re.findall(r"\[PAIR\s+\d+\]", prompt))


# ---------------------------------------------------------------------------
# SmartMockLLM
# ---------------------------------------------------------------------------

class SmartMockLLM(BaseLLM):
    """Prompt-aware mock — returns structurally correct responses with 0 API calls.

    All responses are deterministic for a given prompt (based on SHA-256 hashing
    of extracted content), so the same experiment input always yields the same
    output, making results reproducible without a real LLM.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, prompt: str) -> str:
        """Dispatch to the appropriate handler based on prompt type."""
        p = prompt  # convenience alias

        # 1. Distil batch  (contains TOON encoder directive AND numbered [TEXT N])
        # Matches both original "TOON memory encoder" and compressed "TOON encoder:"
        if ("TOON memory encoder" in p or "TOON encoder" in p) and "[TEXT 1]" in p:
            return self._handle_distil_batch(p)

        # 2. Distil single (contains TOON encoder directive but no [TEXT 1])
        # Also matches compressed "TOON note:" (TEEGPipeline) and "TOON note —" (batcher fallback)
        if "TOON memory encoder" in p or "TOON encoder" in p or "TOON note" in p:
            return self._handle_distil_single(p)

        # 3. Evolve batch  (contains ---VERDICT--- delimiter AND [PAIR 1])
        if "---VERDICT---" in p and "[PAIR 1]" in p:
            return self._handle_evolve_batch(p)

        # 4. Evolve single (contains the judge role header)
        if "memory consistency judge" in p:
            return self._handle_evolve_single(p)

        # 5. Faithfulness judge (contains the strict judge directive)
        if "strict faithfulness judge" in p:
            return self._handle_faithfulness_judge(p)

        # 6. Lost-in-middle needle extraction
        # Detection: "What is the secret code?" is unique to this task; no suffix check needed
        # (lim_extended uses "Answer only with the number." — no bare "Answer:" suffix)
        if "What is the secret code?" in p:
            return self._handle_lost_in_middle(p)

        # 7. TEEG memory query — extract answer from [TEEG MEMORY] block
        if "[TEEG MEMORY]" in p and "QUESTION:" in p:
            return self._handle_teeg_query(p)

        # 8. Generic QA
        return self._handle_generic(p)

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------

    def _handle_distil_single(self, prompt: str) -> str:
        """Return one TOON block for a single distil call."""
        # First: try "Text: ..." prefix — used by both compressed prompt formats
        for line in prompt.splitlines():
            stripped = line.strip()
            if stripped.lower().startswith("text:"):
                snippet = stripped[5:].strip()
                if len(snippet) > 5:
                    return _build_toon(snippet)
        # Fallback: last non-colon line > 20 chars (handles legacy "TOON memory encoder" format)
        lines = [ln.strip() for ln in prompt.splitlines() if ln.strip()]
        text_lines = [
            ln for ln in lines
            if len(ln) > 20 and ":" not in ln and not ln.isupper()
        ]
        snippet = " ".join(text_lines[-3:]) if text_lines else lines[-1] if lines else "mock"
        return _build_toon(snippet)

    def _handle_distil_batch(self, prompt: str) -> str:
        """Return N TOON blocks separated by ``---TOON---``."""
        blocks = _extract_text_blocks(prompt)
        if not blocks:
            # Fallback: return one generic block
            return _build_toon("smart mock batch item")
        toon_blocks = [_build_toon(b) for b in blocks]
        return "\n---TOON---\n".join(toon_blocks)

    def _handle_evolve_single(self, prompt: str) -> str:
        """Return a single RELATION/REASON judge response."""
        return "RELATION: SUPPORTS\nREASON: Smart Mock — notes are consistent."

    def _handle_evolve_batch(self, prompt: str) -> str:
        """Return N verdicts separated by ``---VERDICT---``."""
        n = _count_pairs(prompt)
        if n == 0:
            n = 1
        return "\n---VERDICT---\n".join(["SUPPORTS"] * n)

    def _handle_generic(self, prompt: str) -> str:
        """Return a generic QA-style answer referencing extracted keywords."""
        # Try to extract keywords from the prompt body
        # Strip any [TEEG MEMORY] or [DOCUMENT SUMMARY] tags first
        clean = re.sub(r"\[/?[A-Z ]+\]", " ", prompt)
        keywords = _extract_keywords(clean, max_kw=4)
        kw_str = ", ".join(keywords)
        return (
            f"Smart Mock answer: Based on the context, the answer involves {kw_str}. "
            f"[This is a deterministic mock response; use --model ollama:<name> "
            f"for realistic output.]"
        )

    def _handle_faithfulness_judge(self, prompt: str) -> str:
        """Return a faithfulness verdict by checking ANSWER word coverage in CONTEXT.

        Extracts the CONTEXT and ANSWER sections, then checks whether every
        content word (len > 3, not a stop word) in ANSWER appears in CONTEXT
        via whole-word boundary matching.  Returns ``VERDICT: YES`` when all
        answer content words are found in the context, ``VERDICT: NO`` otherwise.

        This gives correct verdicts for all three canonical faithfulness examples:
          1. "capital of France is Paris" in context "capital is Paris" → YES
          2. "Elon City" absent from Mars context                       → NO
          3. "wrote Hamlet" absent from "was a playwright" context      → NO
        """
        lines = prompt.splitlines()
        section: str = ""
        context_parts: List[str] = []
        answer_parts: List[str] = []

        for line in lines:
            stripped = line.strip()
            if stripped == "CONTEXT:":
                section = "context"
                continue
            if stripped in ("QUESTION:", "ANSWER:", "OUTPUT FORMAT:"):
                section = stripped.rstrip(":")
                continue
            if section == "context":
                context_parts.append(stripped)
            elif section == "ANSWER":
                answer_parts.append(stripped)

        context = " ".join(context_parts).lower()
        answer = " ".join(answer_parts).lower()

        # Extract content words: length > 3, not a stop word
        answer_words = [
            w for w in re.split(r"\W+", answer)
            if len(w) > 3 and w not in _STOP_WORDS
        ]

        if not answer_words:
            # Nothing substantive to check → assume supported
            verdict = "YES"
        else:
            # Every content word must appear as a whole word in the context
            verdict = "YES" if all(
                re.search(r"\b" + re.escape(w) + r"\b", context)
                for w in answer_words
            ) else "NO"

        support_str = "supports" if verdict == "YES" else "does not support"
        return (
            f"Smart Mock: the context {support_str} all answer claims. "
            f"VERDICT: {verdict}"
        )

    def _handle_lost_in_middle(self, prompt: str) -> str:
        """Extract and return the needle value from a lost-in-middle prompt.

        The ``LostInMiddleTask`` always embeds a needle of the form
        ``"The secret code is {VALUE}"`` somewhere in a long filler context,
        then queries ``"What is the secret code?"``.  This handler finds the
        needle via regex and returns the code verbatim regardless of needle
        position (start / middle / end) — scoring 3/3 = 1.000 at zero API cost.

        Detection: prompt contains ``"What is the secret code?"`` (bare check, no suffix needed;
        covers both ``LostInMiddleTask`` and the ``lim_extended`` script format).
        """
        match = re.search(r"secret\s+code\s+is\s+(\S+)", prompt, re.IGNORECASE)
        if match:
            code = match.group(1).rstrip(".,!?;:")
            return f"The secret code is {code}."
        return "Smart Mock: no secret code found in context."

    def _handle_teeg_query(self, prompt: str) -> str:
        """Return a memory-aware answer by extracting content from the TEEG MEMORY block.

        Finds the ``[TEEG MEMORY]`` / ``[/TEEG MEMORY]`` block in the prompt,
        extracts ``content:`` lines from the TOON-encoded notes inside, and
        returns a concise answer referencing that content.  Falls back to
        keywords from the QUESTION block if no content lines are found.

        Detection: prompt contains ``"[TEEG MEMORY]"`` + ``"QUESTION:"``.
        """
        # Extract the TEEG MEMORY block
        memory_match = re.search(
            r"\[TEEG MEMORY\](.*?)\[/TEEG MEMORY\]", prompt, re.DOTALL
        )
        content_lines: List[str] = []
        if memory_match:
            memory_block = memory_match.group(1)
            for line in memory_block.splitlines():
                stripped = line.strip()
                if stripped.startswith("content:"):
                    val = stripped[len("content:"):].strip()
                    if val and val != "(no relevant memory found)":
                        content_lines.append(val)

        if content_lines:
            # Use the first (most relevant) content line as the basis for the answer
            snippet = content_lines[0][:80]
            return f"Based on the memory, {snippet}"

        # Fallback: extract keywords from the QUESTION line
        question_match = re.search(r"QUESTION:\s*(.+)", prompt)
        if question_match:
            q_text = question_match.group(1).strip()
            kws = _extract_keywords(q_text, max_kw=3)
            return f"Smart Mock: based on memory, the answer involves {', '.join(kws)}."
        return "Smart Mock: no relevant memory found for this query."
