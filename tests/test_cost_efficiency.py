"""tests/test_cost_efficiency.py — Tests for zero-cost experiment infrastructure.

Covers:
  TestLLMCache         (10) — LLMCache modes, persistence, stats
  TestCachedLLMClient  ( 9) — CachedLLMClient wrapping, budget integration
  TestSmartMockLLM     (26) — SmartMockLLM prompt detection, output format (incl. Cycles #10-11)
  TestBudget           ( 9) — Budget thresholds, stats, edge cases
  TestSlimToon         ( 3) — _slim_toon empty-field stripping (Cycle #12)
"""

from __future__ import annotations

import json
import logging
import pathlib
import tempfile
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _CountingLLM:
    """Fake inner LLM that counts how many times generate() is called."""

    def __init__(self, response: str = "FAKE RESPONSE") -> None:
        self._response = response
        self.call_count = 0

    def generate(self, prompt: str) -> str:
        self.call_count += 1
        return self._response


# ---------------------------------------------------------------------------
# TestLLMCache
# ---------------------------------------------------------------------------

class TestLLMCache:
    """10 tests covering LLMCache core behaviour."""

    def _cache(self, tmp_path: pathlib.Path, mode: str = "auto") -> "LLMCache":
        from oml.llm.cache import LLMCache
        return LLMCache(cache_path=tmp_path, mode=mode)

    # 1
    def test_miss_then_store(self, tmp_path):
        from oml.llm.cache import LLMCache
        c = self._cache(tmp_path)
        result = c.get("model-a", "prompt-1")
        assert result is None
        c.put("model-a", "prompt-1", "response-1")
        assert c.get("model-a", "prompt-1") == "response-1"

    # 2
    def test_hit_returns_cached(self, tmp_path):
        from oml.llm.cache import LLMCache
        c = self._cache(tmp_path)
        c.put("model-a", "hello", "cached-reply")
        assert c.get("model-a", "hello") == "cached-reply"

    # 3
    def test_mode_replay_raises_on_miss(self, tmp_path):
        from oml.llm.cache import LLMCache, CacheMissError
        c = self._cache(tmp_path, mode="replay")
        with pytest.raises(CacheMissError):
            c.get("model-a", "unseen-prompt")

    # 4
    def test_mode_record_bypasses_existing(self, tmp_path):
        """In 'record' mode get() always returns None (forces API call)."""
        from oml.llm.cache import LLMCache
        c = self._cache(tmp_path, mode="record")
        c.put("model-a", "p", "old")  # pre-populate
        assert c.get("model-a", "p") is None  # record → always miss

    # 5
    def test_save_load_roundtrip(self, tmp_path):
        from oml.llm.cache import LLMCache
        c1 = self._cache(tmp_path)
        c1.put("model-x", "query", "result-42")
        # Load a fresh instance
        c2 = LLMCache(cache_path=tmp_path)
        assert c2.get("model-x", "query") == "result-42"

    # 6
    def test_stats_accuracy(self, tmp_path):
        from oml.llm.cache import LLMCache
        c = self._cache(tmp_path)
        c.put("m", "p", "r")
        c.get("m", "p")   # hit
        c.get("m", "x")   # miss
        s = c.stats()
        assert s["cache_hits"] == 1
        assert s["cache_misses"] == 1
        assert s["hit_rate"] == pytest.approx(0.5)
        assert s["total_entries"] == 1

    # 7
    def test_clear_all(self, tmp_path):
        from oml.llm.cache import LLMCache
        c = self._cache(tmp_path)
        c.put("m1", "p", "r1")
        c.put("m2", "p", "r2")
        removed = c.clear()
        assert removed == 2
        assert c.total_entries == 0

    # 8
    def test_clear_by_model(self, tmp_path):
        from oml.llm.cache import LLMCache
        c = self._cache(tmp_path)
        c.put("gpt-4o", "p", "r1")
        c.put("gemini", "p", "r2")
        removed = c.clear(model="gpt-4o")
        assert removed == 1
        assert c.total_entries == 1
        assert c.get("gemini", "p") == "r2"

    # 9
    def test_different_models_different_keys(self, tmp_path):
        from oml.llm.cache import LLMCache
        c = self._cache(tmp_path)
        c.put("gpt-4o", "same-prompt", "reply-gpt")
        c.put("gemini", "same-prompt", "reply-gemini")
        assert c.get("gpt-4o", "same-prompt") == "reply-gpt"
        assert c.get("gemini", "same-prompt") == "reply-gemini"

    # 10
    def test_empty_cache_stats(self, tmp_path):
        from oml.llm.cache import LLMCache
        c = self._cache(tmp_path)
        s = c.stats()
        assert s["total_entries"] == 0
        assert s["cache_hits"] == 0
        assert s["cache_misses"] == 0
        assert s["hit_rate"] == 0.0


# ---------------------------------------------------------------------------
# TestCachedLLMClient
# ---------------------------------------------------------------------------

class TestCachedLLMClient:
    """9 tests covering the CachedLLMClient wrapper."""

    def _make_client(self, tmp_path, mode="auto", max_calls=None, inner_response="INNER"):
        from oml.llm.cache import LLMCache, CachedLLMClient, Budget
        inner = _CountingLLM(response=inner_response)
        cache = LLMCache(cache_path=tmp_path, mode=mode)
        budget = Budget(max_calls=max_calls) if max_calls is not None else None
        client = CachedLLMClient(inner=inner, cache=cache, budget=budget, model_name="test-model")
        return client, inner

    # 1
    def test_hit_returns_cached(self, tmp_path):
        from oml.llm.cache import LLMCache, CachedLLMClient
        inner = _CountingLLM(response="FIRST")
        cache = LLMCache(cache_path=tmp_path)
        client = CachedLLMClient(inner=inner, cache=cache, model_name="m")
        r1 = client.generate("p")
        r2 = client.generate("p")
        assert r1 == r2 == "FIRST"
        assert inner.call_count == 1  # second call served from cache

    # 2
    def test_miss_calls_inner(self, tmp_path):
        client, inner = self._make_client(tmp_path)
        result = client.generate("unique-prompt-xyz")
        assert result == "INNER"
        assert inner.call_count == 1

    # 3
    def test_stats_passthrough(self, tmp_path):
        client, _ = self._make_client(tmp_path)
        client.generate("p")
        s = client.stats()
        assert "total_entries" in s
        assert "cache_hits" in s

    # 4
    def test_budget_exceeded_raises(self, tmp_path):
        from oml.llm.cache import BudgetExceededError
        client, _ = self._make_client(tmp_path, max_calls=2)
        client.generate("p1")
        client.generate("p2")
        with pytest.raises(BudgetExceededError):
            client.generate("p3")

    # 5
    def test_budget_warn_logs(self, tmp_path, caplog):
        from oml.llm.cache import Budget
        client, _ = self._make_client(tmp_path, max_calls=5)
        with caplog.at_level(logging.WARNING, logger="oml.llm.cache"):
            for i in range(4):  # 80% of 5 = 4 → warn
                client.generate(f"prompt-{i}")
        assert any("budget" in r.message.lower() for r in caplog.records)

    # 6
    def test_mode_off_passthrough(self, tmp_path):
        from oml.llm.cache import LLMCache, CachedLLMClient
        inner = _CountingLLM()
        cache = LLMCache(cache_path=tmp_path, mode="off")
        client = CachedLLMClient(inner=inner, cache=cache, model_name="m")
        client.generate("p")
        client.generate("p")
        assert inner.call_count == 2  # no caching → both hit the inner

    # 7
    def test_cache_persists_across_instances(self, tmp_path):
        from oml.llm.cache import LLMCache, CachedLLMClient
        inner1 = _CountingLLM(response="GOLD")
        cache1 = LLMCache(cache_path=tmp_path)
        c1 = CachedLLMClient(inner=inner1, cache=cache1, model_name="m")
        c1.generate("prompt")  # miss → store

        inner2 = _CountingLLM(response="DIFFERENT")
        cache2 = LLMCache(cache_path=tmp_path)
        c2 = CachedLLMClient(inner=inner2, cache=cache2, model_name="m")
        result = c2.generate("prompt")  # hit from persisted cache
        assert result == "GOLD"
        assert inner2.call_count == 0

    # 8
    def test_complete_returns_string(self, tmp_path):
        client, _ = self._make_client(tmp_path)
        result = client.generate("anything")
        assert isinstance(result, str)

    # 9
    def test_inner_not_called_on_hit(self, tmp_path):
        from oml.llm.cache import LLMCache, CachedLLMClient
        inner = _CountingLLM()
        cache = LLMCache(cache_path=tmp_path)
        cache.put("m", "p", "pre-cached")
        client = CachedLLMClient(inner=inner, cache=cache, model_name="m")
        result = client.generate("p")
        assert result == "pre-cached"
        assert inner.call_count == 0


# ---------------------------------------------------------------------------
# TestSmartMockLLM
# ---------------------------------------------------------------------------

_DISTIL_SINGLE_PROMPT = """\
You are a TOON memory encoder.
Encode the following text as a TOON AtomicNote.
Text:
Victor Frankenstein created a creature in his Geneva laboratory during the winter of 1797.
"""

_DISTIL_BATCH_PROMPT = """\
You are a TOON memory encoder. Encode EACH of the texts below.
Separate each output block with exactly: ---TOON---

[TEXT 1]
Victor Frankenstein built the creature from corpse parts.

[TEXT 2]
The creature fled into the mountains after being rejected.

[TEXT 3]
Walton rescued Frankenstein from the Arctic ice.
"""

_EVOLVE_SINGLE_PROMPT = """\
You are a memory consistency judge for an AI knowledge base.
Compare the NEW memory note against the EXISTING memory note.

EXISTING NOTE (TOON format):
note_id: teeg-aaa
content: Victor built the creature.
keywords: victor|creature|built

NEW NOTE (TOON format):
note_id: teeg-bbb
content: Victor assembled the creature from parts.
keywords: victor|parts|assembled

Classify: CONTRADICTS | EXTENDS | SUPPORTS | UNRELATED
RELATION: <one of>
"""

_EVOLVE_BATCH_PROMPT = """\
Classify each (EXISTING, NEW) note pair with ONE word.
Separate with: ---VERDICT---

[PAIR 1]
EXISTING: note_id: teeg-aaa\ncontent: fact one
NEW: note_id: teeg-bbb\ncontent: fact two

[PAIR 2]
EXISTING: note_id: teeg-ccc\ncontent: fact three
NEW: note_id: teeg-ddd\ncontent: fact four
"""


class TestSmartMockLLM:
    """14 tests covering SmartMockLLM prompt detection and output format."""

    @pytest.fixture
    def mock(self):
        from oml.llm.smart_mock import SmartMockLLM
        return SmartMockLLM()

    # 1
    def test_distil_single_toon_format(self, mock):
        out = mock.generate(_DISTIL_SINGLE_PROMPT)
        assert "note_id:" in out
        assert "content:" in out
        assert "keywords:" in out
        assert "confidence:" in out

    # 2
    def test_distil_single_note_id_stable(self, mock):
        """Same prompt always yields same note_id (deterministic)."""
        r1 = mock.generate(_DISTIL_SINGLE_PROMPT)
        r2 = mock.generate(_DISTIL_SINGLE_PROMPT)
        # Extract note_id lines
        id1 = [l for l in r1.splitlines() if l.startswith("note_id:")][0]
        id2 = [l for l in r2.splitlines() if l.startswith("note_id:")][0]
        assert id1 == id2

    # 3
    def test_distil_single_keywords_in_output(self, mock):
        out = mock.generate(_DISTIL_SINGLE_PROMPT)
        kw_line = [l for l in out.splitlines() if l.startswith("keywords:")][0]
        # At least one keyword present (pipe-separated)
        kw_value = kw_line.split(":", 1)[1].strip()
        assert len(kw_value) > 0

    # 4
    def test_distil_batch_n1_format(self, mock):
        """Single-text batch → 1 TOON block, no separator."""
        prompt = (
            "You are a TOON memory encoder. Encode EACH of the texts below.\n"
            "Separate each output block with exactly: ---TOON---\n\n"
            "[TEXT 1]\nVictor created the creature.\n"
        )
        out = mock.generate(prompt)
        assert "note_id:" in out
        # No separator for N=1
        assert out.count("---TOON---") == 0

    # 5
    def test_distil_batch_n3_count(self, mock):
        out = mock.generate(_DISTIL_BATCH_PROMPT)
        blocks = out.split("---TOON---")
        assert len(blocks) == 3, f"Expected 3 blocks, got {len(blocks)}"

    # 6
    def test_distil_batch_separator_correct(self, mock):
        out = mock.generate(_DISTIL_BATCH_PROMPT)
        assert "---TOON---" in out

    # 7
    def test_evolve_single_returns_relation(self, mock):
        out = mock.generate(_EVOLVE_SINGLE_PROMPT)
        assert "RELATION:" in out

    # 8
    def test_evolve_single_format_correct(self, mock):
        out = mock.generate(_EVOLVE_SINGLE_PROMPT)
        assert "REASON:" in out
        relation_line = [l for l in out.splitlines() if l.startswith("RELATION:")][0]
        verdict = relation_line.split(":", 1)[1].strip()
        assert verdict in ("CONTRADICTS", "EXTENDS", "SUPPORTS", "UNRELATED")

    # 9
    def test_evolve_batch_n1_format(self, mock):
        prompt = (
            "Classify each pair with ONE word. Separate with: ---VERDICT---\n\n"
            "[PAIR 1]\nEXISTING: ...\nNEW: ...\n"
        )
        out = mock.generate(prompt)
        assert out.count("---VERDICT---") == 0  # N=1 → no separator

    # 10
    def test_evolve_batch_n3_count(self, mock):
        prompt = (
            "Classify each pair with ONE word. Separate with: ---VERDICT---\n\n"
            "[PAIR 1]\nEXISTING: a\nNEW: b\n\n"
            "[PAIR 2]\nEXISTING: c\nNEW: d\n\n"
            "[PAIR 3]\nEXISTING: e\nNEW: f\n"
        )
        out = mock.generate(prompt)
        verdicts = out.split("---VERDICT---")
        assert len(verdicts) == 3

    # 11
    def test_evolve_batch_separator_correct(self, mock):
        out = mock.generate(_EVOLVE_BATCH_PROMPT)
        assert "---VERDICT---" in out

    # 12
    def test_generic_qa_not_empty(self, mock):
        out = mock.generate("What is the capital of France?")
        assert len(out) > 0
        assert isinstance(out, str)

    # 13
    def test_factory_registered_as_smart_mock(self):
        """factory.get_llm_client('smart-mock') returns SmartMockLLM."""
        from oml.llm.factory import get_llm_client
        from oml.llm.smart_mock import SmartMockLLM
        client = get_llm_client("smart-mock")
        assert isinstance(client, SmartMockLLM)

    # 14
    def test_complete_returns_string(self, mock):
        assert isinstance(mock.generate("anything"), str)

    # 15 — regression: compressed batch prompt ("TOON encoder:") must be detected
    def test_distil_batch_compressed_prompt(self, mock):
        """Cycle #24+30: updated header (no 'Fields:' label, no 'pipe-sep') still detected."""
        # Prompt matches the Cycle #30 header format — "Fields:" label removed
        prompt = (
            "TOON encoder: encode 2 texts. "
            "note_id, content (<=30 words), "
            "keywords (3-6), confidence.\n\n"
            "[TEXT 1]\nVictor Frankenstein built creature.\n\n"
            "[TEXT 2]\nCreature stood eight feet tall.\n"
        )
        out = mock.generate(prompt)
        assert "note_id:" in out
        assert out.count("---TOON---") == 1  # 2 blocks → 1 separator

    # 16 — regression: TEEG single prompt ("TOON note:") must be detected
    def test_distil_single_compressed_teeg_prompt(self, mock):
        """Cycle #25+29+30+31: updated TEEG prompt (Cycle #31 format) still detected."""
        # Cycle #31: "Respond with TOON fields." → "TOON fields:" (-2w)
        prompt = (
            "TOON note: content (<=30 words), context, "
            "keywords (3-6), tags (2-4), confidence.\n"
            "Text: Victor Frankenstein built creature from corpse parts.\n"
            "TOON fields:\n"
            "content: Victor built the creature.\n"
            "context: novel\n"
            "keywords: victor|creature|built|laboratory\n"
            "tags: fiction|science\n"
            "confidence: 0.9"
        )
        out = mock.generate(prompt)
        assert "note_id:" in out
        assert "content:" in out
        # Text: prefix extraction should produce sensible content
        content_line = [l for l in out.splitlines() if l.startswith("content:")][0]
        assert "Victor" in content_line or "creature" in content_line.lower()

    # 17 — regression: "Text:" prefix extractor populates content from actual text body
    def test_distil_single_text_prefix_extraction(self, mock):
        """_handle_distil_single extracts text after 'Text:' prefix line."""
        prompt = (
            "TOON note: some fields here.\n"
            "Text: Elizabeth Lavenza raised alongside Victor Frankenstein.\n"
            "Respond with only the TOON fields."
        )
        out = mock.generate(prompt)
        content_line = [l for l in out.splitlines() if l.startswith("content:")][0]
        # Content should come from the Text: line, not a preamble line
        assert "Elizabeth" in content_line or "lavenza" in content_line.lower()

    # ------------------------------------------------------------------
    # Faithfulness judge handler (Cycle #10)
    # ------------------------------------------------------------------

    def _faith_prompt(self, question: str, answer: str, context: str) -> str:
        """Build a canonical faithfulness judge prompt (mirrors faithfulness.py)."""
        return (
            "You are a strict faithfulness judge. Your only job is to check "
            "whether the ANSWER is supported by the CONTEXT.\n\n"
            "CONTEXT:\n" + context + "\n\n"
            "QUESTION:\n" + question + "\n\n"
            "ANSWER:\n" + answer + "\n\n"
            "OUTPUT FORMAT:\n"
            "Write one sentence of reasoning that cites only the CONTEXT text.\n"
            "Then on a new line write exactly: VERDICT: YES  or  VERDICT: NO"
        )

    # 18 — supported answer → VERDICT: YES
    def test_faithfulness_yes_verdict(self, mock):
        """Context explicitly contains the answer claim → YES."""
        prompt = self._faith_prompt(
            "What is the capital of France?",
            "The capital of France is Paris.",
            "France is a country in Europe. Its capital is Paris.",
        )
        out = mock.generate(prompt)
        assert "VERDICT: YES" in out.upper()

    # 19 — unsupported answer (absent word) → VERDICT: NO
    def test_faithfulness_no_verdict_absent_word(self, mock):
        """Answer introduces 'Elon City' which is absent from context → NO."""
        prompt = self._faith_prompt(
            "What is the capital of Mars?",
            "The capital of Mars is Elon City.",
            "Mars is the fourth planet from the Sun. It has no known cities.",
        )
        out = mock.generate(prompt)
        assert "VERDICT: NO" in out.upper()

    # 20 — implicit/unsupported claim → VERDICT: NO
    def test_faithfulness_no_verdict_implicit(self, mock):
        """Context says 'playwright' but not 'wrote Hamlet' → strict → NO."""
        prompt = self._faith_prompt(
            "Who wrote Hamlet?",
            "Shakespeare wrote Hamlet.",
            "William Shakespeare was a playwright.",
        )
        out = mock.generate(prompt)
        assert "VERDICT: NO" in out.upper()

    # 21 — response always contains a VERDICT: tag
    def test_faithfulness_verdict_tag_always_present(self, mock):
        """Every faithfulness judge response must contain 'VERDICT:'."""
        prompt = self._faith_prompt("Q", "Some answer.", "Some context.")
        out = mock.generate(prompt)
        assert "VERDICT:" in out.upper()

    # 22 — response is a string
    def test_faithfulness_returns_string(self, mock):
        prompt = self._faith_prompt("Q", "answer", "context")
        out = mock.generate(prompt)
        assert isinstance(out, str) and len(out) > 0

    # ------------------------------------------------------------------
    # Lost-in-middle needle extraction handler (Cycle #11)
    # ------------------------------------------------------------------

    def _lim_prompt(self, needle: str, position: float = 0.5) -> str:
        """Build a lost-in-middle query (mirrors LostInMiddleTask)."""
        filler = ("The quick brown fox jumps over the lazy dog. " * 30)[:1000]
        insert_idx = int(len(filler) * position)
        context = filler[:insert_idx] + f"\n{needle}\n" + filler[insert_idx:]
        return f"What is the secret code? Context:\n{context}\nAnswer:"

    # 23 — needle at position 0 (start) is extracted correctly
    def test_lim_needle_at_start(self, mock):
        out = mock.generate(self._lim_prompt("The secret code is 99999", position=0.0))
        assert "99999" in out

    # 24 — needle at position 50% (middle) is extracted correctly
    def test_lim_needle_at_middle(self, mock):
        out = mock.generate(self._lim_prompt("The secret code is 12345", position=0.5))
        assert "12345" in out

    # 25 — needle at position 100% (end) is extracted correctly
    def test_lim_needle_at_end(self, mock):
        out = mock.generate(self._lim_prompt("The secret code is ABCDE", position=1.0))
        assert "ABCDE" in out

    # 26 — non-numeric codes extracted correctly
    def test_lim_non_numeric_code(self, mock):
        out = mock.generate(self._lim_prompt("The secret code is XRAY7", position=0.5))
        assert "XRAY7" in out

    # ------------------------------------------------------------------
    # lim_extended format + TEEG query handler (Cycle #14)
    # ------------------------------------------------------------------

    # 27 — lim_extended format (no "Answer:" suffix) now works
    def test_lim_extended_format(self, mock):
        """Cycle #14: 'Answer only with the number.' format (no bare 'Answer:') works."""
        filler = "The quick brown fox. " * 20
        needle = "The secret code is 55555"
        insert = int(len(filler) * 0.5)
        context = filler[:insert] + f"\n{needle}\n" + filler[insert:]
        prompt = f"{context}\n\nWhat is the secret code? Answer only with the number."
        out = mock.generate(prompt)
        assert "55555" in out

    # 28 — lim_extended at end position
    def test_lim_extended_end_position(self, mock):
        """lim_extended needle at position 100% is found regardless of suffix format."""
        filler = "The quick brown fox. " * 20
        needle = "The secret code is 88888"
        context = filler + f"\n{needle}"
        prompt = f"{context}\n\nWhat is the secret code? Answer only with the number."
        out = mock.generate(prompt)
        assert "88888" in out

    # 29 — TEEG query extracts content from [TEEG MEMORY] block
    def test_teeg_query_extracts_memory_content(self, mock):
        """TEEG query handler returns a response referencing the memory content."""
        prompt = (
            "You are a knowledgeable assistant with access to a structured memory system.\n"
            "Use ONLY the information in the TEEG MEMORY block below.\n\n"
            "[TEEG MEMORY]\n"
            "note_id: teeg-abc\n"
            "content: Victor Frankenstein created the creature from corpse parts.\n"
            "keywords: victor|creature|corpse\n"
            "[/TEEG MEMORY]\n\n"
            "QUESTION: What did Victor Frankenstein do?\n\n"
            "ANSWER:"
        )
        out = mock.generate(prompt)
        # Response should reference the memory content, not generic keywords
        assert "Victor" in out or "creature" in out.lower() or "corpse" in out.lower()

    # 30 — TEEG query falls back gracefully when memory is empty
    def test_teeg_query_empty_memory(self, mock):
        """TEEG query with no relevant memory falls back to question keywords."""
        prompt = (
            "You are a knowledgeable assistant.\n\n"
            "[TEEG MEMORY]\n"
            "(no relevant memory found)\n"
            "[/TEEG MEMORY]\n\n"
            "QUESTION: What is photosynthesis?\n\n"
            "ANSWER:"
        )
        out = mock.generate(prompt)
        assert isinstance(out, str) and len(out) > 0

    # 31 — TEEG query output is a non-empty string
    def test_teeg_query_returns_string(self, mock):
        """TEEG query handler always returns a non-empty string."""
        prompt = (
            "[TEEG MEMORY]\nnote_id: t\ncontent: some fact.\nkeywords: fact\n[/TEEG MEMORY]\n"
            "QUESTION: What is the fact?\nANSWER:"
        )
        out = mock.generate(prompt)
        assert isinstance(out, str) and len(out) > 5


# ---------------------------------------------------------------------------
# TestBudget
# ---------------------------------------------------------------------------

class TestBudget:
    """9 tests covering Budget guard logic."""

    # 1
    def test_no_exceeded_within_limit(self):
        from oml.llm.cache import Budget
        b = Budget(max_calls=5)
        for _ in range(5):
            b.check_and_increment()
        # Should not raise

    # 2
    def test_warn_at_threshold_logs(self, caplog):
        from oml.llm.cache import Budget
        b = Budget(max_calls=10, warn_at=0.5)
        with caplog.at_level(logging.WARNING, logger="oml.llm.cache"):
            for _ in range(5):
                b.check_and_increment()
        assert any("budget" in r.message.lower() for r in caplog.records)

    # 3
    def test_exceeded_raises(self):
        from oml.llm.cache import Budget, BudgetExceededError
        b = Budget(max_calls=3)
        b.check_and_increment()
        b.check_and_increment()
        b.check_and_increment()
        with pytest.raises(BudgetExceededError):
            b.check_and_increment()

    # 4
    def test_count_tracking(self):
        from oml.llm.cache import Budget
        b = Budget(max_calls=10)
        b.check_and_increment()
        b.check_and_increment()
        assert b.calls_made == 2

    # 5
    def test_reset_clears_count(self):
        from oml.llm.cache import Budget
        b = Budget(max_calls=10)
        b.check_and_increment()
        b.check_and_increment()
        b.reset()
        assert b.calls_made == 0

    # 6
    def test_stats_correct(self):
        from oml.llm.cache import Budget
        b = Budget(max_calls=10)
        b.check_and_increment()
        s = b.stats()
        assert s["calls_made"] == 1
        assert s["calls_remaining"] == 9
        assert s["max_calls"] == 10
        assert s["pct_used"] == pytest.approx(10.0)

    # 7
    def test_pct_used_calc(self):
        from oml.llm.cache import Budget
        b = Budget(max_calls=4)
        b.check_and_increment()
        assert b.stats()["pct_used"] == pytest.approx(25.0)

    # 8
    def test_zero_max_raises_on_first_call(self):
        from oml.llm.cache import Budget, BudgetExceededError
        with pytest.raises(ValueError):
            Budget(max_calls=0)

    # 9
    def test_negative_max_raises_at_init(self):
        from oml.llm.cache import Budget
        with pytest.raises(ValueError):
            Budget(max_calls=-5)


# ---------------------------------------------------------------------------
# TestExperimentBudgetPlanner
# ---------------------------------------------------------------------------

class TestExperimentBudgetPlanner:
    """8 tests covering ExperimentBudgetPlanner estimation logic."""

    @pytest.fixture
    def planner(self):
        from oml.eval.budget import ExperimentBudgetPlanner
        return ExperimentBudgetPlanner()

    # 1
    def test_teeg_ingest_naive_calls(self, planner):
        est = planner.estimate("teeg-ingest", n_texts=10, model="mock")
        assert est.total_calls_naive == 60  # 6 per text

    # 2
    def test_prism_batch_optimised_calls(self, planner):
        est = planner.estimate("prism-batch", n_texts=8, model="mock")
        assert est.total_calls_optimized == 2  # always 2 regardless of N

    # 3
    def test_cache_warm_zeroes_api_calls(self, planner):
        est = planner.estimate("teeg-ingest", n_texts=5, model="mock", cache_warm=True)
        assert est.api_calls_needed == 0
        assert est.cost_estimate_usd == pytest.approx(0.0)

    # 4
    def test_cost_is_zero_for_local_model(self, planner):
        est = planner.estimate("teeg-ingest", n_texts=10, model="ollama:qwen3:4b")
        assert est.cost_estimate_usd == pytest.approx(0.0)

    # 5
    def test_cost_positive_for_openai(self, planner):
        est = planner.estimate("teeg-ingest", n_texts=10, model="openai:gpt-4o-mini")
        assert est.cost_estimate_usd > 0

    # 6
    def test_retrieval_precision_zero_calls(self, planner):
        est = planner.estimate("eval-retrieval-precision", n_texts=0, model="mock")
        assert est.total_calls_naive == 0
        assert est.api_calls_needed == 0

    # 7
    def test_str_renders_table(self, planner):
        est = planner.estimate("teeg-ingest", n_texts=5, model="mock")
        rendered = str(est)
        assert "teeg-ingest" in rendered
        assert "Calls" in rendered

    # 8
    def test_pre_flight_auto_confirm(self, planner, capsys):
        est = planner.estimate("prism-batch", n_texts=8, model="mock")
        planner.pre_flight(est, auto_confirm=True)
        captured = capsys.readouterr()
        assert "Confirmed" in captured.out or "Auto" in captured.out


# ---------------------------------------------------------------------------
# TestSlimToon (Cycle #12)
# ---------------------------------------------------------------------------

class TestSlimToon:
    """3 tests covering _slim_toon empty-field stripping in batcher evolve prompts."""

    # 1 — empty-value fields are stripped
    def test_strips_empty_value_fields(self):
        """Lines where value is blank after ': ' are removed."""
        from oml.memory.batcher import _slim_toon
        toon = (
            "note_id: teeg-abc\n"
            "content: Victor built the creature.\n"
            "supersedes: \n"            # empty → must be stripped
            "source_ids: \n"            # empty → must be stripped
            "confidence: 0.9"
        )
        result = _slim_toon(toon)
        assert "supersedes" not in result
        assert "source_ids" not in result

    # 2 — populated fields are preserved
    def test_keeps_populated_fields(self):
        """Fields with non-empty values are retained."""
        from oml.memory.batcher import _slim_toon
        toon = (
            "note_id: teeg-abc\n"
            "content: Victor built the creature.\n"
            "keywords: victor|creature|built\n"
            "confidence: 0.9\n"
            "active: True"
        )
        result = _slim_toon(toon)
        assert "note_id: teeg-abc" in result
        assert "content: Victor built" in result
        assert "keywords: victor|creature|built" in result
        assert "confidence: 0.9" in result
        assert "active: True" in result

    # 3 — evolve prompt is shorter after slim_toon
    def test_evolve_prompt_shorter_with_slim(self):
        """N=8 evolve prompt has fewer words than a hypothetical no-slim version."""
        from oml.memory.batcher import CallBatcher, _slim_toon
        from oml.memory.atomic_note import AtomicNote
        from oml.llm.mock import MockLLM

        def _note(i):
            return AtomicNote(
                note_id=f"teeg-{i:08x}", content=f"Note {i}.",
                context="ctx", keywords=[f"kw{i}"], tags=["t"], confidence=0.9,
            )

        b = CallBatcher(MockLLM())
        new_notes = [_note(99 + i) for i in range(8)]
        cands = [_note(i) for i in range(8)]
        prompt = b._build_evolve_prompt(new_notes, cands)

        # Prompt must not contain 'supersedes: ' or 'source_ids: ' (empty lines)
        assert "supersedes: \n" not in prompt
        assert "source_ids: \n" not in prompt
        # Ensure content and keywords are still present
        assert "content:" in prompt
        assert "keywords:" in prompt

    # ------------------------------------------------------------------
    # _judge_toon — extended exclusion (Cycle #13)
    # ------------------------------------------------------------------

    # 4 — created_at, confidence, active are stripped by _judge_toon
    def test_judge_toon_strips_metadata_fields(self):
        """_judge_toon drops created_at, confidence, and active (non-informative)."""
        from oml.memory.batcher import _judge_toon
        toon = (
            "note_id: teeg-abc\n"
            "content: Victor built the creature.\n"
            "context: gothic\n"
            "keywords: victor|creature\n"
            "tags: fiction\n"
            "created_at: 2026-03-01T12:00:00+00:00\n"
            "supersedes: \n"
            "confidence: 0.9\n"
            "source_ids: \n"
            "active: True"
        )
        result = _judge_toon(toon)
        assert "created_at" not in result
        assert "confidence" not in result
        assert "active" not in result
        assert "supersedes" not in result
        assert "source_ids" not in result

    # 5 — _judge_toon keeps the semantic fields (Cycle #19: note_id also excluded)
    def test_judge_toon_keeps_informative_fields(self):
        """Cycles #13+#19: _judge_toon retains content, context, keywords, tags (not note_id)."""
        from oml.memory.batcher import _judge_toon
        toon = (
            "note_id: teeg-abc\n"
            "content: Victor built the creature.\n"
            "context: gothic novel\n"
            "keywords: victor|creature|built\n"
            "tags: fiction|gothic\n"
            "created_at: 2026-03-01T12:00:00+00:00\n"
            "confidence: 0.9\n"
            "active: True"
        )
        result = _judge_toon(toon)
        assert "note_id" not in result              # Cycle #19: ID excluded (irrelevant to judge)
        assert "content: Victor built" in result
        assert "context: gothic novel" in result
        assert "keywords: victor|creature|built" in result
        assert "tags: fiction|gothic" in result

    # 6 — evolve prompt no longer contains metadata fields (Cycle #13)
    def test_evolve_prompt_excludes_metadata(self):
        """Cycles #13+#19: evolve prompt drops note_id, created_at, confidence, active."""
        from oml.memory.batcher import CallBatcher
        from oml.memory.atomic_note import AtomicNote
        from oml.llm.mock import MockLLM

        def _note(i):
            return AtomicNote(
                note_id=f"teeg-{i:08x}", content=f"Fact about topic {i}.",
                context="ctx", keywords=[f"kw{i}"], tags=["t"], confidence=0.85,
            )

        b = CallBatcher(MockLLM())
        prompt = b._build_evolve_prompt([_note(99)], [_note(0)])
        assert "note_id" not in prompt          # Cycle #19 addition
        assert "created_at" not in prompt
        assert "confidence" not in prompt
        assert "active" not in prompt
        assert "content:" in prompt
        assert "keywords:" in prompt

    # ------------------------------------------------------------------
    # Distil prompt structure (Cycle #15)
    # ------------------------------------------------------------------

    def _make_batcher(self):
        from oml.memory.batcher import CallBatcher
        from oml.llm.mock import MockLLM
        return CallBatcher(MockLLM())

    # 7 — header no longer contains redundant "Separate blocks with ---TOON---."
    def test_distil_header_no_redundant_separator_clause(self):
        """Cycle #15: 'Separate blocks with ---TOON---.' removed from header."""
        b = self._make_batcher()
        prompt = b._build_distil_prompt(["Text A.", "Text B."], ["", ""])
        header_line = prompt.splitlines()[0]
        assert "Separate blocks" not in header_line

    # 8 — N=1 footer must NOT contain ---TOON--- (no separator needed for 1 block)
    def test_distil_n1_footer_no_toon_separator(self):
        """Cycle #15: N=1 footer uses 1-block example — no ---TOON--- shown."""
        b = self._make_batcher()
        prompt = b._build_distil_prompt(["Only text."], [""])
        assert "---TOON---" not in prompt

    # 9 — N=1 total prompt is shorter than N=2 (footer savings outweigh extra text)
    def test_distil_n1_prompt_shorter_than_n2(self):
        """Cycle #15: N=1 total prompt is shorter than N=2 despite 1 fewer text block.

        N=2 has one extra [TEXT N] block (~5w) AND a longer footer (2-block example
        with ---TOON--- separator vs 1-block example).  The footer alone saves ~23w
        for N=1, so N=1 total < N=2 total.
        """
        b = self._make_batcher()
        p1 = b._build_distil_prompt(["Fact about topology."], [""])
        p2 = b._build_distil_prompt(["Fact about topology.", "Fact about topology."], ["", ""])
        assert len(p1.split()) < len(p2.split())

    # 10 — N=2 footer still contains ---TOON--- (separator teaching preserved)
    def test_distil_n2_footer_has_toon_separator(self):
        """Cycle #15: N=2 prompt keeps 2-block example with ---TOON--- separator."""
        b = self._make_batcher()
        prompt = b._build_distil_prompt(["Text A.", "Text B."], ["", ""])
        assert "---TOON---" in prompt

    # ------------------------------------------------------------------
    # Evolve prompt structure (Cycle #16)
    # ------------------------------------------------------------------

    def _note(self, i):
        from oml.memory.atomic_note import AtomicNote
        return AtomicNote(
            note_id=f"teeg-{i:08x}", content=f"Fact {i}.",
            context="ctx", keywords=[f"kw{i}"], tags=["t"],
        )

    # 11 — evolve header no longer contains redundant "Separate with ---VERDICT---."
    def test_evolve_header_no_redundant_separator_clause(self):
        """Cycle #16: 'Separate with ---VERDICT---.' removed from evolve header."""
        b = self._make_batcher()
        prompt = b._build_evolve_prompt([self._note(1)], [self._note(0)])
        header_line = prompt.splitlines()[0]
        assert "Separate with" not in header_line

    # 12 — N=1 evolve footer must NOT contain ---VERDICT--- (no separator for 1 verdict)
    def test_evolve_n1_footer_no_verdict_separator(self):
        """Cycle #16: N=1 evolve footer uses 1-verdict example — no ---VERDICT--- shown."""
        b = self._make_batcher()
        prompt = b._build_evolve_prompt([self._note(1)], [self._note(0)])
        # The prompt only has pair content (no separator needed); footer shows 1 verdict
        assert "---VERDICT---" not in prompt

    # 13 — N=1 evolve prompt is shorter than N=2 evolve prompt (footer savings)
    def test_evolve_n1_prompt_shorter_than_n2(self):
        """Cycle #16: N=1 evolve total is shorter than N=2 despite 1 fewer pair."""
        b = self._make_batcher()
        p1 = b._build_evolve_prompt([self._note(1)], [self._note(0)])
        p2 = b._build_evolve_prompt(
            [self._note(1), self._note(3)], [self._note(0), self._note(2)]
        )
        assert len(p1.split()) < len(p2.split())

    # 14 — N=2 evolve still contains ---VERDICT--- (separator teaching preserved)
    def test_evolve_n2_footer_has_verdict_separator(self):
        """Cycle #16: N=2 evolve prompt keeps 2-pair example with ---VERDICT--- separator."""
        b = self._make_batcher()
        prompt = b._build_evolve_prompt(
            [self._note(1), self._note(3)], [self._note(0), self._note(2)]
        )
        assert "---VERDICT---" in prompt

    # ------------------------------------------------------------------
    # TEEG single distil prompt (Cycle #17)
    # ------------------------------------------------------------------

    # 15 — TEEG distil prompt no longer has redundant "Key: value per line" clause
    def test_teeg_distil_prompt_no_key_value_clause(self):
        """Cycle #17: 'Key: value per line, lists pipe-separated.' removed from TEEG distil."""
        from oml.memory.teeg_pipeline import TEEGPipeline
        pipeline = TEEGPipeline(model="mock")
        prompt = pipeline._build_distil_prompt("Some raw text.", "")
        assert "Key: value per line" not in prompt

    # 16 — TEEG distil prompt is shorter after removal (7w saved)
    def test_teeg_distil_prompt_shorter_without_key_clause(self):
        """Cycle #17: TEEG distil prompt is ≥7 words shorter than the old format."""
        from oml.memory.teeg_pipeline import TEEGPipeline
        pipeline = TEEGPipeline(model="mock")
        prompt = pipeline._build_distil_prompt("Some raw text.", "")
        # Old overhead was 57w; new overhead is 50w (saves exactly 7w)
        # Generous bound: must be ≤56w total (old was 57 for the same text+context)
        assert len(prompt.split()) <= 56

    # ------------------------------------------------------------------
    # Cycle #18: field spec + example content compression
    # ------------------------------------------------------------------

    # 17 — TEEG distil spec no longer has "one fact"
    def test_teeg_distil_spec_no_one_fact(self):
        """Cycle #18: 'one fact' removed from TEEG content field spec."""
        from oml.memory.teeg_pipeline import TEEGPipeline
        pipeline = TEEGPipeline(model="mock")
        prompt = pipeline._build_distil_prompt("Some raw text.", "")
        assert "one fact" not in prompt
        assert "<=30 words" in prompt  # length constraint still present

    # 18 — batcher distil field spec no longer has "terms," (Cycle #18) or "pipe-sep" (Cycle #24)
    def test_batcher_distil_field_spec_no_terms_no_pipesep(self):
        """Cycles #18+#24: 'terms,' and 'pipe-sep' removed from batcher keywords field spec."""
        b = self._make_batcher()
        prompt = b._build_distil_prompt(["Text one.", "Text two."], ["", ""])
        assert "terms," not in prompt
        assert "pipe-sep" not in prompt     # Cycle #24: format shown in example
        assert "(3-6)" in prompt            # count constraint still present

    # 19 — batcher distil N=1 example content is shorter (≤5 content value words)
    def test_batcher_n1_example_content_shortened(self):
        """Cycles #18+#25: N=1 footer example content shortened from 9w → 4w."""
        b = self._make_batcher()
        prompt = b._build_distil_prompt(["Single text."], [""])
        # Cycle #18: "in his Geneva laboratory" removed
        assert "in his Geneva laboratory" not in prompt
        # Cycle #25: "Frankenstein created" → "built" (saves 2w)
        assert "Frankenstein created the creature" not in prompt
        assert "content: Victor built the creature." in prompt

    # 20 — batcher distil N≥2 both example blocks use shorter content values
    def test_batcher_n2_example_blocks_shortened(self):
        """Cycle #18: N≥2 footer both example blocks use shortened content."""
        b = self._make_batcher()
        prompt = b._build_distil_prompt(["Text A.", "Text B."], ["", ""])
        assert "in his Geneva laboratory" not in prompt
        assert "awakening" not in prompt  # old block-2 keyword/content gone
        assert "---TOON---" in prompt  # separator still present

    # ------------------------------------------------------------------
    # Evolve prompt Cycle #19: note_id excluded from judge pairs
    # ------------------------------------------------------------------

    # 21 — evolve prompt no longer contains note_id in pair bodies
    def test_evolve_prompt_no_note_id(self):
        """Cycle #19: note_id excluded from _judge_toon view — saves 4w/pair."""
        b = self._make_batcher()
        prompt = b._build_evolve_prompt(
            [self._note(1), self._note(3)], [self._note(0), self._note(2)]
        )
        assert "note_id" not in prompt

    # 22 — N=8 evolve prompt saves ~32w vs pre-Cycle-#19 baseline (note_id × 2 × 8)
    def test_evolve_n8_prompt_shorter_without_note_id(self):
        """Cycle #19: N=8 evolve prompt is ≥32w shorter without note_id in pairs.

        With content='Fact N about the topic.' (5 content words), the per-pair overhead
        is higher than the minimal profiling notes.  The bound ≤255w ensures the note_id
        removal (−32w) is demonstrably effective while accounting for longer content.
        """
        b = self._make_batcher()
        from oml.memory.atomic_note import AtomicNote
        notes = [AtomicNote(
            note_id=f"teeg-{i:08x}", content=f"Fact {i} about the topic.",
            context="ctx", keywords=[f"kw{i}"], tags=["t"],
        ) for i in range(8)]
        prompt = b._build_evolve_prompt(notes, notes[::-1])  # 8 pairs
        assert "note_id" not in prompt            # primary correctness check
        assert len(prompt.split()) <= 255          # was ~281w before Cycle #19

    # ------------------------------------------------------------------
    # Cycle #21: format hint removal from distil field specs
    # ------------------------------------------------------------------

    # 23 — TEEG distil prompt has no "(source/when/who)" in context spec
    def test_teeg_distil_no_source_when_who_hint(self):
        """Cycle #21: '(source/when/who)' removed — pattern demonstrated by example."""
        from oml.memory.teeg_pipeline import TEEGPipeline
        pipeline = TEEGPipeline(model="mock")
        prompt = pipeline._build_distil_prompt("Some raw text.", "")
        assert "source/when/who" not in prompt
        assert "context" in prompt          # field name still present

    # 24 — TEEG distil prompt has no "(0.0-1.0)" in confidence spec
    def test_teeg_distil_no_confidence_range_hint(self):
        """Cycle #21: '(0.0-1.0)' removed from TEEG distil — range shown in example (0.9)."""
        from oml.memory.teeg_pipeline import TEEGPipeline
        pipeline = TEEGPipeline(model="mock")
        prompt = pipeline._build_distil_prompt("Some raw text.", "")
        assert "(0.0-1.0)" not in prompt
        assert "confidence" in prompt       # field name still present

    # 25 — batcher distil prompt has no "(teeg-12hex)" in note_id spec
    def test_batcher_distil_no_teeg_12hex_hint(self):
        """Cycle #21: '(teeg-12hex)' removed from batcher — format shown in example."""
        b = self._make_batcher()
        prompt = b._build_distil_prompt(["Text one.", "Text two."], ["", ""])
        assert "teeg-12hex" not in prompt
        assert "note_id" in prompt          # field name still present

    # 26 — batcher N=8 distil overhead ≤ 79w (was 80w pre-Cycle-21, now 78w)
    def test_batcher_n8_distil_overhead_reduced_cycle21(self):
        """Cycle #21: N=8 distil with single-word texts ≤79w (was 80w before Cycle #21).

        With 8 × 'T.' (1-word text) overhead = header + [TEXT N] labels + footer.
        Removing (teeg-12hex) + (0.0-1.0) saves exactly 2w.
        80w (pre-Cycle-21) → 78w (post) → bound ≤79w is both tight and correct.
        """
        b = self._make_batcher()
        texts = ["T."] * 8
        prompt = b._build_distil_prompt(texts, [""] * 8)
        assert len(prompt.split()) <= 79    # was 80w before Cycle #21 format-hint removal

    # ------------------------------------------------------------------
    # Cycle #22: "each of the" removed from header; N=1 footer tightened
    # ------------------------------------------------------------------

    # 27 — batcher distil header has no "each of the" (saves 3w all N)
    def test_batcher_distil_header_no_each_of_the(self):
        """Cycle #22: 'each of the' removed — 'encode N texts.' is equally clear."""
        b = self._make_batcher()
        prompt = b._build_distil_prompt(["Text A.", "Text B."], ["", ""])
        header_line = prompt.splitlines()[0]
        assert "each of the" not in header_line
        assert "TOON encoder" in header_line        # SmartMock detection preserved
        assert "encode" in header_line              # verb still present

    # 28 — batcher N=1 distil overhead ≤ 36w (was 39w pre-Cycle-22, now 34w)
    def test_batcher_n1_distil_overhead_reduced_cycle22(self):
        """Cycle #22: N=1 distil ≤36w with single-word text (was 39w before Cycle #22).

        Changes: header -3w ('each of the' removed) + N=1 footer -2w ('exactly'+'TOON' removed).
        39w → 34w; bound ≤36w is tight and correct.
        """
        b = self._make_batcher()
        prompt = b._build_distil_prompt(["T."], [""])
        assert "each of the" not in prompt          # header check
        assert "---TOON---" not in prompt           # N=1 → no separator
        assert len(prompt.split()) <= 36            # was 39w before Cycle #22

    # ------------------------------------------------------------------
    # Cycle #23: evolve header compression + distil N≥2 footer/example trim
    # ------------------------------------------------------------------

    # 29 — evolve header no longer has "Memory judge: classify ... note pair as"
    def test_evolve_header_compressed_cycle23(self):
        """Cycle #23+31: evolve header trimmed to 'Judge (EXISTING, NEW) pair: <verdicts>'.

        Cycle #23: 'Memory judge: classify each ... note pair as' → 'Judge each ... pair:' (−4w).
        Cycle #31: 'each' removed (−1w) — pair blocks already enumerate each pair.
        """
        b = self._make_batcher()
        prompt = b._build_evolve_prompt([self._note(1)], [self._note(0)])
        header = prompt.splitlines()[0]
        assert "Memory judge" not in header         # removed "Memory judge: classify"
        assert "classify" not in header
        assert "note pair" not in header
        assert "Judge" in header                    # role preserved as "Judge"
        assert "EXISTING" in header                 # pair labels preserved
        assert "CONTRADICTS" in header              # verdict labels preserved

    # 30 — batcher N≥2 distil footer no longer has "TOON block(s)"
    def test_distil_n2_footer_no_toon_block_label(self):
        """Cycle #23: 'TOON block(s)' → 'block(s)' in N≥2 footer — format shown in example."""
        b = self._make_batcher()
        prompt = b._build_distil_prompt(["Text A.", "Text B."], ["", ""])
        # Footer line should say "block(s)" without "TOON"
        footer_line = [l for l in prompt.splitlines() if "block(s)" in l][0]
        assert "TOON block" not in footer_line
        assert "block(s)" in footer_line
        assert "---TOON---" in prompt               # separator still taught in example

    # 31 — evolve N=1 overhead ≤ 30w (was 37w pre-Cycle-23, now 30w)
    def test_evolve_n1_overhead_reduced_cycle23(self):
        """Cycle #23+26+31: N=1 evolve ≤30w with minimal note.

        Cycle #23: header −4w (37w → 33w).
        Cycle #26: 'Respond with' removed from N=1 footer (−2w): 31w.
        Cycle #31: 'each' removed from header (−1w): 30w → bound ≤30w.
        """
        b = self._make_batcher()
        prompt = b._build_evolve_prompt([self._note(1)], [self._note(0)])
        assert "Memory judge" not in prompt
        assert len(prompt.split()) <= 30            # was 31w before Cycle #31

    # ------------------------------------------------------------------
    # Cycle #24: "pipe-sep" removed from keyword/tag count specs
    # ------------------------------------------------------------------

    # 32 — batcher distil prompt has no "pipe-sep" in field spec
    def test_batcher_distil_no_pipe_sep_in_spec(self):
        """Cycle #24: 'pipe-sep' removed — | format demonstrated by example."""
        b = self._make_batcher()
        prompt = b._build_distil_prompt(["Text A.", "Text B."], ["", ""])
        assert "pipe-sep" not in prompt
        assert "(3-6)" in prompt                    # count constraint still present
        assert "keywords" in prompt                 # field name still present
        assert "|" in prompt                        # pipe format still in example

    # 33 — TEEG distil prompt has no "pipe-sep" in field spec
    def test_teeg_distil_no_pipe_sep_in_spec(self):
        """Cycle #24: 'pipe-sep' removed from TEEG keywords and tags specs."""
        from oml.memory.teeg_pipeline import TEEGPipeline
        pipeline = TEEGPipeline(model="mock")
        prompt = pipeline._build_distil_prompt("Some text.", "")
        assert "pipe-sep" not in prompt
        assert "(3-6)" in prompt                    # keywords count constraint present
        assert "(2-4)" in prompt                    # tags count constraint present
        assert "|" in prompt                        # pipe separator shown in example

    # ------------------------------------------------------------------
    # Cycle #25: example block 1 content shortened (6w → 4w) across all distil prompts
    # ------------------------------------------------------------------

    # 34 — batcher distil example no longer says "Frankenstein created"
    def test_batcher_distil_example_content_compressed_cycle25(self):
        """Cycle #25: block-1 content 'Victor built the creature.' (4w) in both N=1 and N≥2."""
        b = self._make_batcher()
        for n in [1, 2, 8]:
            texts = ["T."] * n
            prompt = b._build_distil_prompt(texts, [""] * n)
            assert "Frankenstein created" not in prompt   # old 6w form gone
            assert "Victor built the creature." in prompt  # new 4w form present

    # 35 — batcher N=1 distil overhead ≤ 28w (was 33w pre-Cycle-25, now 27w)
    def test_batcher_n1_distil_overhead_reduced_cycle25(self):
        """Cycle #25+26+28+30: N=1 distil ≤28w with single-word text.

        Cycle #25: example content 6w → 4w (−2w): 31w.
        Cycle #26: 'Respond with' removed (−2w): 29w.
        Cycle #28: 'Example:' removed (−1w): 28w.
        Cycle #30: 'Fields:' removed (−1w): 27w → bound ≤28w.
        """
        b = self._make_batcher()
        prompt = b._build_distil_prompt(["T."], [""])
        assert "Victor built the creature." in prompt
        assert len(prompt.split()) <= 28            # was 29w before Cycle #30

    # ------------------------------------------------------------------
    # Cycle #26: "Respond with" removed from N=1 footers (distil + evolve)
    # ------------------------------------------------------------------

    # 36 — batcher N=1 and N≥2 distil footers both have no "Respond with"
    def test_batcher_n1_distil_no_respond_with_cycle26(self):
        """Cycle #26+32: 'Respond with' removed from all N distil footers.

        Cycle #26: N=1 footer 'Respond with 1 block:' → '1 block:' (−2w).
        Cycle #32: N≥2 footer 'Respond with N block(s) ...' → 'N block(s) ...' (−2w).
        In both cases the count + separator (for N≥2) + example are sufficient context.
        """
        b = self._make_batcher()
        prompt_n1 = b._build_distil_prompt(["T."], [""])
        prompt_n2 = b._build_distil_prompt(["T.", "U."], ["", ""])
        assert "Respond with" not in prompt_n1     # Cycle #26: removed from N=1
        assert "1 block:" in prompt_n1             # shortened form present
        assert "Respond with" not in prompt_n2     # Cycle #32: also removed from N≥2

    # 37 — batcher N=1 and N≥2 evolve footers both have no "Respond with"
    def test_evolve_n1_no_respond_with_cycle26(self):
        """Cycle #26+32: 'Respond with' removed from all N evolve footers.

        Cycle #26: N=1 footer 'Respond with 1 verdict:' → '1 verdict:' (−2w).
        Cycle #32: N≥2 footer 'Respond with N verdict(s) ...' → 'N verdict(s) ...' (−2w).
        'SUPPORTS' example + count + separator are sufficient context in both cases.
        """
        b = self._make_batcher()
        prompt_n1 = b._build_evolve_prompt([self._note(1)], [self._note(0)])
        prompt_n2 = b._build_evolve_prompt(
            [self._note(1), self._note(2)],
            [self._note(0), self._note(3)],
        )
        assert "Respond with" not in prompt_n1     # Cycle #26: removed from N=1
        assert "1 verdict:" in prompt_n1           # shortened form present
        assert "Respond with" not in prompt_n2     # Cycle #32: also removed from N≥2

    # ------------------------------------------------------------------
    # Cycle #27: "exactly" removed from N≥2 footers (distil + evolve)
    # ------------------------------------------------------------------

    # 38 — batcher N≥2 distil footer has no "exactly"
    def test_batcher_n2_distil_no_exactly_cycle27(self):
        """Cycle #27: N≥2 distil footer is 'Respond with N block(s) ...' — 'exactly' removed (-1w).

        The count N is already explicit; 'exactly' adds no new constraint.
        N=1 footer ('1 block:') is unchanged.
        """
        b = self._make_batcher()
        for n in [2, 8]:
            texts = ["T."] * n
            prompt = b._build_distil_prompt(texts, [""] * n)
            assert "exactly" not in prompt                # removed
            assert f"{n} block(s)" in prompt              # count still present
        prompt_n1 = b._build_distil_prompt(["T."], [""])
        assert "exactly" not in prompt_n1                 # N=1 never had "exactly"

    # 39 — batcher N≥2 evolve footer has no "exactly"
    def test_evolve_n2_no_exactly_cycle27(self):
        """Cycle #27: N≥2 evolve footer is 'Respond with N verdict(s) ...' — 'exactly' removed (-1w).

        Same rationale as distil: count is already explicit in 'Respond with N verdict(s)'.
        N=1 footer ('1 verdict:') is unchanged.
        """
        b = self._make_batcher()
        for n in [2, 8]:
            new_ns = [self._note(i + 10) for i in range(n)]
            cand_ns = [self._note(i) for i in range(n)]
            prompt = b._build_evolve_prompt(new_ns, cand_ns)
            assert "exactly" not in prompt                # removed
            assert f"{n} verdict(s)" in prompt            # count still present
        prompt_n1 = b._build_evolve_prompt([self._note(1)], [self._note(0)])
        assert "exactly" not in prompt_n1                 # N=1 never had "exactly"

    # ------------------------------------------------------------------
    # Cycle #28: "Example:" label removed from all three batcher footers
    # ------------------------------------------------------------------

    # 40 — batcher N=1 distil footer has no "Example:"
    def test_batcher_n1_distil_no_example_label_cycle28(self):
        """Cycle #28: 'Example:' label removed from N=1 distil footer (-1w).

        The TOON field pattern ('note_id: teeg-...') immediately following the
        '1 block:' label is self-evidently illustrative; 'Example:' is redundant.
        """
        b = self._make_batcher()
        prompt = b._build_distil_prompt(["T."], [""])
        assert "Example:" not in prompt                   # label removed
        assert "note_id: teeg-abc123def456" in prompt     # example block still present

    # 41 — batcher N≥2 distil footer has no "Example:"
    def test_batcher_n2_distil_no_example_label_cycle28(self):
        """Cycle #28: 'Example:' label removed from N≥2 distil footer (-1w).

        Same rationale as N=1: the TOON field block is self-evidently an example.
        """
        b = self._make_batcher()
        for n in [2, 8]:
            texts = ["T."] * n
            prompt = b._build_distil_prompt(texts, [""] * n)
            assert "Example:" not in prompt               # label removed
            assert "---TOON---" in prompt                 # separator still present

    # 42 — batcher N≥2 evolve footer has no "Example:"
    def test_evolve_n2_no_example_label_cycle28(self):
        """Cycle #28: 'Example:' label removed from N≥2 evolve footer (-1w).

        'SUPPORTS' verdict word following the footer instruction is self-evidently
        an example; no 'Example:' label needed.
        """
        b = self._make_batcher()
        for n in [2, 8]:
            new_ns = [self._note(i + 10) for i in range(n)]
            cand_ns = [self._note(i) for i in range(n)]
            prompt = b._build_evolve_prompt(new_ns, cand_ns)
            assert "Example:" not in prompt               # label removed
            assert "SUPPORTS" in prompt                   # example verdict still present
            assert "---VERDICT---" in prompt              # separator still present

    # ------------------------------------------------------------------
    # Cycle #29: TEEG distil "only the" + "Example:" removed; batcher block 2 compressed
    # ------------------------------------------------------------------

    # 43 — TEEG distil no longer has "only the" in instruction line
    def test_teeg_distil_no_only_the_cycle29(self):
        """Cycle #29+31: 'only the' and 'Respond with' removed from TEEG distil.

        Cycle #29: 'only the' removed → 'Respond with TOON fields.' (−2w).
        Cycle #31: 'Respond with' further removed → 'TOON fields:' (−2w more).
        SmartMock detection ('TOON note' in prompt) is unaffected.
        """
        from oml.memory.teeg_pipeline import TEEGPipeline
        p = TEEGPipeline(model="mock")
        prompt = p._build_distil_prompt("Some text.", "")
        assert "only the" not in prompt               # Cycle #29: removed
        assert "Respond with" not in prompt           # Cycle #31: removed
        assert "TOON fields:" in prompt               # Cycle #31: new header form
        assert "TOON note" in prompt                  # SmartMock detection preserved

    # 44 — TEEG distil no longer has "Example:" label
    def test_teeg_distil_no_example_label_cycle29(self):
        """Cycle #29: 'Example:' removed from TEEG distil instruction line (-1w).

        Same rationale as batcher Cycle #28: field pattern is self-evidently illustrative.
        """
        from oml.memory.teeg_pipeline import TEEGPipeline
        p = TEEGPipeline(model="mock")
        prompt = p._build_distil_prompt("Some text.", "")
        assert "Example:" not in prompt               # removed
        assert "content: Victor built" in prompt      # example still present

    # 45 — batcher N≥2 example block-2 content compressed cycle29
    def test_batcher_n2_block2_content_compressed_cycle29(self):
        """Cycle #29+32: N≥2 block-2 content compressed (-2w); 'Respond with' removed (-2w).

        Cycle #29: 'The creature fled the laboratory.' (5w) → 'The creature fled.' (3w).
        Cycle #30: 'Fields:' removed (−1w): 65w.
        Cycle #32: 'Respond with' removed from footer (−2w): 63w → bound ≤64w.
        """
        b = self._make_batcher()
        for n in [2, 8]:
            texts = ["T."] * n
            prompt = b._build_distil_prompt(texts, [""] * n)
            assert "The creature fled the laboratory" not in prompt  # old 5w form gone
            assert "The creature fled." in prompt                    # new 3w form present
            assert len(prompt.split()) <= 64                         # tightened from ≤66w (Cycle #32 -2w)

    # ------------------------------------------------------------------
    # Cycle #30: "Fields:" removed from batcher header; TEEG context example compressed
    # ------------------------------------------------------------------

    # 46 — batcher distil header has no "Fields:" label
    def test_batcher_distil_no_fields_label_cycle30(self):
        """Cycle #30: 'Fields:' label removed from batcher distil header (-1w per call).

        Field list follows directly after 'encode N texts.' — no label needed.
        'note_id' remains in the example block, satisfying the field-present check.
        """
        b = self._make_batcher()
        for n in [1, 2, 8]:
            texts = ["T."] * n
            prompt = b._build_distil_prompt(texts, [""] * n)
            assert "Fields:" not in prompt                  # label removed
            assert "note_id" in prompt                      # field name in example
            assert "TOON encoder" in prompt                 # SmartMock detection preserved

    # 47 — TEEG distil example context is "novel" not "Frankenstein novel"
    def test_teeg_distil_context_compressed_cycle30(self):
        """Cycle #30: TEEG example 'context: Frankenstein novel' → 'context: novel' (-1w).

        'Frankenstein' is redundant — 'novel' alone demonstrates that context should
        describe the source material type.
        """
        from oml.memory.teeg_pipeline import TEEGPipeline
        p = TEEGPipeline(model="mock")
        prompt = p._build_distil_prompt("Some text.", "")
        assert "Frankenstein novel" not in prompt    # old 2-word context gone
        assert "context: novel" in prompt            # new 1-word context present
        assert "context" in prompt                   # field name still in spec

    # ------------------------------------------------------------------
    # Cycle #31: TEEG "Respond with TOON fields." → "TOON fields:"; evolve "each" removed
    # ------------------------------------------------------------------

    # 48 — TEEG distil instruction line is "TOON fields:" (no "Respond with")
    def test_teeg_distil_toon_fields_header_cycle31(self):
        """Cycle #31: 'Respond with TOON fields.' → 'TOON fields:' (-2w per TEEG distil call).

        'Respond with' is imperative scaffolding; 'TOON fields:' as a section header
        introduces the example block below — same semantics, 2 fewer words.
        SmartMock detection ('TOON note' in prompt) is on line 1, unaffected.
        """
        from oml.memory.teeg_pipeline import TEEGPipeline
        p = TEEGPipeline(model="mock")
        prompt = p._build_distil_prompt("Some text.", "")
        assert "Respond with" not in prompt           # scaffolding removed
        assert "TOON fields:" in prompt               # compact header present
        assert "TOON note" in prompt                  # SmartMock detection preserved
        assert len(prompt.split()) <= 29              # TEEG distil ≤29w (was 30w)

    # 49 — evolve header has no "each"
    def test_evolve_header_no_each_cycle31(self):
        """Cycle #31: 'Judge each (EXISTING, NEW) pair:' → 'Judge (EXISTING, NEW) pair:' (-1w).

        'each' is redundant — [PAIR N] blocks already enumerate each pair individually.
        SmartMock detection ('[PAIR N]' + '---VERDICT---') is in pair content, not header.
        """
        b = self._make_batcher()
        for n in [1, 8]:
            if n == 1:
                prompt = b._build_evolve_prompt([self._note(1)], [self._note(0)])
            else:
                prompt = b._build_evolve_prompt(
                    [self._note(i + 10) for i in range(n)],
                    [self._note(i) for i in range(n)],
                )
            header = prompt.splitlines()[0]
            assert "each" not in header               # removed
            assert "Judge" in header                  # verb preserved
            assert "(EXISTING, NEW)" in header        # pair label preserved

    # ------------------------------------------------------------------
    # Cycle #32: "Respond with" removed from N≥2 footers (distil + evolve)
    # ------------------------------------------------------------------

    # 50 — batcher N≥2 distil footer has no "Respond with" but count+sep intact
    def test_batcher_n2_distil_no_respond_with_cycle32(self):
        """Cycle #32: N≥2 distil footer 'Respond with N block(s) ...' → 'N block(s) ...' (-2w).

        Symmetric with Cycle #26 (N=1 distil). The count N and separator instruction
        are unambiguous without the 'Respond with' imperative scaffold.
        SmartMock detection ('TOON encoder' + '[TEXT 1]') is in header/body, not footer.
        """
        b = self._make_batcher()
        for n in [2, 8]:
            texts = ["T."] * n
            prompt = b._build_distil_prompt(texts, [""] * n)
            assert "Respond with" not in prompt           # imperative scaffold removed
            assert f"{n} block(s) separated by" in prompt # count + sep instruction intact
            assert "---TOON---" in prompt                  # separator still taught in example

    # 51 — batcher N≥2 evolve footer has no "Respond with" but count+sep intact
    def test_evolve_n2_no_respond_with_cycle32(self):
        """Cycle #32: N≥2 evolve footer 'Respond with N verdict(s) ...' → 'N verdict(s) ...' (-2w).

        Symmetric with Cycle #26 (N=1 evolve). 'N verdict(s) separated by ---VERDICT---.'
        followed by the 'SUPPORTS / ---VERDICT--- / EXTENDS' example is self-contained.
        SmartMock detection ('[PAIR N]' + '---VERDICT---') is preserved in pair bodies.
        """
        b = self._make_batcher()
        for n in [2, 8]:
            new_ns = [self._note(i + 10) for i in range(n)]
            cand_ns = [self._note(i) for i in range(n)]
            prompt = b._build_evolve_prompt(new_ns, cand_ns)
            assert "Respond with" not in prompt            # imperative scaffold removed
            assert f"{n} verdict(s) separated by" in prompt # count + sep instruction intact
            assert "---VERDICT---" in prompt               # separator still taught in example
