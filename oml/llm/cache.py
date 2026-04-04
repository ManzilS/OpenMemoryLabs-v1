"""oml/llm/cache.py — Disk-persisted LLM response cache.

Three classes:

``LLMCache``
    Plain JSON dict stored at ``<cache_path>/llm_cache.json``.  Cache key is
    ``SHA-256(model_name + "|||" + prompt)``.  Supports four modes:

    * ``"auto"``    (default) — hit → return cached; miss → call API, store, return.
    * ``"replay"``  — hit → return cached; miss → raise ``CacheMissError``.
    * ``"record"``  — always call API, always overwrite entry (force refresh).
    * ``"off"``     — pass-through; no reads or writes.

``CachedLLMClient``
    Transparent ``BaseLLM`` wrapper: check cache → check budget → call inner
    ``generate()`` → store in cache → return.

``Budget``
    Call-count guard.  Logs a WARNING at ``warn_at × max_calls`` and raises
    ``BudgetExceededError`` once ``max_calls`` is reached.

Environment variable ``OML_CACHE_MODE`` overrides the mode passed to ``LLMCache``
when the cache is constructed via ``CachedLLMClient.from_env()``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import pathlib
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Optional

from oml.llm.base import BaseLLM

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class CacheMissError(RuntimeError):
    """Raised in ``"replay"`` mode when a prompt is not in the cache."""

    def __init__(self, key: str, prompt_preview: str) -> None:
        self.key = key
        self.prompt_preview = prompt_preview
        super().__init__(
            f"Cache miss in replay mode (key={key[:12]}…): {prompt_preview!r}"
        )


class BudgetExceededError(RuntimeError):
    """Raised when the ``Budget`` call-count limit is reached."""

    def __init__(self, calls_made: int, max_calls: int) -> None:
        self.calls_made = calls_made
        self.max_calls = max_calls
        super().__init__(
            f"LLM call budget exhausted: {calls_made}/{max_calls} calls used."
        )


# ---------------------------------------------------------------------------
# Rough pricing table (informational only, USD per 1 000 input tokens)
# ---------------------------------------------------------------------------
_PRICE_PER_1K_IN: Dict[str, float] = {
    "gpt-4o-mini":          0.000150,
    "gpt-4o":               0.005000,
    "gpt-4-turbo":          0.010000,
    "gemini-1.5-flash":     0.000075,
    "gemini-1.5-pro":       0.001250,
    "gemini-2.0-flash":     0.000100,
    "claude-3-haiku":       0.000250,
    "claude-3-sonnet":      0.003000,
    "claude-3-opus":        0.015000,
}


def _price_per_call(model: str, avg_prompt_tokens: int = 400) -> float:
    """Return rough USD cost per call for a given model."""
    for prefix, price in _PRICE_PER_1K_IN.items():
        if prefix in model:
            return price * avg_prompt_tokens / 1000
    return 0.0  # local/unknown model


# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------

class Budget:
    """Call-count budget attached to a ``CachedLLMClient``.

    Args:
        max_calls:  Hard limit.  ``BudgetExceededError`` raised at this count.
        warn_at:    Fraction of ``max_calls`` at which a WARNING is logged.
                    Default: 0.8 (warn at 80%).

    Raises:
        ValueError: If ``max_calls`` is not a positive integer.
    """

    def __init__(self, max_calls: int, warn_at: float = 0.8) -> None:
        if max_calls <= 0:
            raise ValueError(f"max_calls must be positive, got {max_calls}")
        self._max_calls = max_calls
        self._warn_at = warn_at
        self._calls_made: int = 0
        self._warned: bool = False

    # ------------------------------------------------------------------
    def check_and_increment(self, prompt: str = "") -> None:
        """Increment counter; warn or raise as thresholds are crossed.

        Args:
            prompt: The prompt about to be sent (used only for log messages).

        Raises:
            BudgetExceededError: When ``_calls_made`` reaches ``_max_calls``.
        """
        if self._calls_made >= self._max_calls:
            raise BudgetExceededError(self._calls_made, self._max_calls)

        self._calls_made += 1

        if not self._warned and self._calls_made >= self._warn_at * self._max_calls:
            self._warned = True
            logger.warning(
                "LLM call budget at %.0f%%: %d/%d calls used.",
                100 * self._calls_made / self._max_calls,
                self._calls_made,
                self._max_calls,
            )

    def reset(self) -> None:
        """Reset call counter (useful between test cases)."""
        self._calls_made = 0
        self._warned = False

    # ------------------------------------------------------------------
    def stats(self) -> dict:
        pct = 100.0 * self._calls_made / self._max_calls if self._max_calls else 0.0
        return {
            "calls_made": self._calls_made,
            "calls_remaining": max(0, self._max_calls - self._calls_made),
            "max_calls": self._max_calls,
            "pct_used": round(pct, 1),
        }

    # ------------------------------------------------------------------
    @property
    def calls_made(self) -> int:
        return self._calls_made

    @property
    def max_calls(self) -> int:
        return self._max_calls


# ---------------------------------------------------------------------------
# LLMCache
# ---------------------------------------------------------------------------

@dataclass
class _CacheEntry:
    key: str
    model: str
    prompt_preview: str
    response: str
    created_at: str
    hit_count: int = 0

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "model": self.model,
            "prompt_preview": self.prompt_preview,
            "response": self.response,
            "created_at": self.created_at,
            "hit_count": self.hit_count,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "_CacheEntry":
        return cls(
            key=d["key"],
            model=d.get("model", ""),
            prompt_preview=d.get("prompt_preview", ""),
            response=d["response"],
            created_at=d.get("created_at", ""),
            hit_count=d.get("hit_count", 0),
        )


_VALID_MODES = frozenset({"auto", "replay", "record", "off"})


class LLMCache:
    """Disk-persisted prompt → response cache.

    Args:
        cache_path: Directory where ``llm_cache.json`` is stored.
                    Created automatically if absent.
        mode:       One of ``"auto"``, ``"replay"``, ``"record"``, ``"off"``.
                    Reads ``OML_CACHE_MODE`` env var as override when ``None``.
    """

    _FILENAME = "llm_cache.json"

    def __init__(
        self,
        cache_path: str | pathlib.Path = "artifacts",
        mode: Optional[str] = None,
    ) -> None:
        resolved_mode = mode or os.environ.get("OML_CACHE_MODE", "auto")
        if resolved_mode not in _VALID_MODES:
            raise ValueError(
                f"Invalid cache mode {resolved_mode!r}. Choose from {sorted(_VALID_MODES)}."
            )
        self._mode = resolved_mode
        self._path = pathlib.Path(cache_path)
        self._file = self._path / self._FILENAME
        self._entries: Dict[str, _CacheEntry] = {}
        self._hits: int = 0
        self._misses: int = 0
        self.load()

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_key(model: str, prompt: str) -> str:
        raw = f"{model}|||{prompt}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def get(self, model: str, prompt: str) -> Optional[str]:
        """Return cached response or ``None`` (or raise) depending on mode.

        * ``"off"``     → always ``None`` (no cache read)
        * ``"record"``  → always ``None`` (force API call; existing entry ignored)
        * ``"auto"``    → cached value or ``None`` on miss
        * ``"replay"``  → cached value or ``CacheMissError`` on miss
        """
        if self._mode in ("off", "record"):
            return None
        key = self._make_key(model, prompt)
        entry = self._entries.get(key)
        if entry is not None:
            entry.hit_count += 1
            self._hits += 1
            logger.debug("Cache HIT  key=%s… model=%s", key[:12], model)
            return entry.response
        self._misses += 1
        logger.debug("Cache MISS key=%s… model=%s", key[:12], model)
        if self._mode == "replay":
            raise CacheMissError(key, prompt[:80])
        return None

    def put(self, model: str, prompt: str, response: str) -> None:
        """Store a prompt → response pair (no-op if mode is off)."""
        if self._mode == "off":
            return
        key = self._make_key(model, prompt)
        self._entries[key] = _CacheEntry(
            key=key,
            model=model,
            prompt_preview=prompt[:120],
            response=response,
            created_at=datetime.now(timezone.utc).isoformat(),
            hit_count=0,
        )
        self.save()

    def clear(self, model: Optional[str] = None) -> int:
        """Remove entries.  If ``model`` given, only remove that model's entries.

        Returns:
            Number of entries removed.
        """
        if model is None:
            removed = len(self._entries)
            self._entries.clear()
        else:
            to_remove = [k for k, e in self._entries.items() if e.model == model]
            for k in to_remove:
                del self._entries[k]
            removed = len(to_remove)
        if removed:
            self.save()
        return removed

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Persist the in-memory cache to disk."""
        self._path.mkdir(parents=True, exist_ok=True)
        data = {k: e.to_dict() for k, e in self._entries.items()}
        tmp = self._file.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        tmp.replace(self._file)

    def load(self) -> None:
        """Load cache from disk (no-op if file is absent)."""
        if not self._file.exists():
            return
        try:
            raw = json.loads(self._file.read_text(encoding="utf-8"))
            self._entries = {k: _CacheEntry.from_dict(v) for k, v in raw.items()}
            logger.debug("Loaded %d cache entries from %s", len(self._entries), self._file)
        except (json.JSONDecodeError, KeyError) as exc:
            logger.warning("Could not load LLM cache from %s: %s", self._file, exc)
            self._entries = {}

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        total = self._hits + self._misses
        hit_rate = self._hits / total if total else 0.0
        return {
            "total_entries": len(self._entries),
            "cache_hits": self._hits,
            "cache_misses": self._misses,
            "hit_rate": round(hit_rate, 3),
            "estimated_calls_saved": self._hits,
            "mode": self._mode,
            "cache_file": str(self._file),
        }

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def total_entries(self) -> int:
        return len(self._entries)


# ---------------------------------------------------------------------------
# CachedLLMClient
# ---------------------------------------------------------------------------

class CachedLLMClient(BaseLLM):
    """Transparent ``BaseLLM`` wrapper that intercepts every ``generate()`` call.

    Flow::

        generate(prompt)
            ├─ mode == "off"   → inner.generate(prompt) directly
            ├─ cache HIT       → return cached response (0 API calls)
            ├─ budget check    → BudgetExceededError if exhausted
            ├─ mode == "replay"  + MISS → CacheMissError
            └─ API call        → inner.generate(prompt)
                                  → cache.put()
                                  → return response

    Args:
        inner:  Any ``BaseLLM`` instance (Ollama, OpenAI, Gemini, Mock, …).
        cache:  A ``LLMCache`` instance.
        budget: Optional ``Budget`` guard (checked only on API calls, not cache hits).
        model_name: Override for the cache key's model identifier.  Defaults to
                    ``type(inner).__name__``.
    """

    def __init__(
        self,
        inner: BaseLLM,
        cache: LLMCache,
        budget: Optional[Budget] = None,
        model_name: Optional[str] = None,
    ) -> None:
        self._inner = inner
        self._cache = cache
        self._budget = budget
        self._model_name = model_name or type(inner).__name__

    # ------------------------------------------------------------------
    def generate(self, prompt: str) -> str:  # type: ignore[override]
        """Generate a response, using cache when available."""
        # --- 1. Check cache (get() handles off/record/replay/auto modes) ---
        cached = self._cache.get(self._model_name, prompt)
        if cached is not None:
            return cached

        # --- 2. Budget guard (only charged for actual API calls) ---
        if self._budget is not None:
            self._budget.check_and_increment(prompt)

        # --- 3. Call the real LLM ---
        response = self._inner.generate(prompt)

        # --- 4. Store in cache (no-op in off/replay modes) ---
        self._cache.put(self._model_name, prompt, response)

        return response

    # ------------------------------------------------------------------
    def stats(self) -> dict:
        """Return merged cache + budget stats."""
        s = self._cache.stats()
        if self._budget is not None:
            s["budget"] = self._budget.stats()
        return s

    @property
    def cache(self) -> LLMCache:
        return self._cache

    @property
    def budget(self) -> Optional[Budget]:
        return self._budget
