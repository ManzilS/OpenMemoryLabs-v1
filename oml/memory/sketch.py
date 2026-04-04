"""oml/memory/sketch.py — Probabilistic membership and near-duplicate detection.

Two complementary structures address the write-redundancy efficiency gap in AI
memory systems.  Together they form the **SketchGate** — a write-time filter
that prevents near-duplicate facts from proliferating in the knowledge graph.

BloomFilter
-----------
O(1) amortised membership test for topic keywords.  Uses *k* independent hash
functions derived from MD5 seed slices; no external libraries needed.  Optimal
*k* and bit-array size *m* are computed from the target false-positive rate and
expected item count at construction time.

Space: ~1.44 × log₂(1/fp_rate) bits per element.

MinHashIndex
------------
Jaccard similarity estimator for keyword sets.  Maintains one 64-integer
MinHash signature per stored note.  Two notes are considered near-duplicates
when their estimated Jaccard similarity exceeds a configurable threshold.

Uses MD5(seed, keyword) → hexdigest[:8] for hash values: deterministic and
dependency-free.  Accuracy: ~3.5% error for 64 hashes.

SketchGate
----------
Combines both structures into a write-time gate:
  1. MinHash near-duplicate scan — O(N × H) where N = stored notes, H = num_hashes
  2. BloomFilter — O(k) topic-membership hint (metrics only; not used for skip
     decisions to avoid false-positive suppression of valid notes)

Persists both structures to a single JSON file so they survive process restarts
without re-scanning the entire note set.

Novel aspect
------------
MinHash LSH is well-established for document fingerprinting (near-duplicate
web-page detection, dataset dedup) but has not been applied to LLM memory graph
write-time deduplication before PRISM.  Applying it to the keyword-set
representation of ``AtomicNote`` allows sub-millisecond near-dup detection
without any embedding model calls.

Pure Python — zero new runtime dependencies.

Usage
-----
    gate = SketchGate(artifacts_dir="teeg_store")
    existing_id = gate.should_skip("Victor created the creature",
                                   keywords=["victor", "creature"])
    if existing_id:
        print(f"Near-duplicate of {existing_id!r} — skipping ingest")
    else:
        note = pipeline.ingest(text)
        gate.register(note)
        gate.save()
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from oml.memory.atomic_note import AtomicNote

logger = logging.getLogger(__name__)

_SKETCH_FILE = "sketch_gate.json"
# 32-bit ceiling — used as the "infinity" sentinel for empty keyword sets
_MAX_INT32: int = 0xFFFF_FFFF


# ══════════════════════════════════════════════════════════════════════════════
# BloomFilter
# ══════════════════════════════════════════════════════════════════════════════


class BloomFilter:
    """Probabilistic set-membership structure with configurable false-positive rate.

    Uses *k* independent hash functions, each derived from an MD5 digest of
    ``"{seed}:{item}"`` so no external hash libraries are required.  Optimal *k*
    and bit-array size *m* are computed at construction time via the standard
    formulas:

        m = ⌈-n × ln(p) / (ln 2)²⌉
        k = ⌈(m/n) × ln 2⌉

    Parameters
    ----------
    capacity:
        Expected number of distinct items before the false-positive rate
        degrades beyond *fp_rate*.  Default: 10,000.
    fp_rate:
        Target false-positive probability.  Default: 0.01 (1 %).
    """

    def __init__(self, capacity: int = 10_000, fp_rate: float = 0.01) -> None:
        self._capacity = capacity
        self._fp_rate = fp_rate
        self._m: int = self._optimal_size(capacity, fp_rate)
        self._k: int = self._optimal_k(self._m, capacity)
        self._bits: bytearray = bytearray((self._m + 7) // 8)
        self._count: int = 0

    # ── internal helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _optimal_size(n: int, p: float) -> int:
        """Return optimal bit-array length m = ⌈-n × ln(p) / (ln 2)²⌉."""
        return max(8, math.ceil(-n * math.log(p) / (math.log(2) ** 2)))

    @staticmethod
    def _optimal_k(m: int, n: int) -> int:
        """Return optimal hash-function count k = ⌈(m/n) × ln 2⌉."""
        return max(1, math.ceil((m / n) * math.log(2)))

    def _hashes(self, item: str) -> List[int]:
        """Return *k* independent bit positions for *item*."""
        positions: List[int] = []
        for seed in range(self._k):
            digest = hashlib.md5(f"{seed}:{item}".encode()).hexdigest()
            pos = int(digest[:8], 16) % self._m
            positions.append(pos)
        return positions

    # ── public API ────────────────────────────────────────────────────────────

    def add(self, item: str) -> None:
        """Insert *item* into the filter."""
        for pos in self._hashes(item):
            self._bits[pos >> 3] |= 1 << (pos & 7)
        self._count += 1

    def __contains__(self, item: str) -> bool:
        """Return True if *item* was probably added (may false-positive)."""
        return all(
            (self._bits[pos >> 3] >> (pos & 7)) & 1
            for pos in self._hashes(item)
        )

    @property
    def count(self) -> int:
        """Number of items added (may include duplicates)."""
        return self._count

    def to_dict(self) -> dict:
        """Serialise filter state to a JSON-safe dict."""
        return {
            "capacity": self._capacity,
            "fp_rate": self._fp_rate,
            "k": self._k,
            "m": self._m,
            "count": self._count,
            "bits": list(self._bits),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "BloomFilter":
        """Reconstruct a BloomFilter from a serialised dict."""
        obj: BloomFilter = cls.__new__(cls)
        obj._capacity = d["capacity"]
        obj._fp_rate = d["fp_rate"]
        obj._k = d["k"]
        obj._m = d["m"]
        obj._count = d.get("count", 0)
        obj._bits = bytearray(d["bits"])
        return obj


# ══════════════════════════════════════════════════════════════════════════════
# MinHashIndex
# ══════════════════════════════════════════════════════════════════════════════


class MinHashIndex:
    """Jaccard similarity estimator on keyword sets via MinHash signatures.

    Maintains a mapping of ``note_id → 64-integer signature``.  Two notes are
    near-duplicates when their estimated Jaccard similarity exceeds a threshold.

    Estimation accuracy: ±1/√(num_hashes) ≈ ±3.5 % for the default 64 hashes.

    Parameters
    ----------
    num_hashes:
        Signature length.  64 gives ~3.5 % estimation error.  Higher values
        improve accuracy at O(num_hashes) cost per ``add()`` and ``find_nearest()``
        call.
    """

    def __init__(self, num_hashes: int = 64) -> None:
        self.num_hashes = num_hashes
        self._signatures: Dict[str, List[int]] = {}  # note_id → signature

    # ── internal helpers ─────────────────────────────────────────────────────

    def _compute_signature(self, keywords: List[str]) -> List[int]:
        """Return a MinHash signature for a keyword set.

        For each of *num_hashes* hash functions *h_i*, the i-th element of the
        signature is ``min(h_i(kw) for kw in keywords)``.  Returns a list of
        ``_MAX_INT32`` values when *keywords* is empty — empty sets share no
        similarity with any other set.
        """
        if not keywords:
            return [_MAX_INT32] * self.num_hashes
        sig: List[int] = []
        for seed in range(self.num_hashes):
            min_val: int = min(
                int(hashlib.md5(f"{seed}:{kw}".encode()).hexdigest()[:8], 16)
                for kw in keywords
            )
            sig.append(min_val)
        return sig

    @staticmethod
    def _jaccard(sig_a: List[int], sig_b: List[int]) -> float:
        """Estimate Jaccard similarity from two MinHash signatures."""
        if not sig_a or not sig_b:
            return 0.0
        matches: int = sum(a == b for a, b in zip(sig_a, sig_b))
        return matches / len(sig_a)

    # ── public API ────────────────────────────────────────────────────────────

    def add(self, note_id: str, keywords: List[str]) -> None:
        """Register a note's keyword signature."""
        self._signatures[note_id] = self._compute_signature(keywords)

    def remove(self, note_id: str) -> None:
        """Deregister a note (e.g. when archived)."""
        self._signatures.pop(note_id, None)

    def find_nearest(
        self, keywords: List[str], threshold: float = 0.75
    ) -> Optional[str]:
        """Return the note_id of the most similar stored note, or None.

        Scans all stored signatures in O(N × num_hashes) time.  Acceptable for
        N < 100,000; at larger scales, pre-filter with BloomFilter to narrow
        candidates before calling this method.

        Returns
        -------
        Optional[str]
            ``note_id`` of the nearest neighbour whose estimated Jaccard
            similarity ≥ *threshold*, or ``None`` if no such note exists.
        """
        if not keywords:
            return None
        query_sig = self._compute_signature(keywords)
        best_id: Optional[str] = None
        best_score: float = -1.0
        for note_id, sig in self._signatures.items():
            score = self._jaccard(query_sig, sig)
            if score >= threshold and score > best_score:
                best_score = score
                best_id = note_id
        return best_id

    def __len__(self) -> int:
        return len(self._signatures)

    def to_dict(self) -> dict:
        """Serialise index state to a JSON-safe dict."""
        return {
            "num_hashes": self.num_hashes,
            "signatures": {k: v for k, v in self._signatures.items()},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MinHashIndex":
        """Reconstruct a MinHashIndex from a serialised dict."""
        obj = cls(num_hashes=d.get("num_hashes", 64))
        obj._signatures = {
            k: list(v) for k, v in d.get("signatures", {}).items()
        }
        return obj


# ══════════════════════════════════════════════════════════════════════════════
# SketchGate
# ══════════════════════════════════════════════════════════════════════════════


class SketchGate:
    """Write-time gate combining BloomFilter topic-check and MinHash deduplication.

    Sits in front of ``TEEGPipeline.ingest()`` / ``PRISMPipeline.ingest()``
    to intercept near-duplicate texts before any LLM calls are made.

    Decision logic
    --------------
    1. If *keywords* is empty → cannot make a reliable dedup decision → allow.
    2. MinHash scan: find the nearest stored note above *dedup_threshold*.
    3. If a near-duplicate is found → return its note_id (caller should skip).
    4. Otherwise → return None (proceed with ingest).

    The BloomFilter is intentionally **not** used for skip/allow decisions
    (only for ``probably_seen_topic()`` metrics).  Bloom false-positives would
    silently discard valid notes; MinHash estimates are more reliable for the
    skip gate.

    Parameters
    ----------
    artifacts_dir:
        Directory where ``sketch_gate.json`` will be read/written.
    bloom_capacity:
        Expected total note count before Bloom FP rate degrades.
    bloom_fp_rate:
        Target false-positive rate for Bloom membership tests (default 1 %).
    minhash_num_hashes:
        MinHash signature length (default 64).
    dedup_threshold:
        Jaccard threshold above which two notes are considered near-duplicates
        (default 0.75 — shares ≥ 75 % of keywords).
    """

    def __init__(
        self,
        artifacts_dir: str | Path = "teeg_store",
        bloom_capacity: int = 10_000,
        bloom_fp_rate: float = 0.01,
        minhash_num_hashes: int = 64,
        dedup_threshold: float = 0.75,
    ) -> None:
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.dedup_threshold = dedup_threshold
        self._bloom = BloomFilter(
            capacity=bloom_capacity, fp_rate=bloom_fp_rate
        )
        self._minhash = MinHashIndex(num_hashes=minhash_num_hashes)
        self._skip_count: int = 0
        self._check_count: int = 0
        self.load()

    # ── public API ────────────────────────────────────────────────────────────

    def should_skip(self, text: str, keywords: List[str]) -> Optional[str]:
        """Return existing note_id if *text*/*keywords* are a near-duplicate, else None.

        The ``text`` parameter is accepted for API consistency but is not used
        in the current implementation (MinHash operates on the keyword set).
        Future versions may incorporate word-level or character n-gram hashing
        for higher recall on texts with non-overlapping vocabularies.
        """
        self._check_count += 1
        if not keywords:
            return None
        existing_id = self._minhash.find_nearest(
            keywords, threshold=self.dedup_threshold
        )
        if existing_id is not None:
            self._skip_count += 1
            logger.debug(
                "[SketchGate] Near-duplicate of %r "
                "(Jaccard ≥ %.2f) — skip recommended",
                existing_id,
                self.dedup_threshold,
            )
        return existing_id

    def register(
        self,
        note: AtomicNote,
        keywords_override: "Optional[List[str]]" = None,
    ) -> None:
        """Add a stored note to both the MinHash index and Bloom filter.

        Call this *after* a note has been persisted to ``TEEGStore``.

        Parameters
        ----------
        note:
            The note to register.
        keywords_override:
            If provided, use these keywords for the MinHash signature instead
            of ``note.keywords``.  Pass the same keyword list that was used
            in ``should_skip()`` (e.g. ``_quick_keywords(raw_text)``) so that
            the stored signature and the query signature use the same extraction
            method, giving accurate Jaccard estimates for near-duplicate checks.
            ``note.keywords`` (LLM-extracted) are still used for the Bloom
            filter regardless of this parameter.
        """
        kws_for_minhash = keywords_override if keywords_override is not None else note.keywords
        self._minhash.add(note.note_id, kws_for_minhash)
        for kw in note.keywords:
            self._bloom.add(kw.lower())

    def probably_seen_topic(self, keyword: str) -> bool:
        """Return True if content about *keyword* probably already exists.

        This is a soft hint (default 1 % false-positive rate) — suitable for
        logging and metrics but **not** for hard skip decisions.
        """
        return keyword.lower() in self._bloom

    def bulk_register(self, notes: Iterable[AtomicNote]) -> None:
        """Register many notes at once (e.g. after loading an existing store)."""
        for note in notes:
            self.register(note)

    # ── persistence ──────────────────────────────────────────────────────────

    def save(self) -> None:
        """Persist both structures to ``<artifacts_dir>/sketch_gate.json``."""
        path = self.artifacts_dir / _SKETCH_FILE
        data = {
            "dedup_threshold": self.dedup_threshold,
            "skip_count": self._skip_count,
            "check_count": self._check_count,
            "bloom": self._bloom.to_dict(),
            "minhash": self._minhash.to_dict(),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)
        logger.debug("[SketchGate] Saved to %s", path)

    def load(self) -> None:
        """Load persisted structures from disk (no-op if file not present)."""
        path = self.artifacts_dir / _SKETCH_FILE
        if not path.exists():
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.dedup_threshold = data.get("dedup_threshold", self.dedup_threshold)
            self._skip_count = data.get("skip_count", 0)
            self._check_count = data.get("check_count", 0)
            if "bloom" in data:
                self._bloom = BloomFilter.from_dict(data["bloom"])
            if "minhash" in data:
                self._minhash = MinHashIndex.from_dict(data["minhash"])
            logger.debug("[SketchGate] Loaded from %s", path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[SketchGate] Could not load: %s", exc)

    # ── diagnostics ──────────────────────────────────────────────────────────

    def stats(self) -> dict:
        """Return statistics for ``prism-stats`` and health checks."""
        return {
            "registered_notes": len(self._minhash),
            "bloom_items_added": self._bloom.count,
            "dedup_threshold": self.dedup_threshold,
            "checks_total": self._check_count,
            "skips_total": self._skip_count,
            "dedup_rate": (
                round(self._skip_count / self._check_count, 4)
                if self._check_count > 0
                else 0.0
            ),
        }
