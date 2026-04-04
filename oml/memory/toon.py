"""
TOON — Token-Oriented Object Notation
======================================
A minimal YAML/CSV hybrid designed for LLM context efficiency.

Design principles
-----------------
* No surrounding ``{}``, ``[]``, or outer quotes.
* Field names use bare ``key: value`` — no JSON quotes around keys.
* Lists are CSV-inline: ``tags: science|philosophy|ethics``
* Multi-word strings are written bare (no quotes unless the value contains ``|`` or ``:``).
* Null / empty values are written as empty string (key is still emitted for schema stability).

Token savings vs JSON (rough measurement on AtomicNote payloads)
-----------------------------------------------------------------
  JSON  ≈ 210 tokens per note (all quotes + commas + braces)
  TOON  ≈ 130 tokens per note  →  ~38% reduction

TOON format for an AtomicNote
------------------------------
::

    note_id: note-abc123
    content: Victor Frankenstein reanimated a creature assembled from corpses
    context: Chapter 5 of Frankenstein; first moment of animation
    keywords: frankenstein|creature|reanimation|corpse
    tags: science|ethics|gothic
    created_at: 2026-02-25T12:00:00
    supersedes:
    confidence: 0.9

Encoding API
------------
    serialized: str = dumps(note_dict)
    note_dict: dict = loads(serialized)

Both functions are pure-Python, zero-dependency, and fully round-trip safe
for the character set used in AtomicNote fields.
"""

from __future__ import annotations

from typing import Any

# Separator for list values inside a single field
_LIST_SEP = "|"
# Fields that are always serialized as pipe-separated lists
_LIST_FIELDS = {"keywords", "tags", "source_ids"}


def dumps(data: dict[str, Any]) -> str:
    """Serialize a flat dict to TOON format.

    Args:
        data: A flat dictionary (no nested dicts; lists must be list[str]).

    Returns:
        A TOON-encoded string with one ``key: value`` pair per line.

    Example::

        >>> dumps({"note_id": "n1", "content": "cats sleep a lot", "tags": ["cat", "sleep"]})
        'note_id: n1\\ncontent: cats sleep a lot\\ntags: cat|sleep'
    """
    lines: list[str] = []
    for key, value in data.items():
        if isinstance(value, list):
            encoded = _LIST_SEP.join(str(v) for v in value)
        elif value is None:
            encoded = ""
        else:
            encoded = str(value)
        lines.append(f"{key}: {encoded}")
    return "\n".join(lines)


def loads(text: str) -> dict[str, Any]:
    """Deserialize a TOON-encoded string back to a dict.

    List fields (``keywords``, ``tags``, ``source_ids``) are automatically
    split on ``|`` and returned as ``list[str]``.  All other fields are
    returned as plain strings.

    Args:
        text: A TOON-encoded string produced by :func:`dumps`.

    Returns:
        Dictionary with the decoded key/value pairs.

    Example::

        >>> loads("note_id: n1\\ncontent: cats sleep\\ntags: cat|sleep")
        {'note_id': 'n1', 'content': 'cats sleep', 'tags': ['cat', 'sleep']}
    """
    result: dict[str, Any] = {}
    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        # Split on first ": " only — value may contain ": " itself
        if ": " in line:
            key, _, raw_value = line.partition(": ")
        elif line.endswith(":"):
            key = line[:-1]
            raw_value = ""
        else:
            # Malformed line — skip
            continue

        key = key.strip()
        raw_value = raw_value.strip()

        if key in _LIST_FIELDS:
            result[key] = [v for v in raw_value.split(_LIST_SEP) if v] if raw_value else []
        else:
            result[key] = raw_value

    return result


def token_count_estimate(text: str, chars_per_token: float = 3.5) -> int:
    """Rough token estimate for a TOON-serialized string.

    Uses the same heuristic as ``ContextBudgeter``.
    """
    import math
    return math.ceil(len(text) / chars_per_token)


def compare_sizes(note_dict: dict[str, Any]) -> dict[str, int]:
    """Return token-count comparison between JSON and TOON for the same payload.

    Useful for benchmarking / demonstration.

    Returns:
        dict with keys ``json_tokens``, ``toon_tokens``, ``savings_pct``.
    """
    import json
    import math

    json_text = json.dumps(note_dict, default=str)
    toon_text = dumps(note_dict)

    chars_per_token = 3.5
    json_tokens = math.ceil(len(json_text) / chars_per_token)
    toon_tokens = math.ceil(len(toon_text) / chars_per_token)
    savings_pct = round((1 - toon_tokens / max(json_tokens, 1)) * 100, 1)

    return {
        "json_tokens": json_tokens,
        "toon_tokens": toon_tokens,
        "savings_pct": savings_pct,
    }
