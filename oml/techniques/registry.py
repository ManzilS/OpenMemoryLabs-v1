"""Technique registry and built-in presets for OpenMemoryLab.

The :class:`TechniqueRegistry` provides a central name-to-class mapping so
that pipeline configurations can reference techniques by string name rather
than importing concrete classes directly.  This enables YAML/CLI-driven
pipeline composition.

Built-in presets (``PRESETS``) bundle commonly-used technique stacks into
named profiles that a pipeline can load in one call.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Type


class TechniqueRegistry:
    """Name-based registry for technique implementations.

    Techniques are registered with a human-readable *name* and an optional
    *category* tag (e.g. ``"ingest"``, ``"retrieval"``).  Look-ups are
    case-insensitive on the name.

    Example::

        reg = TechniqueRegistry()
        reg.register("toon-distiller", TOONDistiller, category="ingest")
        cls = reg.get("toon-distiller")
    """

    def __init__(self) -> None:
        self._entries: Dict[str, _RegistryEntry] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register(
        self,
        name: str,
        technique_cls: Type[Any],
        *,
        category: Optional[str] = None,
        description: str = "",
    ) -> None:
        """Register a technique class under *name*.

        Parameters
        ----------
        name:
            Unique, case-insensitive identifier.
        technique_cls:
            The class (not an instance) that implements the technique.
        category:
            Optional grouping tag (``"ingest"``, ``"retrieval"``, etc.).
        description:
            Optional one-line description shown in listings.

        Raises
        ------
        ValueError
            If *name* is already registered.
        """
        key = name.lower()
        if key in self._entries:
            raise ValueError(
                f"Technique {name!r} is already registered "
                f"(class={self._entries[key].cls.__name__})"
            )
        self._entries[key] = _RegistryEntry(
            name=name,
            cls=technique_cls,
            category=category,
            description=description,
        )

    def get(self, name: str) -> Type[Any]:
        """Return the technique class registered under *name*.

        Raises
        ------
        KeyError
            If *name* is not found.
        """
        key = name.lower()
        if key not in self._entries:
            available = ", ".join(sorted(self._entries))
            raise KeyError(
                f"Unknown technique {name!r}. Available: {available}"
            )
        return self._entries[key].cls

    def list_available(
        self,
        category: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Return metadata for all registered techniques.

        Parameters
        ----------
        category:
            If given, only return techniques whose category matches
            (case-insensitive).

        Returns
        -------
        list[dict]
            Each dict contains ``"name"``, ``"class"``, ``"category"``,
            and ``"description"`` keys.
        """
        results: List[Dict[str, Any]] = []
        for entry in sorted(self._entries.values(), key=lambda e: e.name):
            if category and (
                entry.category or ""
            ).lower() != category.lower():
                continue
            results.append(
                {
                    "name": entry.name,
                    "class": entry.cls,
                    "category": entry.category,
                    "description": entry.description,
                }
            )
        return results

    def __contains__(self, name: str) -> bool:
        return name.lower() in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        return f"TechniqueRegistry({len(self._entries)} techniques)"


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------


class _RegistryEntry:
    """Immutable bookkeeping record for one registered technique."""

    __slots__ = ("name", "cls", "category", "description")

    def __init__(
        self,
        name: str,
        cls: Type[Any],
        category: Optional[str],
        description: str,
    ) -> None:
        self.name = name
        self.cls = cls
        self.category = category
        self.description = description


# ------------------------------------------------------------------
# Built-in presets
# ------------------------------------------------------------------

PRESETS: Dict[str, List[str]] = {
    # Basic TEEG stack: TOON distiller, LLM judge evolver, keyword seeder,
    # BFS walker, TOON compressor.
    "teeg": [
        "toon-distiller",
        "llm-judge-evolver",
        "keyword-seeder",
        "bfs-walker",
        "toon-compressor",
    ],
    # Full PRISM stack: superset of TEEG with vector seeding, reranking,
    # and multi-hop walking.
    "prism": [
        "toon-distiller",
        "llm-judge-evolver",
        "vector-seeder",
        "keyword-seeder",
        "multihop-walker",
        "cross-encoder-reranker",
        "toon-compressor",
    ],
    # Lightweight: heuristic ingestion, vector retrieval, no graph walking.
    # Suitable for quick demos or resource-constrained environments.
    "lightweight": [
        "heuristic-distiller",
        "vector-seeder",
        "extractive-compressor",
    ],
}
