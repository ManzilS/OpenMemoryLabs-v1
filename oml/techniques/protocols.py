"""Pluggable technique protocols for OpenMemoryLab pipelines.

Each protocol corresponds to one stage of the memory lifecycle.  Concrete
classes that structurally satisfy a protocol can be registered in the
:class:`~oml.techniques.registry.TechniqueRegistry` and composed into
pipeline configurations at runtime.

All protocols are decorated with ``@runtime_checkable`` so that
``isinstance(obj, SomeTechnique)`` works for duck-type validation.
"""

from __future__ import annotations

from typing import Any, Dict, List

try:
    from typing import Protocol, runtime_checkable
except ImportError:  # Python < 3.8
    from typing_extensions import Protocol, runtime_checkable


# ---------------------------------------------------------------------------
# Write-time techniques
# ---------------------------------------------------------------------------


@runtime_checkable
class IngestTechnique(Protocol):
    """Protocol for write-time ingestion techniques.

    An ingest technique converts raw text (and optional surrounding context)
    into one or more structured notes suitable for storage.

    Examples: TOON distiller, heuristic chunker, LLM summariser.
    """

    def process(self, raw_text: str, context: str = "", **kwargs: Any) -> Any:
        """Transform *raw_text* into structured note(s).

        Parameters
        ----------
        raw_text:
            The source text to ingest.
        context:
            Optional context hint (e.g. filename, topic) that may guide
            the distillation.
        **kwargs:
            Implementation-specific options.

        Returns
        -------
        Any
            One or more structured notes.  The exact type depends on the
            concrete implementation (e.g. ``AtomicNote``, ``list[AtomicNote]``).
        """
        ...


@runtime_checkable
class EvolutionTechnique(Protocol):
    """Protocol for note evolution techniques.

    An evolution technique decides how a newly ingested note relates to
    existing memory and returns the updated set of notes (which may include
    archival or supersession actions).

    Examples: LLM judge evolver, timestamp-decay evolver.
    """

    def evolve(
        self,
        new_note: Any,
        existing_notes: List[Any],
        **kwargs: Any,
    ) -> List[Any]:
        """Evolve memory by integrating *new_note* with *existing_notes*.

        Parameters
        ----------
        new_note:
            The freshly ingested note to integrate.
        existing_notes:
            Current active notes that may be affected.
        **kwargs:
            Implementation-specific options (e.g. LLM client, thresholds).

        Returns
        -------
        list
            The updated list of notes after evolution (may include
            deactivated or merged entries).
        """
        ...


@runtime_checkable
class CompressionTechnique(Protocol):
    """Protocol for context compression techniques.

    A compression technique takes a collection of notes and a token budget,
    then produces a compressed textual context block that fits within the
    budget while preserving as much relevant information as possible.

    Examples: TOON encoder, extractive summariser, priority ranker.
    """

    def compress(
        self,
        notes: List[Any],
        budget_tokens: int,
        **kwargs: Any,
    ) -> str:
        """Compress *notes* into a context string within *budget_tokens*.

        Parameters
        ----------
        notes:
            The notes to compress into a context block.
        budget_tokens:
            Maximum number of tokens the output should consume.
        **kwargs:
            Implementation-specific options (e.g. tokeniser, priorities).

        Returns
        -------
        str
            A textual context block ready for inclusion in an LLM prompt.
        """
        ...


# ---------------------------------------------------------------------------
# Read-time techniques
# ---------------------------------------------------------------------------


@runtime_checkable
class RetrievalTechnique(Protocol):
    """Protocol for read-time retrieval techniques.

    A retrieval technique filters and/or re-ranks a set of candidate notes
    against a user query.

    Examples: BM25 scorer, vector similarity, cross-encoder reranker.
    """

    def retrieve(
        self,
        query: str,
        candidates: List[Any],
        **kwargs: Any,
    ) -> List[Any]:
        """Retrieve the most relevant candidates for *query*.

        Parameters
        ----------
        query:
            The user's natural-language question.
        candidates:
            Pool of candidate notes to filter / rank.
        **kwargs:
            Implementation-specific options (e.g. top_k, threshold).

        Returns
        -------
        list
            Ordered list of relevant items (highest relevance first).
        """
        ...


@runtime_checkable
class SeedingTechnique(Protocol):
    """Protocol for seed discovery techniques.

    A seeding technique identifies the initial set of entry-point notes
    (seeds) for a graph walk, given a user query.

    Examples: keyword matcher, vector nearest-neighbour, entity linker.
    """

    def find_seeds(self, query: str, **kwargs: Any) -> List[Any]:
        """Find seed notes for *query*.

        Parameters
        ----------
        query:
            The user's natural-language question.
        **kwargs:
            Implementation-specific options (e.g. top_k, store handle).

        Returns
        -------
        list
            Seed items (notes or note-score tuples) to start traversal from.
        """
        ...


@runtime_checkable
class WalkingTechnique(Protocol):
    """Protocol for graph walking techniques.

    A walking technique traverses a knowledge graph starting from seed
    notes and collects related context.

    Examples: BFS walker, weighted random walk, attention-guided walk.
    """

    def walk(self, seeds: List[Any], **kwargs: Any) -> Dict[str, Any]:
        """Walk the graph from *seeds* and collect context.

        Parameters
        ----------
        seeds:
            Starting points for the traversal (typically from a
            :class:`SeedingTechnique`).
        **kwargs:
            Implementation-specific options (e.g. max_hops, graph handle).

        Returns
        -------
        dict
            A mapping with at least ``"notes"`` (collected notes) and
            optionally ``"paths"``, ``"scores"``, or other traversal
            metadata.
        """
        ...
