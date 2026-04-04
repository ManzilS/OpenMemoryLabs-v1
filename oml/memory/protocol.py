from __future__ import annotations
from typing import Any, List, Tuple
try:
    from typing import Protocol, runtime_checkable
except ImportError:
    from typing_extensions import Protocol, runtime_checkable


@runtime_checkable
class MemoryPipeline(Protocol):
    """Shared protocol for TEEG and PRISM pipelines.

    Both TEEGPipeline and PRISMPipeline satisfy this protocol, making them
    interchangeable anywhere a MemoryPipeline is expected -- in the UI, in
    evaluation scripts, and in user experiments.  Code that only cares about
    the ingest/query/save contract can type-annotate with MemoryPipeline
    rather than importing a concrete class.
    """

    def ingest(
        self,
        text: str,
        context_hint: str = "",
        source_id: str = "",
    ) -> Any:
        """Ingest a single text and return the resulting note (or wrapper)."""
        ...

    def ingest_batch(
        self,
        texts: List[str],
        context_hint: str = "",
    ) -> Any:
        """Ingest a list of texts and return all resulting notes (or batch wrapper)."""
        ...

    def query(
        self,
        question: str,
        top_k: int = 8,
        return_context: bool = False,
    ) -> Tuple:
        """Answer a question from memory, optionally returning the context block."""
        ...

    def search(self, query: str, top_k: int = 5) -> List:
        """Return raw retrieval results without LLM generation."""
        ...

    def save(self) -> None:
        """Persist all memory to disk."""
        ...

    def stats(self) -> dict:
        """Return a dict of pipeline and store statistics."""
        ...
