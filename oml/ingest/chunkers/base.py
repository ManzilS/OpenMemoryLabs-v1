from typing import Protocol, List
from oml.models.schema import Document, Chunk

class ChunkingStrategy(Protocol):
    def segment(self, doc: Document, min_size: int, max_size: int) -> List[Chunk]:
        ...
