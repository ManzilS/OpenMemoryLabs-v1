from oml.ingest.chunkers.base import ChunkingStrategy
from oml.ingest.chunkers.text import RecursiveStrategy
from oml.ingest.chunkers.code import CodeStrategy

def get_strategy(doc_type: str) -> ChunkingStrategy:
    """Factory to get the appropriate chunking strategy."""
    if doc_type == 'code':
        return CodeStrategy()
    # 'text', 'email', etc.
    return RecursiveStrategy()

def segment_document(doc, min_size: int = 100, max_size: int = 1000) -> list:
    """Helper wrapper to segment a document immediately."""
    strategy = get_strategy(doc.doc_type)
    return strategy.segment(doc, min_size, max_size)

__all__ = ["ChunkingStrategy", "RecursiveStrategy", "CodeStrategy", "get_strategy", "segment_document"]
