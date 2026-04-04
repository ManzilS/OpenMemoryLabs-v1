from typing import List
from oml.models.schema import Document, Chunk
from oml.ingest.chunkers.base import ChunkingStrategy

class CodeStrategy(ChunkingStrategy):
    """Respects code structure (functions/classes) via indentation."""
    def segment(self, doc: Document, min_size: int, max_size: int) -> List[Chunk]:
        # Simple indentation-based splitter for now.
        lines = doc.clean_text.splitlines(keepends=True)
        chunks = []
        current_block = []
        current_len = 0
        chunk_idx = 0
        
        for line in lines:
            # Heuristic: If we are at top level (no indent) and chunk is big enough, split.
            is_top_level = len(line) - len(line.lstrip()) == 0 and line.strip()
            
            if is_top_level and current_len >= min_size:
                 # Check if adding this line would exceed max (soft limit)
                 # Or just split here because it's a cleaner break
                 chunks.append(self._create_chunk(doc, current_block, chunk_idx))
                 chunk_idx += 1
                 current_block = []
                 current_len = 0
            
            # Hard limit check
            if current_len + len(line) > max_size:
                 if current_block:
                     chunks.append(self._create_chunk(doc, current_block, chunk_idx))
                     chunk_idx += 1
                     current_block = []
                     current_len = 0
            
            current_block.append(line)
            current_len += len(line)
            
        if current_block:
            chunks.append(self._create_chunk(doc, current_block, chunk_idx))
            
        return chunks

    def _create_chunk(self, doc, lines, idx):
        text = "".join(lines)
        return Chunk(chunk_id=f"{doc.doc_id}_c{idx}", doc_id=doc.doc_id, chunk_text=text, start_char=0, end_char=len(text))
