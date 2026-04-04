from typing import List
from oml.models.schema import Document, Chunk
from oml.ingest.chunkers.base import ChunkingStrategy

class RecursiveStrategy(ChunkingStrategy):
    """Standard recursive text splitter (Paragraph > Sentence > Char)."""
    def segment(self, doc: Document, min_size: int, max_size: int) -> List[Chunk]:
        text = doc.clean_text
        if not text:
            return []
            
        chunks = []
        chunk_idx = 0
        
        atoms = self._split_recursive(text, max_size)
        current_chunk_atoms = []
        current_len = 0
        
        for atom in atoms:
            if current_len + len(atom) > max_size and current_len >= min_size:
                chunks.append(self._create_chunk(doc, current_chunk_atoms, chunk_idx))
                chunk_idx += 1
                current_chunk_atoms = []
                current_len = 0
            current_chunk_atoms.append(atom)
            current_len += len(atom)
            
        if current_chunk_atoms:
            chunks.append(self._create_chunk(doc, current_chunk_atoms, chunk_idx))
            
        return chunks

    def _create_chunk(self, doc, atoms, idx):
        text = "".join(atoms).strip()
        # TODO: Calculate real start/end chars if needed
        return Chunk(chunk_id=f"{doc.doc_id}_c{idx}", doc_id=doc.doc_id, chunk_text=text, start_char=0, end_char=len(text))

    def _split_recursive(self, text: str, max_chars: int) -> List[str]:
        if not text: return []
        if len(text) <= max_chars: return [text]
        for sep in ["\n\n", "\n", ". ", " "]:
            if sep in text:
                splits = [p for p in text.split(sep) if p]
                if splits and max(len(p) for p in splits) < len(text):
                    parts = []
                    for i, p in enumerate(splits):
                        candidate = (sep if i > 0 else "") + p
                        if len(candidate) > max_chars:
                            sub_parts = self._split_recursive(p, max_chars)
                            if sub_parts and i > 0:
                                sub_parts[0] = sep + sub_parts[0]
                                if len(sub_parts[0]) > max_chars:
                                    excess = sub_parts[0][max_chars:]
                                    sub_parts[0] = sub_parts[0][:max_chars]
                                    sub_parts.insert(1, excess)
                            parts.extend(sub_parts)
                        else:
                            parts.append(candidate)
                    return parts
        return [text[i:i+max_chars] for i in range(0, len(text), max_chars)]
