from typing import List, Optional
from oml.memory.context import ContextChunk
from oml.storage.base import BaseStorage
from oml.retrieval.base import SearchResult

class ContextAssembler:
    """Assembles retrieved chunks into ContextChunks, adding summaries and notes natively."""
    def __init__(self, storage: BaseStorage, storage_type: str, reranker=None):
        self.storage = storage
        self.storage_type = storage_type
        self.reranker = reranker

    def assemble(
        self, 
        query: str, 
        results: List[SearchResult], 
        note_results: List[SearchResult], 
        top_k: int,
        graph_context: Optional[str] = None
    ) -> List[ContextChunk]:
        context_chunks = []
        
        # 1. Inject notes
        if note_results and self.storage_type != "lancedb":
            note_ids = [n.chunk_id for n in note_results if n.score > 0.35]
            if note_ids:
                notes_data = self.storage.get_notes_by_ids(note_ids)
                for n_data, n_res in zip(notes_data, note_results):
                    if n_data:
                        context_chunks.append(ContextChunk(
                            chunk_id=n_data.note_id,
                            text=f"[MEMORY NOTE - THREAD SUMMARY]\n{n_data.content}",
                            score=n_res.score + 0.1
                        ))
                        
        # 2. Inject Graph Context
        if graph_context:
            context_chunks.append(ContextChunk(
                chunk_id="knowledge_graph_context",
                text=graph_context,
                score=1.0
            ))

        if not results:
            return context_chunks

        # Fetch underlying chunks
        chunk_ids = [r.chunk_id for r in results]
        chunks_data = self.storage.get_chunks_by_ids(chunk_ids)

        # 3. Reranking
        if self.reranker:
            try:
                doc_texts = []
                valid_results = []
                chunk_map_temp = {c.chunk_id: c.chunk_text for c in chunks_data}
                
                for r in results:
                    txt = chunk_map_temp.get(r.chunk_id)
                    if txt:
                        doc_texts.append(txt)
                        valid_results.append(r)
                
                if valid_results:
                    results = self.reranker.rerank(query, doc_texts, valid_results)
                    results = results[:top_k]
            except Exception as e:
                print(f"Reranking failed (falling back to hybrid scores): {e}")
                results = results[:top_k]

        # 4. Assembling Docs + Chunks
        seen_hashes = set()
        doc_ids = list(set([c.doc_id for c in chunks_data if c]))
        docs_data = self.storage.get_documents_by_ids(doc_ids)
        seen_docs = set()

        for res in results:
            c_data = next((c for c in chunks_data if c.chunk_id == res.chunk_id), None)
            if not c_data: continue
            
            if c_data.doc_id not in seen_docs:
                doc = next((d for d in docs_data if d.doc_id == c_data.doc_id), None)
                if doc and getattr(doc, "summary", None):
                    context_chunks.append(ContextChunk(
                        chunk_id=f"summary_{doc.doc_id}",
                        text=f"[DOCUMENT SUMMARY]\n{doc.summary}",
                        score=res.score + 0.05
                    ))
                seen_docs.add(c_data.doc_id)
            
            h = hash(c_data.chunk_text.strip())
            if h in seen_hashes: continue
            seen_hashes.add(h)
            
            context_chunks.append(ContextChunk(
                chunk_id=res.chunk_id,
                text=c_data.chunk_text,
                score=res.score
            ))
            
        return context_chunks
