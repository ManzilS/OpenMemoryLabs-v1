from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Tuple, Optional

from oml.config import DEFAULT_MODEL
from oml.storage.factory import get_storage
from oml.retrieval.unified import UnifiedRetriever
from oml.memory.context import ContextBudgeter, ContextChunk
from oml.memory.assembler import ContextAssembler

class QueryPipeline:
    """
    Orchestrates the retrieval process: Hybrid Search -> Reranking -> Graph/Note Injection -> Budgeting.
    Designed to be run programmatically, independently of the CLI.
    """
    
    def __init__(
        self,
        storage_type: str,
        artifacts_dir: str = "artifacts"
    ):
        self.storage_type = storage_type
        self.artifacts_dir = Path(artifacts_dir)
        self.storage = get_storage(storage_type)
        self.retriever = UnifiedRetriever(self.storage_type, self.artifacts_dir)

    def run(
        self,
        query: str,
        top_k: int = 5,
        alpha: float = 0.5,
        budget: Optional[int] = None,
        use_rerank: bool = True,
        use_hyde: bool = False,
        use_graph: bool = False,
        model_name: str = DEFAULT_MODEL
    ) -> Tuple[List[ContextChunk], Optional[str], Optional[int]]:
        """
        Runs the full query pipeline and returns the final context chunks.
        Optionally returns the packed prompt and token count if a budget is provided.
        """
        vector_query = None
        if use_hyde:
            from oml.retrieval.hyde import generate_hypothetical_document
            vector_query = generate_hypothetical_document(query, model_name)

        candidate_k = top_k * 5 if use_rerank else top_k
        # Run main search, note search (and optionally graph search) concurrently
        graph_context = None
        with ThreadPoolExecutor(max_workers=3) as _pool:
            _search_fut = _pool.submit(
                self.retriever.search,
                query,
                candidate_k,
                alpha,
                vector_query=vector_query,
            )
            _notes_fut = _pool.submit(self.retriever.search_notes, query, 2)

            if use_graph:
                from oml.retrieval.graph_retriever import GraphRetriever
                g_retriever = GraphRetriever(self.artifacts_dir)
                _graph_fut = _pool.submit(g_retriever.search_graph, query, model_name)
            else:
                _graph_fut = None

            results      = _search_fut.result()
            note_results = _notes_fut.result()
            if _graph_fut is not None:
                graph_context = _graph_fut.result()

        if not results:
            return [], None, None

        reranker = None
        if use_rerank:
            from oml.retrieval.rerank import Reranker
            reranker = Reranker()

        assembler = ContextAssembler(self.storage, self.storage_type, reranker)
        context_chunks = assembler.assemble(
            query=query, 
            results=results, 
            note_results=note_results, 
            top_k=top_k, 
            graph_context=graph_context
        )

        prompt = None
        approx_tokens = None
        
        if budget:
            budgeter = ContextBudgeter()
            prompt, approx_tokens = budgeter.construct_prompt_with_tokens(
                query, context_chunks, max_tokens=budget
            )
            
        return context_chunks, prompt, approx_tokens
