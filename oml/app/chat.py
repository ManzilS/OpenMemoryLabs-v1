import typer
from pathlib import Path
from typing import Optional, Tuple
import uuid
import os

from oml.retrieval.unified import UnifiedRetriever
from oml.retrieval.rerank import Reranker
from oml.storage.factory import get_storage
from oml.memory.context import ContextBudgeter
from oml.memory.assembler import ContextAssembler
from oml.llm import get_llm_client
from oml.storage.events import EventStore
from oml.models.events import ChatEvent, RetrievalEvent
from oml.config import DEFAULT_STORAGE

class ChatSession:
    """
    Maintains state and orchestrates interactions for a RAG chat session.
    Designed to be used by any UI (CLI, Web, API).
    """

    def __init__(
        self,
        model: str,
        storage_type: str = DEFAULT_STORAGE,
        artifacts_dir: str = "artifacts",
        event_db_path: Optional[str] = None,
    ):
        self.model = model
        self.storage_type = storage_type
        self.session_id = str(uuid.uuid4())
        
        self.storage = get_storage(storage_type)
        self.artifacts_dir = Path(artifacts_dir)
        self.retriever = UnifiedRetriever(self.storage_type, self.artifacts_dir)
        
        self.llm = get_llm_client(model)
        self.budgeter = ContextBudgeter()
        
        self.reranker = None
        try:
            self.reranker = Reranker()
        except Exception as e:
            print(f"Warning: Reranker failed to load ({e}). Using standard hybrid search.")
            
        resolved_event_db = event_db_path or os.getenv("OML_EVENTS_DB", "data/oml_events.db")
        self.event_store = EventStore(db_path=resolved_event_db)
        
    def send_message(
        self, 
        query: str, 
        top_k: int = 5, 
        alpha: float = 0.5, 
        budget: int = 4000
    ) -> Tuple[str, str, int]:
        """
        Processes a user message through the RAG pipeline.
        Returns: (AI Response, Packed Prompt, Approx Tokens Used)
        """
        # 1. Retrieve Candidates
        candidate_k = top_k * 5 if self.reranker else top_k
        results = self.retriever.search(query, top_k=candidate_k, alpha=alpha)
        note_results = self.retriever.search_notes(query, top_k=2)
        
        # 2. Context Construction
        assembler = ContextAssembler(self.storage, self.storage_type, self.reranker)
        context_chunks = assembler.assemble(query, results, note_results, top_k=top_k)
                 
        # 3. Log Retrieval Event
        try:
            r_chunk_ids = [r.chunk_id for r in results]
            strategies = ["hybrid"]
            if self.reranker: strategies.append("rerank")
             
            retrieval_event = RetrievalEvent(
                session_id=self.session_id,
                query=query,
                retrieved_chunk_ids=r_chunk_ids,
                strategies_used=strategies
            )
            self.event_store.log_event(retrieval_event)
        except Exception as e:
            print(f"Failed to log retrieval event: {e}")
        
        # 4. Prompt Packing
        packed_prompt, approx_tokens = self.budgeter.construct_prompt_with_tokens(
            query, context_chunks, max_tokens=budget
        )
        
        # 5. Generate Response
        response = self.llm.generate(packed_prompt)
        
        # 6. Log Chat Event
        try:
            chat_event = ChatEvent(
                session_id=self.session_id,
                user_message=query,
                llm_response=response
            )
            self.event_store.log_event(chat_event)
        except Exception as e:
             print(f"Failed to log chat event: {e}")
             
        return response, packed_prompt, approx_tokens


def chat_loop(
    model: str,
    top_k: int = 5,
    alpha: float = 0.5,
    budget: int = 4000,
    show_prompt: bool = False,
    show_tokens: bool = False,
    storage_type: str = DEFAULT_STORAGE,
):
    """
    Interactive RAG Chat Loop UI Driver.
    """
    typer.echo(f"Initializing Chat with model: {model}...")
    
    try:
        session = ChatSession(model=model, storage_type=storage_type)
    except Exception as e:
        typer.secho(str(e), fg=typer.colors.RED)
        raise typer.Exit(1)
        
    print(f"Session ID: {session.session_id}", flush=True)
    print("Ready! Type 'exit' or 'quit' to stop.", flush=True)
    print("--------------------------------------", flush=True)

    while True:
        try:
            print("You: ", end="", flush=True)
            query = input()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...", flush=True)
            break
            
        if query.lower() in ["exit", "quit"]:
            break
            
        print("Thinking...", flush=True)
        
        try:
            response, packed_prompt, approx_tokens = session.send_message(
                query=query,
                top_k=top_k,
                alpha=alpha,
                budget=budget
            )
            
            if show_prompt:
                typer.secho("\n=== Packed Prompt ===", fg=typer.colors.MAGENTA)
                print(packed_prompt)
                typer.secho("=====================\n", fg=typer.colors.MAGENTA)

            if show_tokens:
                typer.secho(
                    f"[ContextBudget] Approx tokens used: {approx_tokens}/{budget}",
                    fg=typer.colors.CYAN,
                )
                
            typer.secho(f"\nAI: {response}", fg=typer.colors.GREEN)
        
        except Exception as e:
            typer.secho(f"Error generating response: {e}", fg=typer.colors.RED)
            
        typer.echo("--------------------------------------")
