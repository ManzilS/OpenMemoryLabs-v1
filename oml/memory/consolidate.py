import json
from pathlib import Path
from typing import List
from oml.config import DEFAULT_MODEL

# Optional: rich for progress bars
try:
    from rich.progress import track
except ImportError:
    def track(iter, description="Processing..."):
        print(description)
        return iter

from sqlalchemy import select

from oml.models.schema import MemoryNote, Document
from oml.storage.sqlite import SQLiteStorage, documents_table, memory_notes_table
from oml.llm import get_llm_client
from oml.retrieval.vector import VectorIndex

def get_unconsolidated_threads(storage: SQLiteStorage) -> List[str]:
    """Find thread_ids that have documents but no memory note."""
    engine = storage.engine
    
    with engine.connect() as conn:
        # Get all doc threads
        doc_stmt = select(documents_table.c.thread_id).distinct().where(documents_table.c.thread_id != None)
        all_threads = set([r[0] for r in conn.execute(doc_stmt) if r[0]])
        
        # Get existing note threads
        note_stmt = select(memory_notes_table.c.thread_id).distinct()
        existing_threads = set([r[0] for r in conn.execute(note_stmt)])
        
    return list(all_threads - existing_threads)

def get_thread_content(storage: SQLiteStorage, thread_id: str) -> List[Document]:
    """Fetch all documents for a specific thread."""
    engine = storage.engine
    stmt = select(documents_table).where(documents_table.c.thread_id == thread_id).order_by(documents_table.c.timestamp)
    
    docs = []
    with engine.connect() as conn:
        rows = conn.execute(stmt).all()
        for row in rows:
            recipients = json.loads(row.recipients) if row.recipients else []
            docs.append(Document(
                doc_id=row.doc_id,
                source=row.source,
                timestamp=row.timestamp,
                author=row.author,
                recipients=recipients,
                subject=row.subject,
                thread_id=row.thread_id,
                raw_text=row.raw_text,
                clean_text=row.clean_text
            ))
    return docs

def consolidate_threads(
    db_path: str,
    model_name: str = DEFAULT_MODEL, 
    limit: int = 10,
    batch_size: int = 1,
    artifacts_dir: str = "artifacts",
):
    """
    Main loop:
    1. Find unconsolidated threads
    2. Fetch content
    3. LLM Summarize
    4. Save Note
    """
    storage = SQLiteStorage(db_path)
    
    print(f"Checking for unconsolidated threads in {db_path}...")
    threads = get_unconsolidated_threads(storage)
    
    if not threads:
        print("No new threads to consolidate.")
        return
        
    print(f"Found {len(threads)} threads pending consolidation.")
    
    # Apply limit
    if limit:
        threads = threads[:limit]
        print(f"Processing top {limit} threads...")

    llm = get_llm_client(model_name)
    
    new_notes = []
    
    for thread_id in track(threads, description="Consolidating..."):
        try:
            docs = get_thread_content(storage, thread_id)
            if not docs:
                continue
                
            # Prepare Context for LLM
            # Concatenate emails chronologically
            context_text = ""
            for d in docs:
                context_text += f"From: {d.author}\nTo: {d.recipients}\nDate: {d.timestamp}\nSubject: {d.subject}\n\n{d.clean_text}\n\n---\n\n"
            
            # Truncate if too long (simple char limit for now, better to use tokenizer later)
            if len(context_text) > 12000:
                context_text = context_text[:12000] + "...(truncated)"
                
            prompt = f"""
            You are a helpful assistant organizing an email archive.
            Summarize the following email thread into 3-5 concise bullet points capturing the key events, decisions, and people involved.
            
            EMAIL THREAD:
            {context_text}
            
            SUMMARY:
            """
            
            summary = llm.generate(prompt)
            
            note = MemoryNote(
                thread_id=thread_id,
                content=summary.strip(),
                source_doc_ids=[d.doc_id for d in docs],
                # Use timestamp of last email as note timestamp
                timestamp=docs[-1].timestamp if docs else None
            )
            
            new_notes.append(note)
            
            # Batch save
            if len(new_notes) >= batch_size:
                storage.upsert_notes(new_notes)
                new_notes = []
                
        except Exception as e:
            print(f"Failed to consolidate thread {thread_id}: {e}")
            
    # Flush remaining
    if new_notes:
        storage.upsert_notes(new_notes)
    
    print("Consolidation complete.")
    
    # Rebuild Index
    rebuild_note_index(storage, artifacts_dir=artifacts_dir)

def rebuild_note_index(storage: SQLiteStorage, artifacts_dir: str = "artifacts"):
    """Rebuilds the MemoryNote vector index."""
    print("Rebuilding MemoryNote Index...")
    notes = storage.get_all_notes()
    
    if not notes:
        print("No notes to index.")
        return
        
    artifacts_path = Path(artifacts_dir)
    artifacts_path.mkdir(parents=True, exist_ok=True)
    index_path = artifacts_path / "notes_vector.index"
    map_path = artifacts_path / "notes_vector_map.json"
    
    # Initialize Index
    # Use same model as vector index for simplicity
    v_index = VectorIndex(index_path, map_path)
    
    # Prepare data
    ids = [n.note_id for n in notes]
    texts = [f"Summary of thread {n.thread_id}: {n.content}" for n in notes]
    
    # Build
    v_index.build(ids, texts)
    v_index.save()
    print("MemoryNote Index updated.")
