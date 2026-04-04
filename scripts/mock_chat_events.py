from oml.storage.events import EventStore
from oml.models.events import ChatEvent, RetrievalEvent
import uuid
import time
import subprocess

def run():
    print("Generating simulated RAG session events...")
    store = EventStore("data/oml_events.db")
    session_id = str(uuid.uuid4())
    
    # 1. Retrieval Event
    r_event = RetrievalEvent(
        type="retrieval",
        session_id=session_id,
        query="Tell me about OpenMemoryLab's architecture.",
        retrieved_chunk_ids=["chunk_1_hybrid", "chunk_2_rerank"],
        strategies_used=["hybrid", "rerank", "hyde"]
    )
    store.log_event(r_event)
    time.sleep(1)
    
    # 2. Chat Event
    c_event = ChatEvent(
        type="chat_turn",
        session_id=session_id,
        user_message="Tell me about OpenMemoryLab's architecture.",
        llm_response="OpenMemoryLab uses a hybrid retrieval system with SQLite and LanceDB storage backends, enhanced by Cross-Encoder reranking."
    )
    store.log_event(c_event)
    print("Events logged successfully.")
    
    # Run the audit
    print("\nRunning Audit Script...")
    subprocess.run(["uv", "run", "--python", "3.11", "scripts/run_event_audit.py"])

if __name__ == "__main__":
    run()
