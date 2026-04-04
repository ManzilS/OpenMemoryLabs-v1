from pathlib import Path
from oml.storage.events import EventStore
import json

LOG_DIR = Path("logs/events")
LOG_DIR.mkdir(parents=True, exist_ok=True)

def run():
    print("Connecting to Event Store...")
    store = EventStore("data/oml_events.db")
    
    sessions = store.get_all_sessions()
    if not sessions:
        print("No sessions found in the event store. Please run `oml chat` and ask a question first to generate data.")
        return
        
    session_id = sessions[0]  # Get the most recent session
    print(f"Auditing latest session: {session_id}\n")
    
    events = store.get_session_events(session_id)
    
    log_path = LOG_DIR / f"audit_session_{session_id}.txt"
    with log_path.open("w", encoding="utf-8") as f:
        f.write(f"--- EVENT AUDIT LOG FOR SESSION {session_id} ---\n\n")
        
        for i, event in enumerate(events, 1):
            e_type = event.get("type", "unknown")
            timestamp = event.get("timestamp", 0)
            f.write(f"[{i}] Event: {e_type.upper()} | Time: {timestamp}\n")
            
            if e_type == "retrieval":
                query = event.get("query", "")
                chunks = event.get("retrieved_chunk_ids", [])
                strategies = event.get("strategies_used", [])
                f.write(f"  User Intention: '{query}'\n")
                f.write(f"  Strategies: {strategies}\n")
                f.write(f"  Retrieved {len(chunks)} chunks: {chunks[:3]}...\n")
                
            elif e_type == "chat_turn":
                msg = event.get("user_message", "")
                resp = event.get("llm_response", "")
                f.write(f"  User: {msg}\n")
                # Preview response
                preview = resp[:150].replace('\n', ' ') + "..." if len(resp) > 150 else resp.replace('\n', ' ')
                f.write(f"  Agent: {preview}\n")
                
            f.write("\n")
            
    print(f"Audit log successfully transcribed at {log_path}")

if __name__ == "__main__":
    run()
