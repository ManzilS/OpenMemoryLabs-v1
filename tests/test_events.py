import pytest
from pathlib import Path
from oml.models.events import ChatEvent, RetrievalEvent
from oml.storage.events import EventStore

@pytest.fixture
def temp_db(tmp_path):
    db_file = tmp_path / "test_events.db"
    return str(db_file)

def test_event_store_initialization(temp_db):
    store = EventStore(temp_db)
    assert Path(temp_db).exists()

def test_chat_event_logging(temp_db):
    store = EventStore(temp_db)
    
    event = ChatEvent(
        session_id="session_123",
        user_message="Hello",
        llm_response="Hi there"
    )
    
    store.log_event(event)
    
    events = store.get_session_events("session_123")
    assert len(events) == 1
    
    loaded_event = events[0]
    assert loaded_event["type"] == "chat_turn"
    assert loaded_event["session_id"] == "session_123"
    assert "user_message" in loaded_event
    assert loaded_event["llm_response"] == "Hi there"

def test_retrieval_event_logging(temp_db):
    store = EventStore(temp_db)
    
    event = RetrievalEvent(
        session_id="session_456",
        query="What is OML?",
        retrieved_chunk_ids=["chunk_1", "chunk_2"],
        strategies_used=["hybrid", "graph"]
    )
    
    store.log_event(event)
    
    events = store.get_session_events("session_456")
    assert len(events) == 1
    
    loaded = events[0]
    assert loaded["type"] == "retrieval"
    assert loaded["session_id"] == "session_456"
    assert loaded["query"] == "What is OML?"
    assert len(loaded["retrieved_chunk_ids"]) == 2
    assert "graph" in loaded["strategies_used"]

def test_get_all_sessions(temp_db):
    store = EventStore(temp_db)
    
    store.log_event(ChatEvent(session_id="sess_A"))
    store.log_event(RetrievalEvent(session_id="sess_B"))
    store.log_event(ChatEvent(session_id="sess_A"))
    
    sessions = store.get_all_sessions()
    assert len(sessions) == 2
    assert "sess_A" in sessions
    assert "sess_B" in sessions
