from dataclasses import dataclass, field
from typing import List, Dict, Any
import time
import uuid

@dataclass
class Event:
    type: str = ""
    session_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

@dataclass
class ChatEvent(Event):
    user_message: str = ""
    llm_response: str = ""
    def __post_init__(self):
        if not self.type:
            self.type = "chat_turn"

@dataclass
class RetrievalEvent(Event):
    query: str = ""
    retrieved_chunk_ids: List[str] = field(default_factory=list)
    strategies_used: List[str] = field(default_factory=list)
    def __post_init__(self):
        if not self.type:
            self.type = "retrieval"
