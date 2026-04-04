import sqlite3
import json
import dataclasses
from pathlib import Path
from typing import List, Dict, Any
from oml.models.events import Event
from oml.config import DEFAULT_EVENTS_DB_PATH

class EventStore:
    def __init__(self, db_path: str = DEFAULT_EVENTS_DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._init_db()

    def _init_db(self):
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id TEXT PRIMARY KEY,
                timestamp REAL,
                type TEXT,
                session_id TEXT,
                payload TEXT
            )
        """)
        self.conn.commit()

    def log_event(self, event: Event):
        cursor = self.conn.cursor()
        payload = json.dumps(dataclasses.asdict(event))
        cursor.execute(
            "INSERT INTO events (id, timestamp, type, session_id, payload) VALUES (?, ?, ?, ?, ?)",
            (event.id, event.timestamp, event.type, event.session_id, payload)
        )
        self.conn.commit()

    def get_session_events(self, session_id: str) -> List[Dict[str, Any]]:
        cursor = self.conn.cursor()
        cursor.execute("SELECT payload FROM events WHERE session_id = ? ORDER BY timestamp ASC", (session_id,))
        rows = cursor.fetchall()
        return [json.loads(row[0]) for row in rows]
        
    def get_all_sessions(self) -> List[str]:
        cursor = self.conn.cursor()
        cursor.execute("SELECT DISTINCT session_id FROM events ORDER BY timestamp DESC")
        return [row[0] for row in cursor.fetchall()]
