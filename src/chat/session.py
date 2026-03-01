"""In-memory session management for chat interactions."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Session:
    """A locate + chat session."""

    id: str
    image_path: str
    pipeline_state: dict[str, Any] = field(default_factory=dict)
    messages: list[dict[str, Any]] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)

    def add_message(self, role: str, content: str, metadata: dict | None = None):
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": time.time(),
            "metadata": metadata or {},
        })
        self.last_active = time.time()

    @property
    def candidates(self) -> list[dict]:
        return self.pipeline_state.get("ranked_candidates", [])

    @property
    def prediction(self) -> dict:
        return self.pipeline_state.get("prediction", {})


class SessionManager:
    """In-memory session store with TTL-based cleanup."""

    def __init__(self, ttl_seconds: int = 3600):
        self._sessions: dict[str, Session] = {}
        self._ttl = ttl_seconds

    def create(self, image_path: str, pipeline_state: dict | None = None) -> Session:
        """Create and store a new session."""
        self._cleanup()
        session = Session(
            id=str(uuid.uuid4()),
            image_path=image_path,
            pipeline_state=pipeline_state or {},
        )
        self._sessions[session.id] = session
        return session

    def get(self, session_id: str) -> Optional[Session]:
        """Retrieve a session by ID (returns None if expired/missing)."""
        self._cleanup()
        return self._sessions.get(session_id)

    def update(self, session_id: str, pipeline_state: dict) -> Optional[Session]:
        """Update session pipeline state."""
        session = self._sessions.get(session_id)
        if session:
            session.pipeline_state = pipeline_state
            session.last_active = time.time()
        return session

    def delete(self, session_id: str) -> bool:
        return self._sessions.pop(session_id, None) is not None

    def _cleanup(self):
        """Remove expired sessions."""
        now = time.time()
        expired = [
            sid for sid, s in self._sessions.items()
            if now - s.last_active > self._ttl
        ]
        for sid in expired:
            del self._sessions[sid]

    @property
    def active_count(self) -> int:
        return len(self._sessions)
