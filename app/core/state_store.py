"""
Session state store.
Primary backend: Redis
Fallback backend: in-memory
"""
from __future__ import annotations

import uuid
from typing import Dict, Optional

from app.config import settings
from app.core.types import SessionState

try:
    import redis
except Exception:  # pragma: no cover - handled by fallback behavior
    redis = None


class InMemoryStateStore:
    """Simple in-memory state store (fallback)."""

    def __init__(self):
        self._sessions: Dict[str, SessionState] = {}

    def get_state(self, session_id: str) -> Optional[SessionState]:
        return self._sessions.get(session_id)

    def create_state(self, session_id: Optional[str] = None) -> SessionState:
        if session_id is None:
            session_id = str(uuid.uuid4())

        state = SessionState(session_id=session_id)
        self._sessions[session_id] = state
        return state

    def save_state(self, state: SessionState):
        self._sessions[state.session_id] = state

    def delete_state(self, session_id: str):
        if session_id in self._sessions:
            del self._sessions[session_id]

    def list_sessions(self) -> list[str]:
        return list(self._sessions.keys())


class RedisStateStore:
    """Redis-backed state store with JSON serialization."""

    def __init__(self, redis_url: str, ttl_seconds: int, key_prefix: str = "nih_chatbot"):
        if redis is None:
            raise RuntimeError("redis package is not installed.")
        self._client = redis.from_url(
            redis_url,
            decode_responses=True,
            socket_timeout=2,
            socket_connect_timeout=2,
            health_check_interval=30,
        )
        self._ttl_seconds = max(0, int(ttl_seconds))
        self._key_prefix = key_prefix.strip() or "nih_chatbot"

    def _key(self, session_id: str) -> str:
        return f"{self._key_prefix}:session:{session_id}"

    def get_state(self, session_id: str) -> Optional[SessionState]:
        raw = self._client.get(self._key(session_id))
        if not raw:
            return None
        try:
            return SessionState.model_validate_json(raw)
        except Exception:
            # Corrupted state should not break the request path.
            return None

    def create_state(self, session_id: Optional[str] = None) -> SessionState:
        if session_id is None:
            session_id = str(uuid.uuid4())
        state = SessionState(session_id=session_id)
        self.save_state(state)
        return state

    def save_state(self, state: SessionState):
        payload = state.model_dump_json()
        key = self._key(state.session_id)
        if self._ttl_seconds > 0:
            self._client.setex(key, self._ttl_seconds, payload)
        else:
            self._client.set(key, payload)

    def delete_state(self, session_id: str):
        self._client.delete(self._key(session_id))

    def list_sessions(self) -> list[str]:
        pattern = f"{self._key_prefix}:session:*"
        ids: list[str] = []
        for key in self._client.scan_iter(match=pattern, count=200):
            ids.append(key.rsplit(":", 1)[-1])
        return ids


def _build_state_store():
    redis_url = (settings.REDIS_URL or "").strip()
    if redis_url and redis is not None:
        try:
            store = RedisStateStore(
                redis_url=redis_url,
                ttl_seconds=settings.STATE_TTL_SECONDS,
                key_prefix=settings.REDIS_KEY_PREFIX,
            )
            # Fail fast if Redis is unreachable.
            store._client.ping()
            return store
        except Exception:
            pass
    return InMemoryStateStore()


# Global state store singleton
state_store = _build_state_store()
