"""
Per-user chat history persistence.

Lives in the same database as user accounts (see auth.py): SQLite locally,
hosted Postgres when DATABASE_URL is set. Two tables:

- conversations: one row per chat session, owned by a username.
- messages:      ordered user/assistant turns within a conversation.

Both the FastAPI backend (app/main.py) and the Streamlit frontend
(frontend_streamlit.py) write through record_exchange(), so history is
captured no matter which entry point served the chat.

Ownership is enforced here: every read/write takes the acting username and
refuses to touch conversations owned by someone else.
"""
from datetime import datetime, timezone

from auth import _IS_POSTGRES, _db, _normalize_username, _sql

TITLE_MAX_CHARS = 36


def _utcnow_text() -> str:
    # Microsecond precision so ORDER BY updated_at is stable even for
    # conversations touched within the same second.
    return datetime.now(timezone.utc).isoformat()


def init_db() -> None:
    """Create history tables if they do not exist. Safe to call repeatedly."""
    if _IS_POSTGRES:
        messages_ddl = """
            CREATE TABLE IF NOT EXISTS messages (
                id SERIAL PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """
    else:
        messages_ddl = """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """
    conversations_ddl = """
        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            username TEXT NOT NULL,
            title TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """
    with _db() as conn:
        conn.execute(conversations_ddl)
        conn.execute(messages_ddl)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_conversations_username "
            "ON conversations(username)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_messages_conversation "
            "ON messages(conversation_id)"
        )


def _derive_title(first_message: str) -> str:
    title = " ".join((first_message or "").split())
    if len(title) > TITLE_MAX_CHARS:
        title = title[:TITLE_MAX_CHARS] + "..."
    return title or "Untitled Chat"


def owner_of(conversation_id: str) -> str | None:
    """Return the owning username, or None if the conversation is unknown."""
    if not conversation_id:
        return None
    init_db()
    with _db() as conn:
        row = conn.execute(
            _sql("SELECT username FROM conversations WHERE id = ?"),
            (conversation_id,),
        ).fetchone()
    return row["username"] if row else None


def record_exchange(
    username: str,
    conversation_id: str,
    user_message: str,
    assistant_reply: str,
) -> None:
    """Append one user/assistant turn, creating the conversation on first use.

    Raises PermissionError if the conversation belongs to another user.
    """
    username = _normalize_username(username)
    now = _utcnow_text()
    init_db()
    with _db() as conn:
        row = conn.execute(
            _sql("SELECT username FROM conversations WHERE id = ?"),
            (conversation_id,),
        ).fetchone()
        if row is None:
            conn.execute(
                _sql(
                    "INSERT INTO conversations (id, username, title, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?)"
                ),
                (conversation_id, username, _derive_title(user_message), now, now),
            )
        elif row["username"] != username:
            raise PermissionError("Conversation belongs to another user.")
        else:
            conn.execute(
                _sql("UPDATE conversations SET updated_at = ? WHERE id = ?"),
                (now, conversation_id),
            )
        for role, content in (("user", user_message), ("assistant", assistant_reply)):
            conn.execute(
                _sql(
                    "INSERT INTO messages (conversation_id, role, content, created_at) "
                    "VALUES (?, ?, ?, ?)"
                ),
                (conversation_id, role, content, now),
            )


def list_conversations(username: str) -> list[dict]:
    """Conversations owned by username, most recently active first."""
    username = _normalize_username(username)
    init_db()
    with _db() as conn:
        rows = conn.execute(
            _sql(
                "SELECT c.id, c.title, c.created_at, c.updated_at, "
                "  (SELECT COUNT(*) FROM messages m WHERE m.conversation_id = c.id) "
                "    AS message_count "
                "FROM conversations c WHERE c.username = ? "
                "ORDER BY c.updated_at DESC"
            ),
            (username,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_messages(username: str, conversation_id: str) -> list[dict] | None:
    """Ordered messages of a conversation the user owns, else None."""
    username = _normalize_username(username)
    init_db()
    with _db() as conn:
        row = conn.execute(
            _sql("SELECT username FROM conversations WHERE id = ?"),
            (conversation_id,),
        ).fetchone()
        if row is None or row["username"] != username:
            return None
        rows = conn.execute(
            _sql(
                "SELECT role, content, created_at FROM messages "
                "WHERE conversation_id = ? ORDER BY id"
            ),
            (conversation_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def delete_conversation(username: str, conversation_id: str) -> bool:
    """Delete a conversation the user owns, with its messages."""
    username = _normalize_username(username)
    init_db()
    with _db() as conn:
        cur = conn.execute(
            _sql("DELETE FROM conversations WHERE id = ? AND username = ?"),
            (conversation_id, username),
        )
        if cur.rowcount == 0:
            return False
        conn.execute(
            _sql("DELETE FROM messages WHERE conversation_id = ?"),
            (conversation_id,),
        )
    return True


def delete_all_for_user(username: str) -> None:
    """Remove every conversation and message owned by username."""
    username = _normalize_username(username)
    init_db()
    with _db() as conn:
        conn.execute(
            _sql(
                "DELETE FROM messages WHERE conversation_id IN "
                "(SELECT id FROM conversations WHERE username = ?)"
            ),
            (username,),
        )
        conn.execute(
            _sql("DELETE FROM conversations WHERE username = ?"), (username,)
        )
