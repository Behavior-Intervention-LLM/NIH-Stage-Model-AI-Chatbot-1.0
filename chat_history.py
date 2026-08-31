"""
Per-user chat history persistence.

Lives in the same database as user accounts (see auth.py): SQLite locally,
hosted Postgres when DATABASE_URL is set. Three tables:

- conversations: one row per chat session, owned by a username.
- messages:      ordered user/assistant turns within a conversation.
- attachments:   text of files uploaded into a conversation, scoped to it.

Both the FastAPI backend (app/main.py) and the Streamlit frontend
(frontend_streamlit.py) write through record_exchange(), so history is
captured no matter which entry point served the chat.

Ownership is enforced here: every read/write takes the acting username and
refuses to touch conversations owned by someone else.

Attachments are conversation working context, not corpus material: they are
never indexed into the vector store (that is load_documents.py's job, run
deliberately against data/documents/). They are persisted here only so a
process restart cannot silently empty the conversation mid-thread. Before
this table, they lived solely in the in-process dict in
app/core/state_store.py, so a restart between "here is my draft" and
"rewrite my aims page" erased the draft with no error and no log, and the
model answered from conversation memory as though it still had the file.
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
    # `username` is carried on the row itself rather than resolved through
    # conversations.username: on the first turn of a chat the conversation row
    # does not exist yet (record_exchange creates it at the end of the turn),
    # so an ownership check that joins through it falls open exactly when the
    # attachment is first stored.
    if _IS_POSTGRES:
        attachments_ddl = """
            CREATE TABLE IF NOT EXISTS attachments (
                id SERIAL PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                username TEXT NOT NULL,
                name TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """
    else:
        attachments_ddl = """
            CREATE TABLE IF NOT EXISTS attachments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                username TEXT NOT NULL,
                name TEXT NOT NULL,
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
        conn.execute(attachments_ddl)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_conversations_username "
            "ON conversations(username)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_messages_conversation "
            "ON messages(conversation_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_attachments_conversation "
            "ON attachments(conversation_id, username)"
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


def save_attachments(
    username: str, conversation_id: str, items: list[dict]
) -> int:
    """Persist uploaded file text against a conversation. Returns rows added.

    Re-saving identical (name, content) is a no-op, mirroring
    SessionState.add_attachment(), so re-sending the same file on every turn
    (which the Streamlit uploader does while the widget holds the file) does
    not accumulate duplicate rows.

    Raises PermissionError if the conversation, or an attachment already on
    it, belongs to another user. An unknown conversation with no attachments
    yet is allowed: on the first turn the conversation row does not exist
    until record_exchange() creates it at the end of the turn.
    """
    username = _normalize_username(username)
    rows = [
        (str(i.get("name") or "attached document"), str(i.get("text") or ""))
        for i in (items or [])
    ]
    rows = [(n, c) for n, c in rows if c.strip()]
    if not conversation_id or not rows:
        return 0

    init_db()
    now = _utcnow_text()
    added = 0
    with _db() as conn:
        owner = conn.execute(
            _sql("SELECT username FROM conversations WHERE id = ?"),
            (conversation_id,),
        ).fetchone()
        if owner is not None and owner["username"] != username:
            raise PermissionError("Conversation belongs to another user.")

        # Second gate, for the first turn where no conversation row exists
        # yet: whoever attached first owns the conversation's attachments.
        held = conn.execute(
            _sql(
                "SELECT DISTINCT username FROM attachments WHERE conversation_id = ?"
            ),
            (conversation_id,),
        ).fetchall()
        if any(r["username"] != username for r in held):
            raise PermissionError("Conversation belongs to another user.")

        existing = {
            (r["name"], r["content"])
            for r in conn.execute(
                _sql(
                    "SELECT name, content FROM attachments "
                    "WHERE conversation_id = ? AND username = ?"
                ),
                (conversation_id, username),
            ).fetchall()
        }
        for name, content in rows:
            if (name, content) in existing:
                continue
            existing.add((name, content))
            conn.execute(
                _sql(
                    "INSERT INTO attachments "
                    "(conversation_id, username, name, content, created_at) "
                    "VALUES (?, ?, ?, ?, ?)"
                ),
                (conversation_id, username, name, content, now),
            )
            added += 1
    return added


def load_attachments(username: str, conversation_id: str) -> list[dict]:
    """Files attached to a conversation the user owns, oldest first.

    Returns [{"name", "text"}] shaped for SessionState.add_attachment(). An
    unknown conversation yields [] rather than raising: a brand-new session
    legitimately has nothing stored yet. Ownership is filtered in the query,
    so another user's files are invisible rather than merely refused.
    """
    username = _normalize_username(username)
    if not conversation_id:
        return []
    init_db()
    with _db() as conn:
        rows = conn.execute(
            _sql(
                "SELECT name, content FROM attachments "
                "WHERE conversation_id = ? AND username = ? ORDER BY id"
            ),
            (conversation_id, username),
        ).fetchall()
    return [{"name": r["name"], "text": r["content"]} for r in rows]


def clear_attachments(username: str, conversation_id: str) -> int:
    """Detach every file from a conversation the user owns. Returns rows removed."""
    username = _normalize_username(username)
    if not conversation_id:
        return 0
    init_db()
    with _db() as conn:
        cur = conn.execute(
            _sql(
                "DELETE FROM attachments WHERE conversation_id = ? AND username = ?"
            ),
            (conversation_id, username),
        )
        return cur.rowcount or 0


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
        conn.execute(
            _sql("DELETE FROM attachments WHERE conversation_id = ?"),
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
            _sql(
                "DELETE FROM attachments WHERE conversation_id IN "
                "(SELECT id FROM conversations WHERE username = ?)"
            ),
            (username,),
        )
        conn.execute(
            _sql("DELETE FROM conversations WHERE username = ?"), (username,)
        )
