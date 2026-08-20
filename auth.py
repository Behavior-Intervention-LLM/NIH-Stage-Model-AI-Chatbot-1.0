"""
User authentication.

Storage backend is chosen by DATABASE_URL:
- postgres://... or postgresql://...  -> hosted Postgres (production: Render,
  Neon, Supabase, ...). All app instances share one user database.
- unset                               -> local SQLite file users.db (development).

Security posture is intentionally austere:
- Passwords are never stored or logged; only salted PBKDF2-HMAC-SHA256 hashes.
- Verification uses constant-time comparison.
- Login failures return a single generic message (no username/password hints).
- Repeated failures per username trigger a temporary lockout.
- API tokens are stored hashed (SHA-256), never raw.

CLI (run from the project root; targets whichever backend DATABASE_URL selects):
    python auth.py add-user <username>      # prompts for password, no echo
    python auth.py list-users
    python auth.py delete-user <username>
"""
import hashlib
import hmac
import os
import secrets
import sqlite3
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

DB_PATH = Path(__file__).parent / "users.db"

DATABASE_URL = (os.getenv("DATABASE_URL") or "").strip()
_IS_POSTGRES = DATABASE_URL.startswith(("postgres://", "postgresql://"))

if _IS_POSTGRES:
    import psycopg
    from psycopg.rows import dict_row

PBKDF2_ITERATIONS = 600_000
SALT_BYTES = 16
HASH_NAME = "sha256"

MAX_FAILED_ATTEMPTS = 5
LOCKOUT_SECONDS = 300

MIN_PASSWORD_LENGTH = 8
MAX_PASSWORD_LENGTH = 128
# Lowercase letters, digits, dot, underscore, hyphen; 3-32 chars.
USERNAME_PATTERN = r"^[a-z0-9._-]{3,32}$"

TOKEN_TTL_SECONDS = 12 * 3600


def _sql(query: str) -> str:
    """Translate '?' placeholders to the active driver's style."""
    return query.replace("?", "%s") if _IS_POSTGRES else query


@contextmanager
def _db():
    """Yield a connection; commit on success, always close."""
    if _IS_POSTGRES:
        conn = psycopg.connect(DATABASE_URL, row_factory=dict_row)
    else:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _utcnow_text() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def init_db() -> None:
    """Create tables if they do not exist. Safe to call repeatedly."""
    if _IS_POSTGRES:
        users_ddl = """
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                username TEXT NOT NULL UNIQUE,
                password_hash BYTEA NOT NULL,
                salt BYTEA NOT NULL,
                iterations INTEGER NOT NULL,
                failed_attempts INTEGER NOT NULL DEFAULT 0,
                locked_until DOUBLE PRECISION NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                last_login TEXT
            )
        """
    else:
        users_ddl = """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE COLLATE NOCASE,
                password_hash BLOB NOT NULL,
                salt BLOB NOT NULL,
                iterations INTEGER NOT NULL,
                failed_attempts INTEGER NOT NULL DEFAULT 0,
                locked_until REAL NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                last_login TEXT
            )
        """
    tokens_ddl = """
        CREATE TABLE IF NOT EXISTS auth_tokens (
            token_hash TEXT PRIMARY KEY,
            username TEXT NOT NULL,
            expires_at DOUBLE PRECISION NOT NULL,
            created_at TEXT NOT NULL
        )
    """ if _IS_POSTGRES else """
        CREATE TABLE IF NOT EXISTS auth_tokens (
            token_hash TEXT PRIMARY KEY,
            username TEXT NOT NULL,
            expires_at REAL NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        )
    """
    with _db() as conn:
        conn.execute(users_ddl)
        conn.execute(tokens_ddl)


def _hash_password(password: str, salt: bytes, iterations: int = PBKDF2_ITERATIONS) -> bytes:
    return hashlib.pbkdf2_hmac(HASH_NAME, password.encode("utf-8"), salt, iterations)


def _normalize_username(username: str) -> str:
    # Usernames are case-insensitive across both backends.
    return (username or "").strip().lower()


def create_user(username: str, password: str) -> None:
    """Create a user. Raises ValueError on invalid input or duplicate username."""
    import re

    username = _normalize_username(username)
    if not username:
        raise ValueError("Username cannot be empty.")
    if not re.match(USERNAME_PATTERN, username):
        raise ValueError(
            "Username must be 3-32 characters: letters, digits, dot, underscore, or hyphen."
        )
    if len(password) < MIN_PASSWORD_LENGTH:
        raise ValueError(f"Password must be at least {MIN_PASSWORD_LENGTH} characters.")
    if len(password) > MAX_PASSWORD_LENGTH:
        raise ValueError(f"Password must be at most {MAX_PASSWORD_LENGTH} characters.")

    salt = secrets.token_bytes(SALT_BYTES)
    password_hash = _hash_password(password, salt)

    init_db()
    try:
        with _db() as conn:
            conn.execute(
                _sql(
                    "INSERT INTO users (username, password_hash, salt, iterations, created_at) "
                    "VALUES (?, ?, ?, ?, ?)"
                ),
                (username, password_hash, salt, PBKDF2_ITERATIONS, _utcnow_text()),
            )
    except Exception as exc:
        if "unique" in str(exc).lower() or isinstance(exc, sqlite3.IntegrityError):
            raise ValueError("Username is already taken.")
        raise


def verify_login(username: str, password: str) -> tuple[bool, str]:
    """
    Check credentials. Returns (ok, message).

    The failure message is deliberately generic so it never reveals whether
    the username exists or which field was wrong.
    """
    generic_fail = "Invalid username or password."
    username = _normalize_username(username)
    if not username or not password:
        return False, generic_fail

    init_db()
    with _db() as conn:
        row = conn.execute(
            _sql("SELECT * FROM users WHERE username = ?"), (username,)
        ).fetchone()

        if row is None:
            # Burn comparable time so missing users aren't distinguishable by timing.
            _hash_password(password, b"\x00" * SALT_BYTES)
            return False, generic_fail

        now = time.time()
        if row["locked_until"] > now:
            remaining = int(row["locked_until"] - now) + 1
            return False, f"Account temporarily locked. Try again in {remaining} seconds."

        candidate = _hash_password(password, bytes(row["salt"]), row["iterations"])
        if hmac.compare_digest(candidate, bytes(row["password_hash"])):
            conn.execute(
                _sql(
                    "UPDATE users SET failed_attempts = 0, locked_until = 0, "
                    "last_login = ? WHERE id = ?"
                ),
                (_utcnow_text(), row["id"]),
            )
            return True, "ok"

        failed = row["failed_attempts"] + 1
        locked_until = now + LOCKOUT_SECONDS if failed >= MAX_FAILED_ATTEMPTS else 0
        conn.execute(
            _sql("UPDATE users SET failed_attempts = ?, locked_until = ? WHERE id = ?"),
            (failed % MAX_FAILED_ATTEMPTS if locked_until else failed, locked_until, row["id"]),
        )
        if locked_until:
            return False, f"Too many failed attempts. Account locked for {LOCKOUT_SECONDS // 60} minutes."
        return False, generic_fail


def change_password(username: str, old_password: str, new_password: str) -> tuple[bool, str]:
    ok, msg = verify_login(username, old_password)
    if not ok:
        return False, msg
    if len(new_password) < MIN_PASSWORD_LENGTH:
        return False, f"Password must be at least {MIN_PASSWORD_LENGTH} characters."
    salt = secrets.token_bytes(SALT_BYTES)
    with _db() as conn:
        conn.execute(
            _sql("UPDATE users SET password_hash = ?, salt = ?, iterations = ? WHERE username = ?"),
            (_hash_password(new_password, salt), salt, PBKDF2_ITERATIONS, _normalize_username(username)),
        )
    return True, "Password updated."


def delete_user(username: str) -> bool:
    username = _normalize_username(username)
    init_db()
    with _db() as conn:
        cur = conn.execute(_sql("DELETE FROM users WHERE username = ?"), (username,))
        deleted = cur.rowcount > 0
    if deleted:
        # Lazy import: chat_history imports this module at load time.
        from chat_history import delete_all_for_user

        delete_all_for_user(username)
    return deleted


def list_users() -> list[dict]:
    """Usernames and metadata only — never credential material."""
    init_db()
    with _db() as conn:
        rows = conn.execute(
            "SELECT username, created_at, last_login FROM users ORDER BY username"
        ).fetchall()
    return [dict(r) for r in rows]


def seed_users_from_env() -> list[str]:
    """Bootstrap accounts from SEED_USERS ('alice:pass1,bob:pass2').

    Intended for first deploys: set SEED_USERS in Render env vars or Streamlit
    secrets, boot once, then remove it. Existing usernames are left untouched.
    Returns the usernames created.
    """
    raw = (os.getenv("SEED_USERS") or "").strip()
    if not raw:
        return []
    created = []
    for pair in raw.split(","):
        if ":" not in pair:
            continue
        username, password = pair.split(":", 1)
        try:
            create_user(username, password)
            created.append(_normalize_username(username))
        except ValueError:
            continue  # already exists or invalid — skip silently
    return created


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def issue_token(username: str) -> str:
    """Create an API token for an already-authenticated user.

    The raw token is returned once and never stored — only its SHA-256 hash.
    """
    token = secrets.token_urlsafe(32)
    expires_at = time.time() + TOKEN_TTL_SECONDS
    init_db()
    with _db() as conn:
        conn.execute(
            _sql("INSERT INTO auth_tokens (token_hash, username, expires_at, created_at) VALUES (?, ?, ?, ?)"),
            (_token_hash(token), _normalize_username(username), expires_at, _utcnow_text()),
        )
        # Opportunistic cleanup of expired tokens.
        conn.execute(_sql("DELETE FROM auth_tokens WHERE expires_at < ?"), (time.time(),))
    return token


def verify_token(token: str) -> str | None:
    """Return the username for a valid, unexpired token, else None."""
    if not token:
        return None
    init_db()
    with _db() as conn:
        row = conn.execute(
            _sql("SELECT username, expires_at FROM auth_tokens WHERE token_hash = ?"),
            (_token_hash(token),),
        ).fetchone()
    if row is None or row["expires_at"] < time.time():
        return None
    return row["username"]


def revoke_token(token: str) -> None:
    with _db() as conn:
        conn.execute(_sql("DELETE FROM auth_tokens WHERE token_hash = ?"), (_token_hash(token),))


def _cli() -> None:
    import getpass

    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    backend = "postgres" if _IS_POSTGRES else f"sqlite ({DB_PATH})"
    command = sys.argv[1]

    if command == "add-user":
        if len(sys.argv) != 3:
            print("Usage: python auth.py add-user <username>")
            sys.exit(1)
        username = sys.argv[2]
        print(f"Backend: {backend}")
        password = getpass.getpass("Password: ")
        confirm = getpass.getpass("Confirm password: ")
        if password != confirm:
            print("Passwords do not match.")
            sys.exit(1)
        try:
            create_user(username, password)
        except ValueError as exc:
            print(f"Error: {exc}")
            sys.exit(1)
        print(f"User '{_normalize_username(username)}' created.")

    elif command == "list-users":
        print(f"Backend: {backend}")
        users = list_users()
        if not users:
            print("No users.")
        for u in users:
            print(f"{u['username']}  (created {u['created_at']}, last login {u['last_login'] or 'never'})")

    elif command == "delete-user":
        if len(sys.argv) != 3:
            print("Usage: python auth.py delete-user <username>")
            sys.exit(1)
        if delete_user(sys.argv[2]):
            print(f"User '{_normalize_username(sys.argv[2])}' deleted.")
        else:
            print("User not found.")

    else:
        print(f"Unknown command: {command}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    _cli()
