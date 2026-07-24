"""
User authentication backed by users.db (SQLite).

Security posture is intentionally austere:
- Passwords are never stored or logged; only salted PBKDF2-HMAC-SHA256 hashes.
- Verification uses constant-time comparison.
- Login failures return a single generic message (no username/password hints).
- Repeated failures per username trigger a temporary lockout.

CLI (run from the project root):
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
from pathlib import Path

DB_PATH = Path(__file__).parent / "users.db"

PBKDF2_ITERATIONS = 600_000
SALT_BYTES = 16
HASH_NAME = "sha256"

MAX_FAILED_ATTEMPTS = 5
LOCKOUT_SECONDS = 300

MIN_PASSWORD_LENGTH = 8

TOKEN_TTL_SECONDS = 12 * 3600


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create the users table if it does not exist. Safe to call repeatedly."""
    with _connect() as conn:
        conn.execute(
            """
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
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS auth_tokens (
                token_hash TEXT PRIMARY KEY,
                username TEXT NOT NULL,
                expires_at REAL NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
            """
        )


def _hash_password(password: str, salt: bytes, iterations: int = PBKDF2_ITERATIONS) -> bytes:
    return hashlib.pbkdf2_hmac(HASH_NAME, password.encode("utf-8"), salt, iterations)


def create_user(username: str, password: str) -> None:
    """Create a user. Raises ValueError on invalid input or duplicate username."""
    username = username.strip()
    if not username:
        raise ValueError("Username cannot be empty.")
    if len(password) < MIN_PASSWORD_LENGTH:
        raise ValueError(f"Password must be at least {MIN_PASSWORD_LENGTH} characters.")

    salt = secrets.token_bytes(SALT_BYTES)
    password_hash = _hash_password(password, salt)

    init_db()
    try:
        with _connect() as conn:
            conn.execute(
                "INSERT INTO users (username, password_hash, salt, iterations) VALUES (?, ?, ?, ?)",
                (username, password_hash, salt, PBKDF2_ITERATIONS),
            )
    except sqlite3.IntegrityError:
        raise ValueError("Username is already taken.")


def verify_login(username: str, password: str) -> tuple[bool, str]:
    """
    Check credentials. Returns (ok, message).

    The failure message is deliberately generic so it never reveals whether
    the username exists or which field was wrong.
    """
    generic_fail = "Invalid username or password."
    username = (username or "").strip()
    if not username or not password:
        return False, generic_fail

    init_db()
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE username = ?", (username,)
        ).fetchone()

        if row is None:
            # Burn comparable time so missing users aren't distinguishable by timing.
            _hash_password(password, b"\x00" * SALT_BYTES)
            return False, generic_fail

        now = time.time()
        if row["locked_until"] > now:
            remaining = int(row["locked_until"] - now) + 1
            return False, f"Account temporarily locked. Try again in {remaining} seconds."

        candidate = _hash_password(password, row["salt"], row["iterations"])
        if hmac.compare_digest(candidate, row["password_hash"]):
            conn.execute(
                "UPDATE users SET failed_attempts = 0, locked_until = 0, "
                "last_login = datetime('now') WHERE id = ?",
                (row["id"],),
            )
            return True, "ok"

        failed = row["failed_attempts"] + 1
        locked_until = now + LOCKOUT_SECONDS if failed >= MAX_FAILED_ATTEMPTS else 0
        conn.execute(
            "UPDATE users SET failed_attempts = ?, locked_until = ? WHERE id = ?",
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
    with _connect() as conn:
        conn.execute(
            "UPDATE users SET password_hash = ?, salt = ?, iterations = ? WHERE username = ?",
            (_hash_password(new_password, salt), salt, PBKDF2_ITERATIONS, username.strip()),
        )
    return True, "Password updated."


def delete_user(username: str) -> bool:
    init_db()
    with _connect() as conn:
        cur = conn.execute("DELETE FROM users WHERE username = ?", (username.strip(),))
        return cur.rowcount > 0


def list_users() -> list[dict]:
    """Usernames and metadata only — never credential material."""
    init_db()
    with _connect() as conn:
        rows = conn.execute(
            "SELECT username, created_at, last_login FROM users ORDER BY username"
        ).fetchall()
    return [dict(r) for r in rows]


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def issue_token(username: str) -> str:
    """Create an API token for an already-authenticated user.

    The raw token is returned once and never stored — only its SHA-256 hash.
    """
    token = secrets.token_urlsafe(32)
    expires_at = time.time() + TOKEN_TTL_SECONDS
    init_db()
    with _connect() as conn:
        conn.execute(
            "INSERT INTO auth_tokens (token_hash, username, expires_at) VALUES (?, ?, ?)",
            (_token_hash(token), username.strip(), expires_at),
        )
        # Opportunistic cleanup of expired tokens.
        conn.execute("DELETE FROM auth_tokens WHERE expires_at < ?", (time.time(),))
    return token


def verify_token(token: str) -> str | None:
    """Return the username for a valid, unexpired token, else None."""
    if not token:
        return None
    init_db()
    with _connect() as conn:
        row = conn.execute(
            "SELECT username, expires_at FROM auth_tokens WHERE token_hash = ?",
            (_token_hash(token),),
        ).fetchone()
    if row is None or row["expires_at"] < time.time():
        return None
    return row["username"]


def revoke_token(token: str) -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM auth_tokens WHERE token_hash = ?", (_token_hash(token),))


def _cli() -> None:
    import getpass

    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1]

    if command == "add-user":
        if len(sys.argv) != 3:
            print("Usage: python auth.py add-user <username>")
            sys.exit(1)
        username = sys.argv[2]
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
        print(f"User '{username}' created.")

    elif command == "list-users":
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
            print(f"User '{sys.argv[2]}' deleted.")
        else:
            print("User not found.")

    else:
        print(f"Unknown command: {command}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    _cli()
