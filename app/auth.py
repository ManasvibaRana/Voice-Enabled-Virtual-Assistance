"""Accounts, passwords and JWTs.

Design notes worth knowing before changing anything here:

* Passwords are bcrypt hashes. bcrypt silently truncates at 72 bytes, so longer
  passwords are rejected rather than quietly weakened.
* Login returns the SAME error for an unknown username and a wrong password.
  Different messages would let anyone enumerate valid roll numbers.
* Tokens are stateless JWTs, but every token carries the user's
  `token_version`. Logging out increments that column, which invalidates every
  outstanding token for the user immediately - real revocation without a
  denylist table that grows forever.
* Every write uses bound parameters against a fixed statement, the same rule
  the reminder tool follows. No SQL here is ever assembled from user input.
"""

import os
import secrets
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Any, TypedDict

import bcrypt
import jwt

from app import config, demo

MIN_PASSWORD_LENGTH = 8
BCRYPT_MAX_BYTES = 72
MAX_FAILED_ATTEMPTS = 5
LOCKOUT_MINUTES = 15
TOKEN_TTL_MINUTES = 60
ALGORITHM = "HS256"

# Deliberately identical for "no such user" and "wrong password".
BAD_CREDENTIALS = "Incorrect username or password."


class User(TypedDict):
    id: int
    username: str
    role: str
    student_id: int | None
    token_version: int


_secret_cache: str | None = None


def jwt_secret() -> str:
    """HS256 signing key. Generated per-process if JWT_SECRET is not set."""
    global _secret_cache
    if _secret_cache is not None:
        return _secret_cache

    configured = os.getenv("JWT_SECRET", "").strip()
    if configured:
        _secret_cache = configured
    else:
        _secret_cache = secrets.token_urlsafe(48)
        print(
            "[auth] JWT_SECRET is not set - generated a random one. Logins will "
            "not survive a restart. Add JWT_SECRET to .env to fix that."
        )
    return _secret_cache


def _connection() -> sqlite3.Connection:
    conn = sqlite3.connect(config.DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def _normalise_name(name: str) -> str:
    """Case and whitespace insensitive form used to compare names."""
    return " ".join((name or "").split()).lower()


def _row_to_user(row: sqlite3.Row | None) -> User | None:
    if row is None:
        return None
    return {
        "id": row["id"],
        "username": row["username"],
        "role": row["role"],
        "student_id": row["student_id"],
        "token_version": row["token_version"],
    }


# ---------------------------------------------------------------------------
# Passwords
# ---------------------------------------------------------------------------


def password_problem(password: str) -> str:
    """Return a complaint about the password, or "" if it is acceptable."""
    if not password or len(password) < MIN_PASSWORD_LENGTH:
        return f"Password must be at least {MIN_PASSWORD_LENGTH} characters."
    if len(password.encode("utf-8")) > BCRYPT_MAX_BYTES:
        # bcrypt ignores anything past 72 bytes, so a longer password would be
        # silently weaker than it looks.
        return f"Password must be at most {BCRYPT_MAX_BYTES} bytes."
    return ""


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("ascii")


def verify_password(password: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("ascii"))
    except (ValueError, TypeError):
        return False


# ---------------------------------------------------------------------------
# Tokens
# ---------------------------------------------------------------------------


def create_token(user: User) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(user["id"]),
        "role": user["role"],
        "tv": user["token_version"],
        "iat": now,
        "exp": now + timedelta(minutes=TOKEN_TTL_MINUTES),
    }
    return jwt.encode(payload, jwt_secret(), algorithm=ALGORITHM)


def decode_token(token: str) -> User | None:
    """Verify a bearer token and return the user, or None. Never raises."""
    if not token:
        return None
    try:
        payload: dict[str, Any] = jwt.decode(
            token, jwt_secret(), algorithms=[ALGORITHM]
        )
    except Exception:  # noqa: BLE001 - expired, tampered, malformed, all the same
        return None

    try:
        user_id = int(payload.get("sub", ""))
    except (TypeError, ValueError):
        return None

    user = get_user(user_id)
    if user is None:
        return None

    # A logout bumped token_version, so this token predates it.
    if payload.get("tv") != user["token_version"]:
        return None
    return user


# ---------------------------------------------------------------------------
# Lookups
# ---------------------------------------------------------------------------


def get_user(user_id: int) -> User | None:
    try:
        with _connection() as conn:
            row = conn.execute(
                "SELECT id, username, role, student_id, token_version "
                "FROM users WHERE id = ?",
                (user_id,),
            ).fetchone()
        return _row_to_user(row)
    except sqlite3.Error as exc:
        print(f"[auth] user lookup failed: {exc}")
        return None


def _get_by_username(conn: sqlite3.Connection, username: str) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM users WHERE LOWER(username) = LOWER(?)", (username,)
    ).fetchone()


# ---------------------------------------------------------------------------
# Signup
# ---------------------------------------------------------------------------


def signup(roll_no: str, name: str, password: str) -> tuple[User | None, str]:
    """Claim a seeded student record. Returns (user, error_message)."""
    roll_no = (roll_no or "").strip()
    name = (name or "").strip()

    if not roll_no or not name:
        return None, "Roll number and name are both required."

    problem = password_problem(password)
    if problem:
        return None, problem

    try:
        with _connection() as conn:
            student = conn.execute(
                "SELECT id, name, roll_no FROM students WHERE LOWER(roll_no) = LOWER(?)",
                (roll_no,),
            ).fetchone()
            if student is None:
                return None, (
                    "That roll number is not on our records. It should look "
                    "like CS04001 or CS06015."
                )

            # Cheap proof they are who they say: the name has to match the
            # record. Not real identity proofing, but it stops someone
            # claiming a roll number they merely guessed.
            # Compared on collapsed whitespace and case, so "karan  verma"
            # matches "Karan Verma" - a typo should not block a real student.
            if _normalise_name(student["name"]) != _normalise_name(name):
                return None, "That name does not match the roll number on record."

            taken = conn.execute(
                "SELECT 1 FROM users WHERE student_id = ?", (student["id"],)
            ).fetchone()
            if taken:
                return None, "An account already exists for that roll number."

            now = datetime.now().isoformat(timespec="seconds")
            cursor = conn.execute(
                "INSERT INTO users (username, password_hash, role, student_id, "
                "created_at) VALUES (?, ?, 'student', ?, ?)",
                (student["roll_no"], hash_password(password), student["id"], now),
            )
            conn.commit()
            return get_user(int(cursor.lastrowid)), ""
    except sqlite3.IntegrityError:
        return None, "An account already exists for that roll number."
    except sqlite3.Error as exc:
        print(f"[auth] signup failed: {exc}")
        return None, "Could not create the account. Please try again."


# ---------------------------------------------------------------------------
# Login / logout
# ---------------------------------------------------------------------------


def login(username: str, password: str) -> tuple[str, User | None, str]:
    """Returns (token, user, error_message)."""
    username = (username or "").strip()
    if not username or not password:
        return "", None, BAD_CREDENTIALS

    try:
        with _connection() as conn:
            row = _get_by_username(conn, username)

            if row is not None and row["locked_until"]:
                try:
                    locked_until = datetime.fromisoformat(row["locked_until"])
                except ValueError:
                    locked_until = None
                if locked_until and locked_until > datetime.now():
                    remaining = int((locked_until - datetime.now()).total_seconds() / 60) + 1
                    return "", None, (
                        f"Too many failed attempts. Try again in {remaining} minute"
                        f"{'s' if remaining != 1 else ''}."
                    )

            ok = row is not None and verify_password(password, row["password_hash"])

            if not ok:
                # In a public demo a stranger must not be able to lock the
                # shared demo logins for everyone else.
                if row is not None and not demo.is_shared_account(row["username"]):
                    attempts = int(row["failed_attempts"]) + 1
                    locked = (
                        (datetime.now() + timedelta(minutes=LOCKOUT_MINUTES)).isoformat(
                            timespec="seconds"
                        )
                        if attempts >= MAX_FAILED_ATTEMPTS
                        else None
                    )
                    conn.execute(
                        "UPDATE users SET failed_attempts = ?, locked_until = ? "
                        "WHERE id = ?",
                        (attempts, locked, row["id"]),
                    )
                    conn.commit()
                # Same message whether the user exists or not.
                return "", None, BAD_CREDENTIALS

            conn.execute(
                "UPDATE users SET failed_attempts = 0, locked_until = NULL, "
                "last_login = ? WHERE id = ?",
                (datetime.now().isoformat(timespec="seconds"), row["id"]),
            )
            conn.commit()

            user = _row_to_user(row)
            assert user is not None
            return create_token(user), user, ""
    except sqlite3.Error as exc:
        print(f"[auth] login failed: {exc}")
        return "", None, "Could not sign in. Please try again."


def logout(user_id: int) -> bool:
    """Invalidate every outstanding token for this user."""
    try:
        with _connection() as conn:
            if demo.ENABLED:
                row = conn.execute("SELECT username FROM users WHERE id = ?", (user_id,)).fetchone()
                if row is not None and demo.is_shared_account(row[0]):
                    # Shared demo login: revoking every token would log out
                    # every other visitor. The client just drops its own token.
                    return True
            conn.execute(
                "UPDATE users SET token_version = token_version + 1 WHERE id = ?",
                (user_id,),
            )
            conn.commit()
        return True
    except sqlite3.Error as exc:
        print(f"[auth] logout failed: {exc}")
        return False


def user_from_header(authorization: str | None) -> User | None:
    """Resolve `Authorization: Bearer <token>` to a user, or None."""
    if not authorization:
        return None
    parts = authorization.split(None, 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    return decode_token(parts[1].strip())
