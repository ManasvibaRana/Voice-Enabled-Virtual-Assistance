"""Safeguards for running as a public demo (e.g. a Hugging Face Space).

A public demo has two problems a local install does not:

* Every visitor shares one free-tier LLM key (NVIDIA NIM allows ~40 requests a
  minute). One person - or a bot - must not be able to use it all up. Each
  question costs about three LLM calls, so the site-wide cap of 12 questions a
  minute stays just under that limit.
* Visitors share the seeded demo logins. Without care, a stranger could lock
  them with five wrong passwords, or log everyone out by pressing "log out".

All of this is OFF for local use. It switches on when DEMO_MODE=1, or
automatically inside a Hugging Face Space (which sets SPACE_ID).
"""

import os
import threading
import time
from collections import deque

ENABLED = os.getenv("DEMO_MODE") == "1" or bool(os.getenv("SPACE_ID"))

QUESTIONS_PER_VISITOR = int(os.getenv("DEMO_QUESTIONS_PER_10_MIN", "15"))
VISITOR_WINDOW_SECONDS = 10 * 60
QUESTIONS_PER_MINUTE = int(os.getenv("DEMO_QUESTIONS_PER_MINUTE", "12"))

# The logins seeded by data/seed_db.py and documented in the README.
SHARED_ACCOUNTS = {"CS06020", "staff"}

VISITOR_LIMIT_MESSAGE = (
    "You've asked a lot of questions in a short time. This is a free public "
    "demo, so please wait a few minutes and try again."
)
SITE_LIMIT_MESSAGE = (
    "The demo is busy right now. Please try again in a minute."
)

_lock = threading.Lock()
_visitors: dict[str, deque[float]] = {}
_site: deque[float] = deque()


def client_ip(headers, fallback: str | None) -> str:
    """The visitor's address. Behind the Space's proxy that is the first
    X-Forwarded-For entry, not the socket peer."""
    forwarded = headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return fallback or "unknown"


def check_question(visitor: str, now: float | None = None) -> str | None:
    """Record a question. Returns None if allowed, or the message to show."""
    if not ENABLED:
        return None
    now = time.monotonic() if now is None else now
    with _lock:
        while _site and now - _site[0] > 60:
            _site.popleft()
        if len(_site) >= QUESTIONS_PER_MINUTE:
            return SITE_LIMIT_MESSAGE

        history = _visitors.setdefault(visitor, deque())
        while history and now - history[0] > VISITOR_WINDOW_SECONDS:
            history.popleft()
        if len(history) >= QUESTIONS_PER_VISITOR:
            return VISITOR_LIMIT_MESSAGE

        history.append(now)
        _site.append(now)

        # Forget visitors who have gone quiet so the table cannot grow forever.
        if len(_visitors) > 5000:
            for key in [k for k, q in _visitors.items() if not q or now - q[-1] > VISITOR_WINDOW_SECONDS]:
                del _visitors[key]
        return None


def is_shared_account(username: str) -> bool:
    """Shared demo logins are exempt from lockout and global logout."""
    return ENABLED and username in SHARED_ACCOUNTS


def reset() -> None:
    """Clear all counters (used by the tests)."""
    with _lock:
        _visitors.clear()
        _site.clear()
