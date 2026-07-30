"""Per-account conversation memory.

Server-side, in-process, keyed by the authenticated account (see
`main._memory_key`) so history cannot be read or poisoned by anyone
supplying a guessed identifier. Keeps
the last MAX_HISTORY_TURNS (question, answer) pairs so follow-up questions
like "and what about last month?" can be resolved against earlier turns.
"""

import threading
import time
from collections import defaultdict, deque
from typing import Deque

from app import config

_lock = threading.Lock()
_sessions: dict[str, Deque[dict[str, str]]] = defaultdict(
    lambda: deque(maxlen=config.MAX_HISTORY_TURNS)
)
_last_seen: dict[str, float] = {}


def _evict_stale(now: float) -> None:
    """Drop idle sessions. Caller must hold the lock.

    Every browser that ever visits creates a session, and without this they
    accumulate for the lifetime of the process.
    """
    cutoff = now - config.SESSION_TTL_SECONDS
    stale = [sid for sid, seen in _last_seen.items() if seen < cutoff]
    for sid in stale:
        _sessions.pop(sid, None)
        _last_seen.pop(sid, None)

    # Hard cap as a backstop against a burst of sessions inside the TTL.
    if len(_last_seen) > config.MAX_SESSIONS:
        oldest = sorted(_last_seen, key=_last_seen.get)  # type: ignore[arg-type]
        for sid in oldest[: len(_last_seen) - config.MAX_SESSIONS]:
            _sessions.pop(sid, None)
            _last_seen.pop(sid, None)


def add_turn(key: str, question: str, answer: str) -> None:
    if not key:
        return
    with _lock:
        now = time.time()
        _evict_stale(now)
        _sessions[key].append({"question": question, "answer": answer})
        _last_seen[key] = now


def get_turns(key: str) -> list[dict[str, str]]:
    if not key:
        return []
    with _lock:
        return list(_sessions.get(key, ()))


def history_text(
    key: str,
    max_chars_per_field: int = 220,
    limit: int | None = None,
) -> str:
    """Condensed history block that gets prepended to router and tool prompts.

    `limit` keeps only the most recent N turns. Smaller models measurably lose
    accuracy when a long history is stapled onto a text-to-SQL prompt - they
    start answering the earlier question instead of the current one - and a
    follow-up like "and what about semester 4?" only ever needs the turn
    immediately before it.
    """
    turns = get_turns(key)
    if limit is not None and limit > 0:
        turns = turns[-limit:]
    if not turns:
        return "(no previous turns)"

    lines: list[str] = []
    for i, turn in enumerate(turns, start=1):
        q = turn["question"].strip().replace("\n", " ")[:max_chars_per_field]
        a = turn["answer"].strip().replace("\n", " ")[:max_chars_per_field]
        lines.append(f"{i}. User: {q}\n   Assistant: {a}")
    return "\n".join(lines)


def clear(key: str) -> None:
    with _lock:
        _sessions.pop(key, None)
        _last_seen.pop(key, None)


def active_sessions() -> int:
    with _lock:
        return len(_sessions)
