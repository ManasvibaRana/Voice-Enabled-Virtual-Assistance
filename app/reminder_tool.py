"""Reminders - the one tool that writes to the database.

Design note, and the reason this is a separate module rather than part of
sql_tool: **the model never writes SQL here.** It only extracts structured
fields (an action, a piece of text, a date) as JSON. Python then performs the
write with bound parameters against a hand-written statement, on a connection
that is opened read-write only for this table's operations.

So the read-only guarantee on the text-to-SQL path stays completely intact:
there is no path from model output to a write statement.
"""

import re
import sqlite3
from datetime import datetime, timedelta
from typing import Any, TypedDict

from app import config, identity, llm

WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]


class ReminderResult(TypedDict):
    ok: bool
    action: str                    # "add" | "list" | "complete"
    reminders: list[dict[str, Any]]
    created: dict[str, Any] | None
    error: str


def _writable_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(config.DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


INTENT_PROMPT = """Extract what the user wants to do with their reminders.
Reply with ONLY JSON, no prose and no fences:
{"action": "add" | "list" | "complete", "text": "<the task, imperative, no date>", "due": "<YYYY-MM-DD or a phrase like tomorrow / next friday / in 3 days, or empty>"}

Examples:
"remind me to submit the AI assignment on Friday"
  -> {"action":"add","text":"submit the AI assignment","due":"friday"}
"what are my reminders"      -> {"action":"list","text":"","due":""}
"mark the lab report as done" -> {"action":"complete","text":"lab report","due":""}
Keep "text" short and free of dates."""


def _parse_due(phrase: str, today: datetime) -> str | None:
    """Turn a loose date phrase into YYYY-MM-DD. None if not understood."""
    value = (phrase or "").strip().lower()
    if not value:
        return None

    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        return value

    if value in {"today", "tonight"}:
        return today.date().isoformat()
    if value == "tomorrow":
        return (today + timedelta(days=1)).date().isoformat()

    match = re.fullmatch(r"in (\d+) (day|days|week|weeks)", value)
    if match:
        count = int(match.group(1))
        days = count * (7 if match.group(2).startswith("week") else 1)
        return (today + timedelta(days=days)).date().isoformat()

    cleaned = value.replace("next ", "").replace("on ", "").strip()
    if cleaned in WEEKDAYS:
        target = WEEKDAYS.index(cleaned)
        ahead = (target - today.weekday()) % 7
        if ahead == 0 or value.startswith("next "):
            ahead = ahead or 7
        return (today + timedelta(days=ahead)).date().isoformat()

    return None


def _list_for(student_id: int) -> list[dict[str, Any]]:
    try:
        with _writable_connection() as conn:
            rows = conn.execute(
                "SELECT id, text, due_date, done FROM reminders "
                "WHERE student_id = ? AND done = 0 "
                "ORDER BY COALESCE(due_date, '9999-12-31'), id",
                (student_id,),
            ).fetchall()
        return [dict(r) for r in rows]
    except sqlite3.Error as exc:
        print(f"[reminders] list failed: {exc}")
        return []


def run(question: str, principal=None, now: datetime | None = None) -> ReminderResult:
    now = now or datetime.now()
    student, error = identity.require_student(principal)
    if student is None:
        return {"ok": False, "action": "list", "reminders": [], "created": None,
                "error": error}

    intent = llm.chat_json(
        [
            {"role": "system", "content": INTENT_PROMPT},
            {"role": "user", "content": question},
        ],
        fallback={"action": "list", "text": "", "due": ""},
        model=config.ROUTER_MODEL,
        max_tokens=150,
    )
    action = str(intent.get("action", "list")).strip().lower()
    if action not in {"add", "list", "complete"}:
        action = "list"

    text = str(intent.get("text", "")).strip()
    due = _parse_due(str(intent.get("due", "")), now)

    try:
        if action == "add":
            if not text:
                return {"ok": False, "action": "add", "reminders": [], "created": None,
                        "error": "I couldn't tell what to remind you about."}
            with _writable_connection() as conn:
                # Bound parameters, fixed statement. No model-authored SQL.
                cursor = conn.execute(
                    "INSERT INTO reminders (student_id, text, due_date, created_at, done) "
                    "VALUES (?, ?, ?, ?, 0)",
                    (student["id"], text, due, now.isoformat(timespec="seconds")),
                )
                conn.commit()
                created = {"id": cursor.lastrowid, "text": text, "due_date": due}
            return {"ok": True, "action": "add", "reminders": _list_for(student["id"]),
                    "created": created, "error": ""}

        if action == "complete":
            with _writable_connection() as conn:
                row = conn.execute(
                    "SELECT id, text FROM reminders WHERE student_id = ? AND done = 0 "
                    "AND LOWER(text) LIKE ? ORDER BY id LIMIT 1",
                    (student["id"], f"%{text.lower()}%"),
                ).fetchone()
                if row is None:
                    return {"ok": True, "action": "complete",
                            "reminders": _list_for(student["id"]), "created": None,
                            "error": "no matching reminder"}
                conn.execute(
                    "UPDATE reminders SET done = 1 WHERE id = ? AND student_id = ?",
                    (row["id"], student["id"]),
                )
                conn.commit()
                done = {"id": row["id"], "text": row["text"]}
            return {"ok": True, "action": "complete",
                    "reminders": _list_for(student["id"]), "created": done, "error": ""}

        return {"ok": True, "action": "list", "reminders": _list_for(student["id"]),
                "created": None, "error": ""}

    except sqlite3.Error as exc:
        print(f"[reminders] write failed: {exc}")
        return {"ok": False, "action": action, "reminders": [], "created": None,
                "error": str(exc)}


def as_text(result: ReminderResult) -> str:
    if not result["ok"]:
        return result["error"]

    pending = result["reminders"]
    listing = (
        "\n".join(
            f"- {r['text']}" + (f" (due {r['due_date']})" if r["due_date"] else "")
            for r in pending
        )
        or "(nothing pending)"
    )

    if result["action"] == "add" and result["created"]:
        due = result["created"]["due_date"]
        # Deliberately does NOT repeat the full pending list here: when it did,
        # the composer saw the new item listed alongside the confirmation and
        # announced "you already have a reminder to ..." instead of confirming
        # that it had just been created.
        others = max(0, len(pending) - 1)
        tail = (
            f" They have {others} other reminder{'s' if others != 1 else ''} pending."
            if others
            else " This is their only pending reminder."
        )
        return (
            "A NEW reminder has just been created and saved for the user: "
            f"\"{result['created']['text']}\""
            + (f", due {due}." if due else ", with no due date.")
            + tail
            + " Confirm that it has been saved."
        )
    if result["action"] == "complete":
        if result["error"] == "no matching reminder" or not result["created"]:
            return f"No pending reminder matched that.\nStill pending:\n{listing}"
        return (
            f"Marked done: {result['created']['text']}.\nStill pending:\n{listing}"
        )
    return f"Pending reminders:\n{listing}"
