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

from app import config, identity, llm, memory

WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]


class ReminderResult(TypedDict):
    ok: bool
    action: str          # add | list | complete | delete | correct | none | duplicate
    reminders: list[dict[str, Any]]
    library: list[dict[str, Any]]
    created: dict[str, Any] | None
    error: str


def _writable_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(config.DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


INTENT_PROMPT = """Extract what the user wants to do with their reminders.
Reply with ONLY JSON, no prose and no fences:
{"action": "add" | "list" | "complete" | "delete" | "correct" | "none",
 "text": "<the task, imperative, no date>",
 "due": "<YYYY-MM-DD, or a phrase like tomorrow / next friday / in 3 days, or empty>"}

The conversation so far is given below. Use it: speech recognition mangles
words, so people correct themselves constantly, and a correction must NEVER
create a second reminder.

  "remind me to submit the AI assignment on Friday"
     -> {"action":"add","text":"submit the AI assignment","due":"friday"}
  "what are my reminders"        -> {"action":"list","text":"","due":""}
  "mark the lab report as done"  -> {"action":"complete","text":"lab report","due":""}
  "delete the library reminder"  -> {"action":"delete","text":"library","due":""}

CORRECTIONS - the user is fixing what you just saved, not adding anything:
  "no, I meant get a book"       -> {"action":"correct","text":"get a book","due":""}
  "not submit, just collect it"  -> {"action":"correct","text":"collect the book","due":""}
  "I said library not liberty"   -> {"action":"correct","text":"...","due":""}
Anything starting with "no", "not", "I meant", "I said", or contradicting the
previous turn is a CORRECTION.

USE "none" when there is no clear, complete task to save:
  "yeah"  "okay so"  "hmm"  "that one"  "get one"  "and that"
A reminder must name something concrete to do. If the words are too vague or
garbled to be a real task, return "none" rather than guessing - a wrong
reminder is worse than asking again.

Keep "text" short, imperative, and free of dates."""


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


# Words that carry no task on their own. "get one" is all filler and was
# saved as a reminder before this existed.
_FILLER = {
    "a", "an", "the", "it", "one", "that", "this", "there", "and", "so", "ok",
    "okay", "yeah", "yes", "no", "um", "uh", "please", "just", "then", "my",
    "me", "i", "to", "for", "get", "do", "have", "add", "some", "thing",
    "things", "stuff", "something", "anything", "reminder", "reminders",
}


def _is_a_real_task(text: str) -> bool:
    """Does this name something concrete to do?

    A reminder needs at least one word carrying meaning. "get one" is two
    filler words and became a reminder; "get one book from library" has
    "book" and "library" and is fine.
    """
    words = re.findall(r"[a-z0-9]+", (text or "").lower())
    if len(words) < 2:
        return False
    return any(word not in _FILLER for word in words)


def _find_similar(student_id: int, text: str) -> dict[str, Any] | None:
    """An existing pending reminder that means the same thing, if any."""
    wanted = {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _FILLER}
    if not wanted:
        return None
    for row in _list_for(student_id):
        have = {w for w in re.findall(r"[a-z0-9]+", row["text"].lower())
                if w not in _FILLER}
        if not have:
            continue
        overlap = len(wanted & have) / max(len(wanted), len(have))
        if overlap >= 0.6:
            return row
    return None


def _library_dues(student_id: int) -> list[dict[str, Any]]:
    """Books still out. A book due back is a real pending task, so it belongs
    next to the reminders even though nobody typed it in."""
    try:
        with _writable_connection() as conn:
            rows = conn.execute(
                "SELECT title, due_date, "
                "  MAX(0, CAST(julianday('now') - julianday(due_date) AS INTEGER)) "
                "    AS days_overdue, "
                "  MAX(0, CAST(julianday(due_date) - julianday('now') AS INTEGER)) "
                "    AS days_left "
                "FROM library_loans WHERE student_id = ? AND returned_on IS NULL "
                "ORDER BY due_date",
                (student_id,),
            ).fetchall()
        return [dict(r) for r in rows]
    except sqlite3.Error as exc:
        print(f"[reminders] library lookup failed: {exc}")
        return []


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


def run(question: str, principal=None, now: datetime | None = None,
        memory_key: str = "") -> ReminderResult:
    now = now or datetime.now()
    student, error = identity.require_student(principal)
    if student is None:
        return {"ok": False, "action": "list", "reminders": [], "library": [],
                "created": None, "error": error}

    history = memory.history_text(memory_key, limit=3)
    intent = llm.chat_json(
        [
            {"role": "system", "content": INTENT_PROMPT},
            {
                "role": "user",
                "content": f"Conversation so far:\n{history}\n\nNow: {question}",
            },
        ],
        fallback={"action": "list", "text": "", "due": ""},
        model=config.ROUTER_MODEL,
        max_tokens=180,
    )
    action = str(intent.get("action", "list")).strip().lower()
    if action not in {"add", "list", "complete", "delete", "correct", "none"}:
        action = "list"

    text = str(intent.get("text", "")).strip()
    due = _parse_due(str(intent.get("due", "")), now)

    # A vague fragment must not become a reminder. This is the guard that stops
    # "get one" and "yeah so" being saved as tasks.
    if action in {"add", "correct"} and not _is_a_real_task(text):
        action = "none"

    try:
        if action == "none":
            return {"ok": True, "action": "none",
                    "reminders": _list_for(student["id"]),
                    "library": _library_dues(student["id"]), "created": None,
                    "error": "unclear"}

        if action == "correct":
            # Replace the most recent reminder rather than adding another one.
            with _writable_connection() as conn:
                row = conn.execute(
                    "SELECT id, text FROM reminders WHERE student_id = ? AND done = 0 "
                    "ORDER BY id DESC LIMIT 1",
                    (student["id"],),
                ).fetchone()
                if row is None:
                    action = "add"  # nothing to correct, so treat it as new
                else:
                    conn.execute(
                        "UPDATE reminders SET text = ?, due_date = COALESCE(?, due_date) "
                        "WHERE id = ? AND student_id = ?",
                        (text, due, row["id"], student["id"]),
                    )
                    conn.commit()
                    return {"ok": True, "action": "correct",
                            "reminders": _list_for(student["id"]),
                    "library": _library_dues(student["id"]),
                            "created": {"id": row["id"], "text": text,
                                        "due_date": due, "was": row["text"]},
                            "error": ""}

        if action == "delete":
            with _writable_connection() as conn:
                row = conn.execute(
                    "SELECT id, text FROM reminders WHERE student_id = ? AND done = 0 "
                    "AND LOWER(text) LIKE ? ORDER BY id DESC LIMIT 1",
                    (student["id"], f"%{text.lower()}%"),
                ).fetchone()
                if row is None:
                    return {"ok": True, "action": "delete",
                            "reminders": _list_for(student["id"]),
                    "library": _library_dues(student["id"]), "created": None,
                            "error": "no matching reminder"}
                conn.execute(
                    "DELETE FROM reminders WHERE id = ? AND student_id = ?",
                    (row["id"], student["id"]),
                )
                conn.commit()
            return {"ok": True, "action": "delete",
                    "reminders": _list_for(student["id"]),
                    "library": _library_dues(student["id"]),
                    "created": {"id": row["id"], "text": row["text"]}, "error": ""}

        if action == "add":
            # Saying the same thing twice - common when speech is misheard and
            # repeated - should not produce two identical reminders.
            existing = _find_similar(student["id"], text)
            if existing is not None:
                return {"ok": True, "action": "duplicate",
                        "reminders": _list_for(student["id"]),
                    "library": _library_dues(student["id"]),
                        "created": {"id": existing["id"], "text": existing["text"],
                                    "due_date": existing["due_date"]},
                        "error": ""}

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
                    "library": _library_dues(student["id"]),
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
                            "reminders": _list_for(student["id"]),
                    "library": _library_dues(student["id"]), "created": None,
                            "error": "no matching reminder"}
                conn.execute(
                    "UPDATE reminders SET done = 1 WHERE id = ? AND student_id = ?",
                    (row["id"], student["id"]),
                )
                conn.commit()
                done = {"id": row["id"], "text": row["text"]}
            return {"ok": True, "action": "complete",
                    "reminders": _list_for(student["id"]),
                    "library": _library_dues(student["id"]), "created": done, "error": ""}

        return {"ok": True, "action": "list", "reminders": _list_for(student["id"]),
                    "library": _library_dues(student["id"]),
                "created": None, "error": ""}

    except sqlite3.Error as exc:
        print(f"[reminders] write failed: {exc}")
        return {"ok": False, "action": action, "reminders": [], "library": [],
                "created": None, "error": str(exc)}


def as_text(result: ReminderResult) -> str:
    if not result["ok"]:
        return result["error"]

    pending = result["reminders"]
    books = result.get("library") or []

    lines = [
        f"- {r['text']}" + (f" (due {r['due_date']})" if r["due_date"] else "")
        for r in pending
    ]
    # A library book due back is a real pending task even though nobody typed
    # it in as a reminder, so it is listed alongside them.
    for b in books:
        if b["days_overdue"] > 0:
            fine = b["days_overdue"] * LIBRARY_FINE_PER_DAY
            lines.append(
                f"- LIBRARY: return '{b['title']}' - it was due {b['due_date']}, "
                f"{b['days_overdue']} days ago, and the fine is now Rs. {fine}"
            )
        else:
            lines.append(
                f"- LIBRARY: return '{b['title']}' by {b['due_date']} "
                f"({b['days_left']} days left)"
            )
    listing = "\n".join(lines) or "(nothing pending)"

    action = result["action"]
    created = result["created"]

    if action == "none":
        return (
            "That was not clear enough to save as a reminder. Ask the user what "
            "they would like to be reminded about, in one short sentence. Do NOT "
            "claim anything was saved."
        )

    if action == "duplicate" and created:
        return (
            f"They already have that reminder: '{created['text']}'. Tell them it is "
            f"already on the list, so nothing was added again.\n"
            f"Everything pending:\n{listing}"
        )

    if action == "correct" and created:
        return (
            f"The most recent reminder was CORRECTED, not added. It used to say "
            f"'{created.get('was', '')}' and now says '{created['text']}'. "
            f"Confirm the correction in one sentence.\n"
            f"Everything pending:\n{listing}"
        )

    if action == "delete":
        if result["error"] == "no matching reminder" or not created:
            return (
                f"No reminder matched that, so nothing was deleted.\n"
                f"Still pending:\n{listing}"
            )
        return (
            f"Deleted the reminder '{created['text']}'. Confirm it is gone.\n"
            f"Still pending:\n{listing}"
        )

    if action == "add" and created:
        due = created["due_date"]
        others = max(0, len(pending) - 1)
        tail = (
            f" They have {others} other reminder{'s' if others != 1 else ''} pending."
            if others
            else " This is their only pending reminder."
        )
        return (
            "A NEW reminder has just been created and saved for the user: "
            f"'{created['text']}'"
            + (f", due {due}." if due else ", with no due date.")
            + tail
            + " Confirm that it has been saved."
        )

    if action == "complete":
        if result["error"] == "no matching reminder" or not created:
            return f"No pending reminder matched that.\nStill pending:\n{listing}"
        return f"Marked done: {created['text']}.\nStill pending:\n{listing}"

    return f"Everything pending:\n{listing}"
