"""Class-schedule tool.

Deliberately hand-written SQL rather than LLM-generated: the shapes of these
questions are few and fixed, so a parameterised query is faster, free, and
cannot be wrong. The LLM is only used to pick out which day is being asked
about, and even that falls back to "today" on any failure.
"""

import sqlite3
from datetime import datetime, timedelta
from typing import Any, TypedDict

from app import config, identity, llm

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
TEACHING_DAYS = set(DAYS[:5])


class TimetableResult(TypedDict):
    ok: bool
    kind: str            # "next" | "day"
    day: str
    classes: list[dict[str, Any]]
    student: str
    semester: int | None
    error: str


def _connection() -> sqlite3.Connection:
    conn = sqlite3.connect(
        f"file:{config.DB_PATH.as_posix()}?mode=ro", uri=True, timeout=10
    )
    conn.row_factory = sqlite3.Row
    return conn


def _classes_for(semester: int, day: str) -> list[dict[str, Any]]:
    try:
        with _connection() as conn:
            rows = conn.execute(
                "SELECT t.day, t.start_time, t.end_time, t.room, s.name AS subject "
                "FROM timetable t JOIN subjects s ON s.id = t.subject_id "
                "WHERE t.semester = ? AND t.day = ? ORDER BY t.start_time",
                (semester, day),
            ).fetchall()
        return [dict(r) for r in rows]
    except sqlite3.Error as exc:
        print(f"[timetable] query failed: {exc}")
        return []


INTENT_PROMPT = """Classify a question about a class timetable. Reply with ONLY \
JSON, no prose and no fences:
{"kind": "next" | "day", "day": "Monday"|"Tuesday"|"Wednesday"|"Thursday"|"Friday"|"Saturday"|"Sunday"|"today"|"tomorrow"}

"next" means they want only their next upcoming class ("what's my next class?").
"day" means they want a whole day's schedule ("what are my classes today?",
"what's my timetable on Friday?"). If no day is named, use "today"."""


def _resolve_intent(question: str, now: datetime) -> tuple[str, str]:
    """Return (kind, weekday_name). Falls back to today's full schedule."""
    today = DAYS[now.weekday()]
    result = llm.chat_json(
        [
            {"role": "system", "content": INTENT_PROMPT},
            {"role": "user", "content": question},
        ],
        fallback={"kind": "day", "day": "today"},
        model=config.ROUTER_MODEL,
        max_tokens=80,
    )

    kind = str(result.get("kind", "day")).strip().lower()
    if kind not in {"next", "day"}:
        kind = "day"

    day = str(result.get("day", "today")).strip().title()
    if day == "Today":
        day = today
    elif day == "Tomorrow":
        day = DAYS[(now.weekday() + 1) % 7]
    elif day not in DAYS:
        day = today
    return kind, day


def run(question: str, principal=None, now: datetime | None = None) -> TimetableResult:
    """Answer a timetable question for the signed-in student."""
    now = now or datetime.now()
    student, error = identity.require_student(principal)
    if student is None:
        return {
            "ok": False, "kind": "day", "day": "", "classes": [],
            "student": "", "semester": None, "error": error,
        }

    kind, day = _resolve_intent(question, now)

    if kind == "next":
        # Walk forward from now until a teaching day has a class left in it.
        for offset in range(0, 8):
            probe = now + timedelta(days=offset)
            probe_day = DAYS[probe.weekday()]
            if probe_day not in TEACHING_DAYS:
                continue
            for cls in _classes_for(student["semester"], probe_day):
                if offset > 0 or cls["start_time"] > now.strftime("%H:%M"):
                    cls = dict(cls, when="today" if offset == 0 else probe_day)
                    return {
                        "ok": True, "kind": "next", "day": probe_day,
                        "classes": [cls], "student": student["name"],
                        "semester": student["semester"], "error": "",
                    }
        return {
            "ok": True, "kind": "next", "day": "", "classes": [],
            "student": student["name"], "semester": student["semester"], "error": "",
        }

    classes = _classes_for(student["semester"], day) if day in TEACHING_DAYS else []
    return {
        "ok": True, "kind": "day", "day": day, "classes": classes,
        "student": student["name"], "semester": student["semester"], "error": "",
    }


def as_text(result: TimetableResult) -> str:
    """Render for the answer-composition prompt."""
    if not result["ok"]:
        return result["error"]
    if not result["classes"]:
        if result["kind"] == "next":
            return "(no upcoming classes were found this week)"
        return f"(there are no classes scheduled on {result['day']})"

    lines = []
    for cls in result["classes"]:
        lines.append(
            f"{cls['day']} {cls['start_time']}-{cls['end_time']}: "
            f"{cls['subject']} in room {cls['room']}"
        )
    # Phrased as "your", not by name: the assistant is talking to this person,
    # and naming them made it answer in the third person.
    header = (
        "Your next class:"
        if result["kind"] == "next"
        else f"Your classes on {result['day']}:"
    )
    return header + "\n" + "\n".join(lines)
