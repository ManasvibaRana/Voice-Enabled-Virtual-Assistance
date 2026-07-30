"""Who the assistant is talking to.

Identity now comes from a **verified JWT**, not from a string the browser hands
us. The previous version kept a `session_id -> student_id` dict and trusted
whatever `session_id` arrived in the request body, which meant anyone could
claim to be any student. Nothing in this module can be influenced by the
model's output or by an unauthenticated caller.
"""

import sqlite3
from typing import Any, TypedDict

from app import config
from app.auth import User


class Student(TypedDict):
    id: int
    name: str
    roll_no: str
    semester: int
    branch: str


class Principal(TypedDict):
    """The authenticated caller: their account plus the student they are."""

    user: User
    student: Student | None


def _connection() -> sqlite3.Connection:
    conn = sqlite3.connect(
        f"file:{config.DB_PATH.as_posix()}?mode=ro", uri=True, timeout=10
    )
    conn.row_factory = sqlite3.Row
    return conn


def get_student(student_id: int) -> Student | None:
    try:
        with _connection() as conn:
            row = conn.execute(
                "SELECT id, name, roll_no, semester, branch FROM students WHERE id = ?",
                (student_id,),
            ).fetchone()
        return dict(row) if row else None  # type: ignore[misc,return-value]
    except sqlite3.Error as exc:
        print(f"[identity] lookup failed: {exc}")
        return None


def principal_for(user: User | None) -> Principal | None:
    """Pair an authenticated user with their student record, if they have one."""
    if user is None:
        return None
    student = (
        get_student(user["student_id"]) if user.get("student_id") is not None else None
    )
    return {"user": user, "student": student}


def is_staff(principal: Principal | None) -> bool:
    return bool(principal) and principal["user"]["role"] == "staff"


def student_scope(principal: Principal | None) -> int | None:
    """The student id that queries must be restricted to, or None for staff.

    Staff see everything. A student sees only themselves. A signed-in account
    with no linked student record also gets None, but callers should treat
    that as "no personal data" rather than "unrestricted".
    """
    if principal is None or is_staff(principal):
        return None
    student = principal.get("student")
    return student["id"] if student else None


def context_line(principal: Principal | None) -> str:
    """One line of prompt context describing who is asking."""
    if principal is None:
        return "Nobody is signed in."

    if is_staff(principal):
        return (
            f"The person asking is {principal['user']['username']}, a member of "
            "staff. They may ask about any student and about class-wide "
            "figures such as rankings and averages."
        )

    student = principal.get("student")
    if student is None:
        return (
            f"The person asking is {principal['user']['username']}. Their "
            "account is not linked to a student record, so they have no "
            "personal marks or attendance."
        )

    return (
        f"The person asking is {student['name']}, roll number "
        f"{student['roll_no']}, semester {student['semester']}, branch "
        f"{student['branch']}. 'I', 'me' and 'my' refer to them. They are a "
        "student and can only see their own records - the database they query "
        "already contains nobody else, so write ordinary queries and never "
        "apologise for missing data about other students."
    )


def require_student(principal: Principal | None) -> tuple[Student | None, str]:
    """(student, error). For tools that are meaningless without a student."""
    if principal is None:
        return None, "You need to sign in first."
    student = principal.get("student")
    if student is None:
        return None, (
            "That is personal to a student, and this account is not linked to "
            "a student record."
        )
    return student, ""


def describe(principal: Principal | None) -> dict[str, Any] | None:
    """Safe summary for the UI - never includes the password hash."""
    if principal is None:
        return None
    student = principal.get("student")
    return {
        "username": principal["user"]["username"],
        "role": principal["user"]["role"],
        "name": student["name"] if student else principal["user"]["username"],
        "roll_no": student["roll_no"] if student else None,
        "semester": student["semester"] if student else None,
        "branch": student["branch"] if student else None,
    }
