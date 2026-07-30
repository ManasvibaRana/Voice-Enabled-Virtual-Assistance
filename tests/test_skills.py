"""Identity, wake-phrase gating, timetable and reminders.

The LLM is stubbed, so these run offline.
"""

import sqlite3
from datetime import datetime

import pytest

from app import auth, config, identity, llm, reminder_tool, timetable_tool

MONDAY_MORNING = datetime(2026, 7, 27, 8, 0)  # a Monday, before the first class


def _students() -> list[dict]:
    with sqlite3.connect(config.DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        return [
            dict(r)
            for r in conn.execute(
                "SELECT id, name, roll_no, semester, branch FROM students "
                "ORDER BY semester, name"
            )
        ]


def principal_for_student(student_id: int) -> identity.Principal:
    """Build an authenticated principal directly, without going via login."""
    user: auth.User = {
        "id": 10_000 + student_id,
        "username": f"test{student_id}",
        "role": "student",
        "student_id": student_id,
        "token_version": 1,
    }
    principal = identity.principal_for(user)
    assert principal is not None
    return principal


@pytest.fixture
def session():
    """No-one signed in. Tools must refuse personal questions."""
    return None


@pytest.fixture(autouse=True)
def clean_reminders():
    yield
    with sqlite3.connect(config.DB_PATH) as conn:
        conn.execute("DELETE FROM reminders")
        conn.commit()


@pytest.fixture
def stub_llm(monkeypatch):
    def install(payload):
        monkeypatch.setattr(llm, "chat_json", lambda *a, **k: dict(payload))

    return install


# --- identity -------------------------------------------------------------


def test_principal_carries_the_student_record():
    student = _students()[0]
    principal = principal_for_student(student["id"])
    assert principal["student"]["id"] == student["id"]
    assert identity.student_scope(principal) == student["id"]
    assert not identity.is_staff(principal)


def test_context_line_mentions_the_student():
    student = _students()[0]
    line = identity.context_line(principal_for_student(student["id"]))
    assert student["name"] in line
    assert student["roll_no"] in line


def test_anonymous_context_says_nobody_is_signed_in():
    assert "nobody is signed in" in identity.context_line(None).lower()


def test_unknown_student_id_yields_no_student_record():
    user: auth.User = {
        "id": 1, "username": "ghost", "role": "student",
        "student_id": 99999, "token_version": 1,
    }
    principal = identity.principal_for(user)
    assert principal is not None
    assert principal["student"] is None
    student, error = identity.require_student(principal)
    assert student is None and error


# --- personal tools refuse anonymous callers -------------------------------


def test_timetable_requires_sign_in(session):
    result = timetable_tool.run("what is my next class?", session)
    assert result["ok"] is False
    assert "sign in" in result["error"].lower()


def test_reminders_require_sign_in(session):
    result = reminder_tool.run("what are my reminders?", session)
    assert result["ok"] is False
    assert "sign in" in result["error"].lower()


# --- timetable ------------------------------------------------------------


def test_next_class_is_the_earliest_upcoming_one(stub_llm):
    stub_llm({"kind": "next", "day": "today"})
    student = next(s for s in _students() if s["semester"] == 6)
    me = principal_for_student(student["id"])

    result = timetable_tool.run("what is my next class?", me, now=MONDAY_MORNING)
    assert result["ok"] and len(result["classes"]) == 1
    assert result["classes"][0]["start_time"] == "09:00"
    assert result["classes"][0]["subject"] == "Compiler Design"


def test_day_schedule_is_ordered_and_for_the_right_semester(stub_llm):
    stub_llm({"kind": "day", "day": "Friday"})
    student = next(s for s in _students() if s["semester"] == 4)
    me = principal_for_student(student["id"])

    result = timetable_tool.run("my timetable on Friday", me, now=MONDAY_MORNING)
    assert result["ok"] and result["classes"]
    times = [c["start_time"] for c in result["classes"]]
    assert times == sorted(times)
    # Semester 4 subjects only.
    assert all(c["subject"] in {"Data Structures", "Database Systems",
                                "Operating Systems"} for c in result["classes"])


# --- reminders ------------------------------------------------------------


def test_add_then_list_then_complete(stub_llm):
    me = principal_for_student(_students()[0]["id"])

    stub_llm({"action": "add", "text": "submit the AI assignment", "due": "friday"})
    added = reminder_tool.run("remind me to submit the AI assignment on Friday",
                              me, now=MONDAY_MORNING)
    assert added["ok"] and added["created"]["text"] == "submit the AI assignment"
    assert added["created"]["due_date"] == "2026-07-31"  # the coming Friday

    stub_llm({"action": "list", "text": "", "due": ""})
    listed = reminder_tool.run("what are my reminders?", me, now=MONDAY_MORNING)
    assert [r["text"] for r in listed["reminders"]] == ["submit the AI assignment"]

    stub_llm({"action": "complete", "text": "AI assignment", "due": ""})
    done = reminder_tool.run("mark the AI assignment as done", me,
                             now=MONDAY_MORNING)
    assert done["ok"] and done["reminders"] == []


def test_reminders_are_private_to_each_student(stub_llm):
    students = _students()
    me = principal_for_student(students[0]["id"])
    other = principal_for_student(students[1]["id"])

    stub_llm({"action": "add", "text": "mine", "due": ""})
    reminder_tool.run("remind me about mine", me, now=MONDAY_MORNING)

    stub_llm({"action": "list", "text": "", "due": ""})
    theirs = reminder_tool.run("what are my reminders?", other, now=MONDAY_MORNING)
    assert theirs["reminders"] == []


# --- grades and CGPA ------------------------------------------------------
#
# These exist because the assistant used to answer "what is my CGPA?" with an
# average percentage - 74.4 - and call it a CGPA. A percentage is out of 100
# and a CGPA is out of 10, so that answer was simply wrong. The views below
# derive both from the published grading policy instead of guessing.

GRADE_BANDS = [(90, "O", 10), (80, "A", 9), (70, "B", 8), (60, "C", 7),
               (50, "D", 6), (40, "E", 5), (0, "F", 0)]


def _hand_computed_grades(student_id: int) -> dict:
    """Independent implementation of docs/exam_grading_policy.md."""
    from app import sql_tool

    weights = sql_tool.EXAM_WEIGHTS
    with sqlite3.connect(config.DB_PATH) as conn:
        rows = conn.execute(
            "SELECT sub.name, sub.credits, m.exam_type, m.score, m.max_score "
            "FROM marks m JOIN subjects sub ON sub.id = m.subject_id "
            "WHERE m.student_id = ?", (student_id,)
        ).fetchall()

    per_subject: dict[str, list] = {}
    for name, credits, exam, score, mx in rows:
        entry = per_subject.setdefault(name, [0.0, 0.0, credits])
        entry[0] += (score / mx) * weights[exam]
        entry[1] += weights[exam]

    out = {}
    for name, (weighted, total_weight, credits) in per_subject.items():
        total = 100.0 * weighted / total_weight
        grade, points = next((g, p) for lo, g, p in GRADE_BANDS if total >= lo)
        out[name] = {"total": round(total, 1), "grade": grade,
                     "points": points, "credits": credits}
    return out


def test_subject_grades_match_the_published_policy():
    from app import sql_tool

    student_id = _students()[0]["id"]
    expected = _hand_computed_grades(student_id)
    conn = sql_tool._read_only_connection(student_id)
    try:
        actual = {
            r["subject"]: {"total": r["total_marks"], "grade": r["grade"],
                           "points": r["grade_points"], "credits": r["credits"]}
            for r in conn.execute("SELECT * FROM subject_grades")
        }
    finally:
        conn.close()
    assert actual == expected


def test_cgpa_is_out_of_ten_not_a_percentage():
    from app import sql_tool

    student_id = _students()[0]["id"]
    conn = sql_tool._read_only_connection(student_id)
    try:
        cgpa = conn.execute("SELECT cgpa FROM cgpa").fetchone()["cgpa"]
        percent = conn.execute(
            "SELECT AVG(100.0 * score / max_score) FROM marks"
        ).fetchone()[0]
    finally:
        conn.close()

    assert 0 <= cgpa <= 10, f"CGPA {cgpa} is not on a 10-point scale"
    assert abs(cgpa - percent) > 1, "CGPA must not simply be the percentage"


def test_cgpa_is_credit_weighted():
    """A 4-credit subject must count more than a 3-credit one."""
    from app import sql_tool

    student_id = _students()[0]["id"]
    expected = _hand_computed_grades(student_id)
    num = sum(v["points"] * v["credits"] for v in expected.values())
    den = sum(v["credits"] for v in expected.values())

    conn = sql_tool._read_only_connection(student_id)
    try:
        cgpa = conn.execute("SELECT cgpa FROM cgpa").fetchone()["cgpa"]
    finally:
        conn.close()
    assert cgpa == pytest.approx(round(num / den, 2))


def test_grade_views_respect_student_isolation():
    """A student must not see anyone else's grades through these views."""
    from app import sql_tool

    me = _students()[0]["id"]
    conn = sql_tool._read_only_connection(me)
    try:
        owners = {r[0] for r in conn.execute("SELECT student_id FROM subject_grades")}
        cgpa_rows = conn.execute("SELECT COUNT(*) FROM cgpa").fetchone()[0]
    finally:
        conn.close()
    assert owners <= {me}, f"grade views leaked students {owners - {me}}"
    assert cgpa_rows == 1


@pytest.mark.parametrize(
    "phrase,expected",
    [
        ("2026-08-05", "2026-08-05"),
        ("tomorrow", "2026-07-28"),
        ("today", "2026-07-27"),
        ("friday", "2026-07-31"),
        ("in 3 days", "2026-07-30"),
        ("in 2 weeks", "2026-08-10"),
        ("", None),
        ("whenever", None),
    ],
)
def test_due_date_parsing(phrase, expected):
    assert reminder_tool._parse_due(phrase, MONDAY_MORNING) == expected


def test_sql_tool_still_cannot_write_to_reminders():
    """The read-only guarantee must survive the new write path."""
    ok, _ = __import__("app.sql_tool", fromlist=["x"]).validate_sql(
        "INSERT INTO reminders (student_id, text) VALUES (1, 'x')"
    )
    assert not ok

    from app import sql_tool

    conn = sql_tool._read_only_connection()
    try:
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            conn.execute("INSERT INTO reminders (student_id, text, created_at) "
                         "VALUES (1, 'x', 'now')")
    finally:
        conn.close()
