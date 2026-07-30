"""Row-level isolation.

This is the security-critical file. A regression here fails *silently*: the
assistant keeps answering, it just answers with somebody else's data. Every
assertion is written from the attacker's point of view - "as this student, can
I see data I should not?"
"""

import sqlite3

import pytest

from app import auth, config, identity, sql_tool

# Queries a student must never be able to get real answers from.
ADVERSARIAL = [
    "SELECT COUNT(*) FROM students",
    "SELECT name FROM students",
    "SELECT COUNT(*) FROM marks",
    "SELECT COUNT(*) FROM attendance",
    "SELECT s.name, AVG(100.0 * m.score / m.max_score) AS p FROM students s "
    "JOIN marks m ON m.student_id = s.id GROUP BY s.id ORDER BY p DESC",
    "SELECT s.name FROM students s JOIN attendance a ON a.student_id = s.id "
    "GROUP BY s.id ORDER BY AVG(a.present) ASC",
]


@pytest.fixture
def student_principal():
    user = auth.decode_token(auth.login("CS06020", "demo1234")[0])
    assert user is not None, "seeded student account missing - re-run seed_db.py"
    return identity.principal_for(user)


@pytest.fixture
def staff_principal():
    user = auth.decode_token(auth.login("staff", "staff1234")[0])
    assert user is not None, "seeded staff account missing - re-run seed_db.py"
    return identity.principal_for(user)


# --- scope resolution -----------------------------------------------------


def test_student_is_scoped_and_staff_is_not(student_principal, staff_principal):
    assert identity.student_scope(student_principal) == student_principal["student"]["id"]
    assert identity.student_scope(staff_principal) is None
    assert identity.is_staff(staff_principal) and not identity.is_staff(student_principal)
    # An unauthenticated caller gets no scope AND no principal to act on.
    assert identity.student_scope(None) is None


# --- the isolation itself -------------------------------------------------


@pytest.mark.parametrize("sql", ADVERSARIAL)
def test_student_sees_only_themselves(student_principal, sql):
    me = student_principal["student"]
    scope = identity.student_scope(student_principal)
    columns, rows, error = sql_tool.execute_sql(sql, scope=scope)
    assert not error, error

    # Any name that comes back must be their own.
    if "name" in [c.lower() for c in columns]:
        index = [c.lower() for c in columns].index("name")
        names = {r[index] for r in rows}
        assert names <= {me["name"]}, f"leaked other students: {names - {me['name']}}"

    # Any count must be the scoped count, never the whole table.
    if columns and columns[0].upper().startswith("COUNT"):
        assert rows[0][0] <= 30, "impossible count"
        if "FROM students" in sql:
            assert rows[0][0] == 1, "student can count the whole cohort"


@pytest.mark.parametrize("sql", ADVERSARIAL)
def test_staff_see_everything(staff_principal, sql):
    columns, rows, error = sql_tool.execute_sql(
        sql, scope=identity.student_scope(staff_principal)
    )
    assert not error, error
    assert rows, "staff query returned nothing"


def test_student_row_counts_are_strictly_smaller(student_principal):
    scope = identity.student_scope(student_principal)
    for table, staff_expected in (("students", 30), ("marks", 150), ("attendance", 200)):
        _c, staff_rows, _e = sql_tool.execute_sql(f"SELECT COUNT(*) FROM {table}", None)
        _c2, mine, _e2 = sql_tool.execute_sql(f"SELECT COUNT(*) FROM {table}", scope)
        assert staff_rows[0][0] == staff_expected
        assert mine[0][0] < staff_rows[0][0], f"{table} not scoped for a student"


# --- the escapes the views cannot stop on their own -----------------------


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT COUNT(*) FROM main.marks",
        "SELECT * FROM main.students",
        "select name from MAIN.students",
        "SELECT * FROM temp.marks",
        "SELECT * FROM sqlite_master",
        "SELECT name FROM sqlite_schema",
        "SELECT username, password_hash FROM users",
        "SELECT * FROM Users",
    ],
)
def test_validator_blocks_view_escapes_and_credentials(sql):
    """A temp view is bypassed by `main.`, so the validator must reject it."""
    ok, reason = sql_tool.validate_sql(sql)
    assert not ok, f"validator allowed an escape: {sql}"
    assert reason


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT s.name, AVG(m.score) FROM students s JOIN marks m ON m.student_id = s.id GROUP BY s.id",
        "SELECT COUNT(*) FROM attendance a WHERE a.present = 1",
        "SELECT t.day, t.start_time FROM timetable t ORDER BY t.start_time",
        "SELECT sub.name FROM subjects sub WHERE sub.semester = 6",
    ],
)
def test_ordinary_aliased_sql_still_passes(sql):
    """The escape check must not break normal `alias.column` references."""
    ok, reason = sql_tool.validate_sql(sql)
    assert ok, f"validator wrongly rejected: {sql} ({reason})"


# --- the prompt must not leak either --------------------------------------


def test_schema_context_is_scoped_for_a_student(student_principal):
    """Sample rows go into the prompt AND the 'show working' panel."""
    me = student_principal["student"]["name"]
    scope = identity.student_scope(student_principal)
    student_ctx = sql_tool.schema_context(force=True, scope=scope)

    with sqlite3.connect(config.DB_PATH) as conn:
        others = [
            r[0]
            for r in conn.execute(
                "SELECT name FROM students WHERE name <> ? LIMIT 10", (me,)
            )
        ]
    leaked = [n for n in others if n in student_ctx]
    assert not leaked, f"schema context leaked other students: {leaked}"


def test_schema_context_never_mentions_the_users_table():
    """Password hashes must never reach the model or the UI."""
    for scope in (None, 20):
        ctx = sql_tool.schema_context(force=True, scope=scope)
        assert "CREATE TABLE users" not in ctx
        assert "password_hash" not in ctx


def _captured_system_prompt(monkeypatch, personal_only: bool) -> str:
    from app import compose, llm

    seen: list = []

    def fake_chat(messages, **_kw):
        seen.append(messages)
        return "ok"

    monkeypatch.setattr(llm, "chat", fake_chat)
    result = {"ok": True, "row_count": 1, "columns": ["n"], "rows": [[1]],
              "sql": "", "error": "", "attempts": []}
    compose.compose_from_sql("q", result, "", personal_only=personal_only)
    return seen[0][0]["content"]


def test_access_rule_is_attached_only_for_students(monkeypatch):
    """Staff must not be told they can only see their own records."""
    assert "ACCESS RULE" in _captured_system_prompt(monkeypatch, True)
    assert "ACCESS RULE" not in _captured_system_prompt(monkeypatch, False)


def test_access_rule_still_permits_a_student_their_own_totals(monkeypatch):
    """The refusal must not swallow legitimate personal questions.

    Over-broad wording once made 'what is my overall percentage?' get refused,
    even though it is the student's own figure.
    """
    rules = _captured_system_prompt(monkeypatch, True)
    for allowed in ["what is my attendance?", "what is my overall percentage?"]:
        assert allowed in rules, f"rule no longer allows {allowed!r}"
    assert "Only refuse when answering would need somebody else's data" in rules


def test_credentials_are_invisible_even_without_the_validator():
    """Defence in depth: the users table is blanked on every connection."""
    _c, rows, error = sql_tool.execute_sql("SELECT * FROM users", scope=None)
    assert not error
    assert rows == []
