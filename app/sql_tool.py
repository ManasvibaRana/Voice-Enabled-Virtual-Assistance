"""Text-to-SQL tool: generate -> validate -> execute (read-only) -> retry once.

Safety guards, in order of defence:
  1. Prompt instruction to emit a single SELECT.
  2. Defensive fence stripping and statement normalisation.
  3. Static validation: must be exactly one statement, must start with SELECT,
     no DML/DDL/PRAGMA/ATTACH keywords, no comment markers, no semicolons
     other than a single trailing one.
  4. The SQLite connection itself is opened read-only via the
     `file:...?mode=ro` URI, so even a validator bug cannot write.
  5. Results are capped at MAX_SQL_ROWS rows.
"""

import re
import sqlite3
from typing import Any, TypedDict

from app import config, identity, llm, memory

# ---------------------------------------------------------------------------
# Connection helpers
# ---------------------------------------------------------------------------


def _read_only_connection(scope: int | None = None) -> sqlite3.Connection:
    """Open app.db strictly read-only. Writes fail at the SQLite layer.

    When `scope` is a student id, temp views are created that SHADOW the base
    tables, so every unqualified `SELECT ... FROM marks` sees only that
    student's rows. This is row-level security enforced by SQLite itself: the
    model can write whatever SQL it likes and still cannot reach another
    student's data. Prompt instructions alone would not be enforcement.

    The one way past a temp view is an explicit schema prefix
    (`SELECT * FROM main.marks`), which is why `validate_sql` rejects those.
    """
    uri = f"file:{config.DB_PATH.as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=10)
    conn.row_factory = sqlite3.Row
    # Blank out the credentials table for EVERY caller, staff included. No
    # question about a college needs it, and a stray SELECT would otherwise put
    # password hashes in front of the model.
    conn.execute("CREATE TEMP VIEW users AS SELECT NULL AS id WHERE 0")
    if scope is not None:
        _apply_student_scope(conn, scope)
    _add_grade_views(conn)
    return conn


# Weights from docs/exam_grading_policy.md: a subject is marked out of 100,
# made up of quiz 20, midterm 30 and final 50.
EXAM_WEIGHTS = {"quiz": 20, "midterm": 30, "final": 50}

# From docs/student_faq.md: "Rs. 5 per book per day".
LIBRARY_FINE_PER_DAY = 5


def _add_grade_views(conn: sqlite3.Connection) -> None:
    """Expose grades and CGPA as real columns, computed from the policy.

    Without this the model was asked for a CGPA, found nothing resembling one,
    computed an average percentage and called it "your CGPA" - a confident,
    wrong answer. The grading scale is published in the exam policy, so it can
    be derived exactly instead of guessed.

    Only the components a student actually has are counted, rescaled to 100,
    so a missing quiz does not silently drag the total down.
    """
    weights = " ".join(
        f"WHEN '{name}' THEN {weight}.0" for name, weight in EXAM_WEIGHTS.items()
    )
    conn.execute(
        f"""
        CREATE TEMP VIEW subject_totals AS
        SELECT m.student_id, m.subject_id,
               100.0 * SUM((1.0 * m.score / m.max_score)
                           * CASE m.exam_type {weights} ELSE 0 END)
                     / NULLIF(SUM(CASE m.exam_type {weights} ELSE 0 END), 0)
               AS total_marks
        FROM marks m
        GROUP BY m.student_id, m.subject_id
        """
    )
    # Grade bands, straight from the policy table.
    band = """
        CASE WHEN t.total_marks >= 90 THEN {o}
             WHEN t.total_marks >= 80 THEN {a}
             WHEN t.total_marks >= 70 THEN {b}
             WHEN t.total_marks >= 60 THEN {c}
             WHEN t.total_marks >= 50 THEN {d}
             WHEN t.total_marks >= 40 THEN {e}
             ELSE {f} END
    """
    conn.execute(
        f"""
        CREATE TEMP VIEW subject_grades AS
        SELECT t.student_id, t.subject_id, s.name AS subject, s.credits,
               ROUND(t.total_marks, 1) AS total_marks,
               {band.format(o="'O'", a="'A'", b="'B'", c="'C'", d="'D'", e="'E'", f="'F'")}
                   AS grade,
               {band.format(o=10, a=9, b=8, c=7, d=6, e=5, f=0)} AS grade_points
        FROM subject_totals t JOIN subjects s ON s.id = t.subject_id
        """
    )
    # Books still out, with days overdue and the fine from the student FAQ
    # (Rs 5 per book per day). Derived rather than left to the model, for the
    # same reason as CGPA: it would otherwise invent a number.
    conn.execute(
        f"""
        CREATE TEMP VIEW library_dues AS
        SELECT l.student_id, l.title, l.issued_on, l.due_date,
               MAX(0, CAST(julianday('now') - julianday(l.due_date) AS INTEGER))
                   AS days_overdue,
               MAX(0, CAST(julianday(l.due_date) - julianday('now') AS INTEGER))
                   AS days_left,
               MAX(0, CAST(julianday('now') - julianday(l.due_date) AS INTEGER))
                   * {LIBRARY_FINE_PER_DAY} AS fine_rupees,
               CASE WHEN julianday('now') > julianday(l.due_date)
                    THEN 'overdue' ELSE 'due' END AS status
        FROM library_loans l
        WHERE l.returned_on IS NULL
        """
    )
    conn.execute(
        """
        CREATE TEMP VIEW cgpa AS
        SELECT student_id,
               ROUND(SUM(grade_points * credits) * 1.0 / NULLIF(SUM(credits), 0), 2)
               AS cgpa
        FROM subject_grades GROUP BY student_id
        """
    )


def _apply_student_scope(conn: sqlite3.Connection, student_id: int) -> None:
    """Shadow the sensitive tables with views filtered to one student."""
    semester_row = conn.execute(
        "SELECT semester FROM main.students WHERE id = ?", (student_id,)
    ).fetchone()
    semester = semester_row["semester"] if semester_row else -1

    statements = [
        "CREATE TEMP VIEW students AS "
        f"SELECT * FROM main.students WHERE id = {int(student_id)}",
        "CREATE TEMP VIEW marks AS "
        f"SELECT * FROM main.marks WHERE student_id = {int(student_id)}",
        "CREATE TEMP VIEW attendance AS "
        f"SELECT * FROM main.attendance WHERE student_id = {int(student_id)}",
        # Not private, but a student only cares about their own semester.
        "CREATE TEMP VIEW timetable AS "
        f"SELECT * FROM main.timetable WHERE semester = {int(semester)}",
        "CREATE TEMP VIEW reminders AS "
        f"SELECT * FROM main.reminders WHERE student_id = {int(student_id)}",
        "CREATE TEMP VIEW library_loans AS "
        f"SELECT * FROM main.library_loans WHERE student_id = {int(student_id)}",
    ]
    for statement in statements:
        conn.execute(statement)


# ---------------------------------------------------------------------------
# Schema context for the prompt
# ---------------------------------------------------------------------------

_schema_cache: dict[int | None, str] = {}


def schema_context(
    sample_rows: int = 3, force: bool = False, scope: int | None = None
) -> str:
    """CREATE TABLE statements plus a few sample rows per table.

    Giving the model real values (e.g. exam_type = 'midterm') stops it from
    inventing column names or value formats.

    The sample rows are read through the SAME scoped connection as the query
    itself. For a student that matters: an unscoped dump would paste other
    students' names straight into the prompt - and into the "show the working"
    panel - which is exactly the leak the temp views exist to prevent.
    """
    if not force and scope in _schema_cache:
        return _schema_cache[scope]

    parts: list[str] = []
    try:
        with _read_only_connection(scope) as conn:
            tables = [
                row["name"]
                for row in conn.execute(
                    # `users` holds password hashes. It must never reach the
                    # prompt, the model, or the "show the working" panel.
                    "SELECT name FROM sqlite_master WHERE type='table' "
                    "AND name NOT LIKE 'sqlite_%' AND name <> 'users' "
                    "ORDER BY name"
                )
            ]
            for table in tables:
                ddl_row = conn.execute(
                    "SELECT sql FROM sqlite_master WHERE type='table' AND name=?",
                    (table,),
                ).fetchone()
                if ddl_row and ddl_row["sql"]:
                    parts.append(ddl_row["sql"].strip() + ";")

                rows = conn.execute(
                    f"SELECT * FROM {table} LIMIT {int(sample_rows)}"
                ).fetchall()
                if rows:
                    cols = rows[0].keys()
                    parts.append(f"-- sample rows from {table}:")
                    parts.append("-- " + " | ".join(cols))
                    for row in rows:
                        parts.append("-- " + " | ".join(str(row[c]) for c in cols))
                parts.append("")

            # Derived views for grades. Showing real rows stops the model
            # inventing a CGPA out of whatever number is nearest.
            parts.append("-- derived views (already computed, just SELECT them):")
            parts.append(
                "-- cgpa(student_id, cgpa)  credit-weighted grade points, out of 10"
            )
            parts.append(
                "-- subject_grades(student_id, subject, credits, total_marks, "
                "grade, grade_points)  total_marks is out of 100"
            )
            parts.append(
                "-- library_dues(student_id, title, due_date, days_overdue, "
                "fine_rupees, status)  books not yet returned; status is "
                "'due' or 'overdue'"
            )
            for view in ("cgpa", "subject_grades", "library_dues"):
                try:
                    rows = conn.execute(f"SELECT * FROM {view} LIMIT 3").fetchall()
                except sqlite3.Error:
                    continue
                if rows:
                    cols = rows[0].keys()
                    parts.append(f"-- sample rows from {view}:")
                    parts.append("-- " + " | ".join(cols))
                    for row in rows:
                        parts.append("-- " + " | ".join(str(row[c]) for c in cols))
            parts.append("")

            # Low-cardinality columns: list every value, so the model matches
            # on exact spellings instead of guessing at LIKE patterns.
            parts.append("-- reference values (use these exact spellings):")
            for table, column in (
                ("subjects", "name"),
                ("students", "branch"),
                ("marks", "exam_type"),
                ("students", "semester"),
            ):
                try:
                    values = [
                        str(r[0])
                        for r in conn.execute(
                            f"SELECT DISTINCT {column} FROM {table} ORDER BY {column}"
                        )
                    ]
                except sqlite3.Error:
                    continue
                if values and len(values) <= 40:
                    parts.append(f"-- {table}.{column}: {', '.join(values)}")
    except sqlite3.Error as exc:
        return (
            "-- schema unavailable: "
            f"{exc}\n-- expected tables: students, subjects, marks, attendance"
        )

    _schema_cache[scope] = "\n".join(parts).strip()
    return _schema_cache[scope]


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

SQL_SYSTEM_PROMPT = """You write SQLite SELECT queries for a college database.

Database schema and sample rows:
{schema}

Hard rules:
- Output ONE SQLite SELECT statement and NOTHING else.
- No explanation, no comments, no markdown code fences, no trailing semicolon.
- Never write INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, ATTACH or PRAGMA.
- Prefer a single table, or a single JOIN, and keep the query simple.
- Attendance: present is 1 for present and 0 for absent, so attendance \
percentage is 100.0 * AVG(a.present).
- RANKING questions - "top", "best", "worst", "highest", "lowest", "who has
  the most/least", "top 3 students". These are the easiest to get wrong, so
  follow all five steps:
  1. Work out the measure being ranked. Marks, grades, performance, "top
     student" and "best student" with no measure named all mean the average
     mark percentage, AVG(100.0 * m.score / m.max_score). Attendance means
     100.0 * AVG(a.present).
  2. INNER JOIN to the table holding that measure (marks or attendance).
     Never LEFT JOIN: students with no rows get a NULL average, SQLite sorts
     NULL first, and you would report a student with no records at all as the
     lowest.
  3. SELECT the student's name AND the computed measure as a second column,
     e.g. SELECT s.name, AVG(100.0 * m.score / m.max_score) AS avg_percent.
     The measure column is required - without it the answer cannot quote a
     number and the ranking cannot be checked.
  4. ORDER BY that computed measure: DESC for top/best/highest, ASC for
     lowest/worst. NEVER ORDER BY id, name or roll_no - that is not a ranking,
     it is an arbitrary row order and the answer would be wrong.
  5. Add GROUP BY s.id, HAVING COUNT(*) >= 3, and LIMIT N. If the question
     names a number ("top three") use that number. If it asks for a single
     result ("who has the highest ...") use LIMIT 5 anyway, so that students
     tied at the best value are visible in the results rather than silently
     cut off - the answer step decides how to report a tie.
  Correct: SELECT s.name, AVG(100.0 * m.score / m.max_score) AS avg_percent
           FROM students s INNER JOIN marks m ON m.student_id = s.id
           WHERE s.semester = 6 GROUP BY s.id HAVING COUNT(*) >= 3
           ORDER BY avg_percent DESC LIMIT 3
- Marks: before writing the query, decide whether the question names a single
  exam_type ('midterm', 'final' or 'quiz'). This decides everything:
  * NAMES one, e.g. "highest midterm score", "average final marks":
    filter WHERE m.exam_type = 'midterm' and use the RAW score column.
    Correct: SELECT MAX(m.score) ... WHERE m.exam_type = 'midterm'.
    Do NOT convert to a percentage here - within one exam_type every row has
    the same max_score, so raw scores are already comparable.
  * Does NOT name one, e.g. "average score in Compiler Design":
    the rows mix quiz (out of 20), midterm (out of 50) and final (out of 100),
    so raw scores are incomparable and you MUST normalise.
    Correct: SELECT AVG(100.0 * m.score / m.max_score) AS avg_percent ...
- "Scored above 80" means above 80 percent, i.e.
  100.0 * score / max_score > 80, unless the question names an exam_type.
- Always return a small result: aggregate with COUNT/AVG/SUM, and when listing \
rows use ORDER BY with LIMIT.
- NEVER select a raw id column (subject_id, student_id, id) as an output
  column. The answer is spoken aloud, and "subject 6" means nothing to a
  person. Join to the parent table and select its name instead.
    Wrong:   SELECT m.subject_id, m.exam_type, m.score FROM marks m
    Correct: SELECT sub.name AS subject, m.exam_type, m.score, m.max_score
             FROM marks m JOIN subjects sub ON sub.id = m.subject_id
- Listing marks: ALWAYS include the subject name and max_score (or the
  percentage). A score of 30 is meaningless on its own - it could be 30 out of
  50 or 30 out of 100 - so a query that returns score without max_score is
  wrong.
- GRADES, CGPA and SGPA come from ready-made views. Never compute them
  yourself and NEVER call an average percentage a CGPA - a percentage is out
  of 100, a CGPA is out of 10, and they are different numbers.
    cgpa(student_id, cgpa)  -- credit-weighted, out of 10
    subject_grades(student_id, subject, credits, total_marks, grade,
                   grade_points)  -- total_marks is out of 100, grade is O..F
  'what is my CGPA?'      -> SELECT cgpa FROM cgpa
  'what grade did I get in AI?' ->
      SELECT subject, grade, total_marks FROM subject_grades
      WHERE subject = 'Artificial Intelligence'
  'what is my percentage?' -> the average of 100.0 * score / max_score from
  marks, which is NOT the CGPA.
- LIBRARY books use the ready-made library_dues view. Never compute overdue
  days or fines yourself.
  'do I have any books to return?' / 'any library dues?' / 'when is my book
  due?' -> SELECT title, due_date, days_overdue, fine_rupees, status
           FROM library_dues
  library_loans also holds returned books; library_dues shows only the ones
  still out.
- "How many students ..." must count DISTINCT students, not rows: use
  COUNT(DISTINCT m.student_id). One student has many marks and attendance rows,
  so a plain COUNT(*) over those tables over-counts. Use COUNT(*) only when
  counting rows of the students table itself.
- Alias computed columns with a readable name, e.g. AS avg_score.
- Prefer matching subjects and branches by their EXACT name from the reference
  values above, e.g. WHERE s.name = 'Artificial Intelligence'.
- Users often abbreviate subjects: AI = Artificial Intelligence,
  OS = Operating Systems, DBMS or DB = Database Systems,
  DS = Data Structures, CN = Computer Networks, CD = Compiler Design.
  Expand the abbreviation to the exact name; never LIKE '%AI%'.
- If you do use LIKE, the pattern literal must be lowercase whenever you wrap
  the column in LOWER(), e.g. LOWER(s.name) LIKE '%compiler%'. A pattern like
  LOWER(s.name) LIKE '%AI%' can never match and is always a bug."""


def generate_sql(question: str, principal=None, memory_key: str = "") -> str:
    history = memory.history_text(memory_key, limit=config.SQL_HISTORY_TURNS)
    scope = identity.student_scope(principal)
    user_prompt = (
        f"Conversation so far (use it to resolve follow-up questions):\n"
        f"{history}\n\n{identity.context_line(principal)}\n\n"
        f"Question: {question}\n\nSQL:"
    )
    raw = llm.chat(
        [
            {
                "role": "system",
                "content": SQL_SYSTEM_PROMPT.format(
                    schema=schema_context(scope=scope)
                ),
            },
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        max_tokens=300,
    )
    return _clean_sql(raw)


def repair_sql(question: str, bad_sql: str, error: str, principal=None) -> str:
    """Second and final attempt: hand the SQLite error back to the model."""
    scope = identity.student_scope(principal)
    user_prompt = (
        f"Question: {question}\n\n"
        f"This query failed:\n{bad_sql}\n\n"
        f"SQLite error: {error}\n\n"
        "Write a corrected single SELECT statement. Output only the SQL."
    )
    raw = llm.chat(
        [
            {
                "role": "system",
                "content": SQL_SYSTEM_PROMPT.format(
                    schema=schema_context(scope=scope)
                ),
            },
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        max_tokens=300,
    )
    return _clean_sql(raw)


def _clean_sql(raw: str) -> str:
    """Strip fences, stray labels and trailing punctuation from a model reply."""
    sql = llm.strip_fences(raw or "")
    # Some models prefix "SQL:" or "Query:" despite instructions.
    sql = re.sub(r"^\s*(sql|sqlite|query|answer)\s*:\s*", "", sql, flags=re.I)
    # Keep only from the first SELECT/WITH onwards if there is leading prose.
    match = re.search(r"\bSELECT\b", sql, flags=re.I)
    if match:
        sql = sql[match.start() :]
    return sql.strip().rstrip(";").strip()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

_FORBIDDEN = (
    "insert",
    "update",
    "delete",
    "drop",
    "alter",
    "create",
    "replace",
    "truncate",
    "attach",
    "detach",
    "pragma",
    "vacuum",
    "reindex",
    "grant",
    "revoke",
)


def validate_sql(sql: str) -> tuple[bool, str]:
    """Return (ok, reason). Rejects anything that is not one plain SELECT."""
    if not sql or not sql.strip():
        return False, "The model returned an empty query."

    stripped = sql.strip()

    if "--" in stripped or "/*" in stripped or "*/" in stripped:
        return False, "Comments are not allowed in generated SQL."

    # Schema-qualified names bypass the temp views that enforce per-student
    # isolation: `SELECT * FROM main.marks` reads the whole table. Word
    # boundaries keep ordinary column aliases such as `s.name` working.
    if re.search(r"\b(main|temp|temporary)\s*\.", stripped, flags=re.I):
        return False, "Schema-qualified table names are not allowed."
    if re.search(r"\bsqlite_[a-z_]*", stripped, flags=re.I):
        return False, "SQLite internal tables are not accessible."
    if re.search(r"\busers\b", stripped, flags=re.I):
        return False, "The credentials table is not queryable."

    # One statement only: a single optional trailing semicolon is tolerated.
    body = stripped.rstrip(";").strip()
    if ";" in body:
        return False, "Multiple SQL statements are not allowed."

    if not re.match(r"^select\b", body, flags=re.I):
        return False, "Only SELECT statements are allowed."

    lowered = body.lower()
    for keyword in _FORBIDDEN:
        if re.search(rf"\b{keyword}\b", lowered):
            return False, f"Statement contains the forbidden keyword '{keyword}'."

    return True, "ok"


def _apply_row_cap(sql: str) -> str:
    """Ensure the query cannot return more than MAX_SQL_ROWS rows."""
    if re.search(r"\blimit\b\s+\d+", sql, flags=re.I):
        return sql
    return f"{sql} LIMIT {config.MAX_SQL_ROWS}"


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


class SQLResult(TypedDict):
    ok: bool
    sql: str
    columns: list[str]
    rows: list[list[Any]]
    row_count: int
    error: str
    attempts: list[dict[str, str]]


def execute_sql(
    sql: str, scope: int | None = None
) -> tuple[list[str], list[list[Any]], str]:
    """Execute a validated SELECT read-only. Returns (columns, rows, error)."""
    try:
        with _read_only_connection(scope) as conn:
            cursor = conn.execute(_apply_row_cap(sql))
            fetched = cursor.fetchmany(config.MAX_SQL_ROWS)
            columns = [d[0] for d in cursor.description] if cursor.description else []
            rows = [[row[c] for c in columns] for row in fetched]
            return columns, rows, ""
    except sqlite3.Error as exc:
        return [], [], str(exc)
    except Exception as exc:  # noqa: BLE001 - never let the app crash
        return [], [], f"unexpected error: {exc}"


def run(question: str, principal=None, memory_key: str = "") -> SQLResult:
    """Full text-to-SQL pipeline with a single repair attempt.

    `principal` decides what the query can see: staff get the whole database,
    a student gets a connection where the tables are already filtered to them.
    """
    attempts: list[dict[str, str]] = []
    scope = identity.student_scope(principal)

    sql = generate_sql(question, principal, memory_key)
    ok, reason = validate_sql(sql)
    attempts.append({"sql": sql, "status": "valid" if ok else f"rejected: {reason}"})

    if ok:
        columns, rows, error = execute_sql(sql, scope)
        if not error:
            return {
                "ok": True,
                "sql": sql,
                "columns": columns,
                "rows": rows,
                "row_count": len(rows),
                "error": "",
                "attempts": attempts,
            }
        attempts[-1]["status"] = f"sqlite error: {error}"
    else:
        error = reason

    # --- one repair attempt -------------------------------------------------
    fixed = repair_sql(question, sql, error, principal)
    ok2, reason2 = validate_sql(fixed)
    attempts.append(
        {"sql": fixed, "status": "valid (retry)" if ok2 else f"rejected: {reason2}"}
    )

    if ok2:
        columns, rows, error2 = execute_sql(fixed, scope)
        if not error2:
            return {
                "ok": True,
                "sql": fixed,
                "columns": columns,
                "rows": rows,
                "row_count": len(rows),
                "error": "",
                "attempts": attempts,
            }
        attempts[-1]["status"] = f"sqlite error: {error2}"
        error = error2
    else:
        error = reason2

    return {
        "ok": False,
        "sql": fixed or sql,
        "columns": [],
        "rows": [],
        "row_count": 0,
        "error": error,
        "attempts": attempts,
    }


def rows_as_text(columns: list[str], rows: list[list[Any]], max_rows: int = 30) -> str:
    """Compact textual rendering of a result set for the composer prompt."""
    if not columns:
        return "(no columns)"
    if not rows:
        return "(no rows matched)"

    lines = [" | ".join(columns)]
    for row in rows[:max_rows]:
        lines.append(" | ".join("NULL" if v is None else str(v) for v in row))
    if len(rows) > max_rows:
        lines.append(f"... and {len(rows) - max_rows} more rows (total {len(rows)})")
    return "\n".join(lines)
