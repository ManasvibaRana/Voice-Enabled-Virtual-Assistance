"""The SQL guards. These are the security-relevant tests - keep them green."""

import sqlite3

import pytest

from app import config, sql_tool


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT COUNT(*) FROM students",
        "select name from students where semester=6",
        "SELECT * FROM students; ",  # one trailing semicolon is tolerated
        "SELECT name FROM students WHERE id IN (SELECT student_id FROM marks)",
    ],
)
def test_accepts_plain_selects(sql):
    ok, reason = sql_tool.validate_sql(sql)
    assert ok, reason


@pytest.mark.parametrize(
    "sql",
    [
        "DROP TABLE students",
        "INSERT INTO students VALUES (1)",
        "UPDATE students SET name='x'",
        "DELETE FROM students",
        "PRAGMA table_info(students)",
        "ATTACH DATABASE 'x' AS y",
        "SELECT 1; DROP TABLE students",  # stacked statements
        "SELECT 1 -- comment",
        "SELECT 1 /* c */",
        "",
    ],
)
def test_rejects_dangerous_sql(sql):
    ok, _ = sql_tool.validate_sql(sql)
    assert not ok, f"validator accepted {sql!r}"


def test_connection_is_genuinely_read_only():
    """Not just prompt guidance: SQLite itself must refuse the write."""
    conn = sql_tool._read_only_connection()
    try:
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            conn.execute("CREATE TABLE hack (x INT)")
    finally:
        conn.close()


def test_row_cap_is_enforced():
    _cols, rows, err = sql_tool.execute_sql("SELECT * FROM attendance")
    assert not err
    assert len(rows) == config.MAX_SQL_ROWS


def test_bad_sql_returns_error_instead_of_raising():
    _cols, _rows, err = sql_tool.execute_sql("SELECT nope FROM students")
    assert err


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("```sql\nSELECT COUNT(*) FROM students;\n```", "SELECT COUNT(*) FROM students"),
        ("SQL: SELECT 1", "SELECT 1"),
        ("Here is the query:\nSELECT 1", "SELECT 1"),
    ],
)
def test_cleans_model_formatting(raw, expected):
    assert sql_tool._clean_sql(raw) == expected


def test_schema_context_exposes_tables_and_values():
    ctx = sql_tool.schema_context(force=True)
    for table in ("students", "subjects", "marks", "attendance"):
        assert table in ctx
    # Low-cardinality values are listed so the model matches exact spellings.
    assert "Artificial Intelligence" in ctx
    assert "midterm" in ctx
