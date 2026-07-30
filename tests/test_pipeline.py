"""Router, SQL retry loop, RAG and memory - with the LLM stubbed.

No network and no API key needed, so these run anywhere.
"""

import pytest

from app import compose, config, llm, memory, rag_tool, router, sql_tool


@pytest.fixture
def scripted(monkeypatch):
    """Feed canned model replies in order and capture the prompts sent."""

    def install(replies):
        remaining = list(replies)
        calls = []

        def fake_chat(messages, temperature=0.2, max_tokens=512, model=None):
            calls.append(messages)
            return remaining.pop(0) if remaining else ""

        monkeypatch.setattr(llm, "chat", fake_chat)
        return calls

    return install


# --- router ---------------------------------------------------------------


@pytest.mark.parametrize(
    "reply,expected",
    [
        ('{"tool": "query_database", "reason": "counts"}', "query_database"),
        ('```json\n{"tool": "search_documents", "reason": "policy"}\n```', "search_documents"),
        ('Sure: {"tool": "query_database", "reason": "x"} hope that helps', "query_database"),
        ("I think you should search the documents.", "search_documents"),  # unparseable
        ('{"tool": "rm -rf", "reason": "nope"}', "search_documents"),      # invalid tool
    ],
)
def test_router_parses_defensively(scripted, reply, expected):
    scripted([reply])
    assert router.route("anything")["tool"] == expected


# --- text to SQL ----------------------------------------------------------


def test_happy_path(scripted):
    scripted(["SELECT COUNT(*) AS n FROM students WHERE semester = 6"])
    result = sql_tool.run("how many students are in semester 6")
    assert result["ok"]
    assert result["rows"][0][0] == 15
    assert len(result["attempts"]) == 1


def test_retries_once_after_sqlite_error(scripted):
    scripted([
        "SELECT COUNT(*) FROM student",  # typo -> sqlite error
        "SELECT COUNT(*) AS n FROM students WHERE semester = 4",
    ])
    result = sql_tool.run("how many in semester 4")
    assert result["ok"]
    assert result["rows"][0][0] == 15
    assert len(result["attempts"]) == 2
    assert "sqlite error" in result["attempts"][0]["status"]


def test_gives_up_gracefully_after_second_failure(scripted):
    scripted(["DROP TABLE students", "DELETE FROM students"])
    result = sql_tool.run("delete everything")
    assert result["ok"] is False
    assert compose.compose_from_sql("q", result) == compose.FALLBACK_DB


def test_injection_leaves_database_intact(scripted):
    scripted(["SELECT 1; DROP TABLE students"] * 2)
    assert sql_tool.run("x")["ok"] is False
    _cols, rows, err = sql_tool.execute_sql("SELECT COUNT(*) FROM students")
    assert not err and rows[0][0] == 30


def test_composer_sees_rows_but_not_sql(scripted):
    scripted(["SELECT COUNT(*) AS n FROM students WHERE semester = 6"])
    result = sql_tool.run("how many students are in semester 6")
    calls = scripted(["There are 15 students."])
    compose.compose_from_sql("q", result)
    prompt = calls[0][1]["content"]
    assert "15" in prompt and "Query results" in prompt
    assert "SELECT" not in prompt


def test_spoken_answer_has_no_markdown(scripted):
    scripted(["**Bold** and `code`\n- bullet"])
    result = {"ok": True, "row_count": 1, "columns": ["n"], "rows": [[1]],
              "sql": "", "error": "", "attempts": []}
    answer = compose.compose_from_sql("q", result)
    assert not any(ch in answer for ch in "*`#")


# --- RAG ------------------------------------------------------------------


class _FakeCollection:
    def __init__(self, distances):
        self._distances = distances

    def query(self, **_kw):
        n = len(self._distances)
        return {
            "documents": [[f"chunk {i}" for i in range(n)]],
            "metadatas": [[{"source": "policy.md"} for _ in range(n)]],
            "distances": [self._distances],
        }


def test_retrieval_maps_scores(monkeypatch):
    monkeypatch.setattr(rag_tool, "get_collection", lambda: _FakeCollection([0.12, 0.44]))
    monkeypatch.setattr(rag_tool, "embed", lambda texts: [[0.0] * 384 for _ in texts])
    result = rag_tool.retrieve("q")
    assert result["ok"]
    assert result["chunks"][0]["score"] == 0.88
    assert result["chunks"][0]["source"] == "policy.md"


def test_weak_matches_are_dropped(monkeypatch):
    """An off-topic question must not be answered from a poor chunk."""
    weak = 1.0 - (config.RAG_MIN_SCORE / 2)
    monkeypatch.setattr(rag_tool, "get_collection", lambda: _FakeCollection([weak, weak]))
    monkeypatch.setattr(rag_tool, "embed", lambda texts: [[0.0] * 384 for _ in texts])
    result = rag_tool.retrieve("who won the world cup")
    assert result["ok"] is False
    assert result["chunks"] == []
    assert compose.compose_from_rag("q", result) == compose.FALLBACK_RAG


# --- memory ---------------------------------------------------------------


def test_history_is_capped_and_isolated():
    memory.clear("a")
    memory.clear("b")
    for i in range(config.MAX_HISTORY_TURNS + 3):
        memory.add_turn("a", f"q{i}", f"a{i}")
    memory.add_turn("b", "other", "other")

    turns = memory.get_turns("a")
    assert len(turns) == config.MAX_HISTORY_TURNS
    assert turns[-1]["question"] == f"q{config.MAX_HISTORY_TURNS + 2}"
    assert len(memory.get_turns("b")) == 1
    memory.clear("a")
    assert memory.get_turns("a") == []


def test_sql_prompt_sees_only_recent_turns():
    """Long histories degrade SQL accuracy, so the SQL prompt truncates."""
    memory.clear("h")
    for i in range(config.MAX_HISTORY_TURNS):
        memory.add_turn("h", f"question number {i}", f"answer number {i}")
    short = memory.history_text("h", limit=config.SQL_HISTORY_TURNS)
    assert "question number 0" not in short
    assert f"question number {config.MAX_HISTORY_TURNS - 1}" in short
    memory.clear("h")


def test_idle_sessions_are_evicted(monkeypatch):
    memory.clear("old")
    memory.add_turn("old", "q", "a")
    assert memory.get_turns("old")

    # Jump past the TTL; the next write should sweep the stale session.
    real_time = memory.time.time
    monkeypatch.setattr(
        memory.time, "time", lambda: real_time() + config.SESSION_TTL_SECONDS + 10
    )
    memory.add_turn("fresh", "q", "a")
    assert memory.get_turns("old") == []
    memory.clear("fresh")
