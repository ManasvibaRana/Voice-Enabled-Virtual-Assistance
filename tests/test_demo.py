"""Public-demo safeguards: per-visitor and site-wide limits, shared logins."""

import sqlite3

import pytest

from app import auth, config, demo


@pytest.fixture
def demo_on(monkeypatch):
    monkeypatch.setattr(demo, "ENABLED", True)
    demo.reset()
    yield
    demo.reset()
    with sqlite3.connect(config.DB_PATH) as conn:
        conn.execute("UPDATE users SET failed_attempts = 0, locked_until = NULL, token_version = 1")
        conn.commit()


def test_off_by_default_for_local_use(monkeypatch):
    monkeypatch.setattr(demo, "ENABLED", False)
    for _ in range(100):
        assert demo.check_question("1.2.3.4") is None


def test_visitor_limit(demo_on, monkeypatch):
    monkeypatch.setattr(demo, "QUESTIONS_PER_MINUTE", 1000)
    for i in range(demo.QUESTIONS_PER_VISITOR):
        assert demo.check_question("1.2.3.4", now=float(i)) is None
    assert demo.check_question("1.2.3.4", now=50.0) == demo.VISITOR_LIMIT_MESSAGE
    # another visitor is unaffected
    assert demo.check_question("5.6.7.8", now=50.0) is None
    # and the first one may ask again once the window has passed
    assert demo.check_question("1.2.3.4", now=50.0 + demo.VISITOR_WINDOW_SECONDS) is None


def test_site_wide_limit(demo_on):
    for i in range(demo.QUESTIONS_PER_MINUTE):
        assert demo.check_question(f"10.0.0.{i}", now=1.0) is None
    assert demo.check_question("10.0.1.1", now=1.0) == demo.SITE_LIMIT_MESSAGE
    assert demo.check_question("10.0.1.1", now=62.0) is None


def test_client_ip_prefers_forwarded_header():
    assert demo.client_ip({"x-forwarded-for": "9.9.9.9, 10.0.0.1"}, "10.0.0.1") == "9.9.9.9"
    assert demo.client_ip({}, "127.0.0.1") == "127.0.0.1"


def test_shared_demo_login_cannot_be_locked(demo_on):
    for _ in range(auth.MAX_FAILED_ATTEMPTS + 2):
        _, user, _ = auth.login("CS06020", "wrong password")
        assert user is None
    _, user, error = auth.login("CS06020", "demo1234")
    assert user is not None, error


def test_shared_demo_logout_does_not_kick_other_visitors(demo_on):
    first, user, _ = auth.login("staff", "staff1234")
    second, _, _ = auth.login("staff", "staff1234")
    assert auth.logout(user["id"])
    assert auth.decode_token(second) is not None
    assert auth.decode_token(first) is not None
