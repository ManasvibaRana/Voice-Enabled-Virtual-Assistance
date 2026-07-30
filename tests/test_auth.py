"""Accounts, passwords and tokens."""

import sqlite3
from datetime import datetime, timedelta, timezone

import jwt
import pytest

from app import auth, config


@pytest.fixture(autouse=True)
def clean_test_users():
    """Remove anything these tests create, leaving the seeded accounts alone."""
    yield
    with sqlite3.connect(config.DB_PATH) as conn:
        conn.execute(
            "DELETE FROM users WHERE username NOT IN ('staff', 'CS06020')"
        )
        conn.execute(
            "UPDATE users SET failed_attempts = 0, locked_until = NULL, "
            "token_version = 1"
        )
        conn.commit()


def _free_student() -> dict:
    """A seeded student with no account yet."""
    with sqlite3.connect(config.DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT id, name, roll_no FROM students WHERE id NOT IN "
            "(SELECT student_id FROM users WHERE student_id IS NOT NULL) LIMIT 1"
        ).fetchone()
    return dict(row)


# --- passwords ------------------------------------------------------------


def test_hash_round_trip():
    hashed = auth.hash_password("correct horse battery")
    assert hashed != "correct horse battery"
    assert auth.verify_password("correct horse battery", hashed)
    assert not auth.verify_password("wrong password", hashed)


def test_hashes_are_salted():
    """Two users with the same password must not share a hash."""
    assert auth.hash_password("samepassword") != auth.hash_password("samepassword")


@pytest.mark.parametrize(
    "password,rejected",
    [
        ("short", True),
        ("12345678", False),
        ("x" * 73, True),      # bcrypt silently truncates past 72 bytes
        ("x" * 72, False),
        ("", True),
    ],
)
def test_password_policy(password, rejected):
    assert bool(auth.password_problem(password)) is rejected


def test_verify_survives_a_corrupt_hash():
    assert auth.verify_password("anything", "not-a-bcrypt-hash") is False


# --- signup ---------------------------------------------------------------


def test_signup_claims_a_student_record():
    student = _free_student()
    user, error = auth.signup(student["roll_no"], student["name"], "goodpassword")
    assert error == ""
    assert user is not None
    assert user["role"] == "student"
    assert user["student_id"] == student["id"]


def test_signup_rejects_unknown_roll_number():
    user, error = auth.signup("ZZ99999", "Nobody At All", "goodpassword")
    assert user is None
    assert "not on our records" in error


@pytest.mark.parametrize("mangle", [
    lambda n: n.upper(),
    lambda n: n.lower(),
    lambda n: "  " + n + "  ",
    lambda n: n.replace(" ", "   "),
])
def test_signup_tolerates_case_and_spacing_in_the_name(mangle):
    """A stray space or a capital should not lock a real student out."""
    student = _free_student()
    user, error = auth.signup(student["roll_no"], mangle(student["name"]), "goodpassword")
    assert error == "", f"rejected {mangle(student['name'])!r}"
    assert user is not None


def test_signup_accepts_a_lowercase_roll_number():
    student = _free_student()
    user, error = auth.signup(student["roll_no"].lower(), student["name"], "goodpassword")
    assert error == "" and user is not None


def test_signup_rejects_name_mismatch():
    student = _free_student()
    user, error = auth.signup(student["roll_no"], "Someone Else", "goodpassword")
    assert user is None
    assert "does not match" in error


def test_signup_rejects_an_already_claimed_record():
    student = _free_student()
    first, error = auth.signup(student["roll_no"], student["name"], "goodpassword")
    assert first is not None and error == ""
    second, error2 = auth.signup(student["roll_no"], student["name"], "otherpassword")
    assert second is None
    assert "already exists" in error2


def test_signup_enforces_the_password_policy():
    student = _free_student()
    user, error = auth.signup(student["roll_no"], student["name"], "short")
    assert user is None
    assert "at least" in error


# --- login ----------------------------------------------------------------


def test_login_succeeds_with_the_seeded_account():
    token, user, error = auth.login("CS06020", "demo1234")
    assert error == "" and user is not None and token


def test_login_is_case_insensitive_on_username():
    _token, user, error = auth.login("cs06020", "demo1234")
    assert user is not None and error == ""


def test_no_user_enumeration():
    """Unknown user and wrong password must be indistinguishable."""
    _t1, u1, e1 = auth.login("CS06020", "definitely-wrong")
    _t2, u2, e2 = auth.login("NOSUCHUSER", "definitely-wrong")
    assert u1 is None and u2 is None
    assert e1 == e2 == auth.BAD_CREDENTIALS


def test_lockout_after_repeated_failures():
    for _ in range(auth.MAX_FAILED_ATTEMPTS):
        auth.login("CS06020", "wrong")
    # Even the CORRECT password is refused while locked out.
    _token, user, error = auth.login("CS06020", "demo1234")
    assert user is None
    assert "Too many failed attempts" in error


def test_successful_login_clears_the_failure_count():
    auth.login("CS06020", "wrong")
    auth.login("CS06020", "wrong")
    _token, user, _error = auth.login("CS06020", "demo1234")
    assert user is not None
    with sqlite3.connect(config.DB_PATH) as conn:
        attempts = conn.execute(
            "SELECT failed_attempts FROM users WHERE username = 'CS06020'"
        ).fetchone()[0]
    assert attempts == 0


# --- tokens ---------------------------------------------------------------


def test_token_round_trip():
    token, user, _error = auth.login("CS06020", "demo1234")
    decoded = auth.decode_token(token)
    assert decoded is not None
    assert decoded["id"] == user["id"]
    assert decoded["role"] == "student"


def test_tampered_token_is_rejected():
    token, _user, _error = auth.login("CS06020", "demo1234")
    assert auth.decode_token(token[:-3] + "aaa") is None
    assert auth.decode_token("not.a.token") is None
    assert auth.decode_token("") is None


def test_token_signed_with_another_key_is_rejected():
    forged = jwt.encode({"sub": "1", "role": "staff", "tv": 1}, "attacker-key",
                        algorithm="HS256")
    assert auth.decode_token(forged) is None


def test_expired_token_is_rejected():
    user = auth.decode_token(auth.login("CS06020", "demo1234")[0])
    past = datetime.now(timezone.utc) - timedelta(minutes=5)
    expired = jwt.encode(
        {"sub": str(user["id"]), "role": user["role"], "tv": user["token_version"],
         "iat": past - timedelta(minutes=1), "exp": past},
        auth.jwt_secret(), algorithm="HS256",
    )
    assert auth.decode_token(expired) is None


def test_logout_invalidates_existing_tokens():
    """The headline property of the token_version design."""
    token, user, _error = auth.login("CS06020", "demo1234")
    assert auth.decode_token(token) is not None

    auth.logout(user["id"])
    assert auth.decode_token(token) is None, "old token still valid after logout"

    # A fresh login still works and issues a usable token.
    new_token, _user2, _error2 = auth.login("CS06020", "demo1234")
    assert auth.decode_token(new_token) is not None


def test_bearer_header_parsing():
    token, _user, _error = auth.login("CS06020", "demo1234")
    assert auth.user_from_header(f"Bearer {token}") is not None
    assert auth.user_from_header(f"bearer {token}") is not None
    assert auth.user_from_header(token) is None            # missing scheme
    assert auth.user_from_header(f"Basic {token}") is None  # wrong scheme
    assert auth.user_from_header(None) is None
