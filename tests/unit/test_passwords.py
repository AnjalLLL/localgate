"""Password hashing and account-carrying session tokens."""

from localgate.core.auth import admin_session_subject, issue_admin_session, verify_admin_session
from localgate.core.passwords import hash_password, password_problem, verify_password


def test_a_hash_verifies_only_its_own_password_and_is_salted():
    stored = hash_password("correct horse")
    assert stored.startswith("scrypt$") and "correct horse" not in stored
    assert verify_password("correct horse", stored)
    assert not verify_password("wrong horse", stored)
    assert hash_password("correct horse") != stored  # random salt


def test_malformed_hashes_never_verify():
    for stored in ("", "plain", "scrypt$1$2$3", "bcrypt$a$b$c$d$e"):
        assert not verify_password("x", stored)


def test_short_passwords_are_refused():
    assert password_problem("short") is not None
    assert password_problem("long enough") is None


def test_session_tokens_carry_the_account_and_reject_tampering():
    token = issue_admin_session("server-secret", 1000.0, subject="ana@example.com")
    assert admin_session_subject(token, "server-secret", 1001.0) == "ana@example.com"
    assert admin_session_subject(token, "other-secret", 1001.0) is None
    expires, encoded, signature = token.split(".")
    forged = f"{expires}.{encoded.swapcase()}.{signature}"
    assert admin_session_subject(forged, "server-secret", 1001.0) is None


def test_admin_key_sessions_still_work_with_no_subject():
    token = issue_admin_session("server-secret", 1000.0)
    assert token.count(".") == 1
    assert admin_session_subject(token, "server-secret", 1001.0) == ""
    assert verify_admin_session(token, "server-secret", 1001.0)
