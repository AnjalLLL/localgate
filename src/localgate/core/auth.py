"""API key generation, hashing, and comparison.

**Why SHA-256 and not bcrypt.** Password hashing is deliberately slow because
passwords are low-entropy and guessable — bcrypt's cost factor is what makes a
dictionary attack impractical. An API key here is 32 bytes from
:func:`secrets.token_urlsafe`: 256 bits of entropy, no dictionary to try, and
brute force off the table regardless of how fast the hash is. Meanwhile the key is
verified by lookup on *every request*, so a deliberately slow hash would add
~100ms of pure cost to the hot path and hand an attacker a cheap DoS. Fast hashing
of a high-entropy secret is correct; bcrypt exists for the opposite case. Recorded
in docs/decisions/0003-sha256-for-api-keys.md.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import secrets

KEY_PREFIX = "lg"
PREFIX_DISPLAY_LENGTH = 11  # e.g. "lg_9f3a2b1c"


def generate_key(prefix: str = KEY_PREFIX) -> str:
    """Return a raw key, shown to the user exactly once. Only its hash is stored."""
    return f"{prefix}_{secrets.token_urlsafe(32)}"


def hash_key(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode()).hexdigest()


def key_prefix(raw_key: str) -> str:
    """The leading fragment kept in plaintext so a key is identifiable in a listing."""
    return raw_key[:PREFIX_DISPLAY_LENGTH]


def constant_time_equals(a: str, b: str) -> bool:
    """Compare two secrets without leaking how much of one the caller got right.

    ``==`` on strings short-circuits at the first differing byte, so the time a
    comparison takes to fail is a function of the common prefix length. Measured
    over enough requests, that reconstructs the secret one byte at a time. This is
    the standard defence.
    """
    return hmac.compare_digest(a.encode(), b.encode())


#: Name of the dashboard's admin session cookie.
ADMIN_COOKIE = "lg_admin"
ADMIN_SESSION_SECONDS = 12 * 60 * 60


def _admin_signature(admin_key: str, expires: int, subject: str = "") -> str:
    message = f"localgate-admin-session:{expires}"
    if subject:
        message += f":{subject}"
    return hmac.new(admin_key.encode(), message.encode(), "sha256").hexdigest()


def issue_admin_session(
    admin_key: str, now: float, lifetime: int = ADMIN_SESSION_SECONDS, subject: str = ""
) -> str:
    """A stateless admin session token: ``<expiry>.<HMAC(admin key, expiry)>``, or
    ``<expiry>.<subject>.<HMAC>`` when an account signed in (``subject`` is its
    email, base64url-encoded so the dots in it can't confuse the parser).

    Signed with the admin key itself, so rotating the key logs every dashboard
    session out, and the key never has to be stored in the browser.
    """
    expires = int(now) + lifetime
    signature = _admin_signature(admin_key, expires, subject)
    if not subject:
        return f"{expires}.{signature}"
    encoded = base64.urlsafe_b64encode(subject.encode()).decode().rstrip("=")
    return f"{expires}.{encoded}.{signature}"


def admin_session_subject(token: str, admin_key: str, now: float) -> str | None:
    """Who a valid session belongs to: the account's email, ``""`` for an
    admin-key sign-in, or ``None`` when the token is invalid or expired."""
    parts = token.split(".")
    if len(parts) == 2:
        expires_text, signature = parts
        subject = ""
    elif len(parts) == 3:
        expires_text, encoded, signature = parts
        try:
            subject = base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4)).decode()
        except (ValueError, UnicodeDecodeError):
            return None
    else:
        return None
    if not expires_text.isdigit() or not signature or int(expires_text) < now:
        return None
    expected = _admin_signature(admin_key, int(expires_text), subject)
    return subject if constant_time_equals(signature, expected) else None


def verify_admin_session(token: str, admin_key: str, now: float) -> bool:
    return admin_session_subject(token, admin_key, now) is not None
