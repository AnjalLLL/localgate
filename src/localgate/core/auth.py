"""API key generation, hashing, and validation."""
import hashlib
import secrets


def generate_key(prefix: str = "lg") -> str:
    return f"{prefix}_{secrets.token_urlsafe(32)}"


def hash_key(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode()).hexdigest()
