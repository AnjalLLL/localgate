"""Optional recoverable copies of API keys, so an admin can view a key again later.

The key itself is still stored only as a hash for authentication. Next to it, an
AES-256-GCM ciphertext of the raw key is kept, encrypted with a key derived from
the admin key (HKDF-SHA256). Consequences, all deliberate:

* Only someone holding the admin key can reveal a key — the same person who could
  simply issue a new one.
* Rotating the admin key makes every stored copy unreadable: a built-in kill switch.
* With the well-known default admin key nothing is stored, since anyone could
  decrypt it.
* The key id is bound in as associated data, so a ciphertext copied onto another
  key's row fails to decrypt.
"""

from __future__ import annotations

import base64
import os

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from localgate.config import INSECURE_ADMIN_KEY

_INFO = b"localgate api-key reveal v1"


def _cipher(admin_key: str) -> AESGCM:
    secret = HKDF(algorithm=hashes.SHA256(), length=32, salt=None, info=_INFO).derive(
        admin_key.encode()
    )
    return AESGCM(secret)


def can_store(admin_key: str) -> bool:
    return bool(admin_key) and admin_key != INSECURE_ADMIN_KEY


def seal(raw_key: str, key_id: str, admin_key: str) -> str | None:
    """The stored form of ``raw_key``, or None when storing it would not be safe."""
    if not can_store(admin_key):
        return None
    nonce = os.urandom(12)
    sealed = _cipher(admin_key).encrypt(nonce, raw_key.encode(), key_id.encode())
    return base64.b64encode(nonce + sealed).decode()


def unseal(stored: str, key_id: str, admin_key: str) -> str | None:
    """The raw key, or None if it can't be recovered (e.g. the admin key changed)."""
    try:
        blob = base64.b64decode(stored)
        return _cipher(admin_key).decrypt(blob[:12], blob[12:], key_id.encode()).decode()
    except (InvalidTag, ValueError):
        return None
