"""Revealing an API key again: admin-only, encrypted at rest, and impossible
for legacy keys, revoked keys, the default admin key or after admin-key rotation."""

from localgate.config import INSECURE_ADMIN_KEY
from localgate.core import key_vault
from localgate.db.repositories.keys import APIKeyRepository


async def test_a_new_key_can_be_revealed_by_the_admin(client, admin_headers):
    created = (await client.post("/admin/keys", headers=admin_headers, json={"name": "a"})).json()
    listed = (await client.get("/admin/keys", headers=admin_headers)).json()
    assert next(k for k in listed if k["id"] == created["id"])["revealable"] is True

    revealed = await client.post(f"/admin/keys/{created['id']}/reveal", headers=admin_headers)
    assert revealed.status_code == 200
    assert revealed.json()["api_key"] == created["api_key"]


async def test_reveal_requires_admin(client, admin_headers, auth_headers):
    created = (await client.post("/admin/keys", headers=admin_headers, json={"name": "a"})).json()
    denied = await client.post(f"/admin/keys/{created['id']}/reveal", headers=auth_headers)
    assert denied.status_code == 401


async def test_the_raw_key_is_not_stored_in_plain_text(app, client, admin_headers, db_session):
    created = (await client.post("/admin/keys", headers=admin_headers, json={"name": "a"})).json()
    row = await APIKeyRepository(db_session).get(created["id"])
    assert row is not None and row.key_ciphertext
    assert created["api_key"] not in row.key_ciphertext


async def test_legacy_and_revoked_keys_cannot_be_revealed(client, admin_headers, db_session):
    legacy, _ = await APIKeyRepository(db_session).create("legacy")  # no admin key: no copy
    resp = await client.post(f"/admin/keys/{legacy.id}/reveal", headers=admin_headers)
    assert resp.status_code == 400 and "can't be shown again" in resp.text

    created = (await client.post("/admin/keys", headers=admin_headers, json={"name": "b"})).json()
    await client.delete(f"/admin/keys/{created['id']}", headers=admin_headers)
    resp = await client.post(f"/admin/keys/{created['id']}/reveal", headers=admin_headers)
    assert resp.status_code == 400 and "revoked" in resp.text


def test_rotating_the_admin_key_makes_copies_unreadable_and_ids_are_bound():
    sealed = key_vault.seal("lg_secret", "key-1", "admin-one")
    assert sealed is not None
    assert key_vault.unseal(sealed, "key-1", "admin-one") == "lg_secret"
    assert key_vault.unseal(sealed, "key-1", "admin-two") is None
    assert key_vault.unseal(sealed, "key-2", "admin-one") is None


def test_nothing_is_stored_with_the_default_admin_key():
    assert key_vault.seal("lg_secret", "key-1", INSECURE_ADMIN_KEY) is None
