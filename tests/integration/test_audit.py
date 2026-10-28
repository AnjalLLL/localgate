"""The admin audit log: key changes and sign-ins are recorded with who did them."""

from localgate.db.repositories.users import UserRepository
from tests.conftest import TEST_ADMIN_KEY

ADMIN = {"X-Admin-Key": TEST_ADMIN_KEY}


async def _actions(client):
    return [
        (e["actor"], e["action"]) for e in (await client.get("/admin/audit", headers=ADMIN)).json()
    ]


async def test_key_lifecycle_is_audited(client):
    created = (await client.post("/admin/keys", headers=ADMIN, json={"name": "phone"})).json()
    await client.patch(f"/admin/keys/{created['id']}", headers=ADMIN, json={"max_tokens": 64})
    await client.post(f"/admin/keys/{created['id']}/reveal", headers=ADMIN)
    await client.delete(f"/admin/keys/{created['id']}", headers=ADMIN)
    events = (await client.get("/admin/audit", headers=ADMIN)).json()
    assert [e["action"] for e in events] == [
        "key.revoked",
        "key.revealed",
        "key.updated",
        "key.created",
    ]
    assert events[2]["details"]["changes"] == {"max_tokens": 64}
    assert all(e["actor"] == "admin-key" and e["target"] == created["id"] for e in events)


async def test_sign_ins_record_the_account_and_failures(client, app):
    async with app.state.session_factory() as session:
        await UserRepository(session).create("ana@example.com", "correct horse")
    await client.post(
        "/admin/session", json={"email": "ana@example.com", "password": "wrong horse"}
    )
    await client.post(
        "/admin/session", json={"email": "ana@example.com", "password": "correct horse"}
    )
    # Signed in by cookie now: the next action is attributed to the account.
    await client.post(
        "/admin/keys", headers={"X-Localgate-Csrf": "1"}, json={"name": "from-dashboard"}
    )
    assert (await _actions(client))[:3] == [
        ("ana@example.com", "key.created"),
        ("ana@example.com", "sign_in"),
        ("ana@example.com", "sign_in.failed"),
    ]
