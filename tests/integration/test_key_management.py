"""Integration tests for API key admin endpoints."""


async def test_create_key_requires_admin_auth(client):
    resp = await client.post("/admin/keys", json={"name": "no-admin-header"})
    assert resp.status_code == 401


async def test_create_key_returns_raw_key_once(client, admin_headers):
    resp = await client.post("/admin/keys", headers=admin_headers, json={"name": "my-key"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["api_key"].startswith("lg_")
    assert data["name"] == "my-key"


async def test_list_keys_shows_created_key(client, admin_headers, api_key):
    key_id, _ = api_key
    resp = await client.get("/admin/keys", headers=admin_headers)
    assert resp.status_code == 200
    ids = [k["id"] for k in resp.json()]
    assert key_id in ids


async def test_revoked_key_cannot_authenticate(client, admin_headers, api_key):
    key_id, raw_key = api_key
    revoke_resp = await client.delete(f"/admin/keys/{key_id}", headers=admin_headers)
    assert revoke_resp.status_code == 200

    chat_resp = await client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {raw_key}"},
        json={"model": "fake-model", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert chat_resp.status_code == 401


async def test_invalid_key_is_rejected(client):
    resp = await client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer lg_totally_made_up"},
        json={"model": "fake-model", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 401
