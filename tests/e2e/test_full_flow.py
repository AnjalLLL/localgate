"""End-to-end: create key -> chat -> chat again in the same session (memory
gets written) -> usage recorded -> revoked key can no longer authenticate.

Exercises the same real request path a user hits through the dashboard, just
via httpx against the FakeBackend instead of a browser + real Ollama.
"""


async def test_full_flow(client, admin_headers):
    # 1. Create a key
    create_resp = await client.post("/admin/keys", headers=admin_headers, json={"name": "e2e-key"})
    assert create_resp.status_code == 200
    key_data = create_resp.json()
    key_id, raw_key = key_data["id"], key_data["api_key"]
    auth_headers = {"Authorization": f"Bearer {raw_key}"}

    # 2. First chat turn
    resp1 = await client.post(
        "/v1/chat/completions",
        headers={**auth_headers, "X-Session-Id": "e2e-session"},
        json={"model": "fake-model", "messages": [{"role": "user", "content": "Remember I like tea"}]},
    )
    assert resp1.status_code == 200

    # 3. Second chat turn, same session
    resp2 = await client.post(
        "/v1/chat/completions",
        headers={**auth_headers, "X-Session-Id": "e2e-session"},
        json={"model": "fake-model", "messages": [{"role": "user", "content": "What do I like?"}]},
    )
    assert resp2.status_code == 200

    # 4. Chat history is retrievable
    history_resp = await client.get("/v1/conversations/e2e-session", headers=auth_headers)
    assert history_resp.status_code == 200
    history = history_resp.json()
    assert len(history) == 4  # 2 user + 2 assistant messages

    # 5. Usage was recorded across both turns
    usage_resp = await client.get(f"/admin/usage/{key_id}", headers=admin_headers)
    assert usage_resp.json()["request_count"] == 2

    # 6. Revoke, then confirm the key is dead
    revoke_resp = await client.delete(f"/admin/keys/{key_id}", headers=admin_headers)
    assert revoke_resp.status_code == 200

    dead_resp = await client.post(
        "/v1/chat/completions",
        headers=auth_headers,
        json={"model": "fake-model", "messages": [{"role": "user", "content": "still there?"}]},
    )
    assert dead_resp.status_code == 401
