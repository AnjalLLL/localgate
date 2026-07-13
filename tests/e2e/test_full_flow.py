"""End to end: create a key, chat twice in a session, read the history back,
check the usage, then revoke the key and confirm it is dead.

This is the same request path a real user takes through the dashboard — just driven
by httpx against the FakeBackend instead of a browser against a live Ollama.
"""


async def test_full_flow(client, admin_headers):
    created = await client.post("/admin/keys", headers=admin_headers, json={"name": "e2e-key"})
    assert created.status_code == 201
    key_id, raw_key = created.json()["id"], created.json()["api_key"]
    auth_headers = {"Authorization": f"Bearer {raw_key}"}
    session_headers = {**auth_headers, "X-Session-Id": "e2e-session"}

    first = await client.post(
        "/v1/chat/completions",
        headers=session_headers,
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "Remember I like tea"}],
        },
    )
    assert first.status_code == 200

    second = await client.post(
        "/v1/chat/completions",
        headers=session_headers,
        json={"model": "fake-model", "messages": [{"role": "user", "content": "What do I like?"}]},
    )
    assert second.status_code == 200

    history = await client.get("/v1/conversations/e2e-session", headers=auth_headers)
    assert history.status_code == 200
    body = history.json()
    assert len(body["messages"]) == 4  # two user turns, two assistant turns
    assert body["memory_chunks"] >= 2

    listed = await client.get("/v1/conversations", headers=auth_headers)
    assert [s["session_id"] for s in listed.json()] == ["e2e-session"]

    usage = await client.get(f"/admin/usage/{key_id}", headers=admin_headers)
    assert usage.json()["request_count"] == 2

    overview = await client.get("/admin/usage", headers=admin_headers)
    assert overview.json()["totals"]["request_count"] == 2

    assert (await client.delete(f"/admin/keys/{key_id}", headers=admin_headers)).status_code == 200

    dead = await client.post(
        "/v1/chat/completions",
        headers=auth_headers,
        json={"model": "fake-model", "messages": [{"role": "user", "content": "still there?"}]},
    )
    assert dead.status_code == 401


async def test_one_key_cannot_read_another_keys_conversation(client, admin_headers, auth_headers):
    """A session owned by someone else must be indistinguishable from one that never
    existed — a 403 would confirm it exists and let a caller enumerate session ids."""
    await client.post(
        "/v1/chat/completions",
        headers={**auth_headers, "X-Session-Id": "private-session"},
        json={"model": "fake-model", "messages": [{"role": "user", "content": "my secret"}]},
    )

    other = await client.post("/admin/keys", headers=admin_headers, json={"name": "other"})
    other_headers = {"Authorization": f"Bearer {other.json()['api_key']}"}

    resp = await client.get("/v1/conversations/private-session", headers=other_headers)
    assert resp.status_code == 404
