"""Per-key limits: allowed models, max_tokens cap, token quota, expiry, and
editing them through the admin API without reissuing the key."""

from datetime import datetime, timedelta, timezone


async def _key(client, admin_headers, **limits) -> tuple[str, dict[str, str]]:
    resp = await client.post(
        "/admin/keys", headers=admin_headers, json={"name": "device", **limits}
    )
    assert resp.status_code == 201, resp.text
    body = resp.json()
    return body["id"], {"Authorization": f"Bearer {body['api_key']}"}


def _chat(model="fake-model", text="hello there", **extra):
    return {"model": model, "messages": [{"role": "user", "content": text}], **extra}


async def test_a_model_outside_the_allowed_list_is_refused(client, admin_headers):
    _, headers = await _key(client, admin_headers, allowed_models=["fake-model"])
    assert (
        await client.post("/v1/chat/completions", headers=headers, json=_chat())
    ).status_code == 200
    refused = await client.post("/v1/chat/completions", headers=headers, json=_chat("big-model"))
    assert refused.status_code == 403
    assert refused.json()["error"]["code"] == "model_not_allowed"


async def test_the_model_listing_shows_only_allowed_models(client, admin_headers, app):
    async def list_models():
        return ["fake-model", "big-model"]

    app.state.backend.list_models = list_models
    _, headers = await _key(client, admin_headers, allowed_models=["fake-model"])
    ids = [m["id"] for m in (await client.get("/v1/models", headers=headers)).json()["data"]]
    assert ids == ["fake-model"]


async def test_max_tokens_is_capped_and_filled_in(client, admin_headers, app):
    _, headers = await _key(client, admin_headers, max_tokens=64)
    await client.post("/v1/chat/completions", headers=headers, json=_chat(max_tokens=5000))
    assert app.state.backend.calls[-1]["max_tokens"] == 64
    await client.post("/v1/chat/completions", headers=headers, json=_chat())
    assert app.state.backend.calls[-1]["max_tokens"] == 64
    await client.post(
        "/v1/chat/completions",
        headers=headers,
        json=_chat(max_tokens=10, options={"num_predict": 9999}),
    )
    assert app.state.backend.calls[-1]["max_tokens"] == 10
    assert app.state.backend.calls[-1]["options"]["num_predict"] == 64


async def test_a_spent_quota_returns_429_until_the_period_resets(client, admin_headers):
    key_id, headers = await _key(client, admin_headers, token_quota=5, quota_period="day")
    first = await client.post(
        "/v1/chat/completions", headers=headers, json=_chat(text="one two three four five six")
    )
    assert first.status_code == 200  # under quota when it started
    over = await client.post("/v1/chat/completions", headers=headers, json=_chat())
    assert over.status_code == 429
    assert over.json()["error"]["code"] == "quota_exceeded"
    assert 1 <= int(over.headers["retry-after"]) <= 24 * 3600

    listed = {k["id"]: k for k in (await client.get("/admin/keys", headers=admin_headers)).json()}
    assert listed[key_id]["tokens_this_period"] >= 5

    # Raising the quota lets the same key straight back in.
    await client.patch(f"/admin/keys/{key_id}", headers=admin_headers, json={"token_quota": 10_000})
    again = await client.post("/v1/chat/completions", headers=headers, json=_chat())
    assert again.status_code == 200


async def test_an_expired_key_is_refused_with_a_clear_reason(client, admin_headers):
    past = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    key_id, headers = await _key(client, admin_headers, expires_at=past)
    refused = await client.post("/v1/chat/completions", headers=headers, json=_chat())
    assert refused.status_code == 401
    assert refused.json()["error"]["code"] == "key_expired"
    listed = (await client.get(f"/admin/keys/{key_id}", headers=admin_headers)).json()
    assert listed["expired"] is True


async def test_patch_changes_only_what_it_names_and_null_clears(client, admin_headers):
    key_id, _ = await _key(client, admin_headers, max_tokens=64, token_quota=1000)
    updated = await client.patch(
        f"/admin/keys/{key_id}",
        headers=admin_headers,
        json={"max_tokens": None, "max_concurrent": 1},
    )
    body = updated.json()
    assert (body["max_tokens"], body["token_quota"], body["max_concurrent"]) == (None, 1000, 1)
    still = await client.patch(
        f"/admin/keys/{key_id}", headers=admin_headers, json={"rate_limit_per_min": 7}
    )
    assert still.json()["token_quota"] == 1000 and still.json()["rate_limit_per_min"] == 7


async def test_a_key_holder_sees_its_own_limits(client, admin_headers):
    _, headers = await _key(client, admin_headers, allowed_models=["fake-model"], token_quota=500)
    mine = (await client.get("/v1/usage", headers=headers)).json()
    assert mine["limits"]["allowed_models"] == ["fake-model"]
    assert mine["limits"]["token_quota"] == 500
    assert mine["tokens_this_period"] == 0
