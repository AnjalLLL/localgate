"""Admin key CRUD, and the guarantee that no route is left unauthenticated."""

from fastapi.routing import APIRoute

from localgate.app import create_app


async def test_create_key_requires_admin_auth(client):
    resp = await client.post("/admin/keys", json={"name": "no-admin-header"})
    assert resp.status_code == 401


async def test_wrong_admin_key_is_rejected(client):
    resp = await client.post("/admin/keys", headers={"X-Admin-Key": "wrong"}, json={"name": "nope"})
    assert resp.status_code == 401


async def test_create_key_returns_the_raw_key_exactly_once(client, admin_headers):
    resp = await client.post("/admin/keys", headers=admin_headers, json={"name": "my-key"})
    assert resp.status_code == 201
    created = resp.json()
    assert created["api_key"].startswith("lg_")
    assert created["key_prefix"] == created["api_key"][:11]

    # Every subsequent read of the same key must not carry the secret.
    listed = (await client.get("/admin/keys", headers=admin_headers)).json()
    assert all("api_key" not in key for key in listed)

    fetched = (await client.get(f"/admin/keys/{created['id']}", headers=admin_headers)).json()
    assert "api_key" not in fetched


async def test_rate_limit_can_be_updated_without_reissuing_the_key(client, admin_headers, api_key):
    key_id, raw_key = api_key
    resp = await client.patch(
        f"/admin/keys/{key_id}", headers=admin_headers, json={"rate_limit_per_min": 1}
    )
    assert resp.status_code == 200
    assert resp.json()["rate_limit_per_min"] == 1

    # The same key still authenticates — it was updated, not replaced.
    chat = await client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {raw_key}"},
        json={"model": "fake-model", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert chat.status_code == 200

    # ...and the new limit of 1/min now bites.
    limited = await client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {raw_key}"},
        json={"model": "fake-model", "messages": [{"role": "user", "content": "again"}]},
    )
    assert limited.status_code == 429
    assert limited.json()["error"]["type"] == "rate_limit_error"


async def test_revoked_key_cannot_authenticate(client, admin_headers, api_key):
    key_id, raw_key = api_key
    assert (await client.delete(f"/admin/keys/{key_id}", headers=admin_headers)).status_code == 200

    resp = await client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {raw_key}"},
        json={"model": "fake-model", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 401


async def test_revoking_a_key_keeps_its_usage_history(client, admin_headers, api_key, auth_headers):
    """Revocation is a flag, not a delete. Deleting the row would silently rewrite
    the usage history the dashboard reports."""
    key_id, _ = api_key
    await client.post(
        "/v1/chat/completions",
        headers=auth_headers,
        json={"model": "fake-model", "messages": [{"role": "user", "content": "hi"}]},
    )
    await client.delete(f"/admin/keys/{key_id}", headers=admin_headers)

    usage = (await client.get(f"/admin/usage/{key_id}", headers=admin_headers)).json()
    assert usage["request_count"] == 1


async def test_unknown_key_id_is_a_clean_404_style_error(client, admin_headers):
    resp = await client.delete("/admin/keys/does-not-exist", headers=admin_headers)
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "key_not_found"


async def test_invalid_key_is_rejected(client):
    resp = await client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer lg_totally_made_up"},
        json={"model": "fake-model", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 401


def _api_routes(router):
    """Every APIRoute reachable from ``router``, recursively.

    FastAPI keeps an included router as a single ``_IncludedRouter`` entry in
    ``app.routes`` rather than flattening its routes into it. Iterating ``app.routes``
    directly therefore finds *no* API routes at all — which made an earlier version of
    the test below pass while checking nothing. Recursing is the difference between a
    guard rail and a decoration.
    """
    for route in getattr(router, "routes", []):
        if isinstance(route, APIRoute):
            yield route
        else:
            yield from _api_routes(route)
        # An _IncludedRouter holds the router it included.
        inner = getattr(route, "original_router", None) or getattr(route, "router", None)
        if inner is not None:
            yield from _api_routes(inner)


def test_the_guard_test_below_actually_sees_routes(settings, tmp_path):
    """Guards the guard: if route discovery silently finds nothing, the auth test
    passes vacuously and a public admin endpoint ships unnoticed."""
    app = create_app(settings, database_config_path=tmp_path / "cfg.json")
    paths = {r.path for r in _api_routes(app.router)}

    assert "/keys" in paths or "/admin/keys" in paths, f"route discovery is broken: {paths}"
    assert len(paths) > 10


def test_every_route_is_guarded(settings, tmp_path):
    """Auth is applied per route (see docs/decisions/0001), so a new route that forgets
    its dependency would be silently public. Every route must require an API key or the
    admin key, or appear on the short list of endpoints that are deliberately open.
    """
    app = create_app(settings, database_config_path=tmp_path / "cfg.json")

    deliberately_public = {
        "/health",  # a readiness probe that needed a key would be useless to a load balancer
        "/health/live",
        "/metrics",  # scrapers can't hold per-target bearer tokens; it exposes no secrets
    }

    unguarded = []
    for route in _api_routes(app.router):
        if route.path in deliberately_public:
            continue
        guards = {dep.call.__name__ for dep in route.dependant.dependencies if dep.call is not None}
        if not guards & {"require_api_key", "enforce_rate_limit", "require_admin"}:
            unguarded.append(f"{','.join(route.methods)} {route.path}")

    assert not unguarded, f"These routes require no authentication: {sorted(unguarded)}"
