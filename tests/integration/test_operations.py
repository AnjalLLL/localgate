"""Health, metrics, caching, export, and the OpenAI-compatible surface routes."""

import pytest

from localgate.app import create_app

# ------------------------------------------------------------------ health


async def test_liveness_checks_nothing_external(client):
    """A liveness probe that failed because *Ollama* was down would have Kubernetes
    restart a perfectly healthy gateway, forever, fixing nothing."""
    resp = await client.get("/health/live")
    assert resp.status_code == 200
    assert resp.json()["status"] == "alive"


async def test_readiness_reports_backend_database_and_warnings(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    body = resp.json()

    assert body["status"] == "ok"
    assert body["backend"]["reachable"] is True
    assert body["database"]["connected"] is True
    assert body["memory"]["enabled"] is True
    # Memory on SQLite scans every chunk in the session — fine locally, worth saying.
    assert any("SQLite" in warning for warning in body["warnings"])


async def test_readiness_warns_when_the_admin_key_is_still_the_placeholder(settings, tmp_path):
    """A gateway reachable on the network with the key from the docs is the failure
    this warning exists to prevent."""
    from httpx import ASGITransport, AsyncClient

    from localgate.config import INSECURE_ADMIN_KEY

    settings.admin_key = INSECURE_ADMIN_KEY
    app = create_app(settings, database_config_path=tmp_path / "cfg.json")

    async with (
        AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client,
        app.router.lifespan_context(app),
    ):
        warnings = (await client.get("/health")).json()["warnings"]

    assert any("admin key" in warning for warning in warnings)


async def test_readiness_is_503_when_the_backend_is_down(client, app, monkeypatch):
    """503 rather than 200-with-a-flag, so a load balancer can act on the status code
    alone without parsing the body."""

    async def unreachable() -> bool:
        return False

    monkeypatch.setattr(app.state.backend, "health", unreachable)

    resp = await client.get("/health")
    assert resp.status_code == 503
    assert resp.json()["status"] == "degraded"


# ----------------------------------------------------------------- metrics


async def test_metrics_are_exposed_in_prometheus_format(client, auth_headers):
    await client.post(
        "/v1/chat/completions",
        headers=auth_headers,
        json={"model": "fake-model", "messages": [{"role": "user", "content": "hi"}]},
    )

    resp = await client.get("/metrics")
    assert resp.status_code == 200
    assert "localgate_requests_total" in resp.text
    assert "localgate_tokens_total" in resp.text


async def test_metrics_label_the_route_template_not_the_concrete_path(client, auth_headers):
    """Labelling with the raw path would mint a new Prometheus time series per session
    id and blow up cardinality."""
    await client.get("/v1/conversations/some-session-id", headers=auth_headers)

    metrics = (await client.get("/metrics")).text
    assert "/v1/conversations/{session_id}" in metrics
    assert "some-session-id" not in metrics


async def test_metrics_keep_the_admin_prefix(client, admin_headers):
    """FastAPI reports an included router's route with its *un-prefixed* path, so a naive
    label reads `/keys` — silently merging admin metrics with any same-named public route."""
    await client.get("/admin/keys", headers=admin_headers)

    metrics = (await client.get("/metrics")).text
    assert 'path="/admin/keys"' in metrics


async def test_unmatched_paths_do_not_explode_metric_cardinality(client):
    """A scanner probing /wp-admin, /.env, /phpmyadmin would otherwise mint one time
    series per URL it tries, until Prometheus falls over."""
    for junk in ("/wp-admin", "/.env", "/phpmyadmin"):
        assert (await client.get(junk)).status_code == 404

    metrics = (await client.get("/metrics")).text
    assert "wp-admin" not in metrics
    assert "phpmyadmin" not in metrics
    assert "<unmatched>" in metrics


async def test_metrics_can_be_turned_off(settings, tmp_path):
    from httpx import ASGITransport, AsyncClient

    settings.metrics_enabled = False
    app = create_app(settings, database_config_path=tmp_path / "cfg.json")

    async with (
        AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client,
        app.router.lifespan_context(app),
    ):
        assert (await client.get("/metrics")).status_code == 404


async def test_every_response_carries_a_request_id(client):
    resp = await client.get("/health/live")
    assert resp.headers["X-Request-ID"]


async def test_an_inbound_request_id_is_preserved(client):
    """A request traced through a proxy should keep one identity across all the logs."""
    resp = await client.get("/health/live", headers={"X-Request-ID": "trace-me"})
    assert resp.headers["X-Request-ID"] == "trace-me"


# ------------------------------------------------------------------- cache


@pytest.fixture
def cached_app(settings, tmp_path):
    settings.cache_enabled = True
    return create_app(settings, database_config_path=tmp_path / "cfg.json")


@pytest.fixture
async def cached_client(cached_app):
    from httpx import ASGITransport, AsyncClient

    async with (
        AsyncClient(transport=ASGITransport(app=cached_app), base_url="http://test") as client,
        cached_app.router.lifespan_context(cached_app),
    ):
        yield client


async def test_an_identical_prompt_is_served_from_cache(cached_client, cached_app, admin_headers):
    created = await cached_client.post(
        "/admin/keys", headers=admin_headers, json={"name": "cache-key"}
    )
    headers = {"Authorization": f"Bearer {created.json()['api_key']}"}
    body = {"model": "fake-model", "messages": [{"role": "user", "content": "same question"}]}

    first = await cached_client.post("/v1/chat/completions", headers=headers, json=body)
    calls_after_first = len(cached_app.state.backend.calls)

    second = await cached_client.post("/v1/chat/completions", headers=headers, json=body)

    assert first.json() == second.json()
    # The backend was not asked a second time. That is the entire point.
    assert len(cached_app.state.backend.calls) == calls_after_first
    assert cached_app.state.cache.stats()["hits"] == 1


async def test_a_different_prompt_still_reaches_the_backend(
    cached_client, cached_app, admin_headers
):
    created = await cached_client.post(
        "/admin/keys", headers=admin_headers, json={"name": "cache-key"}
    )
    headers = {"Authorization": f"Bearer {created.json()['api_key']}"}

    await cached_client.post(
        "/v1/chat/completions",
        headers=headers,
        json={"model": "fake-model", "messages": [{"role": "user", "content": "question one"}]},
    )
    calls = len(cached_app.state.backend.calls)

    await cached_client.post(
        "/v1/chat/completions",
        headers=headers,
        json={"model": "fake-model", "messages": [{"role": "user", "content": "question two"}]},
    )

    assert len(cached_app.state.backend.calls) == calls + 1


async def test_caching_is_off_by_default(app):
    """Two identical requests at temperature 0.8 are *supposed* to differ; a cache
    returns the first one twice. That semantic change must be opt-in."""
    assert app.state.settings.cache_enabled is False


async def test_a_cache_hit_is_still_billed_and_still_recorded(settings, tmp_path):
    """Regression: the cache-hit path used to return early without recording the turn.

    Two things broke. Usage went unbilled — a caller could replay a cached prompt for
    free, and the dashboard would never see it. And the exchange never entered the
    conversation's history, so the next turn found no trace that it had happened. The
    cache is meant to save the *inference*, not the bookkeeping.

    Memory is off here because with it on, storing turn 1 changes the payload of turn 2
    (it now carries retrieved context) and the request correctly misses — which is the
    subject of the next test.
    """
    from httpx import ASGITransport, AsyncClient

    settings.cache_enabled = True
    settings.memory_enabled = False
    app = create_app(settings, database_config_path=tmp_path / "cfg.json")
    admin = {"X-Admin-Key": settings.admin_key}

    async with (
        AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client,
        app.router.lifespan_context(app),
    ):
        created = await client.post("/admin/keys", headers=admin, json={"name": "cache-key"})
        key_id = created.json()["id"]
        headers = {
            "Authorization": f"Bearer {created.json()['api_key']}",
            "X-Session-Id": "cached-session",
        }
        body = {"model": "fake-model", "messages": [{"role": "user", "content": "hello"}]}

        await client.post("/v1/chat/completions", headers=headers, json=body)
        await client.post("/v1/chat/completions", headers=headers, json=body)  # from cache

        assert app.state.cache.stats()["hits"] == 1
        assert len(app.state.backend.calls) == 1  # only one inference happened

        usage = (await client.get(f"/admin/usage/{key_id}", headers=admin)).json()
        history = (await client.get("/v1/conversations/cached-session", headers=headers)).json()

    assert usage["request_count"] == 2  # both billed, not just the one that inferred
    assert usage["total_tokens"] > 0
    assert len(history["messages"]) == 4  # both exchanges are in the history


async def test_stored_memory_invalidates_the_cache_for_the_next_turn(
    cached_client, cached_app, admin_headers
):
    """The cache key is the *augmented* payload — after memory injection.

    So the same prompt asked twice in one session is a miss the second time: the first
    turn is now in memory, so the second request carries retrieved context the first one
    didn't, and is genuinely a different prompt. Keying on the client's raw body instead
    would let a hit serve a response built from stale memory.
    """
    created = await cached_client.post("/admin/keys", headers=admin_headers, json={"name": "k"})
    headers = {
        "Authorization": f"Bearer {created.json()['api_key']}",
        "X-Session-Id": "memory-invalidates",
    }
    body = {"model": "fake-model", "messages": [{"role": "user", "content": "remember this"}]}

    await cached_client.post("/v1/chat/completions", headers=headers, json=body)
    await cached_client.post("/v1/chat/completions", headers=headers, json=body)

    assert cached_app.state.cache.stats()["hits"] == 0
    assert len(cached_app.state.backend.calls) == 2  # the second really did re-infer

    # And the second call carried the memory the first one created.
    system_messages = [
        m["content"]
        for m in cached_app.state.backend.calls[-1]["messages"]
        if m["role"] == "system"
    ]
    assert any("remember this" in content for content in system_messages)


# ------------------------------------------------- models / embeddings / completions


async def test_models_lists_the_backend_and_the_aliases(client, auth_headers, app):
    app.state.settings.model_aliases = {"fast": "phi4-mini"}

    resp = await client.get("/v1/models", headers=auth_headers)
    assert resp.status_code == 200

    ids = [m["id"] for m in resp.json()["data"]]
    assert "fake-model" in ids
    assert "fast" in ids  # a caller passing model:"fast" must see it is a name they can use


async def test_models_requires_a_key(client):
    assert (await client.get("/v1/models")).status_code == 401


async def test_embeddings_returns_vectors_and_bills_the_key(
    client, auth_headers, admin_headers, api_key
):
    key_id, _ = api_key
    resp = await client.post(
        "/v1/embeddings",
        headers=auth_headers,
        json={"model": "fake-embed", "input": ["one", "two"]},
    )
    assert resp.status_code == 200

    body = resp.json()
    assert len(body["data"]) == 2
    assert body["data"][0]["index"] == 0
    assert len(body["data"][0]["embedding"]) > 0

    # Embeddings consume backend capacity, so they are billed like any other call.
    usage = (await client.get(f"/admin/usage/{key_id}", headers=admin_headers)).json()
    assert usage["request_count"] == 1
    assert usage["prompt_tokens"] > 0


async def test_legacy_completions_translates_to_chat_and_back(client, auth_headers, app):
    """Local backends have largely dropped /v1/completions, so localgate implements it
    by translating to the chat route — which keeps older clients working anywhere."""
    resp = await client.post(
        "/v1/completions",
        headers=auth_headers,
        json={"model": "fake-model", "prompt": "once upon a time"},
    )
    assert resp.status_code == 200

    body = resp.json()
    assert body["object"] == "text_completion"
    assert "once upon a time" in body["choices"][0]["text"]
    assert body["usage"]["total_tokens"] > 0

    # The backend saw a chat request, because that is all it speaks.
    assert app.state.backend.calls[-1]["messages"] == [
        {"role": "user", "content": "once upon a time"}
    ]


# ------------------------------------------------------------------ export


async def test_export_returns_everything_but_the_secrets(
    client, admin_headers, auth_headers, api_key
):
    """Nobody should feel locked in. But the export must not carry key hashes."""
    await client.post(
        "/v1/chat/completions",
        headers={**auth_headers, "X-Session-Id": "exported"},
        json={"model": "fake-model", "messages": [{"role": "user", "content": "hi"}]},
    )

    resp = await client.get("/admin/export", headers=admin_headers)
    assert resp.status_code == 200

    body = resp.json()
    assert len(body["api_keys"]) == 1
    assert len(body["usage_records"]) == 1
    assert len(body["conversations"]) == 2  # user + assistant

    assert "key_hash" not in body["api_keys"][0]
    assert "deadbeef" not in resp.text


async def test_export_requires_admin(client):
    assert (await client.get("/admin/export")).status_code == 401


# ------------------------------------------------------------------ config


async def test_config_reports_the_running_state_with_credentials_redacted(client, admin_headers):
    resp = await client.get("/admin/config", headers=admin_headers)
    assert resp.status_code == 200

    body = resp.json()
    assert body["backend"]["type"] == "fake"
    assert "ollama" in body["backend"]["available_types"]
    assert body["security"]["using_default_admin_key"] is False  # the test app sets a real one
    assert body["database"]["url"].startswith("sqlite")


def test_database_url_redaction_hides_the_password():
    from localgate.api.config import redact_database_url

    redacted = redact_database_url("postgresql+asyncpg://user:hunter2@host/db")
    assert "hunter2" not in redacted
    assert "host/db" in redacted
