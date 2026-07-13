"""The /v1/chat/completions pipeline, end to end against the FakeBackend."""

import json

import pytest
from sqlalchemy import select

from localgate.db.models import UsageRecord


async def test_chat_completion_roundtrip(client, auth_headers):
    resp = await client.post(
        "/v1/chat/completions",
        headers={**auth_headers, "X-Session-Id": "session-a"},
        json={"model": "fake-model", "messages": [{"role": "user", "content": "hello there"}]},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["choices"][0]["message"]["content"] == "Echo: hello there"
    assert body["usage"]["total_tokens"] > 0


async def test_missing_auth_is_rejected_in_the_openai_error_shape(client):
    """The OpenAI SDK parses failures out of {"error": {...}}. FastAPI's default
    {"detail": ...} envelope would reach those clients as an unreadable error."""
    resp = await client.post(
        "/v1/chat/completions",
        json={"model": "fake-model", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 401
    error = resp.json()["error"]
    assert error["type"] == "authentication_error"
    assert error["code"] == "missing_api_key"


@pytest.mark.parametrize(
    ("body", "reason"),
    [
        ({"model": "fake-model"}, "messages is required"),
        ({"model": "fake-model", "messages": []}, "messages cannot be empty"),
        (
            {
                "model": "fake-model",
                "messages": [{"role": "user", "content": "hi"}],
                "temperature": 9,
            },
            "temperature is out of range",
        ),
    ],
)
async def test_malformed_bodies_are_422_not_500(client, auth_headers, body, reason):
    """A bad request must be the caller's fault, not a crash: the pre-Pydantic
    handler read body["messages"] directly and turned a missing field into a 500."""
    resp = await client.post("/v1/chat/completions", headers=auth_headers, json=body)
    assert resp.status_code == 422, reason
    assert resp.json()["error"]["type"] == "invalid_request_error"


async def test_usage_is_recorded_after_chat(client, admin_headers, api_key, auth_headers):
    key_id, _ = api_key
    await client.post(
        "/v1/chat/completions",
        headers={**auth_headers, "X-Session-Id": "session-b"},
        json={"model": "fake-model", "messages": [{"role": "user", "content": "count my tokens"}]},
    )
    usage = (await client.get(f"/admin/usage/{key_id}", headers=admin_headers)).json()
    assert usage["request_count"] == 1
    assert usage["total_tokens"] > 0


async def test_streaming_returns_sse_frames_and_terminates_with_done(client, auth_headers):
    async with client.stream(
        "POST",
        "/v1/chat/completions",
        headers={**auth_headers, "X-Session-Id": "stream-1"},
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "stream this"}],
            "stream": True,
        },
    ) as resp:
        assert resp.status_code == 200
        lines = [line async for line in resp.aiter_lines() if line.startswith("data:")]

    assert lines[-1] == "data: [DONE]"
    text = "".join(
        json.loads(line[5:])["choices"][0]["delta"].get("content", "") for line in lines[:-1]
    )
    assert "Echo: stream this" in text


async def test_streaming_still_records_usage_after_the_body_is_sent(client, auth_headers, app):
    """The streaming path persists the turn *after* the response is sent, by which
    point FastAPI has torn down the request's dependencies — so it must open its own
    session rather than reuse the injected one."""
    async with client.stream(
        "POST",
        "/v1/chat/completions",
        headers={**auth_headers, "X-Session-Id": "stream-2"},
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "persist me"}],
            "stream": True,
        },
    ) as resp:
        [line async for line in resp.aiter_lines()]  # drain

    async with app.state.session_factory() as session:
        records = (await session.execute(select(UsageRecord))).scalars().all()

    assert len(records) == 1
    assert records[0].completion_tokens > 0


async def test_model_alias_is_resolved_before_the_backend_sees_it(client, auth_headers, app):
    app.state.settings.model_aliases = {"fast": "phi4-mini"}

    await client.post(
        "/v1/chat/completions",
        headers=auth_headers,
        json={"model": "fast", "messages": [{"role": "user", "content": "hi"}]},
    )

    assert app.state.backend.calls[-1]["model"] == "phi4-mini"


async def test_unknown_fields_are_forwarded_to_the_backend(client, auth_headers, app):
    """Backends keep adding sampling knobs. A gateway that dropped the ones it does
    not know about would quietly change the caller's results."""
    await client.post(
        "/v1/chat/completions",
        headers=auth_headers,
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "hi"}],
            "top_k": 40,
            "repeat_penalty": 1.1,
        },
    )

    forwarded = app.state.backend.calls[-1]
    assert forwarded["top_k"] == 40
    assert forwarded["repeat_penalty"] == 1.1
