"""How the chat pipeline behaves when things break.

The rule this file encodes: a failure in something *optional* (memory, caching,
summarization) must degrade the answer, never withhold it. A failure in something
*essential* (the backend that produces the answer) must be reported clearly, with a
message that says what to do about it.
"""

import json

import httpx
import pytest
from sqlalchemy import select

from localgate.db.models import ConversationMessage, MemoryChunk


@pytest.fixture
def broken_backend(app):
    """The inference backend is down."""

    async def chat(_request):
        raise httpx.ConnectError("Connection refused")

    async def chat_stream(_request):
        raise httpx.ConnectError("Connection refused")
        yield  # pragma: no cover — makes this an async generator

    app.state.backend.chat = chat
    app.state.backend.chat_stream = chat_stream
    return app.state.backend


@pytest.fixture
def no_embeddings(app):
    """The chat model works, but the embedding model isn't pulled."""

    async def embed(_text, _model):
        request = httpx.Request("POST", "http://localhost:11434/api/embeddings")
        response = httpx.Response(404, request=request, text="model not found")
        raise httpx.HTTPStatusError("Not Found", request=request, response=response)

    app.state.backend.embed = embed
    return app.state.backend


async def test_an_unreachable_backend_is_a_502_that_says_what_to_do(
    client, auth_headers, broken_backend
):
    resp = await client.post(
        "/v1/chat/completions",
        headers=auth_headers,
        json={"model": "fake-model", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 502

    error = resp.json()["error"]
    assert error["type"] == "backend_error"
    # Not "Connection refused" — a message the reader can act on.
    assert "Could not reach the inference backend" in error["message"]


async def test_a_backend_failure_mid_stream_is_reported_inside_the_stream(
    client, auth_headers, broken_backend
):
    """The 200 and its headers are already on the wire by the time the backend fails,
    so this cannot become a 502. It has to be reported in-band — and still terminate
    with [DONE], because the client is blocking on that sentinel."""
    async with client.stream(
        "POST",
        "/v1/chat/completions",
        headers=auth_headers,
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
    ) as resp:
        assert resp.status_code == 200  # cannot be retracted
        lines = [line async for line in resp.aiter_lines() if line.startswith("data:")]

    assert lines[-1] == "data: [DONE]"  # the client is never left hanging

    payload = json.loads(lines[0][len("data:") :])
    assert payload["error"]["type"] == "backend_error"
    assert "Could not reach the inference backend" in payload["error"]["message"]


async def test_a_missing_embedding_model_degrades_memory_but_still_answers(
    client, auth_headers, no_embeddings
):
    """The user asked a question. Answering it without memory is a worse answer;
    refusing to answer at all because the *embedding* model is missing is worse still."""
    resp = await client.post(
        "/v1/chat/completions",
        headers={**auth_headers, "X-Session-Id": "no-embed"},
        json={"model": "fake-model", "messages": [{"role": "user", "content": "hello"}]},
    )

    assert resp.status_code == 200
    assert resp.json()["choices"][0]["message"]["content"] == "Echo: hello"


async def test_a_failed_memory_write_still_records_the_conversation_and_usage(
    client, auth_headers, no_embeddings, app
):
    """Embedding happens after the client already has its answer, so losing a chunk
    costs the caller nothing. The history and the billing must still be written."""
    await client.post(
        "/v1/chat/completions",
        headers={**auth_headers, "X-Session-Id": "partial"},
        json={"model": "fake-model", "messages": [{"role": "user", "content": "hello"}]},
    )

    async with app.state.session_factory() as session:
        messages = (
            (
                await session.execute(
                    select(ConversationMessage).where(ConversationMessage.session_id == "partial")
                )
            )
            .scalars()
            .all()
        )
        chunks = (
            (await session.execute(select(MemoryChunk).where(MemoryChunk.session_id == "partial")))
            .scalars()
            .all()
        )

    assert len(messages) == 2  # history survived
    assert chunks == []  # memory did not, and that was survivable


async def test_a_cached_stream_is_replayed_as_sse(settings, tmp_path):
    """A cache hit on a `stream: true` request still has to *look* like a stream: the
    client is parsing SSE and blocking on [DONE]."""
    from httpx import ASGITransport, AsyncClient

    from localgate.app import create_app
    from tests.conftest import TEST_ADMIN_KEY

    settings.cache_enabled = True
    app = create_app(settings, database_config_path=tmp_path / "cfg.json")

    async with (
        AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client,
        app.router.lifespan_context(app),
    ):
        created = await client.post(
            "/admin/keys", headers={"X-Admin-Key": TEST_ADMIN_KEY}, json={"name": "k"}
        )
        headers = {"Authorization": f"Bearer {created.json()['api_key']}"}
        body = {
            "model": "fake-model",
            "messages": [{"role": "user", "content": "cache me"}],
        }

        # Prime the cache with a non-streamed request...
        await client.post("/v1/chat/completions", headers=headers, json=body)
        calls = len(app.state.backend.calls)

        # ...then ask for the same thing as a stream.
        async with client.stream(
            "POST", "/v1/chat/completions", headers=headers, json={**body, "stream": True}
        ) as resp:
            lines = [line async for line in resp.aiter_lines() if line.startswith("data:")]

    assert lines[-1] == "data: [DONE]"
    assert len(app.state.backend.calls) == calls  # served from cache, backend untouched

    chunk = json.loads(lines[0][len("data:") :])
    assert chunk["choices"][0]["delta"]["content"] == "Echo: cache me"


async def test_a_request_with_no_user_message_still_works(client, auth_headers):
    """There is nothing to retrieve memory *for*, but a system-only prompt is a legal
    request and must not trip the retrieval path."""
    resp = await client.post(
        "/v1/chat/completions",
        headers=auth_headers,
        json={"model": "fake-model", "messages": [{"role": "system", "content": "be brief"}]},
    )
    assert resp.status_code == 200


async def test_multimodal_content_parts_do_not_break_token_counting(client, auth_headers):
    """Content is a list, not a string, for multimodal messages. Memory and accounting
    only deal in text, and must flatten rather than crash."""
    resp = await client.post(
        "/v1/chat/completions",
        headers=auth_headers,
        json={
            "model": "fake-model",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "what is in this image?"},
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}},
                    ],
                }
            ],
        },
    )
    assert resp.status_code == 200
