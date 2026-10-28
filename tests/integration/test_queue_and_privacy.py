"""The inference queue through the real endpoints, disconnect cleanup on a
stream, and keys that keep no conversation history."""

import asyncio

import pytest
from sqlalchemy import func, select

from localgate.api import chat as chat_module
from localgate.config import Settings
from localgate.core.scheduler import InferenceScheduler
from localgate.db.models import ConversationMessage, UsageRecord
from tests.conftest import TEST_ADMIN_KEY


@pytest.fixture
def settings() -> Settings:
    return Settings(
        _env_file=None,
        database_url="sqlite+aiosqlite:///:memory:",
        backend_type="fake",
        backend_url="",
        admin_key=TEST_ADMIN_KEY,
        log_level="WARNING",
        max_concurrent_requests=1,
        queue_timeout=0.3,
    )


def _chat(text="hello"):
    return {"model": "fake-model", "messages": [{"role": "user", "content": text}]}


@pytest.fixture
def gate(app):
    """Make the fake backend hold each request until the test lets it go."""
    release = asyncio.Event()
    original = app.state.backend.chat

    async def slow_chat(request):
        await release.wait()
        return await original(request)

    app.state.backend.chat = slow_chat
    return release


async def test_a_busy_gpu_queues_then_answers_503_with_retry_after(client, auth_headers, gate):
    first = asyncio.create_task(
        client.post("/v1/chat/completions", headers=auth_headers, json=_chat())
    )
    await asyncio.sleep(0.05)
    second = await client.post("/v1/chat/completions", headers=auth_headers, json=_chat())
    assert second.status_code == 503
    assert second.json()["error"]["code"] == "queue_timeout"
    assert int(second.headers["retry-after"]) >= 1
    gate.set()
    assert (await first).status_code == 200


async def test_a_queued_request_runs_once_a_slot_frees(client, auth_headers, gate, app):
    app.state.scheduler.queue_timeout = 5
    first = asyncio.create_task(
        client.post("/v1/chat/completions", headers=auth_headers, json=_chat())
    )
    second = asyncio.create_task(
        client.post("/v1/chat/completions", headers=auth_headers, json=_chat())
    )
    await asyncio.sleep(0.05)
    assert app.state.scheduler.stats()["queued"] == 1
    gate.set()
    assert [(await t).status_code for t in (first, second)] == [200, 200]
    assert app.state.scheduler.stats()["active"] == 0


async def test_a_keys_own_concurrency_limit_queues_it(client, admin_headers, gate, app):
    app.state.scheduler.max_concurrent = 4  # plenty of GPU room…
    created = await client.post(
        "/admin/keys", headers=admin_headers, json={"name": "phone", "max_concurrent": 1}
    )
    headers = {"Authorization": f"Bearer {created.json()['api_key']}"}
    first = asyncio.create_task(client.post("/v1/chat/completions", headers=headers, json=_chat()))
    await asyncio.sleep(0.05)
    second = await client.post("/v1/chat/completions", headers=headers, json=_chat())
    assert second.status_code == 503  # …but this key may only run one at a time
    gate.set()
    await first


async def test_a_dropped_stream_frees_its_slot_closes_upstream_and_is_billed(app, client, api_key):
    key_id, _ = api_key
    closed = asyncio.Event()

    async def endless_stream(request):
        try:
            for i in range(1000):
                yield {"choices": [{"delta": {"content": f"word{i} "}}]}
                await asyncio.sleep(0)
        finally:
            closed.set()

    app.state.backend.chat_stream = endless_stream
    scheduler = InferenceScheduler(1, 5, 10)
    lease = await scheduler.acquire(key_id)
    stream = chat_module._stream_and_record(
        lease=lease,
        store_conversations=True,
        backend=app.state.backend,
        settings=app.state.settings,
        cache=app.state.cache,
        payload={"model": "fake-model", "messages": [{"role": "user", "content": "go"}]},
        key="k",
        api_key_id=key_id,
        session_id="s1",
        model="fake-model",
        user_text="go",
        started=0.0,
        session_factory=app.state.session_factory,
    )
    for _ in range(3):
        await stream.__anext__()
    await stream.aclose()  # what the server does when the client disconnects

    assert closed.is_set()
    assert scheduler.active == 0
    await asyncio.gather(*chat_module._pending_writes)
    async with app.state.session_factory() as session:
        billed = await session.scalar(
            select(func.sum(UsageRecord.completion_tokens)).where(UsageRecord.api_key_id == key_id)
        )
    assert billed and billed > 0


async def test_a_key_that_keeps_no_history_stores_no_conversation(client, admin_headers, app):
    created = await client.post(
        "/admin/keys", headers=admin_headers, json={"name": "private", "store_conversations": False}
    )
    key_id, raw = created.json()["id"], created.json()["api_key"]
    headers = {"Authorization": f"Bearer {raw}", "X-Session-Id": "secret-chat"}
    assert (
        await client.post("/v1/chat/completions", headers=headers, json=_chat("my secret"))
    ).status_code == 200
    async with app.state.session_factory() as session:
        stored = await session.scalar(
            select(func.count())
            .select_from(ConversationMessage)
            .where(ConversationMessage.api_key_id == key_id)
        )
        usage = await session.scalar(
            select(func.count()).select_from(UsageRecord).where(UsageRecord.api_key_id == key_id)
        )
    assert stored == 0
    assert usage == 1
