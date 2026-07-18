"""AgentMemory wires the coding agent into the same conversation/RAG tables
`chat.py` uses. These tests exercise the mechanism against a real (in-memory
SQLite) database and the deterministic `FakeBackend`, the same pattern
`tests/integration/test_rag_pipeline.py` uses for the HTTP path.
"""

import pytest
from sqlalchemy import select

from localgate.agent.memory import (
    AgentMemory,
    get_or_create_local_agent_key_id,
    project_session_id,
)
from localgate.backends.fake import FakeBackend
from localgate.config import Settings
from localgate.db.engine import init_models, make_engine, make_session_factory
from localgate.db.models import MemoryChunk
from localgate.db.repositories.conversations import ConversationRepository
from localgate.db.repositories.keys import APIKeyRepository


@pytest.fixture
def settings() -> Settings:
    return Settings(
        _env_file=None,
        database_url="sqlite+aiosqlite:///:memory:",
        backend_type="fake",
        backend_url="",
        memory_enabled=True,
        embedding_model="fake",
        chunk_size=512,
        chunk_overlap=50,
        max_retrieved_chunks=5,
        memory_min_score=0.0,
    )


@pytest.fixture
async def session_factory(settings):
    engine = make_engine(settings.database_url)
    await init_models(engine)
    factory = make_session_factory(engine)
    yield factory
    await engine.dispose()


@pytest.fixture
def backend() -> FakeBackend:
    return FakeBackend()


# --------------------------------------------------------------- session id / key


def test_project_session_id_is_stable_across_calls(tmp_path):
    first = project_session_id(tmp_path)
    second = project_session_id(tmp_path)
    assert first == second


def test_project_session_id_writes_a_marker_file(tmp_path):
    session_id = project_session_id(tmp_path)
    marker = tmp_path / ".localgate" / "session_id"
    assert marker.is_file()
    assert marker.read_text().strip() == session_id


async def test_get_or_create_local_agent_key_id_is_idempotent(session_factory, settings):
    async with session_factory() as db_session:
        first = await get_or_create_local_agent_key_id(db_session, settings)
    async with session_factory() as db_session:
        second = await get_or_create_local_agent_key_id(db_session, settings)
        keys = await APIKeyRepository(db_session).list_all()
    assert first == second
    assert len(keys) == 1


# --------------------------------------------------------------------- record_turn


async def test_record_turn_persists_conversation_messages(session_factory, settings, backend):
    async with session_factory() as db_session:
        api_key_id = await get_or_create_local_agent_key_id(db_session, settings)
    memory = AgentMemory(session_factory, backend, settings, "session-1", api_key_id)

    await memory.record_turn("what does app.py do?", "it defines add()")

    async with session_factory() as db_session:
        messages = await ConversationRepository(db_session).recent("session-1")
    assert [m.role for m in messages] == ["user", "assistant"]
    assert messages[0].content == "what does app.py do?"
    assert messages[1].content == "it defines add()"


async def test_record_turn_stores_memory_chunks_when_enabled(session_factory, settings, backend):
    async with session_factory() as db_session:
        api_key_id = await get_or_create_local_agent_key_id(db_session, settings)
    memory = AgentMemory(session_factory, backend, settings, "session-1", api_key_id)

    await memory.record_turn("remember this", "noted")

    async with session_factory() as db_session:
        chunks = (await db_session.execute(select(MemoryChunk))).scalars().all()
    assert len(chunks) >= 1


async def test_record_turn_skips_memory_chunks_when_disabled(session_factory, settings, backend):
    settings.memory_enabled = False
    async with session_factory() as db_session:
        api_key_id = await get_or_create_local_agent_key_id(db_session, settings)
    memory = AgentMemory(session_factory, backend, settings, "session-1", api_key_id)

    await memory.record_turn("remember this", "noted")

    async with session_factory() as db_session:
        chunks = (await db_session.execute(select(MemoryChunk))).scalars().all()
        messages = await ConversationRepository(db_session).recent("session-1")
    assert chunks == []
    assert len(messages) == 2  # conversation history is kept even with memory off


# ------------------------------------------------------------------------- augment


async def test_augment_is_a_noop_when_memory_disabled(session_factory, settings, backend):
    settings.memory_enabled = False
    async with session_factory() as db_session:
        api_key_id = await get_or_create_local_agent_key_id(db_session, settings)
    memory = AgentMemory(session_factory, backend, settings, "session-1", api_key_id)

    messages = [{"role": "user", "content": "hello"}]
    assert await memory.augment(messages) is messages


async def test_augment_is_a_noop_with_no_user_message(session_factory, settings, backend):
    async with session_factory() as db_session:
        api_key_id = await get_or_create_local_agent_key_id(db_session, settings)
    memory = AgentMemory(session_factory, backend, settings, "session-1", api_key_id)

    messages = [{"role": "system", "content": "you are an assistant"}]
    assert await memory.augment(messages) is messages


async def test_augment_is_a_noop_with_nothing_stored_yet(session_factory, settings, backend):
    async with session_factory() as db_session:
        api_key_id = await get_or_create_local_agent_key_id(db_session, settings)
    memory = AgentMemory(session_factory, backend, settings, "session-1", api_key_id)

    messages = [{"role": "user", "content": "anything"}]
    assert await memory.augment(messages) is messages


async def test_augment_injects_a_matching_recalled_chunk(session_factory, settings, backend):
    async with session_factory() as db_session:
        api_key_id = await get_or_create_local_agent_key_id(db_session, settings)
    memory = AgentMemory(session_factory, backend, settings, "session-1", api_key_id)

    await memory.record_turn(
        "the deploy script lives in scripts/deploy.sh", "got it, noted the path"
    )

    messages = [
        {"role": "system", "content": "You are a coding assistant."},
        {"role": "user", "content": "the deploy script lives in scripts/deploy.sh"},
    ]
    augmented = await memory.augment(messages)

    assert len(augmented) == 3
    assert augmented[1]["role"] == "system"
    assert "Context recalled" in augmented[1]["content"]
    assert augmented[0]["content"] == "You are a coding assistant."  # original system prompt first
    assert augmented[2] == messages[1]  # the live user turn is untouched


async def test_augment_does_not_leak_across_sessions(session_factory, settings, backend):
    async with session_factory() as db_session:
        api_key_id = await get_or_create_local_agent_key_id(db_session, settings)
    session_a = AgentMemory(session_factory, backend, settings, "session-a", api_key_id)
    session_b = AgentMemory(session_factory, backend, settings, "session-b", api_key_id)

    await session_a.record_turn("secret project codename is falcon", "understood")

    messages = [{"role": "user", "content": "secret project codename is falcon"}]
    augmented = await session_b.augment(messages)
    assert augmented is messages  # nothing recalled — session-b has no history
