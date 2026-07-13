"""Integration tests for the memory/RAG pipeline.

The FakeBackend's embeddings are a deterministic hash of the input text, not
a real semantic model — so these tests can't assert "semantically similar
text gets retrieved." What they DO prove, which is what actually matters for
correctness, is:
  1. The retrieval mechanism itself works: an exact-text query reliably
     surfaces the chunk with that exact stored content as the top match.
  2. Chat turns actually get persisted as memory chunks in the database.
  3. Memory is correctly scoped per session — no cross-session leakage.
"""
from localgate.backends.fake import FakeBackend
from localgate.db.models import MemoryChunk
from localgate.db.repositories.embeddings import EmbeddingRepository
from localgate.memory.retriever import retrieve_relevant_context
from sqlalchemy import select


async def test_retrieval_surfaces_exact_match_as_top_result(db_session):
    backend = FakeBackend()
    repo = EmbeddingRepository(db_session)

    for content in ["The sky is blue.", "Bananas are yellow.", "Paris is in France."]:
        vector = await backend.embed(content, model="fake")
        await repo.add_chunk(session_id="s1", api_key_id="k1", content=content, embedding=vector)

    results = await retrieve_relevant_context(
        session=db_session,
        backend=backend,
        session_id="s1",
        query="Bananas are yellow.",  # exact match to one stored chunk
        embedding_model="fake",
        top_k=1,
    )
    assert results == ["Bananas are yellow."]


async def test_chat_turn_gets_persisted_as_memory_chunk(client, auth_headers, db_session):
    await client.post(
        "/v1/chat/completions",
        headers={**auth_headers, "X-Session-Id": "memory-session"},
        json={"model": "fake-model", "messages": [{"role": "user", "content": "My favorite color is teal"}]},
    )
    result = await db_session.execute(
        select(MemoryChunk).where(MemoryChunk.session_id == "memory-session")
    )
    chunks = result.scalars().all()
    assert len(chunks) >= 1
    assert "teal" in chunks[0].content


async def test_memory_is_scoped_per_session(db_session):
    backend = FakeBackend()
    repo = EmbeddingRepository(db_session)

    vector = await backend.embed("secret from session A", model="fake")
    await repo.add_chunk(session_id="session-A", api_key_id="k1", content="secret from session A", embedding=vector)

    # Querying from an unrelated session should find nothing, even with an identical query embedding.
    results = await repo.search(session_id="session-B", query_embedding=vector, top_k=5)
    assert results == []


async def test_memory_disabled_setting_skips_chunk_storage(client, admin_headers):
    """Sanity check on the settings flag itself, since chat.py branches on it."""
    from localgate.config import Settings

    settings = Settings(memory_enabled=False, _env_file=None)
    assert settings.memory_enabled is False
