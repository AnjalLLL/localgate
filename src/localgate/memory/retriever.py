"""Similarity search over a session's stored memory."""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from localgate.backends.base import InferenceBackend
from localgate.db.repositories.embeddings import EmbeddingRepository, RetrievedChunk
from localgate.memory.embedder import embed_text


async def retrieve_relevant_context(
    session: AsyncSession,
    backend: InferenceBackend,
    session_id: str,
    query: str,
    embedding_model: str,
    top_k: int = 5,
    min_score: float = 0.0,
) -> list[RetrievedChunk]:
    """Embed ``query`` and return this session's best-matching stored chunks.

    Retrieval is scoped to a single session deliberately. Letting a query reach
    across sessions would mean one caller's conversation could surface inside
    another's context — a data leak dressed up as a feature.
    """
    if not query or top_k == 0:
        return []
    query_embedding = await embed_text(backend, query, embedding_model)
    repo = EmbeddingRepository(session)
    return await repo.search(session_id, query_embedding, top_k=top_k, min_score=min_score)
