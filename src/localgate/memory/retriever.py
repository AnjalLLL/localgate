"""Similarity search over stored conversation memory for a given session."""
from sqlalchemy.ext.asyncio import AsyncSession

from localgate.backends.base import InferenceBackend
from localgate.db.repositories.embeddings import EmbeddingRepository
from localgate.memory.embedder import embed_text


async def retrieve_relevant_context(
    session: AsyncSession,
    backend: InferenceBackend,
    session_id: str,
    query: str,
    embedding_model: str,
    top_k: int = 5,
) -> list[str]:
    query_embedding = await embed_text(backend, query, embedding_model)
    repo = EmbeddingRepository(session)
    return await repo.search(session_id, query_embedding, top_k)
