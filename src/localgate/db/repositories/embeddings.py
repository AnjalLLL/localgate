"""Data access layer for embedding vectors (JSON-stored; see models.py for the pgvector upgrade note)."""
import math

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from localgate.db.models import MemoryChunk


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class EmbeddingRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def add_chunk(self, session_id: str, api_key_id: str, content: str, embedding: list[float]) -> None:
        self.session.add(
            MemoryChunk(session_id=session_id, api_key_id=api_key_id, content=content, embedding=embedding)
        )
        await self.session.commit()

    async def search(self, session_id: str, query_embedding: list[float], top_k: int = 5) -> list[str]:
        """Brute-force cosine similarity search, scoped to one session.

        Fine up to a few thousand chunks per session on SQLite. On Postgres with
        pgvector, replace this with a native `ORDER BY embedding <=> :query LIMIT :k`
        query — same method signature, so nothing above this layer needs to change.
        """
        stmt = select(MemoryChunk).where(MemoryChunk.session_id == session_id)
        result = await self.session.execute(stmt)
        chunks = result.scalars().all()

        scored = [(c.content, _cosine_similarity(c.embedding, query_embedding)) for c in chunks]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return [content for content, _score in scored[:top_k]]
