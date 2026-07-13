"""Data access for memory chunks and their embedding vectors.

Vectors are JSON arrays and similarity is computed in Python — see
``db/models.py::MemoryChunk`` and docs/decisions/0002 for why, and for the
pgvector upgrade path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from localgate.db.models import MemoryChunk


@dataclass(frozen=True)
class RetrievedChunk:
    """A retrieved chunk and how well it matched.

    The score is carried out of the repository rather than discarded because
    retrieval quality is the hardest thing to tune blind: without the scores,
    "the model didn't remember" and "the model remembered the wrong thing" look
    identical from the outside.
    """

    content: str
    score: float
    kind: str


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity, defined as 0.0 for a zero vector rather than undefined."""
    if len(a) != len(b):
        # Changing the embedding model changes the vector width. Comparing across
        # widths is meaningless, so score it as "no match" instead of raising and
        # taking down a request over chunks written by a previous config.
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class EmbeddingRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def add_chunk(
        self,
        session_id: str,
        api_key_id: str,
        content: str,
        embedding: list[float],
        kind: str = "turn",
    ) -> None:
        self.session.add(
            MemoryChunk(
                session_id=session_id,
                api_key_id=api_key_id,
                content=content,
                embedding=embedding,
                kind=kind,
            )
        )
        await self.session.commit()

    async def search(
        self,
        session_id: str,
        query_embedding: list[float],
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> list[RetrievedChunk]:
        """Rank this session's chunks against the query vector, best first.

        ``min_score`` is the guard against the failure mode that makes RAG worse
        than no RAG: with nothing relevant stored, the top-k are still *returned* —
        just with low scores — and injecting them fills the context window with
        noise. Dropping everything below the floor means an irrelevant memory is no
        memory at all.
        """
        stmt = select(MemoryChunk).where(MemoryChunk.session_id == session_id)
        chunks = (await self.session.execute(stmt)).scalars().all()

        scored = [
            RetrievedChunk(
                content=chunk.content,
                score=cosine_similarity(chunk.embedding, query_embedding),
                kind=chunk.kind,
            )
            for chunk in chunks
        ]
        scored.sort(key=lambda c: c.score, reverse=True)
        return [c for c in scored if c.score >= min_score][:top_k]

    async def count(self, session_id: str) -> int:
        stmt = select(func.count(MemoryChunk.id)).where(MemoryChunk.session_id == session_id)
        return int((await self.session.execute(stmt)).scalar_one())
