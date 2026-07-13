"""``/v1/conversations`` — read back stored chat history.

Every route here is scoped to the calling API key. A session belongs to the key
that created it, and a key may only read its own — see :func:`get_conversation`
for how that is enforced without leaking which sessions exist.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from localgate.api.deps import get_session, require_api_key
from localgate.core.errors import LocalgateError
from localgate.db.models import APIKey
from localgate.db.repositories.conversations import ConversationRepository, SummaryRepository
from localgate.db.repositories.embeddings import EmbeddingRepository

router = APIRouter(tags=["conversations"])


class SessionNotFound(LocalgateError):
    status_code = 404
    error_type = "not_found_error"


@router.get("/v1/conversations")
async def list_conversations(
    limit: int = Query(default=50, ge=1, le=500),
    api_key: APIKey = Depends(require_api_key),
    session: AsyncSession = Depends(get_session),
) -> list[dict]:
    """The calling key's sessions, most recently active first."""
    return await ConversationRepository(session).list_sessions(api_key.id, limit=limit)


@router.get("/v1/conversations/{session_id}")
async def get_conversation(
    session_id: str,
    limit: int = Query(default=200, ge=1, le=1000),
    api_key: APIKey = Depends(require_api_key),
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Full history for one session, plus its rolling summary and memory stats.

    A session owned by a *different* key returns 404 rather than 403. 403 would
    confirm that the session exists, which lets a caller enumerate other keys'
    session ids by probing. To this caller, someone else's session and a session
    that was never created look exactly the same.
    """
    convo_repo = ConversationRepository(session)
    messages = await convo_repo.recent(session_id, limit=limit)
    owned = [m for m in messages if m.api_key_id == api_key.id]

    if not owned:
        raise SessionNotFound(f"No conversation {session_id!r} for this API key.")

    summary = await SummaryRepository(session).latest(session_id)
    chunk_count = await EmbeddingRepository(session).count(session_id)

    return {
        "session_id": session_id,
        "messages": [
            {"role": m.role, "content": m.content, "created_at": m.created_at.isoformat()}
            for m in owned
        ],
        "summary": summary.content if summary else None,
        "memory_chunks": chunk_count,
    }
