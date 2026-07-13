"""GET /v1/conversations/{session_id} — chat history for a session.

Scoped to the calling API key: you can only read back sessions created
under your own key, not anyone else's.
"""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from localgate.api.deps import get_session, require_api_key
from localgate.db.models import APIKey
from localgate.db.repositories.conversations import ConversationRepository

router = APIRouter()


@router.get("/v1/conversations/{session_id}")
async def get_conversation(
    session_id: str,
    api_key: APIKey = Depends(require_api_key),
    session: AsyncSession = Depends(get_session),
):
    repo = ConversationRepository(session)
    messages = await repo.recent(session_id, limit=200)
    owned = [m for m in messages if m.api_key_id == api_key.id]
    if not owned and messages:
        # Session exists but belongs to a different key — don't leak its existence.
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
    return [{"role": m.role, "content": m.content, "created_at": m.created_at.isoformat()} for m in owned]
