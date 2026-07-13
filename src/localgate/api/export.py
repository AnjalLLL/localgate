"""``GET /admin/export`` — take all your data and leave.

Nobody should feel locked into a self-hosted tool. This returns every row localgate
holds — keys (metadata only), usage, conversations, summaries — as one JSON
document that can be archived, diffed, or loaded somewhere else.

Key *hashes* and embedding vectors are excluded. The hashes are secret material
with no value outside this database, and the vectors would multiply the export size
by an order of magnitude while being reproducible from the text with the same
embedding model.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from localgate import __version__
from localgate.api.deps import get_session, require_admin
from localgate.db.models import APIKey, ConversationMessage, ConversationSummary, UsageRecord

router = APIRouter(tags=["admin"], dependencies=[Depends(require_admin)])


@router.get("/export")
async def export_all(session: AsyncSession = Depends(get_session)) -> JSONResponse:
    keys = (await session.execute(select(APIKey))).scalars().all()
    usage = (await session.execute(select(UsageRecord))).scalars().all()
    messages = (await session.execute(select(ConversationMessage))).scalars().all()
    summaries = (await session.execute(select(ConversationSummary))).scalars().all()

    payload: dict[str, Any] = {
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "localgate_version": __version__,
        "api_keys": [
            {
                "id": k.id,
                "name": k.name,
                "key_prefix": k.key_prefix,
                "revoked": k.revoked,
                "rate_limit_per_min": k.rate_limit_per_min,
                "created_at": k.created_at.isoformat(),
                "last_used_at": k.last_used_at.isoformat() if k.last_used_at else None,
            }
            for k in keys
        ],
        "usage_records": [
            {
                "id": u.id,
                "api_key_id": u.api_key_id,
                "model": u.model,
                "prompt_tokens": u.prompt_tokens,
                "completion_tokens": u.completion_tokens,
                "total_tokens": u.total_tokens,
                "latency_ms": u.latency_ms,
                "cached": u.cached,
                "created_at": u.created_at.isoformat(),
            }
            for u in usage
        ],
        "conversations": [
            {
                "id": m.id,
                "session_id": m.session_id,
                "api_key_id": m.api_key_id,
                "role": m.role,
                "content": m.content,
                "created_at": m.created_at.isoformat(),
            }
            for m in messages
        ],
        "summaries": [
            {
                "id": s.id,
                "session_id": s.session_id,
                "content": s.content,
                "covers_until": s.covers_until.isoformat(),
                "message_count": s.message_count,
            }
            for s in summaries
        ],
    }

    return JSONResponse(
        payload,
        headers={"Content-Disposition": 'attachment; filename="localgate-export.json"'},
    )
