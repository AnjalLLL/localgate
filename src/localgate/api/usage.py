"""``/admin/usage`` — gateway-wide token accounting for operators, and
``/v1/usage`` — a caller's own usage, scoped to their key rather than the
admin key. A regular API client has no reason to hold the admin key just to
see its own request/token totals; a self-scoped route is the same shape
`/v1/conversations` already uses for the same reason.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from localgate.api.deps import get_session, require_admin, require_api_key
from localgate.db.models import APIKey
from localgate.db.repositories.usage import UsageRepository

router = APIRouter(tags=["admin"], dependencies=[Depends(require_admin)])
self_router = APIRouter(tags=["usage"])


@self_router.get("/v1/usage")
async def my_usage(
    api_key: APIKey = Depends(require_api_key),
    session: AsyncSession = Depends(get_session),
) -> dict:
    """The calling key's own request/token totals — no admin key required."""
    return await UsageRepository(session).summary_for_key(api_key.id)


@router.get("/usage")
async def usage_overview(
    days: int = Query(default=14, ge=1, le=365),
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Everything the dashboard's usage view needs, in a single round trip."""
    repo = UsageRepository(session)
    return {
        "totals": await repo.totals(),
        "by_key": await repo.by_key(),
        "by_model": await repo.by_model(),
        "daily": await repo.daily_totals(days=days),
    }


@router.get("/usage/{api_key_id}")
async def usage_for_key(api_key_id: str, session: AsyncSession = Depends(get_session)) -> dict:
    return await UsageRepository(session).summary_for_key(api_key_id)
