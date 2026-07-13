"""``/admin/usage`` — token accounting."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from localgate.api.deps import get_session, require_admin
from localgate.db.repositories.usage import UsageRepository

router = APIRouter(tags=["admin"], dependencies=[Depends(require_admin)])


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
