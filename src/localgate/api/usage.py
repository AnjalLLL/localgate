"""Admin usage stats — /admin/usage."""
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from localgate.api.deps import get_session, require_admin
from localgate.db.repositories.usage import UsageRepository

router = APIRouter(dependencies=[Depends(require_admin)])


@router.get("/usage/{api_key_id}")
async def usage_for_key(api_key_id: str, session: AsyncSession = Depends(get_session)):
    repo = UsageRepository(session)
    return await repo.summary_for_key(api_key_id)
