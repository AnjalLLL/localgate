"""Data access layer for usage/token accounting records."""
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from localgate.db.models import UsageRecord


class UsageRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def record(
        self, api_key_id: str, model: str, prompt_tokens: int, completion_tokens: int
    ) -> UsageRecord:
        record = UsageRecord(
            api_key_id=api_key_id,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )
        self.session.add(record)
        await self.session.commit()
        return record

    async def summary_for_key(self, api_key_id: str) -> dict:
        stmt = select(
            func.count(UsageRecord.id),
            func.coalesce(func.sum(UsageRecord.prompt_tokens), 0),
            func.coalesce(func.sum(UsageRecord.completion_tokens), 0),
        ).where(UsageRecord.api_key_id == api_key_id)
        result = await self.session.execute(stmt)
        request_count, prompt_tokens, completion_tokens = result.one()
        return {
            "request_count": request_count,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
