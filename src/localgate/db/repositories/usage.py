"""Data access for usage / token accounting records."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from sqlalchemy import desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from localgate.db.models import APIKey, UsageRecord


class UsageRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def record(
        self,
        api_key_id: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: int = 0,
        cached: bool = False,
    ) -> UsageRecord:
        record = UsageRecord(
            api_key_id=api_key_id,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            latency_ms=latency_ms,
            cached=cached,
        )
        self.session.add(record)
        await self.session.commit()
        return record

    async def summary_for_key(self, api_key_id: str) -> dict:
        stmt = select(
            func.count(UsageRecord.id),
            func.coalesce(func.sum(UsageRecord.prompt_tokens), 0),
            func.coalesce(func.sum(UsageRecord.completion_tokens), 0),
            func.coalesce(func.avg(UsageRecord.latency_ms), 0),
        ).where(UsageRecord.api_key_id == api_key_id)
        request_count, prompt_tokens, completion_tokens, avg_latency = (
            await self.session.execute(stmt)
        ).one()
        return {
            "api_key_id": api_key_id,
            "request_count": request_count,
            "prompt_tokens": int(prompt_tokens),
            "completion_tokens": int(completion_tokens),
            "total_tokens": int(prompt_tokens) + int(completion_tokens),
            "avg_latency_ms": round(float(avg_latency), 1),
        }

    async def totals(self) -> dict:
        """Gateway-wide totals for the dashboard header."""
        stmt = select(
            func.count(UsageRecord.id),
            func.coalesce(func.sum(UsageRecord.total_tokens), 0),
            func.coalesce(func.avg(UsageRecord.latency_ms), 0),
        )
        request_count, total_tokens, avg_latency = (await self.session.execute(stmt)).one()
        return {
            "request_count": request_count,
            "total_tokens": int(total_tokens),
            "avg_latency_ms": round(float(avg_latency), 1),
        }

    async def by_key(self) -> list[dict]:
        """Per-key breakdown, including keys that have never been used.

        The outer join is deliberate: a key with zero requests is exactly the key
        an operator is looking for when auditing, and an inner join would hide it.
        """
        stmt = (
            select(
                APIKey.id,
                APIKey.name,
                APIKey.revoked,
                func.count(UsageRecord.id).label("request_count"),
                func.coalesce(func.sum(UsageRecord.total_tokens), 0).label("total_tokens"),
            )
            .select_from(APIKey)
            .outerjoin(UsageRecord, UsageRecord.api_key_id == APIKey.id)
            .group_by(APIKey.id, APIKey.name, APIKey.revoked)
            .order_by(desc("total_tokens"))
        )
        result = await self.session.execute(stmt)
        return [
            {
                "api_key_id": row.id,
                "name": row.name,
                "revoked": row.revoked,
                "request_count": row.request_count,
                "total_tokens": int(row.total_tokens),
            }
            for row in result
        ]

    async def by_model(self) -> list[dict]:
        stmt = (
            select(
                UsageRecord.model,
                func.count(UsageRecord.id).label("request_count"),
                func.coalesce(func.sum(UsageRecord.total_tokens), 0).label("total_tokens"),
            )
            .group_by(UsageRecord.model)
            .order_by(desc("total_tokens"))
        )
        result = await self.session.execute(stmt)
        return [
            {
                "model": row.model,
                "request_count": row.request_count,
                "total_tokens": int(row.total_tokens),
            }
            for row in result
        ]

    async def daily_totals(self, days: int = 14) -> list[dict]:
        """Tokens per day, for the dashboard's usage-over-time chart.

        Bucketing happens in Python rather than in SQL because date truncation is
        spelled differently on SQLite (``strftime``) and Postgres (``date_trunc``),
        and localgate has to run identically on both. At the volume this query
        covers — a couple of weeks of requests from a self-hosted gateway — the
        difference is not measurable.
        """
        since = datetime.now(timezone.utc) - timedelta(days=days)
        stmt = select(UsageRecord).where(UsageRecord.created_at >= since)
        records = (await self.session.execute(stmt)).scalars().all()

        buckets: dict[str, dict[str, int]] = {}
        for record in records:
            day = record.created_at.date().isoformat()
            bucket = buckets.setdefault(
                day, {"prompt_tokens": 0, "completion_tokens": 0, "requests": 0}
            )
            bucket["prompt_tokens"] += record.prompt_tokens
            bucket["completion_tokens"] += record.completion_tokens
            bucket["requests"] += 1

        return [{"date": day, **totals} for day, totals in sorted(buckets.items())]
