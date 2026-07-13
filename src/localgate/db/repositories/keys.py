"""Data access for API keys — route handlers never touch the database directly."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, cast

from sqlalchemy import CursorResult, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from localgate.core.auth import generate_key, hash_key, key_prefix
from localgate.db.models import APIKey


class APIKeyRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(self, name: str, rate_limit_per_min: int = 60) -> tuple[APIKey, str]:
        """Create a key. Returns ``(record, raw_key)``; the raw key is never recoverable again."""
        raw_key = generate_key()
        key = APIKey(
            name=name,
            key_hash=hash_key(raw_key),
            key_prefix=key_prefix(raw_key),
            rate_limit_per_min=rate_limit_per_min,
        )
        self.session.add(key)
        await self.session.commit()
        await self.session.refresh(key)
        return key, raw_key

    async def get_by_raw_key(self, raw_key: str) -> APIKey | None:
        """Look up an active key by its plaintext value.

        Matching is on the hash in SQL, so a revoked key and a key that never
        existed are indistinguishable to the caller — both simply fail to match.
        """
        stmt = select(APIKey).where(APIKey.key_hash == hash_key(raw_key), APIKey.revoked.is_(False))
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def get(self, key_id: str) -> APIKey | None:
        return await self.session.get(APIKey, key_id)

    async def list_all(self) -> list[APIKey]:
        result = await self.session.execute(select(APIKey).order_by(APIKey.created_at.desc()))
        return list(result.scalars().all())

    async def set_rate_limit(self, key_id: str, rate_limit_per_min: int) -> bool:
        result = cast(
            CursorResult[Any],
            await self.session.execute(
                update(APIKey)
                .where(APIKey.id == key_id)
                .values(rate_limit_per_min=rate_limit_per_min)
            ),
        )
        await self.session.commit()
        return bool(result.rowcount)

    async def revoke(self, key_id: str) -> bool:
        """Revoke a key. Returns whether a key with that id existed."""
        # An UPDATE always returns a CursorResult, but `Session.execute` is typed as
        # returning the general Result, which has no rowcount. The cast is what lets
        # "did this key exist?" be answered without a second SELECT.
        result = cast(
            CursorResult[Any],
            await self.session.execute(
                update(APIKey).where(APIKey.id == key_id).values(revoked=True)
            ),
        )
        await self.session.commit()
        return bool(result.rowcount)

    async def touch_last_used(self, key_id: str) -> None:
        await self.session.execute(
            update(APIKey)
            .where(APIKey.id == key_id)
            .values(last_used_at=datetime.now(timezone.utc))
        )
        await self.session.commit()
