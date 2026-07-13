"""Data access layer for API keys — route handlers never touch the DB directly."""
from datetime import datetime, timezone

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from localgate.core.auth import generate_key, hash_key
from localgate.db.models import APIKey


class APIKeyRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(self, name: str, rate_limit_per_min: int = 60) -> tuple[APIKey, str]:
        """Returns (stored record, raw key). The raw key is shown to the caller exactly once."""
        raw_key = generate_key()
        key = APIKey(name=name, key_hash=hash_key(raw_key), rate_limit_per_min=rate_limit_per_min)
        self.session.add(key)
        await self.session.commit()
        await self.session.refresh(key)
        return key, raw_key

    async def get_by_raw_key(self, raw_key: str) -> APIKey | None:
        stmt = select(APIKey).where(APIKey.key_hash == hash_key(raw_key), APIKey.revoked.is_(False))
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def list_all(self) -> list[APIKey]:
        result = await self.session.execute(select(APIKey))
        return list(result.scalars().all())

    async def revoke(self, key_id: str) -> None:
        await self.session.execute(update(APIKey).where(APIKey.id == key_id).values(revoked=True))
        await self.session.commit()

    async def touch_last_used(self, key_id: str) -> None:
        await self.session.execute(
            update(APIKey).where(APIKey.id == key_id).values(last_used_at=datetime.now(timezone.utc))
        )
        await self.session.commit()
