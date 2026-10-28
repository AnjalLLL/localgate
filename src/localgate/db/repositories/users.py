"""Dashboard accounts (see `db.models.User`)."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from localgate.core.passwords import hash_password, verify_password
from localgate.db.models import User


def normalize_email(email: str) -> str:
    return email.strip().lower()


class UserRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def count(self) -> int:
        result = await self.session.execute(select(func.count()).select_from(User))
        return int(result.scalar_one())

    async def list_all(self) -> list[User]:
        result = await self.session.execute(select(User).order_by(User.created_at))
        return list(result.scalars())

    async def get(self, email: str) -> User | None:
        result = await self.session.execute(
            select(User).where(User.email == normalize_email(email))
        )
        return result.scalar_one_or_none()

    async def create(self, email: str, password: str) -> User:
        user = User(email=normalize_email(email), password_hash=hash_password(password))
        self.session.add(user)
        await self.session.commit()
        return user

    async def set_password(self, user: User, password: str) -> None:
        user.password_hash = hash_password(password)
        await self.session.commit()

    async def delete(self, user: User) -> None:
        await self.session.delete(user)
        await self.session.commit()

    async def authenticate(self, email: str, password: str) -> User | None:
        """The user, when ``password`` is theirs; records the sign-in time."""
        user = await self.get(email)
        if user is None or not verify_password(password, user.password_hash):
            return None
        user.last_login_at = datetime.now(timezone.utc)
        await self.session.commit()
        return user
