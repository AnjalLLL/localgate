"""Async SQLAlchemy engine + session factory.

Works unchanged against sqlite+aiosqlite, postgresql+asyncpg, or a Neon
connection string — the only thing that differs is the LOCALGATE_DATABASE_URL value.
"""
from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from localgate.db.models import Base


def make_engine(database_url: str) -> AsyncEngine:
    return create_async_engine(database_url, echo=False)


def make_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False)


async def init_models(engine: AsyncEngine) -> None:
    """Creates all tables if they don't exist yet.

    This is the zero-friction path for local/dev use. Before a real release,
    replace this with Alembic migrations (scaffolded under db/migrations/)
    so schema changes are versioned instead of implicit.
    """
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
