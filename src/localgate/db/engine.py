"""Async SQLAlchemy engine, session factory, and schema management.

The same code runs unchanged against ``sqlite+aiosqlite``, ``postgresql+asyncpg``,
or a Neon connection string — only ``LOCALGATE_DATABASE_URL`` differs.

Schema is created by **Alembic only**. There is deliberately no ``create_all`` path:
if the schema could be built two ways, the migration history and the ORM models
would drift apart, and the person who found out would be someone upgrading a
database that had never been migrated.
"""

from __future__ import annotations

from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import Connection, inspect, text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import NullPool, StaticPool

MIGRATIONS_DIR = Path(__file__).parent / "migrations"


def make_engine(database_url: str) -> AsyncEngine:
    """Create an async engine tuned for the dialect in ``database_url``."""
    if database_url.startswith("postgresql+asyncpg"):
        # asyncpg caches prepared statements per physical connection. That is fine
        # against plain Postgres, but Neon's pooled endpoint (hostname contains
        # "-pooler") is PgBouncer in transaction-pooling mode, which swaps the
        # underlying connection out between queries — so those cached statements go
        # stale and queries begin failing in a way that looks random. Disabling the
        # cache and using NullPool (let PgBouncer own the pooling instead of stacking
        # a second pool on top of it) is the standard fix. It costs nothing against a
        # direct connection, so it is applied unconditionally rather than by sniffing
        # the hostname for "-pooler".
        return create_async_engine(
            database_url,
            echo=False,
            poolclass=NullPool,
            connect_args={"statement_cache_size": 0},
        )

    if ":memory:" in database_url:
        # Each new connection to an in-memory SQLite database is a *different*, empty
        # database. StaticPool keeps one connection alive for the engine's lifetime,
        # which is what makes the schema created at startup still be there when a
        # request queries it.
        return create_async_engine(database_url, echo=False, poolclass=StaticPool)

    return create_async_engine(database_url, echo=False)


def make_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False)


def alembic_config(connection: Connection | None = None) -> Config:
    """Build an Alembic config pointed at this package's migrations.

    A supplied ``connection`` is passed through to ``env.py`` via
    ``config.attributes``, so migrations run on a connection we already hold rather
    than opening a second one — which matters for in-memory SQLite, where a second
    connection would be a second, empty database.
    """
    config = Config(str(MIGRATIONS_DIR / "alembic.ini"))
    config.set_main_option("script_location", str(MIGRATIONS_DIR))
    if connection is not None:
        config.attributes["connection"] = connection
    return config


#: A table that exists in every version of the schema, used to tell "empty database"
#: apart from "database built before migrations existed".
SENTINEL_TABLE = "api_keys"


def _upgrade_to_head(connection: Connection) -> None:
    """Migrate to head, adopting a pre-migrations database if that is what this is.

    localgate 0.1-0.2 created its schema with ``create_all`` and left no
    ``alembic_version`` behind. Running the migrations against such a database would
    try to ``CREATE TABLE api_keys`` a second time and fail — so those databases are
    *stamped* at 0001 (the revision that reproduces exactly what ``create_all``
    built) and then migrated forward from there like any other.

    The three cases:

    * no tables at all      -> run every migration
    * tables, no stamp      -> stamp 0001, then run the rest (this branch)
    * tables and a stamp    -> run whatever is pending; a no-op when current
    """
    # Never wait indefinitely for a lock. A migration blocked behind another connection
    # (a previous instance during a rolling deploy, an open psql session) would otherwise
    # hang the whole startup with no log line and no timeout — the server sits at
    # "Waiting for application startup" forever, and the only symptom the operator sees is
    # that nothing responds. Failing after 15s with a real error is strictly better.
    if connection.dialect.name == "postgresql":
        connection.execute(text("SET lock_timeout = '15s'"))

    inspector = inspect(connection)
    tables = set(inspector.get_table_names())
    config = alembic_config(connection)

    is_legacy = SENTINEL_TABLE in tables and "alembic_version" not in tables
    if is_legacy:
        command.stamp(config, "0001")

    command.upgrade(config, "head")


async def init_models(engine: AsyncEngine) -> None:
    """Bring the database up to the latest migration.

    Safe to call on every startup: Alembic no-ops once the schema is current.
    """
    async with engine.begin() as conn:
        await conn.run_sync(_upgrade_to_head)


async def current_revision(engine: AsyncEngine) -> str | None:
    """The revision the database is at, or ``None`` if it has never been migrated."""
    from alembic.runtime.migration import MigrationContext

    def _revision(connection: Connection) -> str | None:
        return MigrationContext.configure(connection).get_current_revision()

    async with engine.connect() as conn:
        return await conn.run_sync(_revision)
