"""Alembic environment.

Two things here differ from the stock template, both on purpose:

* **The URL comes from localgate, not from alembic.ini.** ``resolve_database_url``
  applies the same precedence the running gateway does — an admin-established
  database outranks ``.env`` — so ``alembic upgrade head`` can never migrate a
  different database than the one being served.

* **``render_as_batch`` is on.** SQLite cannot ``ALTER TABLE ... DROP COLUMN`` or
  alter a column type. Batch mode makes Alembic emit the copy-table-and-swap dance
  instead, so the same migration script runs on SQLite and Postgres. Without it,
  every future migration that touches an existing column would work in production
  and fail for the local-first users who are the whole point of this project.
"""

from __future__ import annotations

import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from localgate.app import resolve_database_url
from localgate.config import load_settings
from localgate.core.db_config_store import redact_database_url
from localgate.db.models import Base

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def get_url() -> str:
    """The database a bare ``alembic`` invocation should migrate.

    ``-x url=...`` wins, so a developer can target a scratch database explicitly.
    Otherwise this applies the gateway's own precedence, which since localgate
    started reading a *per-user* config file means a bare ``alembic upgrade head``
    in a checkout resolves the user's real database rather than something
    CWD-local. That used to be impossible, so `run_async_migrations` announces
    what it resolved before touching anything.
    """
    forced = context.get_x_argument(as_dictionary=True).get("url")
    if forced:
        return str(forced)
    return resolve_database_url(load_settings())


def run_migrations_offline() -> None:
    """Emit SQL to stdout instead of running it — for DBAs who apply changes by hand."""
    context.configure(
        url=get_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        render_as_batch=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        render_as_batch=True,
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Open our own connection — the path taken by a bare ``alembic upgrade head``."""
    configuration = config.get_section(config.config_ini_section, {})
    url = get_url()
    # Only on this path: the application and `localgate db upgrade` pass in a
    # connection, and they already log where they are pointed. Here nothing else
    # would tell you that you are about to migrate a remote production database.
    print(f"alembic: migrating {redact_database_url(url)}")  # noqa: T201 — operator-facing
    configuration["sqlalchemy.url"] = url

    engine = async_engine_from_config(configuration, prefix="sqlalchemy.", poolclass=pool.NullPool)
    async with engine.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await engine.dispose()


def run_migrations_online() -> None:
    # The application (and `localgate db upgrade`) passes in a connection it already
    # holds. Opening our own instead would be wrong for in-memory SQLite — a second
    # connection there is a second, empty database — and wasteful everywhere else.
    connection = config.attributes.get("connection")
    if connection is not None:
        do_run_migrations(connection)
    else:
        asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
