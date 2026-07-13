"""Migrations, including the upgrade path for databases that predate them."""

from sqlalchemy import inspect, text
from sqlalchemy.ext.asyncio import create_async_engine

from localgate.db.engine import current_revision, init_models, make_engine

# The schema exactly as localgate 0.1-0.2 built it with create_all: no
# alembic_version, no key_prefix, no latency_ms/cached, no kind, no summaries table.
LEGACY_DDL = [
    """CREATE TABLE api_keys (id VARCHAR NOT NULL, name VARCHAR NOT NULL,
        key_hash VARCHAR NOT NULL, rate_limit_per_min INTEGER, revoked BOOLEAN,
        created_at TIMESTAMP, last_used_at TIMESTAMP, PRIMARY KEY (id))""",
    "CREATE UNIQUE INDEX ix_api_keys_key_hash ON api_keys (key_hash)",
    """CREATE TABLE usage_records (id VARCHAR NOT NULL, api_key_id VARCHAR, model VARCHAR,
        prompt_tokens INTEGER, completion_tokens INTEGER, total_tokens INTEGER,
        created_at TIMESTAMP, PRIMARY KEY (id),
        FOREIGN KEY(api_key_id) REFERENCES api_keys (id))""",
    "CREATE INDEX ix_usage_records_api_key_id ON usage_records (api_key_id)",
    """CREATE TABLE conversation_messages (id VARCHAR NOT NULL, session_id VARCHAR,
        api_key_id VARCHAR, role VARCHAR, content TEXT, created_at TIMESTAMP,
        PRIMARY KEY (id), FOREIGN KEY(api_key_id) REFERENCES api_keys (id))""",
    "CREATE INDEX ix_conversation_messages_session_id ON conversation_messages (session_id)",
    "CREATE INDEX ix_conversation_messages_api_key_id ON conversation_messages (api_key_id)",
    """CREATE TABLE memory_chunks (id VARCHAR NOT NULL, session_id VARCHAR, api_key_id VARCHAR,
        content TEXT, embedding JSON, created_at TIMESTAMP, PRIMARY KEY (id),
        FOREIGN KEY(api_key_id) REFERENCES api_keys (id))""",
    "CREATE INDEX ix_memory_chunks_session_id ON memory_chunks (session_id)",
    "CREATE INDEX ix_memory_chunks_api_key_id ON memory_chunks (api_key_id)",
]


async def _columns(engine, table: str) -> set[str]:
    async with engine.connect() as conn:
        return await conn.run_sync(
            lambda sync: {c["name"] for c in inspect(sync).get_columns(table)}
        )


async def _tables(engine) -> set[str]:
    async with engine.connect() as conn:
        return await conn.run_sync(lambda sync: set(inspect(sync).get_table_names()))


async def test_fresh_database_migrates_to_head(tmp_path):
    engine = make_engine(f"sqlite+aiosqlite:///{tmp_path}/fresh.db")
    await init_models(engine)

    assert await current_revision(engine) == "0002"
    assert "conversation_summaries" in await _tables(engine)
    assert "key_prefix" in await _columns(engine, "api_keys")
    await engine.dispose()


async def test_migrating_twice_is_a_no_op(tmp_path):
    """Startup runs migrations every time, so this happens on literally every boot."""
    url = f"sqlite+aiosqlite:///{tmp_path}/twice.db"
    engine = make_engine(url)
    await init_models(engine)
    await init_models(engine)  # must not raise
    assert await current_revision(engine) == "0002"
    await engine.dispose()


async def test_pre_migrations_database_is_adopted_without_losing_data(tmp_path):
    """The upgrade path for anyone already running localgate.

    Databases created before migrations existed have tables but no alembic_version.
    Running the migrations against one naively would try to CREATE TABLE api_keys a
    second time and fail, taking the gateway down on upgrade. It must instead be
    stamped at 0001 and migrated forward, keeping every existing row.
    """
    url = f"sqlite+aiosqlite:///{tmp_path}/legacy.db"

    setup = create_async_engine(url)
    async with setup.begin() as conn:
        for ddl in LEGACY_DDL:
            await conn.execute(text(ddl))
        await conn.execute(
            text(
                "INSERT INTO api_keys (id, name, key_hash, rate_limit_per_min, revoked) "
                "VALUES ('k1', 'my-old-key', 'deadbeef', 60, 0)"
            )
        )
        await conn.execute(
            text(
                "INSERT INTO memory_chunks (id, session_id, api_key_id, content, embedding) "
                "VALUES ('c1', 's1', 'k1', 'an old memory', '[0.1]')"
            )
        )
    await setup.dispose()

    engine = make_engine(url)
    await init_models(engine)  # the real startup path

    assert await current_revision(engine) == "0002"
    assert "key_prefix" in await _columns(engine, "api_keys")
    assert {"latency_ms", "cached"} <= await _columns(engine, "usage_records")
    assert "conversation_summaries" in await _tables(engine)

    async with engine.connect() as conn:
        name = (await conn.execute(text("SELECT name FROM api_keys WHERE id='k1'"))).scalar_one()
        chunk = (
            await conn.execute(text("SELECT content, kind FROM memory_chunks WHERE id='c1'"))
        ).one()

    assert name == "my-old-key"  # the user's key still works
    assert chunk.content == "an old memory"
    assert chunk.kind == "turn"  # backfilled: everything that predates the column is a turn
    await engine.dispose()


async def test_columns_added_to_existing_rows_are_backfilled_not_left_null(tmp_path):
    """Regression: `key_prefix` was added as nullable with no backfill, so every key that
    predated the migration held NULL — and `GET /admin/keys` (whose response model types it
    as `str`) answered 500. The keys page was broken for everyone upgrading, which is the
    exact path the migration exists to support.
    """
    url = f"sqlite+aiosqlite:///{tmp_path}/backfill.db"

    setup = create_async_engine(url)
    async with setup.begin() as conn:
        for ddl in LEGACY_DDL:
            await conn.execute(text(ddl))
        await conn.execute(
            text(
                "INSERT INTO api_keys (id, name, key_hash, rate_limit_per_min, revoked) "
                "VALUES ('k1', 'old', 'hash', 60, 0)"
            )
        )
        await conn.execute(
            text(
                "INSERT INTO usage_records (id, api_key_id, model, prompt_tokens, "
                "completion_tokens, total_tokens) VALUES ('u1', 'k1', 'llama3', 1, 2, 3)"
            )
        )
    await setup.dispose()

    engine = make_engine(url)
    await init_models(engine)

    async with engine.connect() as conn:
        key = (
            await conn.execute(text("SELECT key_prefix FROM api_keys WHERE id='k1'"))
        ).scalar_one()
        usage = (
            await conn.execute(text("SELECT latency_ms, cached FROM usage_records WHERE id='u1'"))
        ).one()

    assert key == ""  # not None: the raw key is unrecoverable, but the column is not NULL
    assert usage.latency_ms == 0
    assert usage.cached in (0, False)
    await engine.dispose()


async def test_the_keys_endpoint_survives_a_migrated_legacy_key(settings, tmp_path):
    """The end-to-end version of the bug above: list the keys of an upgraded database."""
    from httpx import ASGITransport, AsyncClient

    from localgate.app import create_app

    url = f"sqlite+aiosqlite:///{tmp_path}/legacy_api.db"

    setup = create_async_engine(url)
    async with setup.begin() as conn:
        for ddl in LEGACY_DDL:
            await conn.execute(text(ddl))
        await conn.execute(
            text(
                "INSERT INTO api_keys (id, name, key_hash, rate_limit_per_min, revoked, "
                "created_at) VALUES ('k1', 'pre-upgrade-key', 'hash', 60, 0, '2026-01-01')"
            )
        )
    await setup.dispose()

    settings.database_url = url
    app = create_app(settings, database_config_path=tmp_path / "cfg.json")

    async with (
        AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client,
        app.router.lifespan_context(app),
    ):
        resp = await client.get("/admin/keys", headers={"X-Admin-Key": settings.admin_key})

    assert resp.status_code == 200, resp.text
    listed = resp.json()
    assert listed[0]["name"] == "pre-upgrade-key"
    assert listed[0]["key_prefix"] == ""  # unknowable, but it must not be a crash
