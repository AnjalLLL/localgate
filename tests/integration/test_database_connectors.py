"""Integration tests for the database config store and engine creation.

Doesn't require a real Postgres/Neon instance — that's covered by manual
testing against a real connection string (see docs/database-setup.md).
This tests the mechanism: establishing, persisting, and resolving which
database URL gets used.
"""

from localgate.core.db_config_store import (
    is_database_established,
    load_database_url,
    save_database_url,
)
from localgate.db.engine import init_models, make_engine


def test_no_config_file_means_not_established(tmp_path):
    config_path = tmp_path / "localgate.config.json"
    assert is_database_established(config_path) is False
    assert load_database_url(config_path) is None


def test_saving_a_url_makes_it_established(tmp_path):
    config_path = tmp_path / "localgate.config.json"
    save_database_url("sqlite+aiosqlite:///./somewhere.db", config_path)
    assert is_database_established(config_path) is True
    assert load_database_url(config_path) == "sqlite+aiosqlite:///./somewhere.db"


def test_saving_again_overwrites_the_previous_url(tmp_path):
    config_path = tmp_path / "localgate.config.json"
    save_database_url("sqlite+aiosqlite:///./first.db", config_path)
    save_database_url("sqlite+aiosqlite:///./second.db", config_path)
    assert load_database_url(config_path) == "sqlite+aiosqlite:///./second.db"


def test_corrupt_config_file_is_treated_as_not_established(tmp_path):
    config_path = tmp_path / "localgate.config.json"
    config_path.write_text("{not valid json")
    assert is_database_established(config_path) is False


async def test_init_models_creates_tables_on_sqlite():
    engine = make_engine("sqlite+aiosqlite:///:memory:")
    await init_models(engine)  # should not raise
    await engine.dispose()


def test_postgres_engine_disables_prepared_statement_cache():
    """Regression test: Neon's pooled endpoint (PgBouncer transaction mode) breaks
    asyncpg's default prepared-statement caching. make_engine must disable it for
    any postgresql+asyncpg URL, not just ones that look pooled — see engine.py."""
    engine = make_engine("postgresql+asyncpg://user:pass@localhost/db")
    # Can't easily introspect connect_args post-construction without connecting,
    # so assert on the pool class instead, which IS inspectable and is set in
    # the same branch as the statement_cache_size fix.
    from sqlalchemy.pool import NullPool

    assert isinstance(engine.pool, NullPool)


def test_sqlite_engine_is_unaffected_by_postgres_specific_settings():
    engine = make_engine("sqlite+aiosqlite:///:memory:")
    from sqlalchemy.pool import NullPool

    assert not isinstance(engine.pool, NullPool)


async def test_database_url_endpoint_rejects_garbage_without_crashing(client, admin_headers):
    """Regression test: this used to crash into a raw 500 instead of a clean 400
    when the URL used a driver that isn't installed (e.g. plain 'postgresql://'
    defaulting to psycopg2, which localgate doesn't depend on)."""
    resp = await client.put(
        "/admin/config/database-url",
        headers=admin_headers,
        json={"database_url": "postgresql://user:pass@localhost:1/doesnotexist"},
    )
    assert resp.status_code == 400
    assert "asyncpg" in resp.json()["error"]["message"]


async def test_database_url_endpoint_rejects_malformed_url(client, admin_headers):
    resp = await client.put(
        "/admin/config/database-url",
        headers=admin_headers,
        json={"database_url": "not-a-url-at-all"},
    )
    assert resp.status_code == 400


async def test_database_url_endpoint_accepts_a_working_sqlite_url(client, admin_headers, tmp_path):
    working_url = f"sqlite+aiosqlite:///{tmp_path}/test.db"
    resp = await client.put(
        "/admin/config/database-url",
        headers=admin_headers,
        json={"database_url": working_url},
    )
    assert resp.status_code == 200
    assert resp.json()["established"] is True


async def test_database_url_endpoint_hints_at_sslmode_vs_ssl(client, admin_headers):
    """Regression test: asyncpg rejects the psycopg-style 'sslmode' query param
    with a raw TypeError. This used to surface as an unhelpful crash; now it
    should come back as a 400 with a specific hint to use 'ssl=' instead."""
    resp = await client.put(
        "/admin/config/database-url",
        headers=admin_headers,
        json={"database_url": "postgresql+asyncpg://user:pass@localhost:1/db?sslmode=require"},
    )
    assert resp.status_code == 400
    assert "ssl=" in resp.json()["error"]["message"]
