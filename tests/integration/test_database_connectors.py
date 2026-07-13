"""Integration tests for the database config store and engine creation.

Doesn't require a real Postgres/Neon instance — that's covered by manual
testing against a real connection string (see docs/database-setup.md).
This tests the mechanism: establishing, persisting, and resolving which
database URL gets used.
"""
from pathlib import Path

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
