"""Shared fixtures.

Every test app gets:

* an in-memory SQLite database, isolated per test, that never touches real data;
* the deterministic ``FakeBackend``, so no test needs a live Ollama;
* a ``database_config_path`` pointing at a file that does not exist, so a test can
  never pick up a real ``localgate.config.json`` from the developer's working
  directory and start writing to their actual database (see ``app.create_app``).

On top of that, :func:`_hermetic_user_dirs` below makes the *whole* suite unable
to see the developer's real home directory. That is a wall, not a convention:
localgate reads a per-user config file and (soon) keeps its default database in a
per-user data directory, so "remember to monkeypatch it" is not good enough — a
test that forgets would silently read or overwrite real config. It has happened
here before, so the fixture also installs a tripwire that turns any such access
into a loud failure instead.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from localgate import paths
from localgate.app import create_app
from localgate.config import Settings

TEST_ADMIN_KEY = "test-admin-key"

#: Anything that could redirect config/data resolution. Cleared or repointed for
#: every test so a developer's exported `LOCALGATE_*` vars can't change results —
#: which used to be a real difference between a local run and CI.
_HOME_VARS = ("HOME", "USERPROFILE", "XDG_CONFIG_HOME", "XDG_DATA_HOME")


@pytest.fixture(autouse=True)
def _hermetic_user_dirs(tmp_path_factory, monkeypatch):
    """Point every per-user path at a throwaway directory, and make the real one fatal."""
    sandbox = tmp_path_factory.mktemp("home")
    for var in _HOME_VARS:
        monkeypatch.setenv(var, str(sandbox))
    monkeypatch.setenv(paths.ENV_CONFIG_DIR, str(sandbox / "config"))
    monkeypatch.setenv(paths.ENV_DATA_DIR, str(sandbox / "data"))

    # Drop any LOCALGATE_* the developer happens to have exported. Settings reads
    # real env vars at the highest precedence, so `_env_file=None` alone does not
    # isolate a test from them.
    for name in [k for k in os.environ if k.startswith("LOCALGATE_")]:
        if name not in (paths.ENV_CONFIG_DIR, paths.ENV_DATA_DIR):
            monkeypatch.delenv(name, raising=False)

    def _no_real_home() -> Path:
        raise AssertionError(
            "A test tried to resolve the real home directory. Every per-user path "
            "must come from localgate.paths, which this fixture has redirected. "
            "If you see this, something is computing a path outside paths.py."
        )

    monkeypatch.setattr(paths, "_home", _no_real_home)
    return sandbox


@pytest.fixture
def settings() -> Settings:
    return Settings(
        _env_file=None,  # never read the developer's real .env
        database_url="sqlite+aiosqlite:///:memory:",
        backend_type="fake",
        backend_url="",
        admin_key=TEST_ADMIN_KEY,
        log_level="WARNING",
    )


@pytest.fixture
def app(settings, tmp_path):
    return create_app(settings, database_config_path=tmp_path / "localgate-test-config.json")


@pytest.fixture
async def client(app):
    transport = ASGITransport(app=app)
    # The lifespan context is what creates session_factory / backend / rate_limiter /
    # cache — without entering it, every request would fail on a missing app.state.
    async with (
        AsyncClient(transport=transport, base_url="http://test") as ac,
        app.router.lifespan_context(app),
    ):
        yield ac


@pytest.fixture
def admin_headers() -> dict[str, str]:
    return {"X-Admin-Key": TEST_ADMIN_KEY}


@pytest.fixture
async def api_key(client, admin_headers) -> tuple[str, str]:
    """Create a real key through the real admin endpoint. Returns ``(key_id, raw_key)``."""
    resp = await client.post("/admin/keys", headers=admin_headers, json={"name": "test-key"})
    assert resp.status_code == 201, resp.text
    data = resp.json()
    return data["id"], data["api_key"]


@pytest.fixture
def auth_headers(api_key) -> dict[str, str]:
    _, raw_key = api_key
    return {"Authorization": f"Bearer {raw_key}"}


@pytest.fixture
async def db_session(app, client):
    """A raw session for tests that assert directly on stored rows.

    Depends on ``client`` so the lifespan — which creates ``session_factory`` — has
    already run.
    """
    async with app.state.session_factory() as session:
        yield session
