"""Shared pytest fixtures.

Every test app uses:
  - an in-memory SQLite database (isolated per test, never touches real data)
  - the FakeBackend (deterministic, no live Ollama required)
  - a database_config_path pointing at a nonexistent file, so tests NEVER
    accidentally pick up a real localgate.config.json from the working
    directory (see app.py's create_app for why this matters).
"""
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from localgate.app import create_app
from localgate.config import Settings

TEST_ADMIN_KEY = "test-admin-key"


@pytest.fixture
def app():
    settings = Settings(
        database_url="sqlite+aiosqlite:///:memory:",
        backend_type="fake",
        backend_url="",
        admin_key=TEST_ADMIN_KEY,
    )
    isolated_config_path = Path("/tmp/localgate-test-nonexistent-config.json")
    return create_app(settings, database_config_path=isolated_config_path)


@pytest.fixture
async def client(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # Trigger the lifespan so app.state.session_factory / backend / rate_limiter exist
        async with app.router.lifespan_context(app):
            yield ac


@pytest.fixture
def admin_headers():
    return {"X-Admin-Key": TEST_ADMIN_KEY}


@pytest.fixture
async def api_key(client, admin_headers):
    """Creates a real API key through the actual admin endpoint and returns
    (key_id, raw_key) — most integration tests need a working key to call chat/etc.
    """
    resp = await client.post("/admin/keys", headers=admin_headers, json={"name": "test-key"})
    assert resp.status_code == 200
    data = resp.json()
    return data["id"], data["api_key"]


@pytest.fixture
def auth_headers(api_key):
    _, raw_key = api_key
    return {"Authorization": f"Bearer {raw_key}"}


@pytest.fixture
async def db_session(app, client):
    """A raw DB session for tests that need to assert on stored rows directly.
    Depends on `client` so the lifespan (which creates session_factory) has run.
    """
    async with app.state.session_factory() as session:
        yield session
