"""Shared fixtures.

Every test app gets:

* an in-memory SQLite database, isolated per test, that never touches real data;
* the deterministic ``FakeBackend``, so no test needs a live Ollama;
* a ``database_config_path`` pointing at a file that does not exist, so a test can
  never pick up a real ``localgate.config.json`` from the developer's working
  directory and start writing to their actual database (see ``app.create_app``).
"""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from localgate.app import create_app
from localgate.config import Settings

TEST_ADMIN_KEY = "test-admin-key"


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
