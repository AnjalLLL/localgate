"""Shared pytest fixtures — test app instance, async DB session, mock backend."""
import pytest
from httpx import ASGITransport, AsyncClient

from localgate.app import create_app
from localgate.config import Settings


@pytest.fixture
def app():
    return create_app(Settings(database_url="sqlite+aiosqlite:///:memory:"))


@pytest.fixture
async def client(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
