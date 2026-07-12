"""Integration test: does a chat request actually round-trip through the backend?

Run this with a real Ollama instance up (e.g. `ollama run llama3`) — it's not mocked
yet, so it will fail/skip if nothing is listening on localhost:11434. That's
intentional for this phase; a mocked version comes once backends/base.py
has a proper test double.
"""
import httpx
import pytest

from localgate.app import create_app
from localgate.config import Settings


@pytest.fixture
def app():
    return create_app(Settings(database_url="sqlite+aiosqlite:///:memory:"))


async def test_health_reports_backend_status(app):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health")
        assert resp.status_code == 200
        assert "backend_reachable" in resp.json()


async def test_chat_completion_roundtrip(app):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test", timeout=60) as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "llama3",
                "messages": [{"role": "user", "content": "Say hi in one word."}],
            },
        )
        if resp.status_code != 200:
            pytest.skip("No Ollama instance reachable on localhost:11434 — start one to run this test.")
        body = resp.json()
        assert "choices" in body
