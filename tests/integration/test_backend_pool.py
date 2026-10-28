"""Several inference backends behind one gateway: routing by model, least-busy
choice, failover when one is unreachable, and an aggregated model list."""

import json

import httpx
import pytest

from localgate.app import create_app
from localgate.config import Settings
from localgate.core.errors import ConfigurationError
from tests.conftest import TEST_ADMIN_KEY

BACKENDS = json.dumps(
    [
        {
            "name": "big-gpu",
            "type": "fake",
            "url": "a",
            "models": ["llama3:70b", "qwen2.5-coder:7b"],
        },
        {"name": "laptop", "type": "fake", "url": "b", "models": ["qwen2.5-coder:7b", "phi4-mini"]},
    ]
)


@pytest.fixture
def settings() -> Settings:
    return Settings(
        _env_file=None,
        database_url="sqlite+aiosqlite:///:memory:",
        backend_type="fake",
        backend_url="",
        admin_key=TEST_ADMIN_KEY,
        log_level="WARNING",
        backends=BACKENDS,
    )


def _chat(model):
    return {"model": model, "messages": [{"role": "user", "content": "hi"}]}


def _member(app, name):
    return next(m for m in app.state.pool.members if m.name == name)


async def test_each_model_goes_to_a_backend_that_serves_it(client, auth_headers, app):
    big, laptop = _member(app, "big-gpu"), _member(app, "laptop")
    await client.post("/v1/chat/completions", headers=auth_headers, json=_chat("llama3:70b"))
    await client.post("/v1/chat/completions", headers=auth_headers, json=_chat("phi4-mini"))
    assert [c["model"] for c in big.backend.calls] == ["llama3:70b"]
    assert [c["model"] for c in laptop.backend.calls] == ["phi4-mini"]


async def test_a_shared_model_goes_to_the_least_busy_backend(app, client):
    big = _member(app, "big-gpu")
    held = await big.scheduler.acquire("someone")
    assert [m.name for m in await app.state.pool.route("qwen2.5-coder:7b")] == ["laptop", "big-gpu"]
    held.release()


async def test_an_unreachable_backend_fails_over_to_the_next(client, auth_headers, app):
    big, laptop = _member(app, "big-gpu"), _member(app, "laptop")
    held = await laptop.scheduler.acquire("someone")  # make big-gpu the first choice…

    async def down(request):
        raise httpx.ConnectError("connection refused")

    big.backend.chat = down  # …and have it be down
    response = await client.post(
        "/v1/chat/completions", headers=auth_headers, json=_chat("qwen2.5-coder:7b")
    )
    held.release()
    assert response.status_code == 200
    assert laptop.backend.calls[-1]["model"] == "qwen2.5-coder:7b"


async def test_a_stream_fails_over_before_its_first_token(client, auth_headers, app):
    big, laptop = _member(app, "big-gpu"), _member(app, "laptop")
    held = await laptop.scheduler.acquire("someone")

    async def down(request):
        raise httpx.ConnectError("connection refused")
        yield  # pragma: no cover — makes this an async generator

    big.backend.chat_stream = down
    response = await client.post(
        "/v1/chat/completions",
        headers=auth_headers,
        json={**_chat("qwen2.5-coder:7b"), "stream": True},
    )
    held.release()
    assert response.status_code == 200
    assert "Echo" in response.text and "error" not in response.text


async def test_the_model_list_covers_every_backend(client, auth_headers, app):
    async def listing(names):
        return names

    for member in app.state.pool.members:
        member.backend.list_models = lambda names=member.spec.models: listing(names)
    data = (await client.get("/v1/models", headers=auth_headers)).json()["data"]
    owners = {card["id"]: card["owned_by"] for card in data}
    assert owners == {"llama3:70b": "big-gpu", "qwen2.5-coder:7b": "big-gpu", "phi4-mini": "laptop"}


async def test_one_backend_down_still_lists_the_others(client, auth_headers, app):
    big, laptop = _member(app, "big-gpu"), _member(app, "laptop")

    async def down():
        raise httpx.ConnectError("connection refused")

    async def laptop_models():
        return ["phi4-mini"]

    big.backend.list_models = down
    laptop.backend.list_models = laptop_models
    data = (await client.get("/v1/models", headers=auth_headers)).json()["data"]
    assert [card["id"] for card in data] == ["phi4-mini"]


async def test_health_reports_every_backend(client):
    body = (await client.get("/health")).json()
    assert [b["name"] for b in body["backends"]] == ["big-gpu", "laptop"]
    assert all("queued" in b for b in body["backends"])


def test_an_unknown_backend_type_is_refused_at_startup(tmp_path):
    bad = Settings(
        _env_file=None,
        database_url="sqlite+aiosqlite:///:memory:",
        admin_key=TEST_ADMIN_KEY,
        backends='[{"name": "x", "type": "tensor-magic", "url": "http://x"}]',
    )
    with pytest.raises(ConfigurationError, match="tensor-magic"):
        create_app(bad, database_config_path=tmp_path / "cfg.json")
