"""Keys that work on the local network, over the internet, or both — and the
dashboard switch that turns the internet address on and off."""

import asyncio
import subprocess
import sys

import pytest
from httpx import ASGITransport, AsyncClient

from localgate.core import tunnel as tunnel_module
from tests.conftest import TEST_ADMIN_KEY

ADMIN = {"X-Admin-Key": TEST_ADMIN_KEY}
LAN = ("192.168.1.9", 4000)
PUBLIC = ("203.0.113.5", 4000)
TUNNEL = ("127.0.0.1", 4000)  # cloudflared connects from this machine


def _chat():
    return {"model": "fake-model", "messages": [{"role": "user", "content": "hi"}]}


@pytest.fixture
def fake_tunnel(monkeypatch):
    """A tunnel manager whose "tunnel" is a harmless local process."""

    def open_blocking(self):
        process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        return tunnel_module.Tunnel("https://quiet-forest.trycloudflare.com", process)

    monkeypatch.setattr(tunnel_module.TunnelManager, "_open_blocking", open_blocking)


async def _key(client, network):
    resp = await client.post(
        "/admin/keys", headers=ADMIN, json={"name": network, "network_access": network}
    )
    return {"Authorization": f"Bearer {resp.json()['api_key']}"}


async def _call(app, peer, headers, extra=None):
    async with AsyncClient(
        transport=ASGITransport(app=app, client=peer), base_url="http://gw"
    ) as c:
        return await c.post(
            "/v1/chat/completions", headers={**headers, **(extra or {})}, json=_chat()
        )


async def _turn_internet_on(client, app):
    await client.post("/admin/share/internet", headers=ADMIN, json={"enabled": True})
    for _ in range(100):
        if app.state.tunnel.active:
            return
        await asyncio.sleep(0.02)
    raise AssertionError("tunnel never came up")


async def test_a_local_key_works_on_the_lan_but_not_from_the_internet(client, app, fake_tunnel):
    await _turn_internet_on(client, app)
    headers = await _key(client, "local")
    assert (await _call(app, LAN, headers)).status_code == 200
    through_tunnel = await _call(app, TUNNEL, headers, {"CF-Connecting-IP": "198.51.100.4"})
    assert through_tunnel.status_code == 403
    assert through_tunnel.json()["error"]["code"] == "network_not_allowed"
    assert (await _call(app, PUBLIC, headers)).status_code == 403
    app.state.tunnel.stop()


async def test_an_internet_key_works_only_through_the_internet_address(client, app, fake_tunnel):
    await _turn_internet_on(client, app)
    headers = await _key(client, "internet")
    assert (
        await _call(app, TUNNEL, headers, {"CF-Connecting-IP": "198.51.100.4"})
    ).status_code == 200
    assert (await _call(app, LAN, headers)).status_code == 403
    app.state.tunnel.stop()


async def test_a_both_key_works_everywhere(client, app, fake_tunnel):
    await _turn_internet_on(client, app)
    headers = await _key(client, "both")
    assert (await _call(app, LAN, headers)).status_code == 200
    assert (
        await _call(app, TUNNEL, headers, {"CF-Connecting-IP": "198.51.100.4"})
    ).status_code == 200
    app.state.tunnel.stop()


async def test_the_dashboard_switch_opens_and_closes_the_internet_address(client, app, fake_tunnel):
    off = (await client.get("/admin/share", headers=ADMIN)).json()
    assert off["internet"]["enabled"] is False
    await _turn_internet_on(client, app)
    on = (await client.get("/admin/share", headers=ADMIN)).json()
    assert on["internet"]["url"] == "https://quiet-forest.trycloudflare.com"
    assert on["urls"][0] == {"label": "Internet", "url": "https://quiet-forest.trycloudflare.com"}
    closed = await client.post("/admin/share/internet", headers=ADMIN, json={"enabled": False})
    assert closed.json()["enabled"] is False
    actions = [e["action"] for e in (await client.get("/admin/audit", headers=ADMIN)).json()]
    assert actions[:2] == ["share.internet_off", "share.internet_on"]


async def test_cf_headers_mean_nothing_without_a_tunnel(client, app):
    """With the internet address off, a LAN device can't claim to be someone else."""
    headers = await _key(client, "internet")
    spoofed = await _call(app, TUNNEL, headers, {"CF-Connecting-IP": "198.51.100.4"})
    assert (
        spoofed.status_code == 403
    )  # treated as this machine: local, so an internet key is refused
