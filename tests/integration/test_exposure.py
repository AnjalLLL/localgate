"""What a client outside the owner's trusted networks can and can't reach."""

import pytest
from httpx import ASGITransport, AsyncClient

from tests.conftest import TEST_ADMIN_KEY

OUTSIDE = ("203.0.113.5", 4000)  # TEST-NET-3: a public internet address
LAN = ("192.168.1.9", 4000)


@pytest.fixture
async def from_outside(app, client):
    async with AsyncClient(
        transport=ASGITransport(app=app, client=OUTSIDE), base_url="http://gw"
    ) as c:
        yield c


@pytest.fixture
async def from_lan(app, client):
    async with AsyncClient(transport=ASGITransport(app=app, client=LAN), base_url="http://gw") as c:
        yield c


ADMIN = {"X-Admin-Key": TEST_ADMIN_KEY}


@pytest.mark.parametrize(
    "path", ["/admin/keys", "/admin/session/methods", "/docs", "/openapi.json", "/metrics"]
)
async def test_admin_surfaces_are_refused_from_outside(from_outside, path):
    response = await from_outside.get(path, headers=ADMIN)  # even with the admin key
    assert response.status_code == 403
    assert response.json()["error"]["code"] == "untrusted_network"


async def test_the_dashboard_is_refused_from_outside_but_its_public_assets_load(from_outside):
    assert (await from_outside.get("/dashboard/")).status_code == 403
    assert (await from_outside.get("/dashboard/app.js")).status_code == 403
    for asset in ("/dashboard/home.css", "/dashboard/home.js", "/dashboard/font-inter.woff2"):
        assert (await from_outside.get(asset)).status_code == 200, asset


async def test_the_api_works_from_outside(from_outside, auth_headers):
    response = await from_outside.post(
        "/v1/chat/completions",
        headers=auth_headers,
        json={"model": "fake-model", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 200
    assert (await from_outside.get("/")).status_code == 200


async def test_health_is_minimal_from_outside_and_full_on_the_lan(from_outside, from_lan):
    outside = (await from_outside.get("/health")).json()
    assert set(outside) == {"status", "version"}
    lan = (await from_lan.get("/health")).json()
    assert "backend" in lan and "database" in lan
    assert (await from_lan.get("/admin/keys", headers=ADMIN)).status_code == 200


async def test_an_admin_sees_full_health_from_anywhere(app, client):
    # Signed in via the admin key header but from a public address — e.g. through a
    # tunnel that doesn't forward client IPs. Health details are an admin's to see.
    async with AsyncClient(
        transport=ASGITransport(app=app, client=OUTSIDE), base_url="http://gw"
    ) as c:
        body = (await c.get("/health", headers=ADMIN)).json()
    assert "backend" in body
