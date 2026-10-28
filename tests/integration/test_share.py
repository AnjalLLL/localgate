"""The Share page's API: reachable addresses, QR codes, and the public /connect page."""

from tests.conftest import TEST_ADMIN_KEY

ADMIN = {"X-Admin-Key": TEST_ADMIN_KEY}


async def test_share_lists_reachable_addresses(client, monkeypatch):
    monkeypatch.setattr(
        "localgate.api.share.share_urls",
        lambda port, public_url=None: [
            {"label": "Local network", "url": "http://192.168.1.20:8000"}
        ],
    )
    body = (await client.get("/admin/share", headers=ADMIN)).json()
    assert {"label": "Local network", "url": "http://192.168.1.20:8000"} in body["urls"]
    # The address the dashboard was opened at (not loopback here) is offered first.
    assert body["urls"][0] == {"label": "This address", "url": "http://test"}


async def test_share_requires_the_admin(client):
    assert (await client.get("/admin/share")).status_code == 401


async def test_qr_codes_are_svg(client):
    response = await client.post(
        "/admin/share/qr", headers=ADMIN, json={"text": "http://192.168.1.20:8000/connect#key=lg_x"}
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/svg+xml")
    assert response.text.lstrip().startswith("<svg")


async def test_connect_page_is_public_and_locked_down(client):
    response = await client.get("/connect")
    assert response.status_code == 200
    assert "Connect to this gateway" in response.text
    assert "script-src 'self'" in response.headers["content-security-policy"]
    assert (await client.get("/dashboard/connect.js")).status_code == 200
