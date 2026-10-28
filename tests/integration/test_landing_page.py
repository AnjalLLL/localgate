"""The landing page at `/`: a browser gets the guide, an API client gets JSON,
and the page loads its assets under the dashboard's strict CSP."""

from localgate import __version__


async def test_browser_gets_the_landing_page(client):
    response = await client.get("/", headers={"Accept": "text/html,application/xhtml+xml"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "Get started" in response.text
    assert "localgate admin-key" in response.text
    assert "script-src 'self'" in response.headers["content-security-policy"]


async def test_api_client_gets_a_json_pointer(client):
    response = await client.get("/", headers={"Accept": "application/json"})
    assert response.status_code == 200
    assert response.json() == {
        "name": "localgate",
        "version": __version__,
        "api": "/v1",
        "docs": "/docs",
        "dashboard": "/dashboard/",
        "health": "/health",
    }


async def test_landing_page_assets_are_served(client):
    for asset in ("/dashboard/home.css", "/dashboard/home.js"):
        response = await client.get(asset)
        assert response.status_code == 200, asset
