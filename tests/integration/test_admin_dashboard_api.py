"""Dashboard sign-in, cookie + CSRF rules, and the operator endpoints."""

from localgate.agent.loop import ToolLogEntry
from localgate.agent.memory import AgentMemory
from localgate.core.auth import ADMIN_COOKIE, issue_admin_session, verify_admin_session


def test_admin_session_tokens_expire_and_bind_to_the_key():
    token = issue_admin_session("key-1", now=1000, lifetime=60)
    assert verify_admin_session(token, "key-1", now=1030)
    assert not verify_admin_session(token, "key-1", now=1061)
    assert not verify_admin_session(token, "key-2", now=1030)
    assert not verify_admin_session("garbage", "key-1", now=1030)


async def test_sign_in_sets_an_httponly_cookie_and_rejects_bad_keys(client):
    bad = await client.post("/admin/session", json={"admin_key": "nope"})
    assert bad.status_code == 401
    assert (await client.get("/admin/session")).status_code == 401

    good = await client.post("/admin/session", json={"admin_key": "test-admin-key"})
    assert good.status_code == 204
    cookie = good.headers["set-cookie"]
    assert ADMIN_COOKIE in cookie and "HttpOnly" in cookie and "SameSite=strict" in cookie
    assert (await client.get("/admin/session")).json() == {"admin": True, "email": None}

    await client.delete("/admin/session")
    client.cookies.clear()
    assert (await client.get("/admin/session")).status_code == 401


async def test_cookie_auth_requires_the_csrf_header_for_changes(client):
    await client.post("/admin/session", json={"admin_key": "test-admin-key"})
    blocked = await client.post("/admin/keys", json={"name": "x"})
    assert blocked.status_code == 401 and "csrf" in blocked.text.lower()
    allowed = await client.post(
        "/admin/keys", json={"name": "x"}, headers={"X-Localgate-Csrf": "1"}
    )
    assert allowed.status_code == 201


async def test_dashboard_has_strict_security_headers(client):
    resp = await client.get("/dashboard/")
    assert "script-src 'self'" in resp.headers["content-security-policy"]
    assert resp.headers["x-frame-options"] == "DENY"


async def test_overview_models_sessions_and_tool_calls(app, client, admin_headers, auth_headers):
    await client.post(
        "/v1/chat/completions",
        headers={**auth_headers, "X-Session-Id": "s-1"},
        json={"model": "fake-model", "messages": [{"role": "user", "content": "hello"}]},
    )
    memory = AgentMemory(
        app.state.session_factory, app.state.backend, app.state.settings, "agent-s", "owner"
    )
    await memory.record_tool_call(
        ToolLogEntry("r", "agent-s", "qwen", "read_file", {"path": "a"}, True, 5)
    )
    await memory.record_tool_call(
        ToolLogEntry("r", "agent-s", "qwen", "edit_file", {"path": "a"}, False, 7, error="x")
    )

    overview = (await client.get("/admin/overview", headers=admin_headers)).json()
    assert overview["backend"]["healthy"] is True
    assert overview["agent"]["success_rate"] == 0.5

    models = (await client.get("/admin/models", headers=admin_headers)).json()
    assert models[0]["name"] == "fake-model"

    sessions = (await client.get("/admin/sessions", headers=admin_headers)).json()
    entry = next(s for s in sessions if s["session_id"] == "s-1")
    detail = (
        await client.get(f"/admin/sessions/{entry['api_key_id']}/s-1", headers=admin_headers)
    ).json()
    assert detail["messages"][0]["content"] == "hello"
    missing = await client.get("/admin/sessions/nobody/s-1", headers=admin_headers)
    assert missing.status_code == 404

    calls = (await client.get("/admin/tool-calls", headers=admin_headers)).json()
    assert {t["name"]: t["success_rate"] for t in calls["by_tool"]} == {
        "read_file": 1.0,
        "edit_file": 0.0,
    }
    assert calls["by_model"][0] == {"name": "qwen", "calls": 2, "ok": 1, "success_rate": 0.5}

    found = await client.post(
        "/admin/memory/search",
        headers=admin_headers,
        json={"api_key_id": entry["api_key_id"], "session_id": "s-1", "query": "hello"},
    )
    assert found.status_code == 200 and found.json()["results"]
