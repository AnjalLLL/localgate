"""Dashboard accounts: email/password sign-in next to the admin key."""

import pytest

from localgate.api import admin_ops
from localgate.core.auth import ADMIN_COOKIE
from localgate.db.repositories.users import UserRepository


@pytest.fixture(autouse=True)
def _reset_throttle():
    admin_ops._failed_sign_ins.clear()
    yield
    admin_ops._failed_sign_ins.clear()


async def _create_user(app, email="Ana@Example.com", password="correct horse"):
    async with app.state.session_factory() as session:
        await UserRepository(session).create(email, password)


async def test_methods_report_whether_an_account_exists(client, app):
    assert (await client.get("/admin/session/methods")).json() == {"accounts": False}
    await _create_user(app)
    assert (await client.get("/admin/session/methods")).json() == {"accounts": True}


async def test_sign_in_with_email_and_password(client, app):
    await _create_user(app)
    response = await client.post(
        "/admin/session", json={"email": "ana@example.com", "password": "correct horse"}
    )
    assert response.status_code == 204
    assert ADMIN_COOKIE in response.headers["set-cookie"]
    who = await client.get("/admin/session")
    assert who.json() == {"admin": True, "email": "ana@example.com"}
    assert (await client.get("/admin/keys")).status_code == 200


async def test_a_wrong_password_is_refused(client, app):
    await _create_user(app)
    response = await client.post(
        "/admin/session", json={"email": "ana@example.com", "password": "wrong horse"}
    )
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "invalid_credentials"
    assert (await client.get("/admin/session")).status_code == 401


async def test_repeated_failures_are_throttled(client, app):
    await _create_user(app)
    for _ in range(admin_ops._MAX_FAILED_SIGN_INS):
        await client.post(
            "/admin/session", json={"email": "ana@example.com", "password": "nope-nope"}
        )
    blocked = await client.post(
        "/admin/session", json={"email": "ana@example.com", "password": "correct horse"}
    )
    assert blocked.status_code == 429


async def test_the_admin_key_still_signs_in(client, admin_headers):
    key = admin_headers["X-Admin-Key"]
    assert (await client.post("/admin/session", json={"admin_key": key})).status_code == 204
    assert (await client.get("/admin/session")).json() == {"admin": True, "email": None}
