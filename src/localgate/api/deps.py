"""Shared FastAPI dependencies: DB session, API key auth, admin auth, rate limiting.

These are dependencies rather than middleware on purpose — see
docs/decisions/0001-dependencies-over-middleware.md.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator

from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from localgate.core import metrics
from localgate.core.auth import ADMIN_COOKIE, constant_time_equals, verify_admin_session
from localgate.core.errors import AuthenticationError, RateLimitError
from localgate.core.key_policy import as_utc, is_expired
from localgate.db.models import APIKey
from localgate.db.repositories.keys import APIKeyRepository

# auto_error=False so that a missing header reaches our own handler and comes back
# in the OpenAI error envelope, rather than FastAPI's {"detail": "Not authenticated"}
# — which the OpenAI SDK cannot parse.
bearer_scheme = HTTPBearer(auto_error=False, description="Your localgate API key")


async def get_session(request: Request) -> AsyncIterator[AsyncSession]:
    session_factory = request.app.state.session_factory
    async with session_factory() as session:
        yield session


async def require_api_key(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    session: AsyncSession = Depends(get_session),
) -> APIKey:
    """Resolve the bearer token to an active API key, or reject the request."""
    if credentials is None or not credentials.credentials:
        raise AuthenticationError(
            "Missing API key. Pass it as: Authorization: Bearer <api-key>",
            code="missing_api_key",
        )

    repo = APIKeyRepository(session)
    key = await repo.get_by_raw_key(credentials.credentials)
    if key is None:
        raise AuthenticationError(
            "Invalid or revoked API key. Create one with `localgate keys create`.",
            code="invalid_api_key",
        )

    if key.expires_at is not None and is_expired(key):
        raise AuthenticationError(
            f"This API key expired on {as_utc(key.expires_at):%Y-%m-%d} UTC. "
            "Ask the gateway's owner for a new one.",
            code="key_expired",
        )

    await repo.touch_last_used(key.id)
    return key


async def enforce_rate_limit(
    request: Request, api_key: APIKey = Depends(require_api_key)
) -> APIKey:
    """Charge this request against its key's per-minute budget.

    Depends on ``require_api_key`` rather than re-resolving the key itself: the
    limit is *per key*, so it cannot be evaluated until the key is known. FastAPI
    caches a dependency within a request, so the key is still resolved only once
    even though several dependencies ask for it.
    """
    limiter = request.app.state.rate_limiter
    allowed, retry_after = await limiter.check(api_key.id, api_key.rate_limit_per_min)
    if not allowed:
        metrics.rate_limited_total.inc()
        raise RateLimitError(
            f"Rate limit exceeded for this key ({api_key.rate_limit_per_min} requests/minute). "
            "Raise it with `localgate keys update <id> --rate-limit N`, or retry shortly.",
            code="rate_limit_exceeded",
            retry_after=retry_after,
        )
    return api_key


def require_admin(request: Request) -> None:
    """Gate the ``/admin`` surface on the configured admin key.

    Accepted as the ``X-Admin-Key`` header (CLI, scripts), or as the dashboard's
    httpOnly session cookie. A cookie is sent by the browser automatically, so a
    state-changing request authenticated by cookie must also carry the
    ``X-Localgate-Csrf`` header, which a cross-site page cannot set.
    """
    settings = request.app.state.settings
    provided = request.headers.get("x-admin-key")
    if provided and constant_time_equals(provided, settings.admin_key):
        return
    cookie = request.cookies.get(ADMIN_COOKIE)
    if cookie and verify_admin_session(cookie, settings.admin_key, time.time()):
        if request.method in {"GET", "HEAD", "OPTIONS"} or request.headers.get("x-localgate-csrf"):
            return
        raise AuthenticationError(
            "Missing X-Localgate-Csrf header on a cookie-authenticated request.",
            code="csrf_required",
        )
    raise AuthenticationError(
        "Invalid admin key. Pass it as the X-Admin-Key header.",
        code="invalid_admin_key",
    )
