"""Shared FastAPI dependencies: DB session, API key auth, admin auth, rate limiting.

These are dependencies rather than middleware on purpose — see
docs/decisions/0001-dependencies-over-middleware.md.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from localgate.core import metrics
from localgate.core.auth import constant_time_equals
from localgate.core.errors import AuthenticationError, RateLimitError
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
    if not limiter.allow(api_key.id, api_key.rate_limit_per_min):
        metrics.rate_limited_total.inc()
        raise RateLimitError(
            f"Rate limit exceeded for this key ({api_key.rate_limit_per_min} requests/minute). "
            "Raise it with `localgate keys update <id> --rate-limit N`, or retry shortly.",
            code="rate_limit_exceeded",
        )
    return api_key


def require_admin(request: Request) -> None:
    """Gate the ``/admin`` surface on the configured admin key."""
    settings = request.app.state.settings
    provided = request.headers.get("x-admin-key")
    if not provided or not constant_time_equals(provided, settings.admin_key):
        raise AuthenticationError(
            "Invalid admin key. Pass it as the X-Admin-Key header.",
            code="invalid_admin_key",
        )
