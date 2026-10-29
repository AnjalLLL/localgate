"""Per-key policy: what one API key may do, beyond its per-minute rate limit.

An owner sharing a GPU hands each device or person its own key, and each key
can be held to: a set of models, a `max_tokens` cap per request, a token quota
per UTC day or month, an expiry date, and a number of requests in flight
(enforced by the inference queue, see `core/scheduler.py`). Every limit is
optional — NULL in the database means "no limit".
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from localgate.core.errors import PermissionDeniedError, RateLimitError
from localgate.db.models import APIKey
from localgate.db.repositories.usage import UsageRepository

QUOTA_PERIODS = ("day", "month")
NETWORK_ACCESS = ("local", "internet", "both")


def as_utc(moment: datetime) -> datetime:
    """SQLite hands back naive datetimes even for timezone-aware columns."""
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


def is_expired(key: APIKey, now: datetime | None = None) -> bool:
    if key.expires_at is None:
        return False
    return as_utc(key.expires_at) <= (now or datetime.now(timezone.utc))


def period_bounds(period: str, now: datetime) -> tuple[datetime, datetime]:
    """(start, reset) of the quota period containing ``now`` — UTC calendar
    days or months, so "1M tokens a month" means what people expect."""
    if period == "day":
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        return start, start + timedelta(days=1)
    start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    reset = (start + timedelta(days=32)).replace(day=1)
    return start, reset


async def tokens_this_period(
    session: AsyncSession, key: APIKey, now: datetime | None = None
) -> int:
    now = now or datetime.now(timezone.utc)
    start, _ = period_bounds(key.quota_period or "month", now)
    return await UsageRepository(session).tokens_since(key.id, start)


def model_allowed(key: APIKey, requested: str | None, resolved: str) -> bool:
    """A key allowed "fast" may call what "fast" points at, and vice versa."""
    if not key.allowed_models:
        return True
    return resolved in key.allowed_models or (requested or "") in key.allowed_models


async def enforce_key_access(
    key: APIKey,
    requested: str | None,
    model: str,
    session: AsyncSession,
    *,
    check_model: bool = True,
) -> None:
    """Refuse up front — before any memory or embedding work — a model this key
    may not use, or a request past its token quota."""
    if check_model and not model_allowed(key, requested, model):
        allowed = ", ".join(key.allowed_models or [])
        raise PermissionDeniedError(
            f"This API key can't use model {model!r}. Allowed: {allowed}.",
            code="model_not_allowed",
        )
    if key.token_quota is None:
        return
    now = datetime.now(timezone.utc)
    used = await tokens_this_period(session, key, now)
    if used >= key.token_quota:
        _, reset = period_bounds(key.quota_period or "month", now)
        raise RateLimitError(
            f"This API key used its {key.token_quota:,}-token {key.quota_period}ly quota. "
            f"It resets at {reset:%Y-%m-%d %H:%M} UTC.",
            code="quota_exceeded",
            retry_after=(reset - now).total_seconds(),
        )


def clamp_max_tokens(key: APIKey, payload: dict[str, Any]) -> None:
    """Hold the reply length to the key's cap — including Ollama's own
    `options.num_predict`, so the cap can't be sidestepped through it."""
    cap = key.max_tokens
    if cap is None:
        return
    requested = payload.get("max_tokens")
    payload["max_tokens"] = min(requested, cap) if isinstance(requested, int) else cap
    options = payload.get("options")
    if isinstance(options, dict) and isinstance(options.get("num_predict"), int):
        options["num_predict"] = min(options["num_predict"], cap)


def policy_dict(key: APIKey) -> dict[str, Any]:
    """The key's limits, as shown to its owner and to the key holder."""
    return {
        "allowed_models": key.allowed_models,
        "max_tokens": key.max_tokens,
        "token_quota": key.token_quota,
        "quota_period": key.quota_period or "month",
        "expires_at": as_utc(key.expires_at).isoformat() if key.expires_at else None,
        "max_concurrent": key.max_concurrent,
        "store_conversations": key.store_conversations is not False,
        "network_access": key.network_access or "both",
        "rate_limit_per_min": key.rate_limit_per_min,
    }


def check_network_access(key: APIKey, from_internet: bool) -> None:
    """Refuse a key used where its owner didn't allow it."""
    access = key.network_access or "both"
    if access == "local" and from_internet:
        raise PermissionDeniedError(
            "This API key only works on the owner's local network, not over the internet.",
            code="network_not_allowed",
        )
    if access == "internet" and not from_internet:
        raise PermissionDeniedError(
            "This API key only works through the gateway's internet address.",
            code="network_not_allowed",
        )
