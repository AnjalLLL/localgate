"""``/admin/keys`` — API key CRUD."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.ext.asyncio import AsyncSession

from localgate.api.deps import get_session, require_admin
from localgate.core.errors import InvalidRequestError
from localgate.core.key_policy import as_utc, is_expired, tokens_this_period
from localgate.core.logging import get_logger
from localgate.db.models import APIKey
from localgate.db.repositories.audit import audit
from localgate.db.repositories.keys import APIKeyRepository

router = APIRouter(tags=["admin"], dependencies=[Depends(require_admin)])
logger = get_logger(__name__)


class KeyLimits(BaseModel):
    """A key's optional limits. Leaving a field out (or null) means no limit."""

    allowed_models: list[str] | None = None
    max_tokens: int | None = Field(default=None, ge=1)
    token_quota: int | None = Field(default=None, ge=1)
    quota_period: Literal["day", "month"] | None = None
    expires_at: datetime | None = None
    max_concurrent: int | None = Field(default=None, ge=1)
    store_conversations: bool | None = None
    #: Where the key works: the owner's local network, the internet address, or both.
    network_access: Literal["local", "internet", "both"] | None = None

    @field_validator("allowed_models")
    @classmethod
    def _no_empty_list(cls, value: list[str] | None) -> list[str] | None:
        # "Allowed: nothing" is never what someone means — treat it as "any model".
        cleaned = [m.strip() for m in value or [] if m.strip()]
        return cleaned or None

    def columns(self, *, only_set: bool) -> dict[str, Any]:
        """Column values to write. With `only_set`, just the fields the request
        sent — so a PATCH changes what it names and nothing else."""
        names = self.model_fields_set if only_set else set(KeyLimits.model_fields)
        values = {name: getattr(self, name) for name in names if name in KeyLimits.model_fields}
        # These two have defaults in the table; null means "back to the default".
        if "quota_period" in values and values["quota_period"] is None:
            values["quota_period"] = "month"
        if "store_conversations" in values and values["store_conversations"] is None:
            values["store_conversations"] = True
        if "network_access" in values and values["network_access"] is None:
            values["network_access"] = "both"
        if not only_set:
            values = {k: v for k, v in values.items() if v is not None}
        return values


class CreateKeyRequest(KeyLimits):
    name: str = Field(..., min_length=1, max_length=100)
    rate_limit_per_min: int | None = Field(default=None, ge=1)


class UpdateKeyRequest(KeyLimits):
    """Partial update: send only what changes; null clears a limit."""

    rate_limit_per_min: int | None = Field(default=None, ge=1)


class KeyResponse(BaseModel):
    """A key as it can safely be shown — no secret material."""

    id: str
    name: str
    #: Empty for keys issued before 0.6: the column didn't exist, and the prefix cannot
    #: be reconstructed because the raw key is gone by design. Those keys still work;
    #: they just can't be identified by sight in a listing. This must stay tolerant of
    #: NULL — a `str` here turns the keys page into a 500 for everyone upgrading.
    key_prefix: str = ""
    revoked: bool
    rate_limit_per_min: int
    # Tolerant of NULL for the same reason as key_prefix: a listing endpoint that answers
    # 500 because one historical row has a null column is never the right trade.
    created_at: datetime | None = None
    last_used_at: datetime | None = None
    #: Whether an admin can view this key again (POST /admin/keys/{id}/reveal).
    revealable: bool = False
    allowed_models: list[str] | None = None
    max_tokens: int | None = None
    token_quota: int | None = None
    quota_period: str = "month"
    expires_at: datetime | None = None
    expired: bool = False
    max_concurrent: int | None = None
    store_conversations: bool = True
    network_access: str = "both"
    #: Tokens used in the current quota period (day or month, UTC).
    tokens_this_period: int | None = None

    @classmethod
    def of(cls, key: APIKey, tokens_this_period: int | None = None) -> KeyResponse:
        return cls(
            allowed_models=key.allowed_models,
            max_tokens=key.max_tokens,
            token_quota=key.token_quota,
            quota_period=key.quota_period or "month",
            expires_at=as_utc(key.expires_at) if key.expires_at else None,
            expired=is_expired(key),
            max_concurrent=key.max_concurrent,
            store_conversations=key.store_conversations is not False,
            network_access=key.network_access or "both",
            tokens_this_period=tokens_this_period,
            id=key.id,
            name=key.name,
            key_prefix=key.key_prefix or "",
            revoked=key.revoked,
            rate_limit_per_min=key.rate_limit_per_min,
            created_at=key.created_at,
            last_used_at=key.last_used_at,
            revealable=bool(key.key_ciphertext) and not key.revoked,
        )


class CreatedKeyResponse(KeyResponse):
    """The one and only response that ever carries the raw key."""

    api_key: str


@router.post("/keys", response_model=CreatedKeyResponse, status_code=201)
async def create_key(
    body: CreateKeyRequest, request: Request, session: AsyncSession = Depends(get_session)
) -> CreatedKeyResponse:
    """Create a key. Unless reveal is disabled (or the admin key is the insecure
    default), an encrypted copy is kept so an admin can view it again later."""
    settings = request.app.state.settings
    limit = body.rate_limit_per_min or settings.default_rate_limit_per_min

    vault_key = settings.admin_key if settings.api_key_reveal_enabled else None
    key, raw_key = await APIKeyRepository(session).create(
        body.name, limit, admin_key=vault_key, policy=body.columns(only_set=False)
    )
    await audit(request, "key.created", key.id, name=key.name, limits=_loggable(body))
    return CreatedKeyResponse(**KeyResponse.of(key, 0).model_dump(), api_key=raw_key)


@router.get("/keys", response_model=list[KeyResponse])
async def list_keys(session: AsyncSession = Depends(get_session)) -> list[KeyResponse]:
    keys = await APIKeyRepository(session).list_all()
    return [KeyResponse.of(key, await _period_usage(session, key)) for key in keys]


async def _period_usage(session: AsyncSession, key: APIKey) -> int | None:
    """Only worth a query when the key has a quota to measure against."""
    return await tokens_this_period(session, key) if key.token_quota is not None else None


class RevealedKey(BaseModel):
    id: str
    api_key: str


@router.post("/keys/{key_id}/reveal", response_model=RevealedKey)
async def reveal_key(
    key_id: str, request: Request, session: AsyncSession = Depends(get_session)
) -> RevealedKey:
    """Show a key's full value again. POST, so a cookie-authenticated dashboard
    request needs the CSRF header like any other sensitive action."""
    settings = request.app.state.settings
    repo = APIKeyRepository(session)
    key = await repo.get(key_id)
    if key is None:
        raise InvalidRequestError(f"No API key with id {key_id!r}.", code="key_not_found")
    if key.revoked:
        raise InvalidRequestError("This key is revoked.", code="key_revoked")
    if not settings.api_key_reveal_enabled:
        raise InvalidRequestError(
            "Revealing keys is disabled (LOCALGATE_API_KEY_REVEAL_ENABLED=false).",
            code="reveal_disabled",
        )
    raw = await repo.reveal(key_id, settings.admin_key)
    if raw is None:
        raise InvalidRequestError(
            "This key can't be shown again: it was created before keys could be revealed, "
            "while the default admin key was in use, or before the admin key was changed. "
            "Create a new key if you've lost it.",
            code="key_not_recoverable",
        )
    logger.info("api_key_revealed", key_id=key_id)
    await audit(request, "key.revealed", key_id, name=key.name)
    return RevealedKey(id=key_id, api_key=raw)


@router.get("/keys/{key_id}", response_model=KeyResponse)
async def get_key(key_id: str, session: AsyncSession = Depends(get_session)) -> KeyResponse:
    key = await APIKeyRepository(session).get(key_id)
    if key is None:
        raise InvalidRequestError(f"No API key with id {key_id!r}.", code="key_not_found")
    return KeyResponse.of(key, await _period_usage(session, key))


@router.patch("/keys/{key_id}", response_model=KeyResponse)
async def update_key(
    key_id: str,
    body: UpdateKeyRequest,
    request: Request,
    session: AsyncSession = Depends(get_session),
) -> KeyResponse:
    """Change a key's rate limit and other limits without reissuing it. Only the
    fields sent change; null clears a limit."""
    repo = APIKeyRepository(session)
    fields = body.columns(only_set=True)
    if "rate_limit_per_min" in body.model_fields_set:
        if body.rate_limit_per_min is None:
            raise InvalidRequestError("rate_limit_per_min can't be null.", code="invalid_limit")
        fields["rate_limit_per_min"] = body.rate_limit_per_min
    if not await repo.update_policy(key_id, fields):
        raise InvalidRequestError(f"No API key with id {key_id!r}.", code="key_not_found")

    key = await repo.get(key_id)
    if key is None:  # pragma: no cover — it was updated one statement ago
        raise InvalidRequestError(f"No API key with id {key_id!r}.", code="key_not_found")
    await session.refresh(key)
    await audit(request, "key.updated", key_id, name=key.name, changes=_loggable(body))
    return KeyResponse.of(key, await _period_usage(session, key))


@router.delete("/keys/{key_id}")
async def revoke_key(
    key_id: str, request: Request, session: AsyncSession = Depends(get_session)
) -> dict:
    """Revoke a key.

    Revocation sets a flag rather than deleting the row: usage records reference the
    key, and deleting it would silently rewrite the history the usage dashboard
    reports.
    """
    if not await APIKeyRepository(session).revoke(key_id):
        raise InvalidRequestError(f"No API key with id {key_id!r}.", code="key_not_found")
    await audit(request, "key.revoked", key_id)
    return {"id": key_id, "revoked": True}


def _loggable(body: BaseModel) -> dict[str, Any]:
    """The fields a request set, JSON-safe, for the audit log."""
    return body.model_dump(mode="json", include=body.model_fields_set)
