"""``/admin/keys`` — API key CRUD."""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from localgate.api.deps import get_session, require_admin
from localgate.core.errors import InvalidRequestError
from localgate.db.models import APIKey
from localgate.db.repositories.keys import APIKeyRepository

router = APIRouter(tags=["admin"], dependencies=[Depends(require_admin)])


class CreateKeyRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    rate_limit_per_min: int | None = Field(default=None, ge=1)


class UpdateKeyRequest(BaseModel):
    rate_limit_per_min: int = Field(..., ge=1)


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

    @classmethod
    def of(cls, key: APIKey) -> KeyResponse:
        return cls(
            id=key.id,
            name=key.name,
            key_prefix=key.key_prefix or "",
            revoked=key.revoked,
            rate_limit_per_min=key.rate_limit_per_min,
            created_at=key.created_at,
            last_used_at=key.last_used_at,
        )


class CreatedKeyResponse(KeyResponse):
    """The one and only response that ever carries the raw key."""

    api_key: str


@router.post("/keys", response_model=CreatedKeyResponse, status_code=201)
async def create_key(
    body: CreateKeyRequest, request: Request, session: AsyncSession = Depends(get_session)
) -> CreatedKeyResponse:
    """Create a key. The raw value is returned **once** and is not recoverable after that."""
    settings = request.app.state.settings
    limit = body.rate_limit_per_min or settings.default_rate_limit_per_min

    key, raw_key = await APIKeyRepository(session).create(body.name, limit)
    return CreatedKeyResponse(**KeyResponse.of(key).model_dump(), api_key=raw_key)


@router.get("/keys", response_model=list[KeyResponse])
async def list_keys(session: AsyncSession = Depends(get_session)) -> list[KeyResponse]:
    return [KeyResponse.of(key) for key in await APIKeyRepository(session).list_all()]


@router.get("/keys/{key_id}", response_model=KeyResponse)
async def get_key(key_id: str, session: AsyncSession = Depends(get_session)) -> KeyResponse:
    key = await APIKeyRepository(session).get(key_id)
    if key is None:
        raise InvalidRequestError(f"No API key with id {key_id!r}.", code="key_not_found")
    return KeyResponse.of(key)


@router.patch("/keys/{key_id}", response_model=KeyResponse)
async def update_key(
    key_id: str, body: UpdateKeyRequest, session: AsyncSession = Depends(get_session)
) -> KeyResponse:
    """Change a key's rate limit without reissuing it."""
    repo = APIKeyRepository(session)
    if not await repo.set_rate_limit(key_id, body.rate_limit_per_min):
        raise InvalidRequestError(f"No API key with id {key_id!r}.", code="key_not_found")

    key = await repo.get(key_id)
    if key is None:  # pragma: no cover — it was updated one statement ago
        raise InvalidRequestError(f"No API key with id {key_id!r}.", code="key_not_found")
    return KeyResponse.of(key)


@router.delete("/keys/{key_id}")
async def revoke_key(key_id: str, session: AsyncSession = Depends(get_session)) -> dict:
    """Revoke a key.

    Revocation sets a flag rather than deleting the row: usage records reference the
    key, and deleting it would silently rewrite the history the usage dashboard
    reports.
    """
    if not await APIKeyRepository(session).revoke(key_id):
        raise InvalidRequestError(f"No API key with id {key_id!r}.", code="key_not_found")
    return {"id": key_id, "revoked": True}
