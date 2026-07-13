"""Admin CRUD for API keys — /admin/keys."""
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from localgate.api.deps import get_session, require_admin
from localgate.db.repositories.keys import APIKeyRepository

router = APIRouter(dependencies=[Depends(require_admin)])


class CreateKeyRequest(BaseModel):
    name: str
    rate_limit_per_min: int = 60


class KeyResponse(BaseModel):
    id: str
    name: str
    revoked: bool
    rate_limit_per_min: int


@router.post("/keys")
async def create_key(body: CreateKeyRequest, session: AsyncSession = Depends(get_session)):
    repo = APIKeyRepository(session)
    key, raw_key = await repo.create(body.name, body.rate_limit_per_min)
    return {
        "id": key.id,
        "name": key.name,
        "api_key": raw_key,  # shown exactly once — the caller must store it now
        "rate_limit_per_min": key.rate_limit_per_min,
    }


@router.get("/keys")
async def list_keys(session: AsyncSession = Depends(get_session)) -> list[KeyResponse]:
    repo = APIKeyRepository(session)
    keys = await repo.list_all()
    return [
        KeyResponse(id=k.id, name=k.name, revoked=k.revoked, rate_limit_per_min=k.rate_limit_per_min)
        for k in keys
    ]


@router.delete("/keys/{key_id}")
async def revoke_key(key_id: str, session: AsyncSession = Depends(get_session)):
    repo = APIKeyRepository(session)
    await repo.revoke(key_id)
    return {"revoked": key_id}
