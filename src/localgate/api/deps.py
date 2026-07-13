"""Shared FastAPI dependencies: DB session, API key auth, admin auth."""
from collections.abc import AsyncIterator

from fastapi import Depends, HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from localgate.db.models import APIKey
from localgate.db.repositories.keys import APIKeyRepository


async def get_session(request: Request) -> AsyncIterator[AsyncSession]:
    session_factory = request.app.state.session_factory
    async with session_factory() as session:
        yield session


def _extract_bearer_token(request: Request) -> str:
    header = request.headers.get("authorization", "")
    if not header.lower().startswith("bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or malformed Authorization header. Use: Bearer <api-key>",
        )
    return header.split(" ", 1)[1].strip()


async def require_api_key(
    request: Request, session: AsyncSession = Depends(get_session)
) -> APIKey:
    raw_key = _extract_bearer_token(request)
    repo = APIKeyRepository(session)
    key = await repo.get_by_raw_key(raw_key)
    if key is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or revoked API key")
    await repo.touch_last_used(key.id)
    return key


def require_admin(request: Request) -> None:
    settings = request.app.state.settings
    provided = request.headers.get("x-admin-key")
    if not provided or provided != settings.admin_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid admin key")
