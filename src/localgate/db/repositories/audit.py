"""The admin audit log (see `db.models.AuditEvent`): who did what, when, from where."""

from __future__ import annotations

import time
from typing import Any

from fastapi import Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from localgate.core.auth import ADMIN_COOKIE, admin_session_subject
from localgate.core.logging import get_logger
from localgate.db.models import AuditEvent

logger = get_logger(__name__)


class AuditRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def record(
        self,
        actor: str,
        action: str,
        target: str | None = None,
        details: dict[str, Any] | None = None,
        ip: str | None = None,
    ) -> None:
        self.session.add(
            AuditEvent(actor=actor, action=action, target=target, details=details or None, ip=ip)
        )
        await self.session.commit()

    async def recent(self, limit: int = 100, offset: int = 0) -> list[AuditEvent]:
        result = await self.session.execute(
            select(AuditEvent).order_by(AuditEvent.at.desc()).limit(limit).offset(offset)
        )
        return list(result.scalars())


def actor_of(request: Request) -> str:
    """Who made an admin request: the signed-in account's email, else "admin-key"."""
    cookie = request.cookies.get(ADMIN_COOKIE)
    if cookie and not request.headers.get("x-admin-key"):
        subject = admin_session_subject(cookie, request.app.state.settings.admin_key, time.time())
        if subject:
            return subject
    return "admin-key"


async def audit(
    request: Request,
    action: str,
    target: str | None = None,
    *,
    actor: str | None = None,
    **details: Any,
) -> None:
    """Record one admin action from a request. Auditing must never break the
    action it records, so a failure here is logged and swallowed."""
    factory: async_sessionmaker[AsyncSession] = request.app.state.session_factory
    try:
        async with factory() as session:
            await AuditRepository(session).record(
                actor or actor_of(request),
                action,
                target,
                {k: v for k, v in details.items() if v is not None},
                request.client.host if request.client else None,
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning("audit_write_failed", action=action, error=str(exc))
