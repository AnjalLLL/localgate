"""Persistence for the agent's session ledger (see `agent/context.py`)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from localgate.db.models import SessionLedgerRecord


class LedgerRepository:
    """One ledger per owner + session. Both keys are always part of the lookup,
    the same tenant rule every conversation and memory query follows."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get(self, session_id: str, api_key_id: str) -> dict[str, Any] | None:
        record = await self._record(session_id, api_key_id)
        return dict(record.data) if record is not None else None

    async def save(self, session_id: str, api_key_id: str, data: dict[str, Any]) -> None:
        record = await self._record(session_id, api_key_id)
        if record is None:
            record = SessionLedgerRecord(session_id=session_id, api_key_id=api_key_id, data=data)
            self.session.add(record)
        else:
            record.data = data
            record.updated_at = datetime.now(timezone.utc)
        await self.session.commit()

    async def _record(self, session_id: str, api_key_id: str) -> SessionLedgerRecord | None:
        result = await self.session.execute(
            select(SessionLedgerRecord).where(
                SessionLedgerRecord.session_id == session_id,
                SessionLedgerRecord.api_key_id == api_key_id,
            )
        )
        return result.scalar_one_or_none()
