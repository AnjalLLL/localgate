"""Repository for tool call logging."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from localgate.db.models import ToolCallLog


class ToolCallRepository:
    """Manages tool call logging for debugging and heuristics."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(
        self,
        session_id: str,
        tool_name: str,
        arguments: dict[str, Any],
        success: bool,
        duration_ms: int,
        error: str | None = None,
        timed_out: bool = False,
    ) -> ToolCallLog:
        """Create a new tool call log entry."""
        log = ToolCallLog(
            session_id=session_id,
            tool_name=tool_name,
            arguments=arguments,
            success=success,
            duration_ms=duration_ms,
            error=error,
            timed_out=timed_out,
        )
        self.session.add(log)
        await self.session.commit()
        await self.session.refresh(log)
        return log

    async def get_recent_by_session(self, session_id: str, limit: int = 50) -> list[ToolCallLog]:
        """Get recent tool calls for a session, most recent first."""
        result = await self.session.execute(
            select(ToolCallLog)
            .where(ToolCallLog.session_id == session_id)
            .order_by(ToolCallLog.created_at.desc())
            .limit(limit)
        )
        return list(result.scalars().all())

    async def get_failure_count(self, session_id: str, tool_name: str, since: datetime) -> int:
        """Count recent failures for a specific tool in this session."""
        result = await self.session.execute(
            select(ToolCallLog).where(
                ToolCallLog.session_id == session_id,
                ToolCallLog.tool_name == tool_name,
                ToolCallLog.success == False,  # noqa: E712
                ToolCallLog.created_at >= since,
            )
        )
        return len(list(result.scalars().all()))

    async def get_timeout_count(self, session_id: str, tool_name: str, since: datetime) -> int:
        """Count recent timeouts for a specific tool in this session."""
        result = await self.session.execute(
            select(ToolCallLog).where(
                ToolCallLog.session_id == session_id,
                ToolCallLog.tool_name == tool_name,
                ToolCallLog.timed_out == True,  # noqa: E712
                ToolCallLog.created_at >= since,
            )
        )
        return len(list(result.scalars().all()))
