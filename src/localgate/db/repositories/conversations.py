"""Data access for stored conversation turns and rolling summaries."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from localgate.db.models import ConversationMessage, ConversationSummary


class ConversationRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def add_message(
        self, session_id: str, api_key_id: str, role: str, content: str
    ) -> ConversationMessage:
        message = ConversationMessage(
            session_id=session_id, api_key_id=api_key_id, role=role, content=content
        )
        self.session.add(message)
        await self.session.commit()
        await self.session.refresh(message)
        return message

    async def recent(self, session_id: str, limit: int = 20) -> list[ConversationMessage]:
        """The newest ``limit`` messages, returned oldest-first for display."""
        stmt = (
            select(ConversationMessage)
            .where(ConversationMessage.session_id == session_id)
            .order_by(desc(ConversationMessage.created_at))
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        return list(reversed(result.scalars().all()))

    async def messages_after(
        self, session_id: str, after: datetime | None
    ) -> list[ConversationMessage]:
        """Every message newer than ``after``, oldest-first. ``None`` means all of them."""
        stmt = select(ConversationMessage).where(ConversationMessage.session_id == session_id)
        if after is not None:
            stmt = stmt.where(ConversationMessage.created_at > after)
        stmt = stmt.order_by(ConversationMessage.created_at)
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def count(self, session_id: str) -> int:
        stmt = select(func.count(ConversationMessage.id)).where(
            ConversationMessage.session_id == session_id
        )
        return int((await self.session.execute(stmt)).scalar_one())

    async def list_sessions(self, api_key_id: str | None = None, limit: int = 50) -> list[dict]:
        """Session index for the dashboard: id, message count, last activity."""
        stmt = (
            select(
                ConversationMessage.session_id,
                func.count(ConversationMessage.id).label("message_count"),
                func.max(ConversationMessage.created_at).label("last_message_at"),
            )
            .group_by(ConversationMessage.session_id)
            .order_by(desc("last_message_at"))
            .limit(limit)
        )
        if api_key_id is not None:
            stmt = stmt.where(ConversationMessage.api_key_id == api_key_id)

        result = await self.session.execute(stmt)
        return [
            {
                "session_id": row.session_id,
                "message_count": row.message_count,
                "last_message_at": row.last_message_at.isoformat() if row.last_message_at else None,
            }
            for row in result
        ]


class SummaryRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def latest(self, session_id: str) -> ConversationSummary | None:
        stmt = (
            select(ConversationSummary)
            .where(ConversationSummary.session_id == session_id)
            .order_by(desc(ConversationSummary.covers_until))
            .limit(1)
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def add(
        self,
        session_id: str,
        api_key_id: str,
        content: str,
        covers_until: datetime,
        message_count: int,
    ) -> ConversationSummary:
        summary = ConversationSummary(
            session_id=session_id,
            api_key_id=api_key_id,
            content=content,
            covers_until=covers_until,
            message_count=message_count,
        )
        self.session.add(summary)
        await self.session.commit()
        await self.session.refresh(summary)
        return summary
