"""Data access layer for stored conversation turns."""
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from localgate.db.models import ConversationMessage


class ConversationRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def add_message(self, session_id: str, api_key_id: str, role: str, content: str) -> None:
        self.session.add(
            ConversationMessage(session_id=session_id, api_key_id=api_key_id, role=role, content=content)
        )
        await self.session.commit()

    async def recent(self, session_id: str, limit: int = 20) -> list[ConversationMessage]:
        stmt = (
            select(ConversationMessage)
            .where(ConversationMessage.session_id == session_id)
            .order_by(ConversationMessage.created_at.desc())
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        return list(reversed(result.scalars().all()))
