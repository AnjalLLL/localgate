"""SQLAlchemy ORM models: APIKey, UsageRecord, ConversationMessage, MemoryChunk."""
import uuid
from datetime import datetime, timezone

from sqlalchemy import JSON, Boolean, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class APIKey(Base):
    __tablename__ = "api_keys"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String, nullable=False)
    key_hash: Mapped[str] = mapped_column(String, unique=True, index=True, nullable=False)
    rate_limit_per_min: Mapped[int] = mapped_column(Integer, default=60)
    revoked: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class UsageRecord(Base):
    __tablename__ = "usage_records"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    api_key_id: Mapped[str] = mapped_column(ForeignKey("api_keys.id"), index=True)
    model: Mapped[str] = mapped_column(String)
    prompt_tokens: Mapped[int] = mapped_column(Integer, default=0)
    completion_tokens: Mapped[int] = mapped_column(Integer, default=0)
    total_tokens: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class ConversationMessage(Base):
    """One turn of chat history, used both as the audit log and as raw material for memory chunks."""

    __tablename__ = "conversation_messages"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    session_id: Mapped[str] = mapped_column(String, index=True)
    api_key_id: Mapped[str] = mapped_column(ForeignKey("api_keys.id"), index=True)
    role: Mapped[str] = mapped_column(String)  # "user" | "assistant"
    content: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class MemoryChunk(Base):
    """A chunk of past conversation plus its embedding, used for retrieval-based context extension.

    Embeddings are stored as JSON float arrays for portability (works on SQLite with zero
    extensions). On Postgres, swap this column for a pgvector `Vector` type and use native
    ANN search instead of the Python-side cosine similarity in memory/retriever.py — that's
    the documented upgrade path once you outgrow SQLite for a given deployment.
    """

    __tablename__ = "memory_chunks"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    session_id: Mapped[str] = mapped_column(String, index=True)
    api_key_id: Mapped[str] = mapped_column(ForeignKey("api_keys.id"), index=True)
    content: Mapped[str] = mapped_column(Text)
    embedding: Mapped[list] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
