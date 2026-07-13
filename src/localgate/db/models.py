"""SQLAlchemy ORM models.

The same schema is created on SQLite and Postgres. Where the two differ, the
portable option wins — this is a tool people run locally with zero setup, and a
schema that only works once you have a Postgres extension installed would defeat
that. See :class:`MemoryChunk` for the one place that tradeoff is load-bearing.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
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
    #: First few characters of the raw key ("lg_9f3a…"). The full key is never
    #: recoverable, so without this there is no way to tell which key on a listing
    #: corresponds to the one an operator is holding.
    key_prefix: Mapped[str] = mapped_column(String, default="")
    rate_limit_per_min: Mapped[int] = mapped_column(Integer, default=60)
    revoked: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class UsageRecord(Base):
    """One row per completed request — the source of truth for token accounting."""

    __tablename__ = "usage_records"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    api_key_id: Mapped[str] = mapped_column(ForeignKey("api_keys.id"), index=True)
    model: Mapped[str] = mapped_column(String, index=True)
    prompt_tokens: Mapped[int] = mapped_column(Integer, default=0)
    completion_tokens: Mapped[int] = mapped_column(Integer, default=0)
    total_tokens: Mapped[int] = mapped_column(Integer, default=0)
    latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    cached: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    # Usage is almost always queried as "this key, over this time range".
    __table_args__ = (Index("ix_usage_key_created", "api_key_id", "created_at"),)


class ConversationMessage(Base):
    """One turn of chat history: both the audit log and the raw material for memory."""

    __tablename__ = "conversation_messages"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    session_id: Mapped[str] = mapped_column(String, index=True)
    api_key_id: Mapped[str] = mapped_column(ForeignKey("api_keys.id"), index=True)
    role: Mapped[str] = mapped_column(String)  # "user" | "assistant"
    content: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    __table_args__ = (Index("ix_message_session_created", "session_id", "created_at"),)


class ConversationSummary(Base):
    """A rolling summary of the older part of a session.

    Retrieval alone degrades on long sessions: chunk-level similarity search can
    surface individual exchanges but loses the through-line ("we settled on
    Postgres two hours ago"). A summary preserves that narrative in a form that
    stays small, and ``covers_until`` records where it stops so the next summary
    picks up from there instead of re-summarizing the whole history.
    """

    __tablename__ = "conversation_summaries"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    session_id: Mapped[str] = mapped_column(String, index=True)
    api_key_id: Mapped[str] = mapped_column(ForeignKey("api_keys.id"), index=True)
    content: Mapped[str] = mapped_column(Text)
    #: Timestamp of the newest message this summary accounts for.
    covers_until: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    message_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class MemoryChunk(Base):
    """A chunk of past conversation plus its embedding — the unit of RAG retrieval.

    Embeddings are stored as JSON float arrays rather than a native vector type so
    the schema is identical on SQLite (no extensions, no setup) and Postgres. The
    cost is that similarity search is a Python-side scan (see
    ``db/repositories/embeddings.py``), which is fine into the low thousands of
    chunks per session and not beyond. Postgres users who outgrow that should move
    this column to ``pgvector`` and push the ranking into SQL; the repository
    method signature is designed not to change when they do. That tradeoff is
    recorded in docs/decisions/0002-json-embeddings-over-pgvector.md.
    """

    __tablename__ = "memory_chunks"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    session_id: Mapped[str] = mapped_column(String, index=True)
    api_key_id: Mapped[str] = mapped_column(ForeignKey("api_keys.id"), index=True)
    #: "turn" for a verbatim exchange, "summary" for condensed older history.
    kind: Mapped[str] = mapped_column(String, default="turn", index=True)
    content: Mapped[str] = mapped_column(Text)
    embedding: Mapped[list] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
