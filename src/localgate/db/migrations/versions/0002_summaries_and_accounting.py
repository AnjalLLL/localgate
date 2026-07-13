"""Add key prefixes, request accounting, chunk kinds, and conversation summaries.

Every column added here is nullable or has a server-side default, so this applies to
a database with existing rows without needing a backfill. ``key_prefix`` is left
empty on pre-existing keys because it cannot be recovered — the raw key is gone by
definition, which is the point of storing only the hash. Those keys simply show a
blank prefix in listings; new ones show theirs.

Revision ID: 0002
Revises: 0001
Create Date: 2026-07-14
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # batch_alter_table emits the copy-and-swap dance on SQLite (which cannot ALTER
    # a column) and a plain ALTER on Postgres, so one script serves both.
    with op.batch_alter_table("api_keys") as batch:
        batch.add_column(sa.Column("key_prefix", sa.String(), nullable=True))
    op.create_index("ix_api_keys_revoked", "api_keys", ["revoked"])

    with op.batch_alter_table("usage_records") as batch:
        batch.add_column(sa.Column("latency_ms", sa.Integer(), nullable=True))
        batch.add_column(sa.Column("cached", sa.Boolean(), nullable=True))
    op.create_index("ix_usage_records_model", "usage_records", ["model"])
    op.create_index("ix_usage_key_created", "usage_records", ["api_key_id", "created_at"])

    with op.batch_alter_table("memory_chunks") as batch:
        batch.add_column(sa.Column("kind", sa.String(), nullable=True))
    op.create_index("ix_memory_chunks_kind", "memory_chunks", ["kind"])

    op.create_index(
        "ix_message_session_created", "conversation_messages", ["session_id", "created_at"]
    )

    op.create_table(
        "conversation_summaries",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("session_id", sa.String(), nullable=True),
        sa.Column("api_key_id", sa.String(), nullable=True),
        sa.Column("content", sa.Text(), nullable=True),
        sa.Column("covers_until", sa.DateTime(timezone=True), nullable=True),
        sa.Column("message_count", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["api_key_id"], ["api_keys.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_conversation_summaries_session_id", "conversation_summaries", ["session_id"]
    )
    op.create_index(
        "ix_conversation_summaries_api_key_id", "conversation_summaries", ["api_key_id"]
    )

    # Backfill the columns added above, so no existing row is left holding a NULL that
    # the application layer would then have to special-case forever.
    #
    # `key_prefix` is set to '' rather than to a real prefix because the raw key is gone
    # by design — only its hash was ever stored. Those keys keep working; they just
    # cannot be identified by sight in a listing.
    op.execute("UPDATE memory_chunks SET kind = 'turn' WHERE kind IS NULL")
    op.execute("UPDATE api_keys SET key_prefix = '' WHERE key_prefix IS NULL")
    op.execute("UPDATE usage_records SET latency_ms = 0 WHERE latency_ms IS NULL")
    op.execute("UPDATE usage_records SET cached = FALSE WHERE cached IS NULL")


def downgrade() -> None:
    op.drop_table("conversation_summaries")
    op.drop_index("ix_message_session_created", table_name="conversation_messages")
    op.drop_index("ix_memory_chunks_kind", table_name="memory_chunks")
    with op.batch_alter_table("memory_chunks") as batch:
        batch.drop_column("kind")
    op.drop_index("ix_usage_key_created", table_name="usage_records")
    op.drop_index("ix_usage_records_model", table_name="usage_records")
    with op.batch_alter_table("usage_records") as batch:
        batch.drop_column("cached")
        batch.drop_column("latency_ms")
    op.drop_index("ix_api_keys_revoked", table_name="api_keys")
    with op.batch_alter_table("api_keys") as batch:
        batch.drop_column("key_prefix")
