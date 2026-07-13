"""Add key prefixes, request accounting, chunk kinds, and conversation summaries.

**Every step here is idempotent, and that is not defensive padding — it is required.**

This migration is the first thing a pre-migrations database (0.1-0.2, schema built by
``create_all``, no ``alembic_version``) runs when it is adopted. Those databases are not
guaranteed to match revision 0001 exactly: a partially-applied run, a hand-edited column,
or a create_all from an intermediate commit all leave a schema that is *nearly* 0001 but
not quite. A migration that blindly ``CREATE TABLE``s or ``ADD COLUMN``s against one of
those aborts the transaction, and the gateway fails to start.

So each step asks the database what already exists and skips what is already there. The
migration converges on the target schema from any of those starting points rather than
demanding one exact starting point.

Every column added is nullable and then backfilled, so this applies to a table with
existing rows without a rewrite. ``key_prefix`` is backfilled to '' rather than to a real
prefix because the raw key is gone by design — only its hash was ever stored. Those keys
keep working; they just cannot be identified by sight in a listing.

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


def _inspector() -> sa.Inspector:
    return sa.inspect(op.get_bind())


def _has_table(table: str) -> bool:
    return table in _inspector().get_table_names()


def _has_column(table: str, column: str) -> bool:
    if not _has_table(table):
        return False
    return column in {c["name"] for c in _inspector().get_columns(table)}


def _has_index(table: str, index: str) -> bool:
    if not _has_table(table):
        return False
    return index in {i["name"] for i in _inspector().get_indexes(table)}


def _add_column(table: str, column: sa.Column) -> None:
    if _has_column(table, column.name):
        return
    # batch_alter_table emits the copy-and-swap dance on SQLite (which cannot ALTER a
    # column) and a plain ALTER on Postgres, so one script serves both.
    with op.batch_alter_table(table) as batch:
        batch.add_column(column)


def _create_index(name: str, table: str, columns: list[str]) -> None:
    if not _has_index(table, name):
        op.create_index(name, table, columns)


def _drop_index(name: str, table: str) -> None:
    if _has_index(table, name):
        op.drop_index(name, table_name=table)


def upgrade() -> None:
    _add_column("api_keys", sa.Column("key_prefix", sa.String(), nullable=True))
    _create_index("ix_api_keys_revoked", "api_keys", ["revoked"])

    _add_column("usage_records", sa.Column("latency_ms", sa.Integer(), nullable=True))
    _add_column("usage_records", sa.Column("cached", sa.Boolean(), nullable=True))
    _create_index("ix_usage_records_model", "usage_records", ["model"])
    _create_index("ix_usage_key_created", "usage_records", ["api_key_id", "created_at"])

    _add_column("memory_chunks", sa.Column("kind", sa.String(), nullable=True))
    _create_index("ix_memory_chunks_kind", "memory_chunks", ["kind"])

    _create_index(
        "ix_message_session_created", "conversation_messages", ["session_id", "created_at"]
    )

    if not _has_table("conversation_summaries"):
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
    _create_index("ix_conversation_summaries_session_id", "conversation_summaries", ["session_id"])
    _create_index("ix_conversation_summaries_api_key_id", "conversation_summaries", ["api_key_id"])

    # Backfill, so no existing row is left holding a NULL that the application layer would
    # then have to special-case forever. (It once did: a NULL key_prefix turned
    # GET /admin/keys into a 500 for everyone upgrading.)
    op.execute("UPDATE memory_chunks SET kind = 'turn' WHERE kind IS NULL")
    op.execute("UPDATE api_keys SET key_prefix = '' WHERE key_prefix IS NULL")
    op.execute("UPDATE usage_records SET latency_ms = 0 WHERE latency_ms IS NULL")
    op.execute("UPDATE usage_records SET cached = FALSE WHERE cached IS NULL")


def downgrade() -> None:
    if _has_table("conversation_summaries"):
        op.drop_table("conversation_summaries")

    _drop_index("ix_message_session_created", "conversation_messages")
    _drop_index("ix_memory_chunks_kind", "memory_chunks")
    _drop_index("ix_usage_key_created", "usage_records")
    _drop_index("ix_usage_records_model", "usage_records")
    _drop_index("ix_api_keys_revoked", "api_keys")

    with op.batch_alter_table("memory_chunks") as batch:
        batch.drop_column("kind")
    with op.batch_alter_table("usage_records") as batch:
        batch.drop_column("cached")
        batch.drop_column("latency_ms")
    with op.batch_alter_table("api_keys") as batch:
        batch.drop_column("key_prefix")
