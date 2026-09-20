"""Scope conversation and memory lookups by API-key owner.

Revision ID: 0004
Revises: 0003
Create Date: 2026-09-20
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _has_index(table: str, index: str) -> bool:
    return index in {item["name"] for item in sa.inspect(op.get_bind()).get_indexes(table)}


INDEXES = (
    (
        "conversation_messages",
        "ix_message_owner_session_created",
        ["api_key_id", "session_id", "created_at"],
    ),
    (
        "conversation_summaries",
        "ix_summary_owner_session_created",
        ["api_key_id", "session_id", "created_at"],
    ),
    (
        "memory_chunks",
        "ix_memory_owner_session_created",
        ["api_key_id", "session_id", "created_at"],
    ),
)


def upgrade() -> None:
    for table, name, columns in INDEXES:
        if not _has_index(table, name):
            op.create_index(name, table, columns)


def downgrade() -> None:
    for table, name, _columns in reversed(INDEXES):
        if _has_index(table, name):
            op.drop_index(name, table_name=table)
