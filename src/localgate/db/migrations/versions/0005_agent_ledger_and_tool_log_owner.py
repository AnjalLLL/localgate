"""Session ledgers for agent context compaction; owner/run/model on tool logs.

Revision ID: 0005
Revises: 0004
Create Date: 2026-10-28
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0005"
down_revision: str | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

TOOL_LOG_COLUMNS = ("api_key_id", "run_id", "model")


def _inspector() -> sa.Inspector:
    return sa.inspect(op.get_bind())


def _has_table(table: str) -> bool:
    return table in _inspector().get_table_names()


def _columns(table: str) -> set[str]:
    return {column["name"] for column in _inspector().get_columns(table)}


def _has_index(table: str, index: str) -> bool:
    if not _has_table(table):
        return False
    return index in {item["name"] for item in _inspector().get_indexes(table)}


def upgrade() -> None:
    existing = _columns("tool_call_logs")
    # batch_alter_table so SQLite (no ALTER ... ADD with constraints) and
    # Postgres take the same migration.
    with op.batch_alter_table("tool_call_logs") as batch:
        for name in TOOL_LOG_COLUMNS:
            if name not in existing:
                batch.add_column(sa.Column(name, sa.String(), nullable=True))
    if not _has_index("tool_call_logs", "ix_toolcall_owner_created"):
        op.create_index("ix_toolcall_owner_created", "tool_call_logs", ["api_key_id", "created_at"])

    if not _has_table("session_ledgers"):
        op.create_table(
            "session_ledgers",
            sa.Column("id", sa.String(), nullable=False),
            sa.Column("api_key_id", sa.String(), nullable=False),
            sa.Column("session_id", sa.String(), nullable=False),
            sa.Column("data", sa.JSON(), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
            sa.PrimaryKeyConstraint("id"),
        )
    for name, columns, unique in (
        ("ix_session_ledgers_api_key_id", ["api_key_id"], False),
        ("ix_session_ledgers_session_id", ["session_id"], False),
        ("ux_ledger_owner_session", ["api_key_id", "session_id"], True),
    ):
        if not _has_index("session_ledgers", name):
            op.create_index(name, "session_ledgers", columns, unique=unique)


def downgrade() -> None:
    if _has_table("session_ledgers"):
        op.drop_table("session_ledgers")
    if _has_index("tool_call_logs", "ix_toolcall_owner_created"):
        op.drop_index("ix_toolcall_owner_created", table_name="tool_call_logs")
    existing = _columns("tool_call_logs")
    with op.batch_alter_table("tool_call_logs") as batch:
        for name in TOOL_LOG_COLUMNS:
            if name in existing:
                batch.drop_column(name)
