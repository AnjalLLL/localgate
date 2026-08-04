"""Add tool_call_logs table for debugging and heuristics.

Revision ID: 0003
Revises: 0002
Create Date: 2026-08-04
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy import JSON, Boolean, DateTime, Integer, String, Text

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _inspector() -> sa.Inspector:
    return sa.inspect(op.get_bind())


def _has_table(table: str) -> bool:
    return table in _inspector().get_table_names()


def _has_index(table: str, index: str) -> bool:
    if not _has_table(table):
        return False
    return index in {idx["name"] for idx in _inspector().get_indexes(table)}


def upgrade() -> None:
    """Create tool_call_logs table if it doesn't exist."""
    if not _has_table("tool_call_logs"):
        op.create_table(
            "tool_call_logs",
            sa.Column("id", String(), nullable=False),
            sa.Column("session_id", String(), nullable=False),
            sa.Column("tool_name", String(), nullable=False),
            sa.Column("arguments", JSON(), nullable=False),
            sa.Column("success", Boolean(), nullable=False),
            sa.Column("duration_ms", Integer(), nullable=False),
            sa.Column("error", Text(), nullable=True),
            sa.Column("timed_out", Boolean(), nullable=False),
            sa.Column("created_at", DateTime(timezone=True), nullable=False),
            sa.PrimaryKeyConstraint("id"),
        )

    # Add indexes if they don't exist
    if not _has_index("tool_call_logs", "ix_tool_call_logs_session_id"):
        op.create_index(
            "ix_tool_call_logs_session_id",
            "tool_call_logs",
            ["session_id"],
        )

    if not _has_index("tool_call_logs", "ix_tool_call_logs_tool_name"):
        op.create_index(
            "ix_tool_call_logs_tool_name",
            "tool_call_logs",
            ["tool_name"],
        )

    if not _has_index("tool_call_logs", "ix_toolcall_session_created"):
        op.create_index(
            "ix_toolcall_session_created",
            "tool_call_logs",
            ["session_id", "created_at"],
        )


def downgrade() -> None:
    """Drop tool_call_logs table and its indexes."""
    if _has_index("tool_call_logs", "ix_toolcall_session_created"):
        op.drop_index("ix_toolcall_session_created", table_name="tool_call_logs")

    if _has_index("tool_call_logs", "ix_tool_call_logs_tool_name"):
        op.drop_index("ix_tool_call_logs_tool_name", table_name="tool_call_logs")

    if _has_index("tool_call_logs", "ix_tool_call_logs_session_id"):
        op.drop_index("ix_tool_call_logs_session_id", table_name="tool_call_logs")

    if _has_table("tool_call_logs"):
        op.drop_table("tool_call_logs")
