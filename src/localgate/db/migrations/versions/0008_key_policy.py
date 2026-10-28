"""Per-key policy: allowed models, max_tokens cap, token quota, expiry,
concurrency and a "don't store conversations" switch.

Revision ID: 0008
Revises: 0007
Create Date: 2026-10-29
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0008"
down_revision: str | None = "0007"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _columns() -> list[sa.Column[object]]:
    """Fresh Column objects each call — a Column can belong to one table only."""
    return [
        sa.Column("allowed_models", sa.JSON(), nullable=True),
        sa.Column("max_tokens", sa.Integer(), nullable=True),
        sa.Column("token_quota", sa.Integer(), nullable=True),
        sa.Column("quota_period", sa.String(), nullable=False, server_default="month"),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("max_concurrent", sa.Integer(), nullable=True),
        sa.Column("store_conversations", sa.Boolean(), nullable=False, server_default=sa.true()),
    ]


def _existing() -> set[str]:
    return {c["name"] for c in sa.inspect(op.get_bind()).get_columns("api_keys")}


def upgrade() -> None:
    existing = _existing()
    missing = [column for column in _columns() if column.name not in existing]
    if missing:
        with op.batch_alter_table("api_keys") as batch:
            for column in missing:
                batch.add_column(column)


def downgrade() -> None:
    existing = _existing()
    present = [column.name for column in _columns() if column.name in existing]
    if present:
        with op.batch_alter_table("api_keys") as batch:
            for name in present:
                batch.drop_column(name)
