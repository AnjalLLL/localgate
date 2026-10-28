"""Encrypted, admin-revealable copy of new API keys.

Revision ID: 0006
Revises: 0005
Create Date: 2026-10-28
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0006"
down_revision: str | None = "0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _columns() -> set[str]:
    return {c["name"] for c in sa.inspect(op.get_bind()).get_columns("api_keys")}


def upgrade() -> None:
    if "key_ciphertext" not in _columns():
        with op.batch_alter_table("api_keys") as batch:
            batch.add_column(sa.Column("key_ciphertext", sa.Text(), nullable=True))


def downgrade() -> None:
    if "key_ciphertext" in _columns():
        with op.batch_alter_table("api_keys") as batch:
            batch.drop_column("key_ciphertext")
