"""Per-key network access: a key can work on the owner's local network, over the
internet (the `serve --public` tunnel), or both.

Revision ID: 0010
Revises: 0009
Create Date: 2026-10-29
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0010"
down_revision: str | None = "0009"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _columns() -> set[str]:
    return {c["name"] for c in sa.inspect(op.get_bind()).get_columns("api_keys")}


def upgrade() -> None:
    if "network_access" not in _columns():
        with op.batch_alter_table("api_keys") as batch:
            batch.add_column(
                sa.Column("network_access", sa.String(), nullable=False, server_default="both")
            )


def downgrade() -> None:
    if "network_access" in _columns():
        with op.batch_alter_table("api_keys") as batch:
            batch.drop_column("network_access")
