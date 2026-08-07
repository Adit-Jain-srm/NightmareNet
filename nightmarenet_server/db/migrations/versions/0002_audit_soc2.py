"""Add SOC 2 audit columns, indexes, and append-only enforcement.

Revision ID: 0002_audit_soc2
Revises: 0001_initial
Create Date: 2026-08-07
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0002_audit_soc2"
down_revision: Union[str, None] = "0001_initial"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("audit_logs") as batch:
        batch.alter_column("org_id", existing_type=sa.String(length=36), nullable=True)
        batch.alter_column(
            "resource_id",
            existing_type=sa.String(length=36),
            type_=sa.String(length=128),
            existing_nullable=False,
        )
        batch.add_column(sa.Column("actor_role", sa.String(length=32), nullable=True))
        batch.add_column(sa.Column("ip_address", sa.String(length=64), nullable=True))
        batch.add_column(sa.Column("request_id", sa.String(length=64), nullable=True))

    op.create_index("ix_audit_logs_request_id", "audit_logs", ["request_id"], unique=False)
    op.create_index(
        "ix_audit_logs_user_id_timestamp",
        "audit_logs",
        ["user_id", "timestamp"],
        unique=False,
    )
    op.create_index(
        "ix_audit_logs_resource_type_resource_id_timestamp",
        "audit_logs",
        ["resource_type", "resource_id", "timestamp"],
        unique=False,
    )

    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.execute(
            """
            CREATE OR REPLACE FUNCTION audit_logs_append_only() RETURNS trigger AS $$
            BEGIN
              RAISE EXCEPTION 'audit_logs is append-only';
            END;
            $$ LANGUAGE plpgsql;
            """
        )
        op.execute(
            """
            CREATE TRIGGER trg_audit_logs_no_update
              BEFORE UPDATE OR DELETE ON audit_logs
              FOR EACH ROW EXECUTE FUNCTION audit_logs_append_only();
            """
        )


def downgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.execute("DROP TRIGGER IF EXISTS trg_audit_logs_no_update ON audit_logs")
        op.execute("DROP FUNCTION IF EXISTS audit_logs_append_only()")

    op.drop_index("ix_audit_logs_resource_type_resource_id_timestamp", table_name="audit_logs")
    op.drop_index("ix_audit_logs_user_id_timestamp", table_name="audit_logs")
    op.drop_index("ix_audit_logs_request_id", table_name="audit_logs")

    with op.batch_alter_table("audit_logs") as batch:
        batch.drop_column("request_id")
        batch.drop_column("ip_address")
        batch.drop_column("actor_role")
        batch.alter_column(
            "resource_id",
            existing_type=sa.String(length=128),
            type_=sa.String(length=36),
            existing_nullable=False,
        )
        batch.alter_column("org_id", existing_type=sa.String(length=36), nullable=False)
