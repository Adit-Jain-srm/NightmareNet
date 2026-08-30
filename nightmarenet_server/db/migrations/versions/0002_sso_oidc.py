"""Add SSO / OIDC fields and per-org provider configuration.

Revision ID: 0002_sso_oidc
Revises: 0001_initial
Create Date: 2026-08-07
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0002_sso_oidc"
down_revision: Union[str, None] = "0001_initial"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("users") as batch:
        batch.add_column(sa.Column("sso_provider", sa.String(length=64), nullable=True))
        batch.add_column(sa.Column("external_id", sa.String(length=255), nullable=True))

    op.create_index("ix_users_sso_provider", "users", ["sso_provider"], unique=False)
    op.create_index("ix_users_external_id", "users", ["external_id"], unique=False)
    op.create_index(
        "ix_users_sso_provider_external_id",
        "users",
        ["sso_provider", "external_id"],
        unique=True,
    )

    op.create_table(
        "sso_providers",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("org_id", sa.String(length=36), sa.ForeignKey("orgs.id"), nullable=False),
        sa.Column("name", sa.String(length=128), nullable=False, server_default="default"),
        sa.Column("issuer", sa.String(length=512), nullable=False),
        sa.Column("metadata_url", sa.String(length=512), nullable=True),
        sa.Column("client_id", sa.String(length=255), nullable=False),
        sa.Column("client_secret", sa.String(length=255), nullable=True),
        sa.Column("role_claim", sa.String(length=64), nullable=False, server_default="groups"),
        sa.Column("role_mapping_json", sa.Text(), nullable=False, server_default="{}"),
        sa.Column("enabled", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    op.create_index("ix_sso_providers_org_id", "sso_providers", ["org_id"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_sso_providers_org_id", table_name="sso_providers")
    op.drop_table("sso_providers")
    op.drop_index("ix_users_sso_provider_external_id", table_name="users")
    op.drop_index("ix_users_external_id", table_name="users")
    op.drop_index("ix_users_sso_provider", table_name="users")
    with op.batch_alter_table("users") as batch:
        batch.drop_column("external_id")
        batch.drop_column("sso_provider")
