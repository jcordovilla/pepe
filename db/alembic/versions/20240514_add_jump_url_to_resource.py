"""add jump_url field to resources table

Revision ID: 20240514_add_jump_url_to_resource
Revises: 20240514_add_name_description
Create Date: 2025-05-14
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '20240514_add_jump_url_to_resource'
down_revision = '20240514_add_name_description'
branch_labels = None
depends_on = None

def upgrade():
    op.add_column('resources', sa.Column('jump_url', sa.String, nullable=True))

def downgrade():
    op.drop_column('resources', 'jump_url')
