"""add name and description fields to resources table

Revision ID: 20240514_add_name_description
Revises: 5fcb4fdc70c5
Create Date: 2025-05-14
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '20240514_add_name_description'
down_revision = '5fcb4fdc70c5'
branch_labels = None
depends_on = None

def upgrade():
    op.add_column('resources', sa.Column('name', sa.String, nullable=True))
    op.add_column('resources', sa.Column('description', sa.Text, nullable=True))

def downgrade():
    op.drop_column('resources', 'name')
    op.drop_column('resources', 'description')
