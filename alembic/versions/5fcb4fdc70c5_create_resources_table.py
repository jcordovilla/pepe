"""create resources table

Revision ID: 5fcb4fdc70c5
Revises: 
Create Date: 2025-05-12 16:47:48.643798

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '5fcb4fdc70c5'
down_revision = None
branch_labels = None
depends_on = None

def upgrade():
    op.create_table(
        'resources',
        sa.Column('id', sa.Integer, primary_key=True, autoincrement=True),
        sa.Column('message_id', sa.String, unique=True, nullable=False),
        sa.Column('guild_id', sa.String, nullable=True),
        sa.Column('channel_id', sa.String, nullable=True),
        sa.Column('url', sa.Text, nullable=False),
        sa.Column('type', sa.String, nullable=False),
        sa.Column('tag', sa.String, nullable=False),
        sa.Column('author', sa.JSON, nullable=True),
        sa.Column('timestamp', sa.DateTime, nullable=False),
        sa.Column('context_snippet', sa.Text, nullable=True),
        sa.Column('metadata', sa.JSON, nullable=True),
    )

def downgrade():
    op.drop_table('resources')
