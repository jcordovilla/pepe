"""add_comprehensive_discord_message_fields

Revision ID: 0e9209c88554
Revises: 20240514_add_jump_url_to_resource
Create Date: 2025-06-07 17:35:28.450695

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '0e9209c88554'
down_revision: Union[str, None] = '20240514_add_jump_url_to_resource'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Add new essential metadata fields
    op.add_column('messages', sa.Column('edited_at', sa.DateTime(), nullable=True))
    op.add_column('messages', sa.Column('type', sa.String(), nullable=True))
    op.add_column('messages', sa.Column('flags', sa.Integer(), nullable=False, server_default='0'))
    op.add_column('messages', sa.Column('tts', sa.Boolean(), nullable=False, server_default='0'))
    op.add_column('messages', sa.Column('pinned', sa.Boolean(), nullable=False, server_default='0'))
    
    # Add rich content fields
    op.add_column('messages', sa.Column('embeds', sa.JSON(), nullable=True))
    op.add_column('messages', sa.Column('attachments', sa.JSON(), nullable=True))
    op.add_column('messages', sa.Column('stickers', sa.JSON(), nullable=True))
    op.add_column('messages', sa.Column('components', sa.JSON(), nullable=True))
    
    # Add reply/thread context fields
    op.add_column('messages', sa.Column('reference', sa.JSON(), nullable=True))
    op.add_column('messages', sa.Column('thread', sa.JSON(), nullable=True))
    
    # Add advanced metadata fields
    op.add_column('messages', sa.Column('webhook_id', sa.String(), nullable=True))
    op.add_column('messages', sa.Column('application_id', sa.String(), nullable=True))
    op.add_column('messages', sa.Column('application', sa.JSON(), nullable=True))
    op.add_column('messages', sa.Column('activity', sa.JSON(), nullable=True))
    op.add_column('messages', sa.Column('poll', sa.JSON(), nullable=True))
    
    # Add raw mention arrays
    op.add_column('messages', sa.Column('raw_mentions', sa.JSON(), nullable=True))
    op.add_column('messages', sa.Column('raw_channel_mentions', sa.JSON(), nullable=True))
    op.add_column('messages', sa.Column('raw_role_mentions', sa.JSON(), nullable=True))
    
    # Add derived content fields
    op.add_column('messages', sa.Column('clean_content', sa.Text(), nullable=True))
    op.add_column('messages', sa.Column('system_content', sa.Text(), nullable=True))
    
    # Add additional mention data fields
    op.add_column('messages', sa.Column('channel_mentions', sa.JSON(), nullable=True))
    op.add_column('messages', sa.Column('role_mentions', sa.JSON(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    # Remove all added columns in reverse order
    op.drop_column('messages', 'role_mentions')
    op.drop_column('messages', 'channel_mentions')
    op.drop_column('messages', 'system_content')
    op.drop_column('messages', 'clean_content')
    op.drop_column('messages', 'raw_role_mentions')
    op.drop_column('messages', 'raw_channel_mentions')
    op.drop_column('messages', 'raw_mentions')
    op.drop_column('messages', 'poll')
    op.drop_column('messages', 'activity')
    op.drop_column('messages', 'application')
    op.drop_column('messages', 'application_id')
    op.drop_column('messages', 'webhook_id')
    op.drop_column('messages', 'thread')
    op.drop_column('messages', 'reference')
    op.drop_column('messages', 'components')
    op.drop_column('messages', 'stickers')
    op.drop_column('messages', 'attachments')
    op.drop_column('messages', 'embeds')
    op.drop_column('messages', 'pinned')
    op.drop_column('messages', 'tts')
    op.drop_column('messages', 'flags')
    op.drop_column('messages', 'type')
    op.drop_column('messages', 'edited_at')
