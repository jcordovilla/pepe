from .helpers import build_jump_url, validate_ids
from .logger import setup_logging

# Discord ID patterns
DISCORD_ID_PATTERN = r'^\d{17,20}$'

def validate_discord_id(id_value, id_type: str = "ID") -> bool:
    """Validate a Discord ID format."""
    if id_value is None:
        return False
    id_str = str(id_value).strip()
    import re
    return bool(re.match(DISCORD_ID_PATTERN, id_str))

def validate_channel_id(channel_id) -> bool:
    """Validate a Discord channel ID."""
    return validate_discord_id(channel_id, "Channel")

def validate_guild_id(guild_id) -> bool:
    """Validate a Discord guild ID."""
    return validate_discord_id(guild_id, "Guild")

def validate_channel_name(channel_name) -> bool:
    """Validate a Discord channel name."""
    if not channel_name:
        return False
    name = channel_name.lstrip('#')
    if not (2 <= len(name) <= 100):
        return False
    import re
    return bool(re.search(r'[\w-]', name)) and not re.search(r'[@/\\]', name)

__all__ = [
    'build_jump_url', 'validate_ids', 'setup_logging',
    'validate_discord_id', 'validate_channel_id', 'validate_guild_id', 'validate_channel_name'
]
