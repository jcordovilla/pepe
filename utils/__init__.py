from .helpers import build_jump_url, validate_ids
from .logger  import setup_logging

"""
Utility functions for the Discord bot.
"""
from typing import Optional, Union
import re

# Discord ID patterns
DISCORD_ID_PATTERN = r'^\d{17,20}$'  # Discord IDs are 17-20 digits
CHANNEL_ID_PATTERN = DISCORD_ID_PATTERN
GUILD_ID_PATTERN = DISCORD_ID_PATTERN

def validate_discord_id(id_value: Union[str, int], id_type: str = "ID") -> bool:
    """
    Validate a Discord ID format.
    
    Args:
        id_value: The ID to validate (can be string or int)
        id_type: The type of ID for error messages (e.g., "Channel", "Guild")
        
    Returns:
        bool: True if valid, False otherwise
    """
    if id_value is None:
        return False
        
    # Convert to string for consistent validation
    id_str = str(id_value).strip()
    
    # Check if it matches Discord's ID pattern
    if not re.match(DISCORD_ID_PATTERN, id_str):
        return False
        
    # Additional validation for specific ID types
    if id_type.lower() == "channel":
        return re.match(CHANNEL_ID_PATTERN, id_str) is not None
    elif id_type.lower() == "guild":
        return re.match(GUILD_ID_PATTERN, id_str) is not None
        
    return True

def validate_channel_id(channel_id: Optional[Union[str, int]]) -> bool:
    """
    Validate a Discord channel ID.
    
    Args:
        channel_id: The channel ID to validate
        
    Returns:
        bool: True if valid, False otherwise
    """
    return validate_discord_id(channel_id, "Channel")

def validate_guild_id(guild_id: Optional[Union[str, int]]) -> bool:
    """
    Validate a Discord guild ID.
    
    Args:
        guild_id: The guild ID to validate
        
    Returns:
        bool: True if valid, False otherwise
    """
    return validate_discord_id(guild_id, "Guild")

def validate_channel_name(channel_name: Optional[str]) -> bool:
    """
    Validate a Discord channel name, allowing emoji and Unicode at the start.
    
    Args:
        channel_name: The channel name to validate
        
    Returns:
        bool: True if valid, False otherwise
    """
    if not channel_name:
        return False
        
    # Remove leading # if present
    name = channel_name.lstrip('#')
    
    # Channel names must be 2-100 characters
    if not (2 <= len(name) <= 100):
        return False
        
    # Allow emoji/unicode at the start, then require at least one alphanumeric/underscore/hyphen after
    # Discord allows a wide range of Unicode, so we check for at least one valid trailing part
    # Accepts: emoji + dash + text, or just text, etc.
    # Example: '🏘general-chat', '📚ai-philosophy-ethics', 'general', 'ai'
    # Regex: start with any unicode, must contain at least one [\w-] after
    if not re.search(r'[\w-]', name):
        return False
        
    # Disallow forbidden characters (e.g., @, /, etc.)
    if re.search(r'[@/\\]', name):
        return False
        
    return True

# Export all validation functions
__all__ = [
    'validate_discord_id',
    'validate_channel_id',
    'validate_guild_id',
    'validate_channel_name'
]
