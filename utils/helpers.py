# utils.py
"""
Utility functions for Discord AI agent toolset.
"""
from typing import Any


def build_jump_url(guild_id: int, channel_id: int, message_id: int) -> str:
    """
    Construct a Discord message jump URL given numeric IDs.

    Raises:
        ValueError: If any ID is not a positive integer.
    """
    validate_ids(guild_id=guild_id, channel_id=channel_id, message_id=message_id)
    return f"https://discord.com/channels/{guild_id}/{channel_id}/{message_id}"


def validate_ids(**kwargs: Any) -> None:
    """
    Ensure that each provided keyword argument is a positive integer.

    Raises:
        ValueError: If any value is not an int > 0.
    """
    for name, val in kwargs.items():
        if not isinstance(val, int) or val <= 0:
            raise ValueError(f"Invalid ID for {name}: {val}")
