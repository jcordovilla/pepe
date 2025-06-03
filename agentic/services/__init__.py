"""
Services module for agentic system.

This module provides various services for the agentic system including
channel resolution, user management, and other utilities.
"""

from .channel_resolver import ChannelResolver, ChannelInfo

__all__ = [
    'ChannelResolver',
    'ChannelInfo'
]
