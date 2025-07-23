"""
Interfaces for Agentic RAG Components

Defines interfaces for integrating the agentic framework with external systems.
"""

from .agent_api import AgentAPI
from .discord_interface import DiscordInterface
__all__ = ["AgentAPI", "DiscordInterface"]
