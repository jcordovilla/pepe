"""
Agentic RAG Framework for Discord Bot

This package contains the core agentic capabilities for the Discord RAG bot,
including multi-agent orchestration, memory systems, and reasoning components.
"""

__version__ = "1.0.0"
__author__ = "Jose Cordovilla"

from .agents import AgentOrchestrator
from .memory import ConversationMemory
from .reasoning import TaskPlanner

__all__ = [
    "AgentOrchestrator",
    "ConversationMemory",
    "TaskPlanner"
]
