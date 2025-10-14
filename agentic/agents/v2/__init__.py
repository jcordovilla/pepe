"""
V2 Agents Package

Lean, focused agents for the Discord bot system.
"""

from .router_agent import RouterAgent
from .qa_agent import QAAgent
from .stats_agent import StatsAgent
from .digest_agent import DigestAgent
from .trend_agent import TrendAgent
from .structure_agent import StructureAgent
from .selfcheck_agent import SelfCheckAgent

def register_agents():
    """
    Register all v2 agents and return the registry.
    
    Returns:
        Dict mapping agent role names to agent classes
    """
    return {
        "router": RouterAgent,
        "qa": QAAgent,
        "stats": StatsAgent,
        "digest": DigestAgent,
        "trend": TrendAgent,
        "structure": StructureAgent,
        "selfcheck": SelfCheckAgent,
    }

__all__ = [
    "RouterAgent",
    "QAAgent", 
    "StatsAgent",
    "DigestAgent",
    "TrendAgent",
    "StructureAgent",
    "SelfCheckAgent",
    "register_agents"
] 