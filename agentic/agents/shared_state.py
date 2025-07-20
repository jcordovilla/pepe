"""
Shared Agent State Management

Provides thread-safe shared state management for agent coordination.
Fixes the state isolation problem where each agent works on a copy.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class StateUpdateType(Enum):
    """Types of state updates for tracking"""
    QUERY_INTERPRETATION = "query_interpretation"
    SEARCH_RESULTS = "search_results"
    ANALYSIS_RESULTS = "analysis_results"
    DIGEST_RESULTS = "digest_results"
    ERROR = "error"
    METADATA = "metadata"


@dataclass
class StateUpdate:
    """Represents a state update with metadata"""
    timestamp: datetime
    update_type: StateUpdateType
    key: str
    value: Any
    source_agent: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class SharedAgentState:
    """
    Thread-safe shared state management for agent coordination.
    
    Provides:
    - Atomic state updates with locking
    - State update history and tracking
    - Proper result propagation between agents
    - Context preservation across workflow steps
    """
    
    def __init__(self, initial_state: Dict[str, Any]):
        self._state = initial_state.copy()
        self._lock = asyncio.Lock()
        self._update_history: List[StateUpdate] = []
        self._subscribers: List[str] = []
        
        logger.info("SharedAgentState initialized")
    
    async def update(self, updates: Dict[str, Any], source_agent: str, update_type: StateUpdateType = StateUpdateType.METADATA) -> None:
        """
        Atomically update the shared state.
        
        Args:
            updates: Dictionary of updates to apply
            source_agent: Name of the agent making the update
            update_type: Type of update for tracking
        """
        async with self._lock:
            for key, value in updates.items():
                self._state[key] = value
                
                # Record the update
                state_update = StateUpdate(
                    timestamp=datetime.utcnow(),
                    update_type=update_type,
                    key=key,
                    value=value,
                    source_agent=source_agent
                )
                self._update_history.append(state_update)
                
                logger.debug(f"State updated by {source_agent}: {key} = {type(value).__name__}")
    
    async def get(self, key: str, default: Any = None) -> Any:
        """Get a value from the shared state."""
        async with self._lock:
            return self._state.get(key, default)
    
    async def get_all(self) -> Dict[str, Any]:
        """Get a copy of the entire state."""
        async with self._lock:
            return self._state.copy()
    
    async def merge_results(self, results: Dict[str, Any], source_agent: str, update_type: StateUpdateType) -> None:
        """
        Merge results into the state, handling lists and dictionaries properly.
        
        Args:
            results: Results to merge
            source_agent: Name of the agent providing results
            update_type: Type of results being merged
        """
        async with self._lock:
            for key, value in results.items():
                if key in self._state:
                    # Handle different types of merging
                    if isinstance(value, list) and isinstance(self._state[key], list):
                        # Merge lists
                        self._state[key].extend(value)
                    elif isinstance(value, dict) and isinstance(self._state[key], dict):
                        # Merge dictionaries
                        self._state[key].update(value)
                    else:
                        # Replace value
                        self._state[key] = value
                else:
                    # New key
                    self._state[key] = value
                
                # Record the update
                state_update = StateUpdate(
                    timestamp=datetime.utcnow(),
                    update_type=update_type,
                    key=key,
                    value=value,
                    source_agent=source_agent
                )
                self._update_history.append(state_update)
    
    async def propagate_to_subtask(self, subtask_id: str) -> Dict[str, Any]:
        """
        Create a context-rich state for a specific subtask.
        
        Args:
            subtask_id: ID of the subtask
            
        Returns:
            State dictionary with relevant context for the subtask
        """
        async with self._lock:
            # Create subtask-specific state with full context
            subtask_state = self._state.copy()
            
            # Add subtask-specific metadata
            subtask_state["subtask_id"] = subtask_id
            subtask_state["context_timestamp"] = datetime.utcnow().isoformat()
            subtask_state["available_context"] = list(self._state.keys())
            
            return subtask_state
    
    async def get_update_history(self, limit: int = 50) -> List[StateUpdate]:
        """Get recent state update history."""
        async with self._lock:
            return self._update_history[-limit:]
    
    async def get_agent_context(self, agent_name: str) -> Dict[str, Any]:
        """
        Get context relevant to a specific agent.
        
        Args:
            agent_name: Name of the agent
            
        Returns:
            Context dictionary for the agent
        """
        async with self._lock:
            # Filter state to what's relevant for this agent
            agent_context = {}
            
            # Always include core context
            core_keys = ["user_context", "query_interpretation", "task_plan", "metadata"]
            for key in core_keys:
                if key in self._state:
                    agent_context[key] = self._state[key]
            
            # Include agent-specific results
            if agent_name == "SearchAgent":
                agent_context["search_results"] = self._state.get("search_results", [])
                agent_context["search_parameters"] = self._state.get("search_parameters", {})
            elif agent_name == "AnalysisAgent":
                agent_context["analysis_results"] = self._state.get("analysis_results", {})
                agent_context["search_results"] = self._state.get("search_results", [])
            elif agent_name == "DigestAgent":
                agent_context["digest_results"] = self._state.get("digest_results", {})
                agent_context["search_results"] = self._state.get("search_results", [])
            
            return agent_context
    
    async def validate_state(self) -> Dict[str, Any]:
        """
        Validate the current state for consistency.
        
        Returns:
            Validation results
        """
        async with self._lock:
            validation = {
                "is_valid": True,
                "errors": [],
                "warnings": [],
                "state_keys": list(self._state.keys()),
                "update_count": len(self._update_history)
            }
            
            # Check for required keys
            required_keys = ["user_context", "query_interpretation"]
            for key in required_keys:
                if key not in self._state:
                    validation["errors"].append(f"Missing required key: {key}")
                    validation["is_valid"] = False
            
            # Check for data consistency
            if "search_results" in self._state and not isinstance(self._state["search_results"], list):
                validation["errors"].append("search_results should be a list")
                validation["is_valid"] = False
            
            if "analysis_results" in self._state and not isinstance(self._state["analysis_results"], dict):
                validation["errors"].append("analysis_results should be a dictionary")
                validation["is_valid"] = False
            
            return validation 