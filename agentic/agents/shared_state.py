"""
Shared Agent State Management

Provides thread-safe shared state management for agent coordination.
Handles state updates, result propagation, and context preservation across workflow steps.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Set
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field
from collections import defaultdict

logger = logging.getLogger(__name__)


class StateUpdateType(Enum):
    """Types of state updates for tracking and validation."""
    QUERY_ANALYSIS = "query_analysis"
    SEARCH_RESULTS = "search_results"
    ANALYSIS_RESULTS = "analysis_results"
    EXECUTION_PLAN = "execution_plan"
    SUBTASK_RESULT = "subtask_result"
    ERROR_RECOVERY = "error_recovery"
    METADATA_UPDATE = "metadata_update"


@dataclass
class StateUpdate:
    """Represents a state update with metadata."""
    timestamp: datetime
    source_agent: str
    update_type: StateUpdateType
    updates: Dict[str, Any]
    description: str
    success: bool = True
    error_message: Optional[str] = None


class SharedAgentState:
    """
    Thread-safe shared state management for agent coordination.
    
    Provides:
    - Atomic state updates with locking
    - State update history and tracking
    - Proper result propagation between agents
    - Context preservation across workflow steps
    - State validation for consistency
    """
    
    def __init__(self, initial_state: Dict[str, Any]):
        self._state = initial_state.copy()
        self._lock = asyncio.Lock()
        self._update_history: List[StateUpdate] = []
        self._subtask_contexts: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self._validation_errors: List[str] = []
        
        logger.info("SharedAgentState initialized")
    
    async def update(self, updates: Dict[str, Any], source_agent: str, 
                    update_type: StateUpdateType, description: str = "") -> bool:
        """
        Update shared state atomically.
        
        Args:
            updates: Dictionary of updates to apply
            source_agent: Name of the agent making the update
            update_type: Type of update for tracking
            description: Human-readable description of the update
            
        Returns:
            True if update was successful
        """
        async with self._lock:
            try:
                # Apply updates
                for key, value in updates.items():
                    if key in self._state:
                        if isinstance(self._state[key], dict) and isinstance(value, dict):
                            self._state[key].update(value)
                        elif isinstance(self._state[key], list) and isinstance(value, list):
                            self._state[key].extend(value)
                        else:
                            self._state[key] = value
                    else:
                        self._state[key] = value
                
                # Record update
                state_update = StateUpdate(
                    timestamp=datetime.utcnow(),
                    source_agent=source_agent,
                    update_type=update_type,
                    updates=updates.copy(),
                    description=description
                )
                self._update_history.append(state_update)
                
                logger.debug(f"State updated by {source_agent}: {description}")
                return True
                
            except Exception as e:
                error_msg = f"Failed to update state: {e}"
                logger.error(f"{source_agent}: {error_msg}")
                
                # Record failed update
                state_update = StateUpdate(
                    timestamp=datetime.utcnow(),
                    source_agent=source_agent,
                    update_type=update_type,
                    updates=updates,
                    description=description,
                    success=False,
                    error_message=str(e)
                )
                self._update_history.append(state_update)
                
                return False
    
    async def merge_results(self, results: Dict[str, Any], source_agent: str, 
                           update_type: StateUpdateType) -> bool:
        """
        Merge agent results into shared state with proper aggregation.
        
        Args:
            results: Results to merge
            source_agent: Name of the agent providing results
            update_type: Type of results being merged
            
        Returns:
            True if merge was successful
        """
        async with self._lock:
            try:
                # Handle different result types
                if update_type == StateUpdateType.SEARCH_RESULTS:
                    if "search_results" not in self._state:
                        self._state["search_results"] = []
                    self._state["search_results"].extend(results.get("search_results", []))
                    
                elif update_type == StateUpdateType.ANALYSIS_RESULTS:
                    if "analysis_results" not in self._state:
                        self._state["analysis_results"] = {}
                    self._state["analysis_results"].update(results.get("analysis_results", {}))
                    
                elif update_type == StateUpdateType.SUBTASK_RESULT:
                    if "subtask_results" not in self._state:
                        self._state["subtask_results"] = {}
                    self._state["subtask_results"].update(results)
                    
                else:
                    # Generic merge
                    for key, value in results.items():
                        if key in self._state:
                            if isinstance(self._state[key], dict) and isinstance(value, dict):
                                self._state[key].update(value)
                            elif isinstance(self._state[key], list) and isinstance(value, list):
                                self._state[key].extend(value)
                            else:
                                self._state[key] = value
                        else:
                            self._state[key] = value
                
                # Record merge
                state_update = StateUpdate(
                    timestamp=datetime.utcnow(),
                    source_agent=source_agent,
                    update_type=update_type,
                    updates=results.copy(),
                    description=f"Merged {update_type.value} results from {source_agent}"
                )
                self._update_history.append(state_update)
                
                logger.debug(f"Results merged from {source_agent}: {update_type.value}")
                return True
                
            except Exception as e:
                error_msg = f"Failed to merge results: {e}"
                logger.error(f"{source_agent}: {error_msg}")
                return False
    
    async def propagate_to_subtask(self, subtask_id: str) -> Dict[str, Any]:
        """
        Get context-rich state for a specific subtask.
        
        Args:
            subtask_id: ID of the subtask
            
        Returns:
            State dictionary with full context for the subtask
        """
        async with self._lock:
            # Create context-rich state for subtask
            subtask_state = self._state.copy()
            
            # Add subtask-specific context
            if subtask_id in self._subtask_contexts:
                subtask_state.update(self._subtask_contexts[subtask_id])
            
            # Add workflow context
            subtask_state.update({
                "subtask_id": subtask_id,
                "workflow_context": {
                    "total_updates": len(self._update_history),
                    "last_update": self._update_history[-1].timestamp if self._update_history else None,
                    "active_agents": list(set(update.source_agent for update in self._update_history))
                }
            })
            
            logger.debug(f"State propagated to subtask {subtask_id}")
            return subtask_state
    
    async def validate_state(self) -> Dict[str, Any]:
        """
        Validate state consistency and return validation results.
        
        Returns:
            Dictionary with validation results and any issues found
        """
        async with self._lock:
            validation_result = {
                "is_valid": True,
                "issues": [],
                "warnings": [],
                "state_summary": {}
            }
            
            # Check required fields
            required_fields = ["messages", "user_context"]
            for field in required_fields:
                if field not in self._state:
                    validation_result["is_valid"] = False
                    validation_result["issues"].append(f"Missing required field: {field}")
            
            # Check data consistency
            if "search_results" in self._state and not isinstance(self._state["search_results"], list):
                validation_result["is_valid"] = False
                validation_result["issues"].append("search_results must be a list")
            
            if "analysis_results" in self._state and not isinstance(self._state["analysis_results"], dict):
                validation_result["is_valid"] = False
                validation_result["issues"].append("analysis_results must be a dictionary")
            
            # Check for potential issues
            if len(self._update_history) > 100:
                validation_result["warnings"].append("Large number of state updates detected")
            
            # Generate state summary
            validation_result["state_summary"] = {
                "total_fields": len(self._state),
                "total_updates": len(self._update_history),
                "last_update": self._update_history[-1].timestamp if self._update_history else None,
                "active_agents": list(set(update.source_agent for update in self._update_history))
            }
            
            if validation_result["issues"]:
                logger.warning(f"State validation found issues: {validation_result['issues']}")
            elif validation_result["warnings"]:
                logger.info(f"State validation warnings: {validation_result['warnings']}")
            else:
                logger.debug("State validation passed")
            
            return validation_result
    
    def get_state(self) -> Dict[str, Any]:
        """Get current state (read-only)."""
        return self._state.copy()
    
    def get_update_history(self) -> List[StateUpdate]:
        """Get state update history."""
        return self._update_history.copy()
    
    def get_agent_activity(self) -> Dict[str, int]:
        """Get activity summary by agent."""
        activity = defaultdict(int)
        for update in self._update_history:
            activity[update.source_agent] += 1
        return dict(activity)
    
    async def clear_history(self):
        """Clear update history (useful for long-running workflows)."""
        async with self._lock:
            self._update_history.clear()
            logger.info("State update history cleared")
    
    async def cleanup(self):
        """Clean up resources."""
        async with self._lock:
            self._state.clear()
            self._update_history.clear()
            self._subtask_contexts.clear()
            self._validation_errors.clear()
            logger.info("SharedAgentState cleaned up") 