"""
Error Recovery Agent

Handles failed subtasks with intelligent retry logic and fallback strategies.
Fixes the issue where failed subtasks break the entire workflow.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from enum import Enum

from .base_agent import BaseAgent, AgentRole, AgentState, SubTask, TaskStatus, agent_registry

logger = logging.getLogger(__name__)


class ErrorType(Enum):
    """Types of errors that can occur"""
    TIMEOUT = "timeout"
    NO_RESULTS = "no_results"
    LLM_ERROR = "llm_error"
    VECTOR_STORE_ERROR = "vector_store_error"
    NETWORK_ERROR = "network_error"
    VALIDATION_ERROR = "validation_error"
    UNKNOWN = "unknown"


class RecoveryStrategy(Enum):
    """Available recovery strategies"""
    RETRY_SIMPLIFIED = "retry_simplified"
    ALTERNATIVE_APPROACH = "alternative_approach"
    FALLBACK_BASIC = "fallback_basic"
    SKIP_AND_CONTINUE = "skip_and_continue"
    ABORT_WORKFLOW = "abort_workflow"


class ErrorRecoveryAgent(BaseAgent):
    """
    Agent responsible for handling failed subtasks and implementing recovery strategies.
    
    This agent:
    - Analyzes error types and patterns
    - Implements intelligent retry logic
    - Provides fallback strategies
    - Maintains workflow continuity
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(AgentRole.ANALYZER, config)
        
        # Recovery configuration
        self.max_retries = config.get("max_retries", 3)
        self.retry_delay = config.get("retry_delay", 1.0)
        self.timeout_threshold = config.get("timeout_threshold", 30.0)
        
        # Error pattern matching
        self.error_patterns = {
            ErrorType.TIMEOUT: [
                "timeout", "timed out", "time out", "deadline exceeded",
                "request timeout", "connection timeout"
            ],
            ErrorType.NO_RESULTS: [
                "no results", "no matches", "empty results", "no data found",
                "zero results", "no documents found"
            ],
            ErrorType.LLM_ERROR: [
                "llm error", "model error", "generation failed", "api error",
                "invalid response", "json parse error"
            ],
            ErrorType.VECTOR_STORE_ERROR: [
                "vector store", "chromadb", "embedding", "similarity search",
                "index error", "collection error"
            ],
            ErrorType.NETWORK_ERROR: [
                "network", "connection", "http", "request failed",
                "connection refused", "dns resolution"
            ]
        }
        
        logger.info(f"ErrorRecoveryAgent initialized with max_retries={self.max_retries}")
        
        # Register this agent
        agent_registry.register_agent(self)
    
    def can_handle(self, task: SubTask) -> bool:
        """Determine if this agent can handle the given task."""
        if not task or not task.task_type:
            return False
        
        recovery_types = ["error_recovery", "retry_subtask", "fallback_strategy"]
        task_type = task.task_type.lower() if task.task_type else ""
        return any(recovery_type in task_type for recovery_type in recovery_types)
    
    async def process(self, state: AgentState) -> AgentState:
        """
        Process error recovery tasks.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with recovery results
        """
        try:
            # Handle current subtask (orchestrator mode)
            if "current_subtask" in state and state["current_subtask"] is not None:
                subtask = state["current_subtask"]
                if self.can_handle(subtask):
                    logger.info(f"Processing error recovery for subtask: {subtask.id}")
                    
                    # Analyze the error and determine recovery strategy
                    error_type = self._classify_error(subtask.error)
                    strategy = self._determine_recovery_strategy(error_type, subtask)
                    
                    # Execute recovery strategy
                    recovered_subtask = await self._execute_recovery_strategy(
                        subtask, strategy, state
                    )
                    
                    # Update state with recovery results
                    state["recovery_results"] = {
                        "original_subtask_id": subtask.id,
                        "error_type": error_type.value,
                        "recovery_strategy": strategy.value,
                        "recovered_subtask": recovered_subtask,
                        "recovery_timestamp": datetime.utcnow().isoformat()
                    }
                    
                    # Update the original subtask
                    subtask.status = TaskStatus.COMPLETED
                    subtask.result = recovered_subtask.result
                    subtask.error = None
                    
                    logger.info(f"Error recovery completed: {strategy.value}")
                    return state
            
            # Fallback: process all recovery subtasks
            subtasks = state.get("subtasks", [])
            recovery_subtasks = [task for task in subtasks if self.can_handle(task)]
            
            if not recovery_subtasks:
                logger.warning("No error recovery subtasks found")
                return state
            
            recovery_results = {}
            
            for subtask in recovery_subtasks:
                logger.info(f"Processing recovery subtask: {subtask.task_type}")
                
                error_type = self._classify_error(subtask.error)
                strategy = self._determine_recovery_strategy(error_type, subtask)
                
                recovered_subtask = await self._execute_recovery_strategy(
                    subtask, strategy, state
                )
                
                recovery_results[subtask.id] = {
                    "error_type": error_type.value,
                    "strategy": strategy.value,
                    "recovered_subtask": recovered_subtask
                }
                
                # Update subtask status
                subtask.status = TaskStatus.COMPLETED
                subtask.result = recovered_subtask.result
                subtask.error = None
            
            # Update state
            state["recovery_results"] = recovery_results
            state["metadata"]["error_recovery_agent"] = {
                "recovery_time": datetime.utcnow().isoformat(),
                "subtasks_recovered": len(recovery_results),
                "strategies_used": [r["strategy"] for r in recovery_results.values()]
            }
            
            logger.info(f"Error recovery completed: {len(recovery_results)} subtasks recovered")
            return state
            
        except Exception as e:
            logger.error(f"Error in error recovery agent: {e}")
            state["errors"] = state.get("errors", [])
            state["errors"].append(f"Error recovery failed: {str(e)}")
            return state
    
    def _classify_error(self, error_message: str) -> ErrorType:
        """
        Classify the error type based on error message patterns.
        
        Args:
            error_message: Error message to classify
            
        Returns:
            Classified error type
        """
        if not error_message:
            return ErrorType.UNKNOWN
        
        error_lower = error_message.lower()
        
        for error_type, patterns in self.error_patterns.items():
            for pattern in patterns:
                if pattern in error_lower:
                    return error_type
        
        return ErrorType.UNKNOWN
    
    def _determine_recovery_strategy(self, error_type: ErrorType, subtask: SubTask) -> RecoveryStrategy:
        """
        Determine the appropriate recovery strategy based on error type and subtask.
        
        Args:
            error_type: Type of error that occurred
            subtask: The failed subtask
            
        Returns:
            Recovery strategy to use
        """
        # Check retry count
        retry_count = getattr(subtask, 'retry_count', 0)
        if retry_count >= self.max_retries:
            return RecoveryStrategy.FALLBACK_BASIC
        
        # Determine strategy based on error type
        if error_type == ErrorType.TIMEOUT:
            return RecoveryStrategy.RETRY_SIMPLIFIED
        elif error_type == ErrorType.NO_RESULTS:
            return RecoveryStrategy.ALTERNATIVE_APPROACH
        elif error_type == ErrorType.LLM_ERROR:
            return RecoveryStrategy.RETRY_SIMPLIFIED
        elif error_type == ErrorType.VECTOR_STORE_ERROR:
            return RecoveryStrategy.ALTERNATIVE_APPROACH
        elif error_type == ErrorType.NETWORK_ERROR:
            return RecoveryStrategy.RETRY_SIMPLIFIED
        else:
            return RecoveryStrategy.FALLBACK_BASIC
    
    async def _execute_recovery_strategy(
        self, 
        subtask: SubTask, 
        strategy: RecoveryStrategy, 
        state: AgentState
    ) -> SubTask:
        """
        Execute the determined recovery strategy.
        
        Args:
            subtask: The failed subtask
            strategy: Recovery strategy to execute
            state: Current agent state
            
        Returns:
            Recovered subtask
        """
        logger.info(f"Executing recovery strategy: {strategy.value} for subtask {subtask.id}")
        
        if strategy == RecoveryStrategy.RETRY_SIMPLIFIED:
            return await self._retry_simplified(subtask, state)
        elif strategy == RecoveryStrategy.ALTERNATIVE_APPROACH:
            return await self._alternative_approach(subtask, state)
        elif strategy == RecoveryStrategy.FALLBACK_BASIC:
            return await self._fallback_basic(subtask, state)
        elif strategy == RecoveryStrategy.SKIP_AND_CONTINUE:
            return await self._skip_and_continue(subtask, state)
        else:
            return await self._abort_workflow(subtask, state)
    
    async def _retry_simplified(self, subtask: SubTask, state: AgentState) -> SubTask:
        """Retry the subtask with simplified parameters."""
        logger.info(f"Retrying subtask {subtask.id} with simplified parameters")
        
        # Create simplified version of the subtask
        simplified_subtask = SubTask(
            id=f"{subtask.id}_retry_{getattr(subtask, 'retry_count', 0) + 1}",
            description=f"Simplified retry: {subtask.description}",
            agent_role=subtask.agent_role,
            task_type=subtask.task_type,
            parameters=self._simplify_parameters(subtask.parameters),
            dependencies=subtask.dependencies,
            status=TaskStatus.PENDING
        )
        
        # Add retry metadata
        simplified_subtask.retry_count = getattr(subtask, 'retry_count', 0) + 1
        simplified_subtask.original_subtask_id = subtask.id
        
        # Add delay before retry
        await asyncio.sleep(self.retry_delay)
        
        return simplified_subtask
    
    async def _alternative_approach(self, subtask: SubTask, state: AgentState) -> SubTask:
        """Try an alternative approach for the subtask."""
        logger.info(f"Trying alternative approach for subtask {subtask.id}")
        
        # Map task types to alternative approaches
        alternative_mapping = {
            "semantic_search": "keyword_search",
            "keyword_search": "filtered_search",
            "filtered_search": "semantic_search",
            "summarize": "extract_insights",
            "extract_insights": "classify_content"
        }
        
        alternative_task_type = alternative_mapping.get(subtask.task_type, "semantic_search")
        
        alternative_subtask = SubTask(
            id=f"{subtask.id}_alternative",
            description=f"Alternative approach: {subtask.description}",
            agent_role=subtask.agent_role,
            task_type=alternative_task_type,
            parameters=self._adapt_parameters_for_alternative(subtask.parameters, alternative_task_type),
            dependencies=subtask.dependencies,
            status=TaskStatus.PENDING
        )
        
        alternative_subtask.original_subtask_id = subtask.id
        alternative_subtask.recovery_strategy = "alternative_approach"
        
        return alternative_subtask
    
    async def _fallback_basic(self, subtask: SubTask, state: AgentState) -> SubTask:
        """Create a basic fallback subtask."""
        logger.info(f"Creating basic fallback for subtask {subtask.id}")
        
        # Create a basic search as fallback
        fallback_subtask = SubTask(
            id=f"{subtask.id}_fallback",
            description=f"Basic fallback: {subtask.description}",
            agent_role=AgentRole.SEARCHER,
            task_type="semantic_search",
            parameters={
                "query": state.get("user_context", {}).get("query", ""),
                "k": 5,  # Reduced results
                "filters": {}
            },
            dependencies=[],
            status=TaskStatus.PENDING
        )
        
        fallback_subtask.original_subtask_id = subtask.id
        fallback_subtask.recovery_strategy = "fallback_basic"
        
        return fallback_subtask
    
    async def _skip_and_continue(self, subtask: SubTask, state: AgentState) -> SubTask:
        """Skip the subtask and continue with workflow."""
        logger.info(f"Skipping subtask {subtask.id} and continuing workflow")
        
        # Mark as completed with empty result
        subtask.status = TaskStatus.COMPLETED
        subtask.result = {"skipped": True, "reason": "Error recovery strategy"}
        subtask.error = None
        
        return subtask
    
    async def _abort_workflow(self, subtask: SubTask, state: AgentState) -> SubTask:
        """Abort the entire workflow."""
        logger.error(f"Aborting workflow due to unrecoverable error in subtask {subtask.id}")
        
        # Mark workflow as failed
        state["workflow_status"] = "failed"
        state["failure_reason"] = f"Unrecoverable error in subtask {subtask.id}"
        
        subtask.status = TaskStatus.FAILED
        subtask.error = "Workflow aborted due to unrecoverable error"
        
        return subtask
    
    def _simplify_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Simplify parameters for retry."""
        simplified = parameters.copy()
        
        # Reduce complexity
        if "k" in simplified:
            simplified["k"] = min(simplified["k"], 5)  # Reduce result count
        
        if "max_tokens" in simplified:
            simplified["max_tokens"] = min(simplified["max_tokens"], 1000)  # Reduce token limit
        
        if "filters" in simplified:
            # Keep only essential filters
            essential_filters = ["time_range", "channel_id"]
            simplified["filters"] = {
                k: v for k, v in simplified["filters"].items() 
                if k in essential_filters
            }
        
        return simplified
    
    def _adapt_parameters_for_alternative(self, parameters: Dict[str, Any], alternative_task_type: str) -> Dict[str, Any]:
        """Adapt parameters for alternative task type."""
        adapted = parameters.copy()
        
        if alternative_task_type == "keyword_search":
            # Extract keywords from query for keyword search
            query = adapted.get("query", "")
            if query:
                # Simple keyword extraction (could be enhanced with NLP)
                keywords = [word.lower() for word in query.split() if len(word) > 3]
                adapted["keywords"] = keywords[:5]  # Limit to 5 keywords
        
        elif alternative_task_type == "filtered_search":
            # Ensure we have basic filters
            if "filters" not in adapted:
                adapted["filters"] = {}
        
        return adapted 