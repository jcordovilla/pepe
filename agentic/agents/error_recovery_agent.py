"""
Error Recovery Agent

Handles failed subtasks with intelligent retry logic and fallback strategies.
Provides error classification, recovery strategies, and workflow continuity.
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
from dataclasses import dataclass
from datetime import datetime, timedelta

from .base_agent import BaseAgent, AgentRole, AgentState, SubTask, TaskStatus

logger = logging.getLogger(__name__)


class ErrorType(Enum):
    """Types of errors that can occur during task execution."""
    TIMEOUT = "timeout"
    NO_RESULTS = "no_results"
    LLM_ERROR = "llm_error"
    NETWORK_ERROR = "network_error"
    VALIDATION_ERROR = "validation_error"
    RESOURCE_ERROR = "resource_error"
    UNKNOWN_ERROR = "unknown_error"


class RecoveryStrategy(Enum):
    """Available recovery strategies."""
    RETRY_SIMPLIFIED = "retry_simplified"
    ALTERNATIVE_APPROACH = "alternative_approach"
    FALLBACK_BASIC = "fallback_basic"
    SKIP_AND_CONTINUE = "skip_and_continue"
    ABORT_WORKFLOW = "abort_workflow"


@dataclass
class RecoveryAttempt:
    """Represents a recovery attempt with metadata."""
    timestamp: datetime
    error_type: ErrorType
    original_error: str
    strategy: RecoveryStrategy
    success: bool
    result: Optional[Any] = None
    new_error: Optional[str] = None
    duration: float = 0.0


class ErrorRecoveryAgent(BaseAgent):
    """
    Handles failed subtasks with intelligent retry logic and fallback strategies.
    
    Provides:
    - Error classification and analysis
    - Intelligent recovery strategy selection
    - Multiple fallback approaches
    - Workflow continuity management
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(AgentRole.RESOURCE_MANAGER, config)
        
        # Recovery configuration
        self.max_retries = config.get("max_retries", 3)
        self.retry_delay = config.get("retry_delay", 1.0)
        self.timeout_threshold = config.get("timeout_threshold", 30.0)
        self.enable_aggressive_recovery = config.get("enable_aggressive_recovery", True)
        
        # Recovery history
        self.recovery_history: List[RecoveryAttempt] = []
        self.error_patterns: Dict[str, int] = {}
        
        logger.info("ErrorRecoveryAgent initialized")
    
    async def process(self, state: AgentState) -> AgentState:
        """
        Process error recovery for failed subtasks.
        
        Args:
            state: Current agent state with error information
            
        Returns:
            Updated state with recovery results
        """
        try:
            # Extract error information from state
            failed_subtasks = self._get_failed_subtasks(state)
            
            if not failed_subtasks:
                logger.info("No failed subtasks to recover")
                return state
            
            logger.info(f"Attempting to recover {len(failed_subtasks)} failed subtasks")
            
            # Process each failed subtask
            recovery_results = {}
            for subtask in failed_subtasks:
                recovery_result = await self._recover_subtask(subtask, state)
                recovery_results[subtask.id] = recovery_result
            
            # Update state with recovery results
            state["error_recovery_results"] = recovery_results
            state["recovery_summary"] = self._generate_recovery_summary()
            
            logger.info(f"Error recovery completed: {len(recovery_results)} subtasks processed")
            return state
            
        except Exception as e:
            logger.error(f"Error in error recovery process: {e}")
            state["errors"].append(f"Error recovery failed: {str(e)}")
            return state
    
    def can_handle(self, task: SubTask) -> bool:
        """Determine if this agent can handle the given task."""
        return task.status == TaskStatus.FAILED
    
    async def _recover_subtask(self, subtask: SubTask, state: AgentState) -> Dict[str, Any]:
        """
        Attempt to recover a specific failed subtask.
        
        Args:
            subtask: The failed subtask to recover
            state: Current agent state
            
        Returns:
            Recovery result with success status and details
        """
        start_time = time.time()
        
        try:
            # Classify the error
            error_type = self._classify_error(subtask.error or "Unknown error")
            
            # Determine recovery strategy
            strategy = self._determine_recovery_strategy(error_type, subtask)
            
            # Execute recovery strategy
            recovery_result = await self._execute_recovery_strategy(subtask, strategy, state)
            
            # Record recovery attempt
            duration = time.time() - start_time
            recovery_attempt = RecoveryAttempt(
                timestamp=datetime.utcnow(),
                error_type=error_type,
                original_error=subtask.error or "Unknown error",
                strategy=strategy,
                success=recovery_result["success"],
                result=recovery_result.get("result"),
                new_error=recovery_result.get("error"),
                duration=duration
            )
            self.recovery_history.append(recovery_attempt)
            
            # Update error patterns
            self.error_patterns[error_type.value] = self.error_patterns.get(error_type.value, 0) + 1
            
            logger.info(f"Recovery attempt for subtask {subtask.id}: {strategy.value} - {'SUCCESS' if recovery_result['success'] else 'FAILED'}")
            
            return recovery_result
            
        except Exception as e:
            logger.error(f"Error during subtask recovery {subtask.id}: {e}")
            return {
                "success": False,
                "error": f"Recovery process failed: {str(e)}",
                "strategy": "none"
            }
    
    def _classify_error(self, error_message: str) -> ErrorType:
        """
        Classify error based on error message patterns.
        
        Args:
            error_message: The error message to classify
            
        Returns:
            Classified error type
        """
        error_lower = error_message.lower()
        
        # Timeout errors
        if any(keyword in error_lower for keyword in ["timeout", "timed out", "deadline exceeded"]):
            return ErrorType.TIMEOUT
        
        # No results errors
        if any(keyword in error_lower for keyword in ["no results", "empty", "not found", "no data"]):
            return ErrorType.NO_RESULTS
        
        # LLM errors
        if any(keyword in error_lower for keyword in ["llm", "model", "generation", "token", "openai"]):
            return ErrorType.LLM_ERROR
        
        # Network errors
        if any(keyword in error_lower for keyword in ["network", "connection", "http", "api", "request"]):
            return ErrorType.NETWORK_ERROR
        
        # Validation errors
        if any(keyword in error_lower for keyword in ["validation", "invalid", "format", "schema"]):
            return ErrorType.VALIDATION_ERROR
        
        # Resource errors
        if any(keyword in error_lower for keyword in ["memory", "resource", "quota", "limit"]):
            return ErrorType.RESOURCE_ERROR
        
        return ErrorType.UNKNOWN_ERROR
    
    def _determine_recovery_strategy(self, error_type: ErrorType, subtask: SubTask) -> RecoveryStrategy:
        """
        Determine the best recovery strategy based on error type and subtask.
        
        Args:
            error_type: Classified error type
            subtask: The failed subtask
            
        Returns:
            Selected recovery strategy
        """
        # Strategy mapping based on error type
        strategy_mapping = {
            ErrorType.TIMEOUT: RecoveryStrategy.RETRY_SIMPLIFIED,
            ErrorType.NO_RESULTS: RecoveryStrategy.ALTERNATIVE_APPROACH,
            ErrorType.LLM_ERROR: RecoveryStrategy.RETRY_SIMPLIFIED,
            ErrorType.NETWORK_ERROR: RecoveryStrategy.RETRY_SIMPLIFIED,
            ErrorType.VALIDATION_ERROR: RecoveryStrategy.ALTERNATIVE_APPROACH,
            ErrorType.RESOURCE_ERROR: RecoveryStrategy.FALLBACK_BASIC,
            ErrorType.UNKNOWN_ERROR: RecoveryStrategy.SKIP_AND_CONTINUE
        }
        
        # Get base strategy
        strategy = strategy_mapping.get(error_type, RecoveryStrategy.SKIP_AND_CONTINUE)
        
        # Apply context-specific adjustments
        if self._is_critical_subtask(subtask):
            if strategy == RecoveryStrategy.SKIP_AND_CONTINUE:
                strategy = RecoveryStrategy.FALLBACK_BASIC
        
        # Check if we've tried too many times
        if self._get_retry_count(subtask.id) >= self.max_retries:
            strategy = RecoveryStrategy.SKIP_AND_CONTINUE
        
        return strategy
    
    async def _execute_recovery_strategy(self, subtask: SubTask, strategy: RecoveryStrategy, state: AgentState) -> Dict[str, Any]:
        """
        Execute the selected recovery strategy.
        
        Args:
            subtask: The failed subtask
            strategy: The recovery strategy to execute
            state: Current agent state
            
        Returns:
            Recovery result
        """
        try:
            if strategy == RecoveryStrategy.RETRY_SIMPLIFIED:
                return await self._retry_simplified(subtask, state)
            elif strategy == RecoveryStrategy.ALTERNATIVE_APPROACH:
                return await self._alternative_approach(subtask, state)
            elif strategy == RecoveryStrategy.FALLBACK_BASIC:
                return await self._fallback_basic(subtask, state)
            elif strategy == RecoveryStrategy.SKIP_AND_CONTINUE:
                return await self._skip_and_continue(subtask, state)
            elif strategy == RecoveryStrategy.ABORT_WORKFLOW:
                return await self._abort_workflow(subtask, state)
            else:
                return {"success": False, "error": f"Unknown recovery strategy: {strategy}"}
                
        except Exception as e:
            logger.error(f"Error executing recovery strategy {strategy}: {e}")
            return {"success": False, "error": f"Strategy execution failed: {str(e)}"}
    
    async def _retry_simplified(self, subtask: SubTask, state: AgentState) -> Dict[str, Any]:
        """Retry with simplified parameters."""
        try:
            # Simplify the task parameters
            simplified_params = self._simplify_parameters(subtask.parameters)
            
            # Add delay before retry
            await asyncio.sleep(self.retry_delay)
            
            # Retry with simplified parameters
            # This would typically involve calling the original agent with simplified params
            result = await self._execute_simplified_task(subtask, simplified_params, state)
            
            return {
                "success": True,
                "result": result,
                "strategy": "retry_simplified",
                "parameters_simplified": True
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Simplified retry failed: {str(e)}",
                "strategy": "retry_simplified"
            }
    
    async def _alternative_approach(self, subtask: SubTask, state: AgentState) -> Dict[str, Any]:
        """Try an alternative approach for the task."""
        try:
            # Determine alternative approach based on task type
            alternative_task = self._create_alternative_task(subtask)
            
            # Execute alternative task
            result = await self._execute_alternative_task(alternative_task, state)
            
            return {
                "success": True,
                "result": result,
                "strategy": "alternative_approach",
                "alternative_used": True
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Alternative approach failed: {str(e)}",
                "strategy": "alternative_approach"
            }
    
    async def _fallback_basic(self, subtask: SubTask, state: AgentState) -> Dict[str, Any]:
        """Use basic search as fallback."""
        try:
            # Create basic search task
            basic_search = self._create_basic_search_task(subtask)
            
            # Execute basic search
            result = await self._execute_basic_search(basic_search, state)
            
            return {
                "success": True,
                "result": result,
                "strategy": "fallback_basic",
                "basic_search_used": True
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Basic fallback failed: {str(e)}",
                "strategy": "fallback_basic"
            }
    
    async def _skip_and_continue(self, subtask: SubTask, state: AgentState) -> Dict[str, Any]:
        """Skip the failed subtask and continue workflow."""
        try:
            # Mark subtask as skipped
            subtask.status = TaskStatus.CANCELLED
            subtask.result = {"status": "skipped", "reason": "error_recovery"}
            
            return {
                "success": True,
                "result": {"status": "skipped"},
                "strategy": "skip_and_continue",
                "workflow_continued": True
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Skip and continue failed: {str(e)}",
                "strategy": "skip_and_continue"
            }
    
    async def _abort_workflow(self, subtask: SubTask, state: AgentState) -> Dict[str, Any]:
        """Abort the entire workflow."""
        try:
            # Mark all subtasks as cancelled
            if "subtasks" in state:
                for task in state["subtasks"]:
                    task.status = TaskStatus.CANCELLED
            
            return {
                "success": False,
                "result": {"status": "workflow_aborted"},
                "strategy": "abort_workflow",
                "workflow_aborted": True
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Workflow abort failed: {str(e)}",
                "strategy": "abort_workflow"
            }
    
    def _get_failed_subtasks(self, state: AgentState) -> List[SubTask]:
        """Get list of failed subtasks from state."""
        failed_subtasks = []
        
        if "subtasks" in state:
            for subtask in state["subtasks"]:
                if subtask.status == TaskStatus.FAILED:
                    failed_subtasks.append(subtask)
        
        return failed_subtasks
    
    def _is_critical_subtask(self, subtask: SubTask) -> bool:
        """Determine if a subtask is critical to workflow success."""
        critical_types = ["search", "query_interpretation", "execution_plan"]
        return subtask.task_type in critical_types
    
    def _get_retry_count(self, subtask_id: str) -> int:
        """Get the number of recovery attempts for a subtask."""
        return len([attempt for attempt in self.recovery_history 
                   if hasattr(attempt, 'subtask_id') and attempt.subtask_id == subtask_id])
    
    def _simplify_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Simplify task parameters for retry."""
        simplified = parameters.copy()
        
        # Reduce complexity
        if "k" in simplified:
            simplified["k"] = min(simplified["k"], 5)  # Limit results
        
        if "max_tokens" in simplified:
            simplified["max_tokens"] = min(simplified["max_tokens"], 1000)  # Limit tokens
        
        # Remove complex filters
        if "filters" in simplified:
            simplified["filters"] = {}
        
        return simplified
    
    def _create_alternative_task(self, subtask: SubTask) -> SubTask:
        """Create an alternative task based on the original."""
        # This would implement task-specific alternatives
        # For now, return a simplified version
        return SubTask(
            id=f"{subtask.id}_alternative",
            description=f"Alternative approach for: {subtask.description}",
            agent_role=subtask.agent_role,
            task_type="basic_search",  # Fallback to basic search
            parameters={"query": subtask.parameters.get("query", "")},
            dependencies=[]
        )
    
    def _create_basic_search_task(self, subtask: SubTask) -> SubTask:
        """Create a basic search task as fallback."""
        return SubTask(
            id=f"{subtask.id}_basic_search",
            description=f"Basic search fallback for: {subtask.description}",
            agent_role=AgentRole.SEARCHER,
            task_type="basic_search",
            parameters={"query": subtask.parameters.get("query", ""), "k": 5},
            dependencies=[]
        )
    
    async def _execute_simplified_task(self, subtask: SubTask, simplified_params: Dict[str, Any], state: AgentState) -> Any:
        """Execute task with simplified parameters."""
        # This would call the original agent with simplified parameters
        # For now, return a mock result
        return {"status": "simplified_execution", "parameters": simplified_params}
    
    async def _execute_alternative_task(self, alternative_task: SubTask, state: AgentState) -> Any:
        """Execute alternative task."""
        # This would execute the alternative task
        # For now, return a mock result
        return {"status": "alternative_execution", "task_type": alternative_task.task_type}
    
    async def _execute_basic_search(self, basic_search: SubTask, state: AgentState) -> Any:
        """Execute basic search task."""
        # This would execute a basic search
        # For now, return a mock result
        return {"status": "basic_search", "results": []}
    
    def _generate_recovery_summary(self) -> Dict[str, Any]:
        """Generate summary of recovery attempts."""
        if not self.recovery_history:
            return {"total_attempts": 0, "success_rate": 0.0}
        
        total_attempts = len(self.recovery_history)
        successful_attempts = len([attempt for attempt in self.recovery_history if attempt.success])
        success_rate = successful_attempts / total_attempts if total_attempts > 0 else 0.0
        
        # Strategy effectiveness
        strategy_stats = {}
        for strategy in RecoveryStrategy:
            strategy_attempts = [attempt for attempt in self.recovery_history if attempt.strategy == strategy]
            if strategy_attempts:
                strategy_success = len([a for a in strategy_attempts if a.success])
                strategy_stats[strategy.value] = {
                    "attempts": len(strategy_attempts),
                    "successes": strategy_success,
                    "success_rate": strategy_success / len(strategy_attempts)
                }
        
        return {
            "total_attempts": total_attempts,
            "successful_attempts": successful_attempts,
            "success_rate": success_rate,
            "strategy_effectiveness": strategy_stats,
            "error_patterns": self.error_patterns,
            "recent_attempts": [
                {
                    "timestamp": attempt.timestamp.isoformat(),
                    "error_type": attempt.error_type.value,
                    "strategy": attempt.strategy.value,
                    "success": attempt.success,
                    "duration": attempt.duration
                }
                for attempt in self.recovery_history[-5:]  # Last 5 attempts
            ]
        }
    
    def get_recovery_stats(self) -> Dict[str, Any]:
        """Get recovery statistics."""
        return {
            "total_recoveries": len(self.recovery_history),
            "success_rate": self._generate_recovery_summary()["success_rate"],
            "most_common_error": max(self.error_patterns.items(), key=lambda x: x[1])[0] if self.error_patterns else None,
            "average_recovery_time": sum(attempt.duration for attempt in self.recovery_history) / len(self.recovery_history) if self.recovery_history else 0.0
        } 