"""
Base Agent Framework

Defines the core agent interface and state management for the agentic RAG system.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, TypedDict
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import asyncio
import logging

logger = logging.getLogger(__name__)


class AgentRole(Enum):
    """Defines the different agent roles in the system"""
    ORCHESTRATOR = "orchestrator"
    PLANNER = "planner"
    SEARCHER = "searcher"
    ANALYZER = "analyzer"
    DIGESTER = "digester"
    AGGREGATOR = "aggregator"
    RESOURCE_MANAGER = "resource_manager"
    CAPABILITY = "capability"  # Added for capability agent


class TaskStatus(Enum):
    """Status of a task or subtask"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class SubTask:
    """Represents a subtask in a larger execution plan"""
    id: str
    description: str
    agent_role: AgentRole
    task_type: str
    parameters: Dict[str, Any]
    dependencies: List[str]
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None
    created_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()


@dataclass
class ExecutionPlan:
    """Represents a complete execution plan for a user query"""
    id: str
    query: str
    subtasks: List[SubTask]
    dependencies: Optional[Dict[str, List[str]]] = None
    status: TaskStatus = TaskStatus.PENDING
    created_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()


class AgentState(TypedDict):
    """State interface for LangGraph agents"""
    messages: List[Dict[str, Any]]
    search_results: List[Dict[str, Any]]
    conversation_history: List[Dict[str, Any]]
    user_context: Dict[str, Any]
    task_plan: Optional[ExecutionPlan]
    current_step: int
    current_subtask: Optional[SubTask]  # Added for tracking current subtask
    analysis_results: Optional[Dict[str, Any]]  # Added for analysis results
    errors: List[str]  # Added for error tracking
    metadata: Dict[str, Any]
    response: Optional[str]  # Added for final response
    # Additional fields for agent coordination
    query_analysis: Optional[Dict[str, Any]]
    query_interpretation: Optional[Dict[str, Any]]  # Added for query interpretation results
    intent: Optional[str]
    entities: Optional[Dict[str, Any]]
    complexity_score: Optional[float]
    execution_plan: Optional[ExecutionPlan]
    subtasks: Optional[List[SubTask]]
    dependencies: Optional[Dict[str, List[str]]]
    # Routing fields for agent orchestration
    next_agent: Optional[str]
    agent_args: Optional[Dict[str, Any]]
    routing_result: Optional[Dict[str, Any]]


class BaseAgent(ABC):
    """
    Abstract base class for all agents in the system.
    
    Each agent should implement specific capabilities while maintaining
    consistency in state management and communication.
    """
    
    def __init__(self, role: AgentRole, config: Dict[str, Any]):
        self.role = role
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{role.value}")
        self._state = {}
        
    @abstractmethod
    async def process(self, state: AgentState) -> AgentState:
        """
        Process the current state and return updated state.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated agent state
        """
        pass
    
    async def run(self, **kwargs) -> Any:
        """
        Run method for compatibility with orchestrator.
        Delegates to process method or handles direct input.
        
        Args:
            **kwargs: Arguments passed to the agent
            
        Returns:
            Agent result
        """
        # If this is called with a state object, use process
        if len(kwargs) == 1 and "state" in kwargs:
            return await self.process(kwargs["state"])
        
        # Otherwise, create a minimal state and call process
        # This is for agents that expect direct input like router_agent
        minimal_state = {
            "messages": [],
            "search_results": [],
            "conversation_history": [],
            "user_context": kwargs,
            "task_plan": None,
            "current_step": 0,
            "current_subtask": None,
            "analysis_results": {},
            "errors": [],
            "metadata": {},
            "response": None,
            "query_analysis": {},
            "query_interpretation": {},
            "intent": None,
            "entities": {},
            "complexity_score": None,
            "execution_plan": None,
            "subtasks": [],
            "dependencies": {},
            "next_agent": None,
            "agent_args": None,
            "routing_result": None
        }
        
        result_state = await self.process(minimal_state)
        return result_state.get("response", "No response generated")
    
    @abstractmethod
    def can_handle(self, task: SubTask) -> bool:
        """
        Determine if this agent can handle the given task.
        
        Args:
            task: Task to evaluate
            
        Returns:
            True if agent can handle the task
        """
        pass
    
    async def validate_input(self, state: AgentState) -> bool:
        """
        Validate input state before processing.
        
        Args:
            state: State to validate
            
        Returns:
            True if state is valid
        """
        required_keys = ["messages", "user_context"]
        return all(key in state for key in required_keys)
    
    def update_metadata(self, state: AgentState, key: str, value: Any) -> AgentState:
        """
        Update metadata in the agent state.
        
        Args:
            state: Current state
            key: Metadata key
            value: Metadata value
            
        Returns:
            Updated state
        """
        if "metadata" not in state:
            state["metadata"] = {}
        state["metadata"][key] = value
        return state
    
    def log_performance(self, operation: str, duration: float, success: bool):
        """
        Log performance metrics for monitoring.
        
        Args:
            operation: Name of the operation
            duration: Time taken in seconds
            success: Whether operation succeeded
        """
        self.logger.info(
            f"Agent {self.role.value} - {operation}: "
            f"duration={duration:.3f}s, success={success}"
        )

    async def cleanup(self):
        """Clean up agent resources and memory."""
        try:
            logger.info(f"Cleaning up {self.__class__.__name__} resources...")
            
            # Clear any cached data
            if hasattr(self, '_cache') and self._cache:
                await self._cache.clear()
            
            # Clear memory if available
            if hasattr(self, 'memory') and self.memory:
                await self.memory.clear()
            
            # Close any open connections
            if hasattr(self, 'llm_client') and self.llm_client:
                await self.llm_client.close()
            
            # Clear internal state
            if hasattr(self, '_state'):
                self._state.clear()
            
            logger.info(f"{self.__class__.__name__} cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during {self.__class__.__name__} cleanup: {e}")
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage statistics for the agent."""
        try:
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            
            return {
                "rss_mb": memory_info.rss / 1024 / 1024,  # Resident Set Size in MB
                "vms_mb": memory_info.vms / 1024 / 1024,  # Virtual Memory Size in MB
                "percent": process.memory_percent(),
                "agent_type": self.__class__.__name__,
                "cache_size": len(self._cache) if hasattr(self, '_cache') else 0,
                "state_size": len(self._state) if hasattr(self, '_state') else 0
            }
            
        except Exception as e:
            logger.error(f"Error getting memory usage: {e}")
            return {"error": str(e)}
    
    def set_services(self, **services):
        """Set shared services for the agent."""
        try:
            for service_name, service_instance in services.items():
                if hasattr(self, service_name):
                    setattr(self, service_name, service_instance)
                    logger.debug(f"Set {service_name} for {self.__class__.__name__}")
            
        except Exception as e:
            logger.error(f"Error setting services for {self.__class__.__name__}: {e}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on the agent."""
        try:
            health_status = {
                "agent_type": self.__class__.__name__,
                "status": "healthy",
                "timestamp": datetime.utcnow().isoformat(),
                "memory_usage": self.get_memory_usage(),
                "services": {}
            }
            
            # Check LLM client
            if hasattr(self, 'llm_client') and self.llm_client:
                try:
                    llm_health = await self.llm_client.health_check()
                    health_status["services"]["llm_client"] = llm_health
                except Exception as e:
                    health_status["services"]["llm_client"] = {"status": "unhealthy", "error": str(e)}
                    health_status["status"] = "degraded"
            
            # Check cache
            if hasattr(self, '_cache') and self._cache:
                try:
                    cache_stats = self._cache.get_stats()
                    health_status["services"]["cache"] = {"status": "healthy", "stats": cache_stats}
                except Exception as e:
                    health_status["services"]["cache"] = {"status": "unhealthy", "error": str(e)}
                    health_status["status"] = "degraded"
            
            # Check memory
            if hasattr(self, 'memory') and self.memory:
                try:
                    memory_stats = await self.memory.get_stats()
                    health_status["services"]["memory"] = {"status": "healthy", "stats": memory_stats}
                except Exception as e:
                    health_status["services"]["memory"] = {"status": "unhealthy", "error": str(e)}
                    health_status["status"] = "degraded"
            
            return health_status
            
        except Exception as e:
            logger.error(f"Error in health check for {self.__class__.__name__}: {e}")
            return {
                "agent_type": self.__class__.__name__,
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }


class AgentRegistry:
    """
    Registry for managing agent instances and routing tasks.
    """
    
    def __init__(self):
        self._agents: Dict[AgentRole, BaseAgent] = {}
        self.logger = logging.getLogger(f"{__name__}.registry")
    
    def register_agent(self, agent: BaseAgent):
        """Register an agent with the registry"""
        self._agents[agent.role] = agent
        self.logger.info(f"Registered agent: {agent.role.value}")
    
    def get_agent(self, role: AgentRole) -> Optional[BaseAgent]:
        """Get agent by role"""
        return self._agents.get(role)
    
    def find_capable_agent(self, task: SubTask) -> Optional[BaseAgent]:
        """Find an agent capable of handling the given task"""
        for agent in self._agents.values():
            if agent.can_handle(task):
                return agent
        return None
    
    def list_agents(self) -> List[AgentRole]:
        """List all registered agent roles"""
        return list(self._agents.keys())


# Global agent registry instance
agent_registry = AgentRegistry()
