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
    RESOURCE_MANAGER = "resource_manager"


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
