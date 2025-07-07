"""
Planning Agent

Specialized agent for query analysis and execution planning.
"""

from typing import Dict, List, Any, Optional
import logging
from datetime import datetime

from .base_agent import BaseAgent, AgentRole, AgentState, SubTask, agent_registry
from ..reasoning.task_planner import TaskPlanner

logger = logging.getLogger(__name__)


class PlanningAgent(BaseAgent):
    """
    Agent responsible for analyzing queries and creating execution plans.
    
    This agent:
    - Analyzes user queries to extract intent and entities
    - Creates detailed execution plans
    - Optimizes task sequences for efficiency
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(AgentRole.PLANNER, config)
        
        self.task_planner = TaskPlanner(config.get("task_planner", {}))
        
        # Planning configuration
        self.max_subtasks = config.get("max_subtasks", 10)
        self.complexity_threshold = config.get("complexity_threshold", 0.7)
        
        logger.info(f"PlanningAgent initialized with max_subtasks={self.max_subtasks}")
        
        # Register this agent
        agent_registry.register_agent(self)
    
    async def process(self, state: AgentState) -> AgentState:
        """
        Process the query and create an execution plan.
        
        Args:
            state: Current agent state containing query and context
            
        Returns:
            Updated state with query analysis and execution plan
        """
        try:
            query = state.get("query", "")
            if not query:
                raise ValueError("No query provided for planning")
            
            # Step 1: Analyze the query
            logger.info(f"Analyzing query: {query[:100]}...")
            
            # Update state with analysis
            state["complexity_score"] = 0.5
            
            # Step 2: Create execution plan
            logger.info(f"Creating execution plan for intent: {state['intent']}")
            execution_plan = await self.task_planner.create_plan(
                query=query,
                analysis=state,
                context=state.get("context", {})
            )
            
            # Update state with plan
            state["execution_plan"] = execution_plan
            state["subtasks"] = execution_plan.subtasks
            state["dependencies"] = execution_plan.dependencies
            
            # Step 3: Validate and optimize plan
            optimized_plan = await self._optimize_plan(execution_plan, state)
            state["execution_plan"] = optimized_plan
            state["subtasks"] = optimized_plan.subtasks
            
            # Update metadata
            state["metadata"]["planning_agent"] = {
                "analysis_time": datetime.utcnow().isoformat(),
                "subtask_count": len(optimized_plan.subtasks),
                "complexity_score": state.get("complexity_score", 0.5),
                "optimization_applied": optimized_plan != execution_plan
            }
            
            logger.info(f"Planning completed: {len(optimized_plan.subtasks)} subtasks generated")
            return state
            
        except Exception as e:
            logger.error(f"Error in planning agent: {e}")
            state["errors"] = state.get("errors", [])
            state["errors"].append(f"Planning error: {str(e)}")
            return state
    
    def can_handle(self, task: SubTask) -> bool:
        """
        Determine if this agent can handle the given task.
        
        Args:
            task: Task to evaluate
            
        Returns:
            True if task is related to planning or analysis
        """
        if not task or not task.task_type:
            return False
            
        planning_types = ["analyze", "plan", "decompose", "optimize"]
        task_type = task.task_type.lower() if task.task_type else ""
        return any(planning_type in task_type for planning_type in planning_types)
    
    async def _optimize_plan(self, plan: Any, state: AgentState) -> Any:
        """
        Optimize the execution plan for better performance.
        
        Args:
            plan: Original execution plan
            state: Current agent state
            
        Returns:
            Optimized execution plan
        """
        try:
            # Check if optimization is needed
            if len(plan.subtasks) <= 2:
                return plan
            
            # Identify parallelizable tasks
            parallelizable = []
            sequential = []
            
            for subtask in plan.subtasks:
                if self._can_parallelize(subtask, plan.dependencies):
                    parallelizable.append(subtask)
                else:
                    sequential.append(subtask)
            
            # Reorder tasks for optimal execution
            optimized_subtasks = []
            
            # Add independent tasks first (can run in parallel)
            optimized_subtasks.extend(parallelizable)
            
            # Add dependent tasks in order
            optimized_subtasks.extend(sequential)
            
            # Create optimized plan
            optimized_plan = plan.__class__(
                user_id=plan.user_id,
                query=plan.query,
                subtasks=optimized_subtasks,
                dependencies=plan.dependencies,
                metadata={
                    **plan.metadata,
                    "optimized": True,
                    "optimization_time": datetime.utcnow().isoformat()
                }
            )
            
            return optimized_plan
            
        except Exception as e:
            logger.error(f"Error optimizing plan: {e}")
            return plan
    
    def _can_parallelize(self, subtask: SubTask, dependencies: Dict[str, List[str]]) -> bool:
        """
        Check if a subtask can be executed in parallel.
        
        Args:
            subtask: Subtask to check
            dependencies: Task dependencies mapping
            
        Returns:
            True if task can be parallelized
        """
        # Tasks with no dependencies can be parallelized
        task_deps = dependencies.get(subtask.id, [])
        return len(task_deps) == 0
    
    async def estimate_complexity(self, query: str) -> float:
        """
        Estimate the complexity of a query.
        
        Args:
            query: User query to analyze
            
        Returns:
            Complexity score between 0.0 and 1.0
        """
        try:
            return 0.5
        except Exception as e:
            logger.error(f"Error estimating complexity: {e}")
            return 0.5
    
    def get_planning_metrics(self) -> Dict[str, Any]:
        """
        Get planning performance metrics.
        
        Returns:
            Dictionary containing planning metrics
        """
        return {
            "agent_type": "planning",
            "role": self.role.value,
            "max_subtasks": self.max_subtasks,
            "complexity_threshold": self.complexity_threshold,
            "capabilities": [
                "query_analysis",
                "execution_planning", 
                "plan_optimization",
                "complexity_estimation"
            ]
        }
