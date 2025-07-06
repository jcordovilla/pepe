"""
Task Planner

Creates execution plans by decomposing complex queries into manageable subtasks
and determining the optimal execution strategy.
"""

import uuid
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging
import time

from ..agents.base_agent import SubTask, ExecutionPlan, AgentRole, TaskStatus

logger = logging.getLogger(__name__)


class TaskPlanner:
    """
    Creates execution plans for complex queries by decomposing them into
    subtasks and determining execution dependencies and priorities.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.max_subtasks = config.get("max_subtasks", 10)
        self.timeout_seconds = config.get("timeout_seconds", 30)
        self.default_k = config.get("default_k", 10)
        
        # Task templates for common patterns
        self.task_templates = {
            "capability": [
                {"role": AgentRole.ANALYZER, "description": "Generate capability information", "task_type": "capability_response"}
            ],
            "search": [
                {"role": AgentRole.SEARCHER, "description": "Search for relevant messages"}
            ],
            "summarize": [
                {"role": AgentRole.SEARCHER, "description": "Retrieve messages in time range"},
                {"role": AgentRole.ANALYZER, "description": "Summarize retrieved messages"}
            ],
            "digest": [
                {"role": AgentRole.SEARCHER, "description": "Retrieve messages for digest period"},
                {"role": AgentRole.ANALYZER, "description": "Generate comprehensive digest"}
            ],
            "weekly_digest": [
                {"role": AgentRole.SEARCHER, "description": "Retrieve last week's messages", "task_type": "filtered_search"},
                {"role": AgentRole.ANALYZER, "description": "Generate weekly digest", "task_type": "weekly_digest"}
            ],
            "monthly_digest": [
                {"role": AgentRole.SEARCHER, "description": "Retrieve last month's messages", "task_type": "filtered_search"},
                {"role": AgentRole.ANALYZER, "description": "Generate monthly digest", "task_type": "monthly_digest"}
            ],
            "analyze": [
                {"role": AgentRole.SEARCHER, "description": "Search for relevant data"},
                {"role": AgentRole.ANALYZER, "description": "Analyze patterns and insights"}
            ],
            "resource_search": [
                {"role": AgentRole.RESOURCE_MANAGER, "description": "Search for shared resources"},
                {"role": AgentRole.ANALYZER, "description": "Classify and organize resources"}
            ],
            "data_availability": [
                {"role": AgentRole.SEARCHER, "description": "Query data availability metrics"}
            ],
            "reactions": [
                {"role": AgentRole.SEARCHER, "description": "Search for messages with reactions", "task_type": "reaction_search"}
            ]
        }
        
        logger.info("Task planner initialized")
    
    async def create_plan(
        self, 
        query: str, 
        analysis: Dict[str, Any], 
        context: Dict[str, Any]
    ) -> ExecutionPlan:
        """
        Create an execution plan based on query analysis.
        
        Args:
            query: User's query
            analysis: Query analysis results
            context: Additional context
            
        Returns:
            Execution plan with subtasks
        """
        try:
            # Generate unique plan ID
            plan_id = f"plan_{int(time.time())}_{uuid.uuid4().hex[:8]}"
            
            # Create subtasks based on intent
            subtasks = []
            
            intent = analysis.get("intent")
            
            if intent == "capability":
                # Handle capability/meta queries
                subtask = SubTask(
                    id=f"capability_{uuid.uuid4().hex[:8]}",
                    description="Generate capability information and help documentation",
                    agent_role=AgentRole.ANALYZER,
                    task_type="capability_response",
                    parameters={
                        "query": query,
                        "response_type": "capability"
                    },
                    dependencies=[],
                    created_at=datetime.utcnow()
                )
                subtasks.append(subtask)
                
            elif intent in self.task_templates:
                # Use template-based planning for known intents
                template = self.task_templates[intent]
                for i, task_def in enumerate(template):
                    subtask = SubTask(
                        id=f"{intent}_{i}_{uuid.uuid4().hex[:8]}",
                        description=task_def["description"],
                        agent_role=task_def["role"],
                        task_type=task_def.get("task_type", intent),
                        parameters=self._build_task_parameters(query, analysis, analysis.get("entities", []), context),
                        dependencies=self._determine_dependencies(i, template),
                        created_at=datetime.utcnow()
                    )
                    subtasks.append(subtask)
                    
            elif analysis.get("intent") == "search":
                # Extract entities and build filters
                entities = analysis.get("entities", [])
                filters = {}
                k = self.default_k
                
                # Build filters from entities
                for entity in entities:
                    entity_type = entity["type"]
                    entity_value = entity["value"]
                    
                    if entity_type == "channel":
                        # Use channel_id if available, resolve to channel name
                        channel_id = entity.get("channel_id")
                        if channel_id:
                            # Import channel resolver to get the full channel name
                            from ..services.channel_resolver import ChannelResolver
                            resolver = ChannelResolver()
                            resolved_name = resolver.resolve_channel_id_to_name(channel_id)
                            
                            if resolved_name:
                                filters["channel_name"] = resolved_name
                                logger.info(f"Resolved channel ID {channel_id} to '{resolved_name}'")
                            else:
                                # Fallback to channel_id filter if resolution fails
                                filters["channel_id"] = channel_id
                                logger.warning(f"Could not resolve channel ID {channel_id}, using as filter")
                        else:
                            filters["channel_name"] = entity_value
                            logger.warning(f"Channel ID not resolved for '{entity_value}', using channel_name filter")
                    elif entity_type == "user":
                        filters["author_username"] = entity_value
                    elif entity_type == "count":
                        try:
                            k = int(entity_value)
                        except (ValueError, TypeError):
                            k = self.default_k
                    elif entity_type == "time_range":
                        start = entity.get("start")
                        end = entity.get("end")
                        if start and end:
                            filters["timestamp"] = {"$gte": start, "$lte": end}
                
                # Determine search type based on query intent
                search_type = "search"  # Default semantic search
                sort_by = None
                
                # Check if this is a temporal query (last X, recent X, etc.)
                query_lower = query.lower()
                temporal_keywords = ["last", "recent", "latest", "newest", "chronological"]
                if any(keyword in query_lower for keyword in temporal_keywords):
                    search_type = "filtered_search"  # Use filtered search for temporal queries
                    sort_by = "timestamp"  # Sort by timestamp (newest first)
                    logger.info(f"Detected temporal query, using filtered search with timestamp sorting")
                
                # Create search subtask
                subtask = SubTask(
                    id=f"search_{uuid.uuid4().hex[:8]}",
                    description="Search for relevant messages",
                    agent_role=AgentRole.SEARCHER,
                    task_type=search_type,  # Use determined search type
                    parameters={
                        "query": query,
                        "filters": filters,
                        "k": k,
                        "sort_by": sort_by  # Add sorting parameter
                    },
                    dependencies=[],
                    created_at=datetime.utcnow()
                )
                subtasks.append(subtask)
            
            # Create execution plan
            plan = ExecutionPlan(
                id=plan_id,
                query=query,
                subtasks=subtasks,
                created_at=datetime.utcnow()
            )
            
            logger.info(f"Created execution plan with {len(subtasks)} subtasks for intent: {analysis.get('intent')}")
            return plan
            
        except Exception as e:
            logger.error(f"Error creating execution plan: {e}")
            raise
    
    def _build_task_parameters(
        self, 
        query: str, 
        analysis: Dict[str, Any], 
        entities: List[Dict[str, Any]], 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build parameters for task execution"""
        parameters = {
            "query": query,
            "original_analysis": analysis,
            "context": context
        }
        
        # Extract specific parameters from entities
        for entity in entities:
            entity_type = entity["type"]
            entity_value = entity["value"]
            
            if entity_type == "channel":
                parameters["channel_name"] = entity_value.lstrip("#")
            elif entity_type == "user":
                parameters["author_name"] = entity_value.lstrip("@")
            elif entity_type == "keyword":
                parameters["keyword"] = entity_value.strip("'\"")
            elif entity_type == "count":
                try:
                    parameters["k"] = int(entity_value)
                except (ValueError, TypeError):
                    parameters["k"] = 5
            elif entity_type == "time_range":
                parameters["time_filter"] = entity_value
        
        # Set defaults
        parameters.setdefault("k", 5)
        parameters.setdefault("as_json", False)
        
        return parameters
    
    def _determine_dependencies(self, task_index: int, template: List[Dict[str, Any]]) -> List[str]:
        """Determine task dependencies based on position and type"""
        dependencies = []
        
        # Simple linear dependency for now
        # More sophisticated dependency analysis can be added
        if task_index > 0:
            # Depend on the previous task
            dependencies.append(f"task_{task_index - 1}")
        
        return dependencies
    
    async def _handle_complex_query(
        self, 
        query: str, 
        analysis: Dict[str, Any], 
        plan_id: str
    ) -> List[SubTask]:
        """Handle complex queries that might need additional subtasks"""
        additional_tasks = []
        
        # Check for multiple intents or sub-queries
        sub_queries = analysis.get("sub_queries", [])
        if sub_queries:
            for i, sub_query in enumerate(sub_queries):
                task = SubTask(
                    id=f"{plan_id}_sub_{i}",
                    description=f"Process sub-query: {sub_query}",
                    agent_role=AgentRole.SEARCHER,
                    parameters={"query": sub_query},
                    dependencies=[]
                )
                additional_tasks.append(task)
        
        # Add synthesis task if multiple subtasks
        if len(additional_tasks) > 1:
            synthesis_task = SubTask(
                id=f"{plan_id}_synthesis",
                description="Synthesize results from multiple subtasks",
                agent_role=AgentRole.ANALYZER,
                parameters={"synthesis_mode": True},
                dependencies=[task.id for task in additional_tasks]
            )
            additional_tasks.append(synthesis_task)
        
        return additional_tasks
    
    async def optimize_plan(self, plan: ExecutionPlan) -> ExecutionPlan:
        """Optimize the execution plan for better performance"""
        try:
            # Remove redundant tasks
            optimized_subtasks = []
            seen_descriptions = set()
            
            for subtask in plan.subtasks:
                if subtask.description not in seen_descriptions:
                    optimized_subtasks.append(subtask)
                    seen_descriptions.add(subtask.description)
            
            # Reorder tasks to minimize dependencies
            optimized_subtasks = self._reorder_by_dependencies(optimized_subtasks)
            
            plan.subtasks = optimized_subtasks
            logger.info(f"Optimized plan: {len(optimized_subtasks)} tasks (was {len(plan.subtasks)})")
            
            return plan
            
        except Exception as e:
            logger.error(f"Error optimizing plan: {str(e)}")
            return plan
    
    def _reorder_by_dependencies(self, subtasks: List[SubTask]) -> List[SubTask]:
        """Reorder subtasks to respect dependencies"""
        # Simple topological sort for task dependencies
        ordered = []
        remaining = subtasks.copy()
        
        while remaining:
            # Find tasks with no unmet dependencies
            ready_tasks = []
            for task in remaining:
                if not task.dependencies or all(
                    dep_id in [t.id for t in ordered] for dep_id in task.dependencies
                ):
                    ready_tasks.append(task)
            
            if not ready_tasks:
                # Break circular dependencies by taking the first task
                ready_tasks = [remaining[0]]
            
            # Add ready tasks to ordered list
            for task in ready_tasks:
                ordered.append(task)
                remaining.remove(task)
        
        return ordered
    
    def estimate_execution_time(self, plan: ExecutionPlan) -> float:
        """Estimate total execution time for the plan"""
        # Simple estimation based on task count and type
        base_time_per_task = 2.0  # seconds
        
        total_time = 0
        for subtask in plan.subtasks:
            if subtask.agent_role == AgentRole.SEARCHER:
                total_time += base_time_per_task
            elif subtask.agent_role == AgentRole.ANALYZER:
                total_time += base_time_per_task * 1.5
            elif subtask.agent_role == AgentRole.RESOURCE_MANAGER:
                total_time += base_time_per_task * 0.8
            else:
                total_time += base_time_per_task
        
        return total_time
