"""
Agent Orchestrator

Coordinates between multiple specialized agents to handle complex queries.
Uses LangGraph for state management and workflow orchestration.
"""

import asyncio
import time
from typing import Dict, List, Any, Optional
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from .base_agent import BaseAgent, AgentRole, AgentState, SubTask, ExecutionPlan, TaskStatus, agent_registry
from ..memory.conversation_memory import ConversationMemory
from ..reasoning.query_analyzer import QueryAnalyzer
from ..reasoning.task_planner import TaskPlanner

logger = logging.getLogger(__name__)


class AgentOrchestrator:
    """
    Main orchestrator that coordinates between specialized agents.
    
    Uses LangGraph to manage stateful workflows and agent coordination.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.conversation_memory = ConversationMemory(config.get("memory", {}))
        self.query_analyzer = QueryAnalyzer(config.get("query_analyzer", {}))
        self.task_planner = TaskPlanner(config.get("task_planner", {}))
        
        # Initialize LangGraph workflow
        self.checkpointer = MemorySaver()
        self.workflow = self._build_workflow()
        self.app = self.workflow.compile(checkpointer=self.checkpointer)
        
        logger.info("Agent Orchestrator initialized")
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow for agent coordination"""
        workflow = StateGraph(AgentState)
        
        # Add nodes for each step in the process
        workflow.add_node("analyze_query", self._analyze_query_node)
        workflow.add_node("plan_execution", self._plan_execution_node)
        workflow.add_node("execute_plan", self._execute_plan_node)
        workflow.add_node("synthesize_results", self._synthesize_results_node)
        
        # Define the flow
        workflow.set_entry_point("analyze_query")
        workflow.add_edge("analyze_query", "plan_execution")
        workflow.add_edge("plan_execution", "execute_plan")
        workflow.add_edge("execute_plan", "synthesize_results")
        workflow.add_edge("synthesize_results", END)
        
        return workflow
    
    async def process_query(self, query: str, user_id: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a user query through the agentic workflow.
        
        Args:
            query: User's natural language query
            user_id: Unique identifier for the user
            context: Additional context (channel, guild, etc.)
            
        Returns:
            Processed response with results and metadata
        """
        start_time = time.time()
        
        try:
            # Initialize state
            initial_state: AgentState = {
                "messages": [{"role": "user", "content": query}],
                "search_results": [],
                "conversation_history": await self.conversation_memory.get_history(user_id),
                "user_context": {
                    "user_id": user_id,
                    "query": query,
                    "timestamp": datetime.utcnow().isoformat(),
                    **(context or {})
                },
                "task_plan": None,
                "current_step": 0,
                "metadata": {
                    "start_time": start_time,
                    "version": "1.0.0"
                }
            }
            
            # Execute workflow
            config = {"configurable": {"thread_id": f"user_{user_id}"}}
            final_state = await self.app.ainvoke(initial_state, config)
            
            # Store conversation in memory
            await self.conversation_memory.add_interaction(
                user_id, query, final_state.get("response", "")
            )
            
            duration = time.time() - start_time
            logger.info(f"Query processed successfully in {duration:.3f}s")
            
            return {
                "response": final_state.get("response", ""),
                "results": final_state.get("search_results", []),
                "metadata": {
                    **final_state.get("metadata", {}),
                    "duration": duration,
                    "success": True
                }
            }
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Error processing query: {str(e)}", exc_info=True)
            
            return {
                "response": f"I encountered an error processing your query: {str(e)}",
                "results": [],
                "metadata": {
                    "duration": duration,
                    "success": False,
                    "error": str(e)
                }
            }
    
    async def _analyze_query_node(self, state: AgentState) -> AgentState:
        """Analyze the user query to understand intent and extract entities"""
        try:
            query = state["user_context"]["query"]
            analysis = await self.query_analyzer.analyze(query, state["user_context"])
            
            state["metadata"]["query_analysis"] = analysis
            state["metadata"]["intent"] = analysis.get("intent", "unknown")
            state["metadata"]["entities"] = analysis.get("entities", [])
            
            logger.info(f"Query analyzed: intent={analysis.get('intent')}")
            return state
            
        except Exception as e:
            logger.error(f"Error in query analysis: {str(e)}")
            state["metadata"]["analysis_error"] = str(e)
            return state
    
    async def _plan_execution_node(self, state: AgentState) -> AgentState:
        """Create an execution plan based on query analysis"""
        try:
            query = state["user_context"]["query"]
            analysis = state["metadata"].get("query_analysis", {})
            
            # Create execution plan
            plan = await self.task_planner.create_plan(query, analysis, state["user_context"])
            state["task_plan"] = plan
            
            logger.info(f"Execution plan created with {len(plan.subtasks)} subtasks")
            return state
            
        except Exception as e:
            logger.error(f"Error in execution planning: {str(e)}")
            state["metadata"]["planning_error"] = str(e)
            return state
    
    async def _execute_plan_node(self, state: AgentState) -> AgentState:
        """Execute the planned subtasks using appropriate agents"""
        try:
            plan = state["task_plan"]
            if not plan:
                logger.warning("No execution plan found")
                return state
            
            results = []
            
            # Execute subtasks in dependency order
            for subtask in plan.subtasks:
                agent = agent_registry.find_capable_agent(subtask)
                if not agent:
                    logger.warning(f"No agent found for task: {subtask.description}")
                    subtask.status = TaskStatus.FAILED
                    subtask.error = "No capable agent found"
                    continue
                
                try:
                    # Prepare task-specific state
                    task_state = state.copy()
                    task_state["current_subtask"] = subtask
                    
                    # Execute subtask
                    result_state = await agent.process(task_state)
                    
                    # Update subtask with results
                    subtask.status = TaskStatus.COMPLETED
                    subtask.result = result_state.get("task_result")
                    results.append(subtask.result)
                    
                    # Merge results back to main state
                    if "search_results" in result_state:
                        state["search_results"].extend(result_state["search_results"])
                    
                except Exception as e:
                    logger.error(f"Error executing subtask {subtask.id}: {str(e)}")
                    subtask.status = TaskStatus.FAILED
                    subtask.error = str(e)
            
            plan.status = TaskStatus.COMPLETED
            state["metadata"]["execution_results"] = results
            
            logger.info(f"Plan execution completed with {len(results)} results")
            return state
            
        except Exception as e:
            logger.error(f"Error in plan execution: {str(e)}")
            state["metadata"]["execution_error"] = str(e)
            return state
    
    async def _synthesize_results_node(self, state: AgentState) -> AgentState:
        """Synthesize results from all subtasks into a coherent response"""
        try:
            search_results = state.get("search_results", [])
            plan = state.get("task_plan")
            
            if not search_results and not plan:
                state["response"] = "I couldn't find any relevant information for your query."
                return state
            
            # Basic synthesis logic - can be enhanced with LLM-based synthesis
            if search_results:
                if len(search_results) == 1:
                    state["response"] = f"I found 1 relevant result for your query."
                else:
                    state["response"] = f"I found {len(search_results)} relevant results for your query."
            else:
                state["response"] = "I processed your query but didn't find specific results."
            
            logger.info("Results synthesized successfully")
            return state
            
        except Exception as e:
            logger.error(f"Error in result synthesis: {str(e)}")
            state["response"] = f"I encountered an error while preparing your response: {str(e)}"
            return state
    
    async def get_conversation_history(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get conversation history for a user"""
        return await self.conversation_memory.get_history(user_id, limit)
    
    async def clear_conversation_history(self, user_id: str):
        """Clear conversation history for a user"""
        await self.conversation_memory.clear_history(user_id)
