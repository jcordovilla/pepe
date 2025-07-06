"""
Agent Orchestrator

Coordinates between multiple specialized agents to handle complex queries.
Uses LangGraph for state management and workflow orchestration.
"""

import asyncio
import time
from typing import Dict, List, Any, Optional, Set
from datetime import datetime
import logging

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from .base_agent import BaseAgent, AgentRole, AgentState, SubTask, ExecutionPlan, TaskStatus, agent_registry
from ..memory.conversation_memory import ConversationMemory
from .query_interpreter_agent import QueryInterpreterAgent
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
        self.query_interpreter_agent = QueryInterpreterAgent(config.get("query_interpreter", {}))
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
        """Interpret the user query using LLM to understand intent and extract entities"""
        try:
            # Use QueryInterpreterAgent to interpret the query
            interpretation_state = await self.query_interpreter_agent.process(state)
            
            # Extract interpretation results
            interpretation = interpretation_state.get("query_interpretation", {})
            
            # Update state with interpretation results
            state["query_interpretation"] = interpretation
            state["metadata"]["intent"] = interpretation.get("intent", "unknown")
            state["metadata"]["entities"] = interpretation.get("entities", [])
            state["metadata"]["interpretation_confidence"] = interpretation.get("confidence", 0.0)
            state["metadata"]["interpretation_rationale"] = interpretation.get("rationale", "")
            
            logger.info(f"Query interpreted: intent={interpretation.get('intent')}, confidence={interpretation.get('confidence', 0.0)}")
            return state
            
        except Exception as e:
            logger.error(f"Error in query interpretation: {str(e)}")
            state["metadata"]["interpretation_error"] = str(e)
            return state
    
    async def _plan_execution_node(self, state: AgentState) -> AgentState:
        """Create an execution plan based on LLM query interpretation"""
        try:
            query = state["user_context"]["query"]
            # Use the interpretation results instead of analysis
            interpretation_data = {
                "query_interpretation": state.get("query_interpretation", {})
            }
            
            # Create execution plan using interpretation
            plan = await self.task_planner.create_plan(query, interpretation_data, state["user_context"])
            state["task_plan"] = plan
            
            logger.info(f"Execution plan created with {len(plan.subtasks)} subtasks from LLM interpretation")
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

            async def _run_subtask(subtask: SubTask):
                """Execute a single subtask in isolation."""
                agent = agent_registry.find_capable_agent(subtask)
                if not agent:
                    logger.warning(f"No agent found for task: {subtask.description}")
                    subtask.status = TaskStatus.FAILED
                    subtask.error = "No capable agent found"
                    return subtask.id, None, []

                try:
                    task_state = state.copy()
                    task_state["current_subtask"] = subtask
                    result_state = await agent.process(task_state)

                    subtask.status = TaskStatus.COMPLETED
                    subtask.result = result_state.get("task_result")
                    extra_results = result_state.get("search_results", [])
                    
                    # Extract analysis results if present
                    analysis_results = result_state.get("analysis_results", {})
                    if analysis_results:
                        state["analysis_results"] = state.get("analysis_results", {})
                        state["analysis_results"].update(analysis_results)
                    
                    return subtask.id, subtask.result, extra_results
                except Exception as e:
                    logger.error(f"Error executing subtask {subtask.id}: {str(e)}")
                    subtask.status = TaskStatus.FAILED
                    subtask.error = str(e)
                    return subtask.id, None, []

            completed: Set[str] = set()
            result_map: Dict[str, Any] = {}

            while len(completed) < len(plan.subtasks):
                ready = [
                    t for t in plan.subtasks
                    if t.id not in completed and all(dep in completed for dep in t.dependencies)
                ]

                if not ready:
                    # Prevent deadlock on circular dependencies
                    ready = [next(t for t in plan.subtasks if t.id not in completed)]

                gather_results = await asyncio.gather(*[_run_subtask(t) for t in ready])

                for sid, res, extras in gather_results:
                    completed.add(sid)
                    result_map[sid] = res
                    if extras:
                        state["search_results"].extend(extras)

            for task in plan.subtasks:
                results.append(result_map.get(task.id))

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
            # Ensure all collections are properly initialized
            search_results = state.get("search_results", []) or []
            plan = state.get("task_plan")
            query = state.get("user_context", {}).get("query", "")
            analysis_results = state.get("analysis_results", {}) or {}
            
            # Check if this is a capability response
            if "capability_response" in analysis_results:
                capability_data = analysis_results["capability_response"]
                if capability_data and isinstance(capability_data, dict):
                    state["response"] = capability_data.get("capability_response", "I can help you search and analyze Discord conversations.")
                else:
                    state["response"] = "I can help you search and analyze Discord conversations."
                logger.info("Capability response synthesized successfully")
                return state
            
            # Handle case where no results or plan exists
            if not search_results and not plan and not analysis_results:
                state["response"] = "I can help you search and analyze Discord conversations. What would you like to know about?"
                return state
            
            # Return the actual search results for the agent API to process
            if search_results and len(search_results) > 0:
                # Create a response that includes both summary and results
                result_count = len(search_results)
                if result_count == 1:
                    summary = "I found 1 relevant result for your query."
                else:
                    summary = f"I found {result_count} relevant results for your query."
                
                # The response will be used as the answer, and search_results contains the actual data
                state["response"] = summary
                # Ensure search_results are preserved for the agent API
                logger.info(f"Results synthesized successfully: {result_count} results")
            else:
                # Provide a helpful response when no specific results are found
                state["response"] = "I can help you search through Discord conversations. Try asking me to find specific topics, users, or messages. For example: 'Find messages about Python programming' or 'Show me recent discussions about AI'."
            
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
