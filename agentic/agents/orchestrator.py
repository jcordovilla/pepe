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
from .v2 import register_agents

logger = logging.getLogger(__name__)


class AgentOrchestrator:
    """
    Main orchestrator that coordinates between specialized agents.
    
    Uses LangGraph to manage stateful workflows and agent coordination.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.conversation_memory = ConversationMemory(config.get("memory", {}))
        
        # Initialize v2 agent registry
        self.agent_registry = register_agents()
        
        # Initialize LangGraph workflow
        self.checkpointer = MemorySaver()
        self.workflow = self._build_workflow()
        self.app = self.workflow.compile(checkpointer=self.checkpointer)
        
        logger.info("Agent Orchestrator initialized with v2 agents")
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow for agent coordination"""
        workflow = StateGraph(AgentState)
        
        # Add nodes for each step in the process
        workflow.add_node("route_query", self._route_query_node)
        workflow.add_node("execute_agent", self._execute_agent_node)
        workflow.add_node("validate_result", self._validate_result_node)
        
        # Define the flow
        workflow.set_entry_point("route_query")
        workflow.add_edge("route_query", "execute_agent")
        workflow.add_edge("execute_agent", "validate_result")
        workflow.add_edge("validate_result", END)
        
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
            logger.info(f"Orchestrator processing query: '{query[:50]}...' for user {user_id}")
            
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
                "current_subtask": None,
                "analysis_results": {},
                "errors": [],
                "metadata": {
                    "start_time": start_time,
                    "version": "2.0.0"
                },
                "response": None,
                "query_analysis": {},
                "query_interpretation": {},
                "intent": None,
                "entities": {},
                "complexity_score": None,
                "execution_plan": None,
                "subtasks": [],
                "dependencies": {}
            }
            
            # Execute the workflow with proper configurable parameters for the checkpointer
            configurable = {
                "thread_id": f"discord_{user_id}",
                "thread_ts": str(int(start_time))
            }
            final_state = await self.app.ainvoke(initial_state, {"configurable": configurable})
            
            # Extract results
            response = final_state.get("response", "No response generated")
            errors = final_state.get("errors", [])
            
            # Calculate duration
            duration = time.time() - start_time
            
            # Store conversation turn
            await self.conversation_memory.add_interaction(
                user_id=user_id,
                query=query,
                response=response,
                metadata={
                    "duration": duration,
                    "agent_used": final_state.get("next_agent", "unknown"),
                    "errors": errors
                }
            )
            
            return {
                "response": response,
                "results": final_state.get("analysis_results", {}),
                "metadata": {
                    "duration": duration,
                    "success": len(errors) == 0,
                    "agent_used": final_state.get("next_agent", "unknown"),
                    "errors": errors,
                    "version": "2.0.0"
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
    
    async def _route_query_node(self, state: AgentState) -> AgentState:
        """Route the query to the appropriate agent using the router agent."""
        try:
            query = state["user_context"]["query"]
            user_id = state["user_context"]["user_id"]
            
            logger.info(f"Routing query: {query[:50]}...")
            
            # Use router agent to determine which agent should handle the query
            router_agent_class = self.agent_registry.get("router")
            if not router_agent_class:
                raise ValueError("Router agent not found in registry")
            
            router_agent = router_agent_class(self.config.get("router", {}))
            
            # Route the query
            routing_result = await router_agent.run(
                command=query,
                payload={
                    "user_id": user_id,
                    "context": state["user_context"]
                }
            )
            
            # Update state with routing results
            state["routing_result"] = routing_result
            state["next_agent"] = routing_result.get("next_agent", "qa")
            state["agent_args"] = routing_result.get("args", {})
            
            logger.info(f"Query routed to: {state['next_agent']}")
            
            return state
            
        except Exception as e:
            logger.error(f"Error in query routing: {e}")
            state["errors"].append(f"Query routing failed: {str(e)}")
            # Default to QA agent
            state["next_agent"] = "qa"
            state["agent_args"] = {"query": state["user_context"]["query"]}
            return state
    
    async def _execute_agent_node(self, state: AgentState) -> AgentState:
        """Execute the appropriate agent based on routing results."""
        try:
            next_agent_name = state.get("next_agent", "qa")
            agent_args = state.get("agent_args", {})
            
            logger.info(f"Executing agent: {next_agent_name}")
            
            # Get the agent class from registry
            agent_class = self.agent_registry.get(next_agent_name)
            if not agent_class:
                raise ValueError(f"Agent '{next_agent_name}' not found in registry")
            
            # Create and execute the agent
            agent = agent_class(self.config)
            
            # Inject services from service container if available
            if hasattr(self, 'service_container'):
                self.service_container.inject_services(agent)
            
            # Execute the agent
            result = await agent.run(**agent_args)
            
            # Update state with agent result
            state["agent_result"] = result
            state["response"] = result if isinstance(result, str) else str(result)
            
            logger.info(f"Agent {next_agent_name} execution completed")
            
            return state
            
        except Exception as e:
            logger.error(f"Error in agent execution: {e}")
            state["errors"].append(f"Agent execution failed: {str(e)}")
            state["response"] = f"Sorry, I encountered an error while processing your request: {str(e)}"
            return state
    
    async def _validate_result_node(self, state: AgentState) -> AgentState:
        """Validate the result using the self-check agent."""
        try:
            response = state.get("response", "")
            if not response:
                return state
            
            logger.info("Validating result with self-check agent")
            
            # Get self-check agent
            selfcheck_agent_class = self.agent_registry.get("selfcheck")
            if not selfcheck_agent_class:
                logger.warning("Self-check agent not found, skipping validation")
                return state
            
            selfcheck_agent = selfcheck_agent_class(self.config)
            
            # Inject services from service container if available
            if hasattr(self, 'service_container'):
                self.service_container.inject_services(selfcheck_agent)
            
            # Prepare context for validation
            context = {
                "query": state["user_context"]["query"],
                "agent_used": state.get("next_agent", "unknown"),
                "agent_args": state.get("agent_args", {}),
                "search_results": state.get("search_results", [])
            }
            
            # Validate the result
            validation_passes = await selfcheck_agent.run(
                text=response,
                context=context
            )
            
            # Update state with validation results
            state["validation_passes"] = validation_passes
            
            if not validation_passes:
                logger.warning("Result failed validation, adding warning")
                state["response"] = f"{response}\n\n*Note: This response may need verification.*"
            
            logger.info(f"Validation completed: passes={validation_passes}")
            
            return state
            
        except Exception as e:
            logger.error(f"Error in result validation: {e}")
            # Don't fail the workflow for validation errors
            state["validation_error"] = str(e)
            return state
    
    async def get_conversation_history(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get conversation history for a user."""
        return await self.conversation_memory.get_history(user_id, limit)
    
    async def clear_conversation_history(self, user_id: str):
        """Clear conversation history for a user."""
        await self.conversation_memory.clear_history(user_id)
