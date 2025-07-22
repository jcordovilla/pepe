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
                    "version": "1.0.0"
                },
                "response": None,
                "query_analysis": None,
                "query_interpretation": {},
                "intent": None,
                "entities": None,
                "complexity_score": None,
                "execution_plan": None,
                "subtasks": None,
                "dependencies": None
            }
            
            logger.info("Starting LangGraph workflow execution...")
            
            # Execute workflow
            config = {"configurable": {"thread_id": f"user_{user_id}"}}
            try:
                # Add timeout to prevent hanging
                final_state = await asyncio.wait_for(
                    self.app.ainvoke(initial_state, config),
                    timeout=30.0  # 30 second timeout
                )
                logger.info("LangGraph workflow completed successfully")
            except asyncio.TimeoutError:
                logger.error("LangGraph workflow timed out after 30 seconds")
                return {
                    "response": "I'm taking too long to process your query. Please try again with a simpler question.",
                    "results": [],
                    "metadata": {
                        "duration": time.time() - start_time,
                        "success": False,
                        "error": "Workflow timeout"
                    }
                }
            except Exception as workflow_error:
                logger.error(f"LangGraph workflow failed: {workflow_error}")
                # Return a fallback response
                return {
                    "response": f"I encountered an error in the workflow: {str(workflow_error)}",
                    "results": [],
                    "metadata": {
                        "duration": time.time() - start_time,
                        "success": False,
                        "error": str(workflow_error)
                    }
                }
            
            logger.info("LangGraph workflow completed, processing results...")
            
            # Debug: Check what's in the final state
            logger.info(f"Final state keys: {list(final_state.keys())}")
            logger.info(f"Final state response: {final_state.get('response', 'NO_RESPONSE')}")
            logger.info(f"Final state search_results: {len(final_state.get('search_results', []))} results")
            logger.info(f"Final state subtask_results: {list(final_state.get('subtask_results', {}).keys())}")
            
            # Store conversation in memory
            response_text = final_state.get("response", "") or "No response generated"
            # Ensure response_text is a string, not a dict
            if isinstance(response_text, dict):
                response_text = str(response_text)
            elif not isinstance(response_text, str):
                response_text = str(response_text)
            
            await self.conversation_memory.add_interaction(
                user_id, query, response_text
            )
            
            duration = time.time() - start_time
            logger.info(f"Query processed successfully in {duration:.3f}s")
            
            # Prepare subtask execution summary for output
            subtask_summaries = []
            subtask_results = final_state.get("subtask_results", {})
            for subtask_id, result in subtask_results.items():
                subtask_obj = result.get("current_subtask", {})
                # Handle both dataclass and dict
                if hasattr(subtask_obj, "task_type"):
                    task_type = getattr(subtask_obj, "task_type", "unknown")
                    description = getattr(subtask_obj, "description", "unknown")
                else:
                    task_type = subtask_obj.get("task_type", "unknown")
                    description = subtask_obj.get("description", "unknown")
                summary = {
                    "id": subtask_id,
                    "type": task_type,
                    "description": description,
                    "status": result.get("status", "unknown"),
                    "error": result.get("error", None)
                }
                subtask_summaries.append(summary)

            return {
                "response": final_state.get("response", ""),
                "results": final_state.get("search_results", []),
                "subtasks": subtask_summaries,
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
            logger.info("Starting query analysis node...")
            
            # Use QueryInterpreterAgent to interpret the query
            logger.info("Calling QueryInterpreterAgent.process()...")
            interpretation_state = await self.query_interpreter_agent.process(state)
            
            # Extract interpretation results
            interpretation = interpretation_state.get("query_interpretation", {})
            logger.info(f"Received interpretation from agent: {interpretation.get('intent', 'unknown')} with {len(interpretation.get('subtasks', []))} subtasks")
            
            # Debug: Log the interpretation details
            logger.info(f"Analyze query node - interpretation: {interpretation}")
            logger.info(f"Analyze query node - interpretation keys: {list(interpretation.keys()) if interpretation else 'None'}")
            
            # Update state with interpretation results
            state["query_interpretation"] = interpretation
            state["metadata"]["intent"] = interpretation.get("intent", "unknown")
            state["metadata"]["entities"] = interpretation.get("entities", [])
            state["metadata"]["interpretation_confidence"] = interpretation.get("confidence", 0.0)
            state["metadata"]["interpretation_rationale"] = interpretation.get("rationale", "")
            
            # Debug: Log what we're putting in state
            logger.info(f"Analyze query node - state['query_interpretation'] after update: {state.get('query_interpretation', {})}")
            logger.info(f"Analyze query node - state['query_interpretation'] keys: {list(state.get('query_interpretation', {}).keys()) if state.get('query_interpretation') else 'None'}")
            
            logger.info(f"Query analysis completed: intent={interpretation.get('intent')}, confidence={interpretation.get('confidence', 0.0)}")
            logger.info("Analyze query node completed successfully")
            return state
            
        except Exception as e:
            logger.error(f"Error in query interpretation: {str(e)}")
            state["metadata"]["interpretation_error"] = str(e)
            # Set a fallback interpretation to avoid empty query_interpretation
            logger.warning("Setting fallback interpretation in orchestrator due to error")
            state["query_interpretation"] = {
                "intent": "search",
                "entities": [],
                "subtasks": [
                    {
                        "task_type": "semantic_search",
                        "description": "Search for relevant messages",
                        "parameters": {
                            "query": state.get("user_context", {}).get("query", ""),
                            "filters": {},
                            "k": 10
                        },
                        "dependencies": []
                    }
                ],
                "confidence": 0.5,
                "rationale": "Fallback interpretation due to error"
            }
            logger.info("Analyze query node completed with fallback")
            return state
    
    async def _plan_execution_node(self, state: AgentState) -> AgentState:
        """Create an execution plan based on LLM query interpretation"""
        try:
            logger.info("Starting plan execution node...")
            query = state["user_context"]["query"]
            
            # Debug: Log what's in the state
            query_interpretation = state.get("query_interpretation", {})
            logger.info(f"Plan execution node - query_interpretation from state: {query_interpretation}")
            logger.info(f"Plan execution node - query_interpretation keys: {list(query_interpretation.keys()) if query_interpretation else 'None'}")
            
            # Use the interpretation results instead of analysis
            interpretation_data = {
                "query_interpretation": query_interpretation
            }
            
            logger.info(f"Plan execution node - interpretation_data being passed to task_planner: {interpretation_data}")
            
            # Create execution plan using interpretation
            plan = await self.task_planner.create_plan(query, interpretation_data, state["user_context"])
            state["task_plan"] = plan
            
            logger.info(f"Execution plan created with {len(plan.subtasks)} subtasks from LLM interpretation")
            
            # Debug: Log the plan details
            if plan and plan.subtasks:
                for i, subtask in enumerate(plan.subtasks):
                    logger.info(f"  Plan Subtask {i+1}: {subtask.task_type} - {subtask.description}")
            
            logger.info("Plan execution node completed successfully")
            return state
            
        except Exception as e:
            logger.error(f"Error in execution planning: {str(e)}")
            state["metadata"]["planning_error"] = str(e)
            logger.info("Plan execution node completed with error")
            return state
    
    async def _execute_plan_node(self, state: AgentState) -> AgentState:
        """Execute the planned subtasks using appropriate agents with shared state and error recovery"""
        try:
            logger.info("Starting execute plan node...")
            plan = state["task_plan"]
            if not plan:
                logger.warning("No execution plan found")
                logger.info("Execute plan node completed with no plan")
                return state
            
            # Extract subtasks from plan and add to state for agent access
            subtasks = plan.subtasks if plan.subtasks else []
            state["subtasks"] = subtasks
            
            logger.info(f"Executing plan with {len(subtasks)} subtasks")
            
            # Debug: Log the subtasks being executed
            for i, subtask in enumerate(subtasks):
                logger.info(f"  Subtask {i+1}: {subtask.task_type} - {subtask.description}")
            
            # Initialize service container if not already done
            from ..services.service_container import get_service_container
            service_container = get_service_container()
            
            # Ensure services are initialized
            if not hasattr(service_container, '_initialized') or not service_container._initialized:
                logger.info("Initializing service container...")
                await service_container.initialize_services()
            
            # Initialize shared state for coordination
            from .shared_state import SharedAgentState, StateUpdateType
            shared_state = SharedAgentState(state)

            # Initialize error recovery agent
            from .error_recovery_agent import ErrorRecoveryAgent
            recovery_agent = ErrorRecoveryAgent(self.config.get("error_recovery", {}))

            results = []
            completed: Set[str] = set()
            failed_subtasks: List[SubTask] = []
            result_map: Dict[str, Any] = {}

            async def _run_subtask(subtask: SubTask):
                """Execute a single subtask with shared state and error recovery."""
                try:
                    # Get appropriate agent for this subtask
                    agent = self._get_agent_for_subtask(subtask)
                    # Inject shared services into agent
                    service_container.inject_services(agent)
                    # Get context-rich state for this subtask
                    task_state = await shared_state.propagate_to_subtask(subtask.id)
                    # Set current_subtask in the state
                    task_state["current_subtask"] = subtask
                    # Execute subtask
                    result_state = await agent.process(task_state)
                    # Update shared state with results
                    if result_state.get("search_results"):
                        await shared_state.merge_results(
                            {"search_results": result_state["search_results"]},
                            f"{type(agent).__name__}",
                            StateUpdateType.SEARCH_RESULTS
                        )
                    if result_state.get("analysis_results"):
                        await shared_state.merge_results(
                            {"analysis_results": result_state["analysis_results"]},
                            f"{type(agent).__name__}",
                            StateUpdateType.ANALYSIS_RESULTS
                        )
                    # Always store a complete result dict for this subtask
                    result_map[subtask.id] = {
                        "current_subtask": subtask,
                        "status": "success",
                        "error": None,
                        "result": result_state.get("search_results") or result_state.get("analysis_results") or result_state.get("digest_results") or result_state.get("response")
                    }
                    completed.add(subtask.id)
                    # Clean up agent resources
                    await agent.cleanup()
                    logger.info(f"Subtask {subtask.id} completed successfully")
                except Exception as e:
                    logger.error(f"Subtask {subtask.id} failed: {e}")
                    failed_subtasks.append(subtask)
                    result_map[subtask.id] = {
                        "current_subtask": subtask,
                        "status": "failed",
                        "error": str(e),
                        "result": None
                    }

            # Execute subtasks with dependency management
            while len(completed) < len(subtasks):
                # Find ready subtasks (dependencies satisfied)
                ready = [
                    subtask for subtask in subtasks 
                    if subtask.id not in completed and 
                    all(dep in completed for dep in subtask.dependencies)
                ]
                
                if not ready:
                    # Check for circular dependencies
                    remaining = [s.id for s in subtasks if s.id not in completed]
                    logger.warning(f"No ready subtasks, remaining: {remaining}")
                    break
                
                # Execute ready subtasks
                tasks = [asyncio.create_task(_run_subtask(subtask)) for subtask in ready]
                completed_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Handle results
                for i, result in enumerate(completed_results):
                    if isinstance(result, Exception):
                        logger.error(f"Subtask {ready[i].id} failed with exception: {result}")
                        failed_subtasks.append(ready[i])
                        result_map[ready[i].id] = {"error": str(result), "status": "failed"}

            # Handle failed subtasks with error recovery
            if failed_subtasks:
                logger.info(f"Attempting recovery for {len(failed_subtasks)} failed subtasks")
                
                recovery_state = AgentState({
                    "failed_subtasks": failed_subtasks,
                    "shared_state": shared_state,
                    "result_map": result_map
                })
                
                recovery_result = await recovery_agent.process(recovery_state)
                
                # Update subtask statuses based on recovery result
                recovered_subtasks = recovery_result.get("recovered_subtasks", [])
                for subtask in recovered_subtasks:
                    if subtask.id in [s.id for s in failed_subtasks]:
                        failed_subtasks = [s for s in failed_subtasks if s.id != subtask.id]
                        completed.add(subtask.id)

            # Update state with final results
            state["subtask_results"] = result_map
            state["completed_subtasks"] = list(completed)
            state["failed_subtasks"] = [s.id for s in failed_subtasks if s.status == TaskStatus.FAILED]
            state["shared_state_summary"] = await shared_state.validate_state()
            
            # Add performance metrics
            state["performance_metrics"] = {
                "total_subtasks": len(subtasks),
                "completed_count": len(completed),
                "failed_count": len(failed_subtasks),
                "success_rate": len(completed) / len(subtasks) if subtasks else 0,
                "recovery_attempts": len(recovery_result.get("recovery_attempts", [])) if failed_subtasks else 0
            }
            
            logger.info(f"Plan execution completed: {len(completed)}/{len(subtasks)} subtasks successful")
            logger.info("Execute plan node completed successfully")
            return state
            
        except Exception as e:
            logger.error(f"Error in plan execution: {e}")
            state["error"] = str(e)
            state["status"] = "failed"
            logger.info("Execute plan node completed with error")
            return state
    
    async def _synthesize_results_node(self, state: AgentState) -> AgentState:
        """Synthesize final results from all subtasks using result aggregator"""
        try:
            logger.info("Starting synthesize results node...")
            plan = state["task_plan"]
            if not plan:
                logger.warning("No execution plan found for synthesis")
                logger.info("Synthesize results node completed with no plan")
                return state
            
            # Use result aggregator to properly combine all results
            from .result_aggregator import ResultAggregator
            
            # Create aggregation subtask
            aggregation_subtask = SubTask(
                id="final_aggregation",
                description="Aggregate all results from completed subtasks",
                agent_role=AgentRole.ANALYZER,
                task_type="aggregate_results",
                parameters={},
                dependencies=[],
                status=TaskStatus.PENDING
            )
            
            # Set up aggregation state
            aggregation_state = state.copy()
            # Ensure all required keys are present for state validation
            if "query_interpretation" not in aggregation_state:
                aggregation_state["query_interpretation"] = state.get("query_interpretation", {})
            aggregation_state["current_subtask"] = aggregation_subtask
            
            # Initialize result aggregator
            aggregator = ResultAggregator({})
            
            # Perform aggregation
            aggregation_result = await aggregator.process(aggregation_state)
            
            # Extract final response
            final_response = aggregation_result.get("response", "No response generated")
            aggregated_results = aggregation_result.get("metadata", {}).get("result_aggregator", {})
            
            # Update state with final results
            state["response"] = final_response
            state["aggregated_results"] = aggregated_results
            
            # Validate state consistency
            from .shared_state import SharedAgentState
            
            # Ensure query_interpretation exists for validation
            validation_state = state.copy()
            if "query_interpretation" not in validation_state:
                validation_state["query_interpretation"] = validation_state.get("query_interpretation", {})
            
            shared_state = SharedAgentState(validation_state)
            validation = await shared_state.validate_state()
            
            if not validation["is_valid"]:
                logger.warning(f"State validation issues: {validation['errors']}")
                state["validation_warnings"] = validation["warnings"]
            elif validation.get("warnings"):
                logger.info(f"State validation warnings: {validation['warnings']}")
                state["validation_warnings"] = validation["warnings"]
            
            logger.info(f"Results synthesis completed: response generated with {len(final_response)} characters")
            logger.info("Synthesize results node completed successfully")
            return state
            
        except Exception as e:
            logger.error(f"Error in results synthesis: {str(e)}")
            state["errors"] = state.get("errors", [])
            state["errors"].append(f"Synthesis error: {str(e)}")
            logger.info("Synthesize results node completed with error")
            return state
    
    async def get_conversation_history(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get conversation history for a user"""
        return await self.conversation_memory.get_history(user_id, limit)
    
    async def clear_conversation_history(self, user_id: str):
        """Clear conversation history for a user"""
        await self.conversation_memory.clear_history(user_id)

    def _get_agent_for_subtask(self, subtask: SubTask) -> BaseAgent:
        """Get the appropriate agent for a given subtask."""
        try:
            # Use agent registry to find capable agent
            agent = agent_registry.find_capable_agent(subtask)
            if not agent:
                logger.warning(f"No agent found for task: {subtask.description}")
                # Return a default agent or raise exception
                raise ValueError(f"No capable agent found for subtask: {subtask.id}")
            
            return agent
            
        except Exception as e:
            logger.error(f"Error getting agent for subtask {subtask.id}: {e}")
            raise
