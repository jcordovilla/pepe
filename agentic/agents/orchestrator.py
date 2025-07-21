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
            final_state = await self.app.ainvoke(initial_state, config)
            
            logger.info("LangGraph workflow completed, processing results...")
            
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
            logger.info("Starting query analysis node...")
            
            # Use QueryInterpreterAgent to interpret the query
            logger.info("Calling QueryInterpreterAgent.process()...")
            interpretation_state = await self.query_interpreter_agent.process(state)
            
            # Extract interpretation results
            interpretation = interpretation_state.get("query_interpretation", {})
            logger.info(f"Received interpretation from agent: {interpretation.get('intent', 'unknown')} with {len(interpretation.get('subtasks', []))} subtasks")
            
            # Update state with interpretation results
            state["query_interpretation"] = interpretation
            state["metadata"]["intent"] = interpretation.get("intent", "unknown")
            state["metadata"]["entities"] = interpretation.get("entities", [])
            state["metadata"]["interpretation_confidence"] = interpretation.get("confidence", 0.0)
            state["metadata"]["interpretation_rationale"] = interpretation.get("rationale", "")
            
            logger.info(f"Query analysis completed: intent={interpretation.get('intent')}, confidence={interpretation.get('confidence', 0.0)}")
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
                "rationale": "Fallback interpretation due to orchestrator error"
            }
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
        """Execute the planned subtasks using appropriate agents with shared state"""
        try:
            plan = state["task_plan"]
            if not plan:
                logger.warning("No execution plan found")
                return state
            
            # Initialize shared state
            from .shared_state import SharedAgentState, StateUpdateType
            shared_state = SharedAgentState(state)
            
            results = []

            async def _run_subtask(subtask: SubTask):
                """Execute a single subtask with shared state."""
                agent = agent_registry.find_capable_agent(subtask)
                if not agent:
                    logger.warning(f"No agent found for task: {subtask.description}")
                    subtask.status = TaskStatus.FAILED
                    subtask.error = "No capable agent found"
                    return subtask.id, None, []

                try:
                    # Get context-rich state for this subtask
                    task_state = await shared_state.propagate_to_subtask(subtask.id)
                    task_state["current_subtask"] = subtask
                    
                    # Execute the subtask
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
                    
                    if result_state.get("digest_results"):
                        await shared_state.merge_results(
                            {"digest_results": result_state["digest_results"]},
                            f"{type(agent).__name__}",
                            StateUpdateType.DIGEST_RESULTS
                        )

                    subtask.status = TaskStatus.COMPLETED
                    subtask.result = result_state.get("task_result")
                    
                    return subtask.id, subtask.result, []
                    
                except Exception as e:
                    logger.error(f"Error executing subtask {subtask.id}: {str(e)}")
                    subtask.status = TaskStatus.FAILED
                    subtask.error = str(e)
                    
                    # Try error recovery
                    try:
                        from .error_recovery_agent import ErrorRecoveryAgent
                        recovery_agent = ErrorRecoveryAgent({})
                        
                        recovery_state = {
                            "current_subtask": subtask,
                            "user_context": state.get("user_context", {}),
                            "query_interpretation": state.get("query_interpretation", {})
                        }
                        
                        recovery_result = await recovery_agent.process(recovery_state)
                        
                        if recovery_result.get("recovery_results"):
                            logger.info(f"Error recovery successful for subtask {subtask.id}")
                            subtask.status = TaskStatus.COMPLETED
                            subtask.result = recovery_result["recovery_results"].get("recovered_subtask", {}).get("result")
                            subtask.error = None
                            
                            return subtask.id, subtask.result, []
                    
                    except Exception as recovery_error:
                        logger.error(f"Error recovery failed for subtask {subtask.id}: {recovery_error}")
                    
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
        """Synthesize final results from all subtasks using result aggregator"""
        try:
            plan = state["task_plan"]
            if not plan:
                logger.warning("No execution plan found for synthesis")
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
            final_response = aggregation_result.get("final_response", {})
            aggregated_results = aggregation_result.get("aggregated_results", {})
            
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
            
            logger.info(f"Results synthesis completed: {final_response.get('type', 'unknown')} response generated")
            return state
            
        except Exception as e:
            logger.error(f"Error in results synthesis: {str(e)}")
            state["errors"] = state.get("errors", [])
            state["errors"].append(f"Synthesis error: {str(e)}")
            return state
    
    async def get_conversation_history(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get conversation history for a user"""
        return await self.conversation_memory.get_history(user_id, limit)
    
    async def clear_conversation_history(self, user_id: str):
        """Clear conversation history for a user"""
        await self.conversation_memory.clear_history(user_id)
