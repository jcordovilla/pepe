"""
Router Agent

Lightweight dispatcher that routes commands to appropriate agents.
Replaces QueryInterpreter + Planner with a simpler, more direct approach.
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

from ..base_agent import BaseAgent, AgentRole, AgentState
from ...services.llm_client import UnifiedLLMClient

logger = logging.getLogger(__name__)


class RouterAgent(BaseAgent):
    """
    Router agent that dispatches commands to appropriate agents.
    
    Input: dict(command: str, payload: any)
    Output: AgentResult(next_agent: str, args: dict)
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(AgentRole.ORCHESTRATOR, config)
        self.llm_client = UnifiedLLMClient(config.get("llm", {}))
        
        # Command routing patterns
        self.command_patterns = {
            "ask": "qa",
            "question": "qa", 
            "search": "qa",
            "find": "qa",
            "list": "qa",
            "users": "qa",
            "who": "qa",
            "stats": "stats",
            "statistics": "stats",
            "metrics": "stats",
            "digest": "digest",
            "summary": "digest",
            "summarize": "digest",
            "trend": "trend",
            "trends": "trend",
            "structure": "structure",
            "channels": "structure",
            "organize": "structure"
        }
        
        logger.info("RouterAgent initialized")
    
    def signature(self) -> Dict[str, Any]:
        """Return agent signature for registration."""
        return {
            "role": "router",
            "description": "Routes commands to appropriate agents",
            "input_schema": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Command to route"},
                    "payload": {"type": "object", "description": "Command payload"}
                },
                "required": ["command", "payload"]
            },
            "output_schema": {
                "type": "object", 
                "properties": {
                    "next_agent": {"type": "string", "description": "Next agent to call"},
                    "args": {"type": "object", "description": "Arguments for next agent"}
                }
            }
        }
    
    async def run(self, **kwargs) -> Dict[str, Any]:
        """
        Route the command to the appropriate agent.
        
        Args:
            command: Command string
            payload: Command payload
            
        Returns:
            Dict with next_agent and args
        """
        command = kwargs.get("command", "").lower().strip()
        payload = kwargs.get("payload", {})
        
        logger.info(f"RouterAgent processing command: {command}")
        
        # Try pattern matching first
        next_agent = self._pattern_match(command)
        
        # If no pattern match, use LLM for intelligent routing
        if not next_agent:
            next_agent = await self._llm_route(command, payload)
        
        # Prepare arguments for the next agent
        args = self._prepare_args(command, payload, next_agent)
        
        return {
            "next_agent": next_agent,
            "args": args
        }
    
    def _pattern_match(self, command: str) -> Optional[str]:
        """Match command against known patterns."""
        for pattern, agent in self.command_patterns.items():
            if pattern in command:
                return agent
        return None
    
    async def _llm_route(self, command: str, payload: Dict[str, Any]) -> str:
        """Use LLM to intelligently route the command."""
        prompt = f"""You are a command router for a Discord bot. Route this command to the appropriate agent:

Command: "{command}"
Payload: {payload}

Available agents:
- qa: For specific questions, searches, finding specific information, listing users with specific skills/experience, finding particular messages or content
- stats: For statistics, metrics, data analysis, numerical summaries
- digest: For general summaries, overviews, "what happened" type questions, general activity summaries
- trend: For trend analysis, patterns over time, "how has X changed" questions
- structure: For channel organization, structure analysis, "how is this organized" questions

Examples:
- "list users with experience in cybersecurity" → qa (specific information search)
- "find messages about Python" → qa (specific search)
- "who knows about machine learning" → qa (specific user search)
- "what happened this week" → digest (general summary)
- "give me a summary of recent activity" → digest (general overview)
- "how many messages were sent" → stats (statistics)
- "what are the trending topics" → trend (trend analysis)

Respond with ONLY the agent name (qa, stats, digest, trend, or structure)."""

        try:
            response = await self.llm_client.generate(prompt, max_tokens=10, temperature=0.1)
            agent = response.strip().lower()
            
            # Validate the response
            if agent in ["qa", "stats", "digest", "trend", "structure"]:
                return agent
            else:
                logger.warning(f"LLM returned invalid agent: {agent}, defaulting to qa")
                return "qa"
                
        except Exception as e:
            logger.error(f"LLM routing failed: {e}, defaulting to qa")
            return "qa"
    
    def _prepare_args(self, command: str, payload: Dict[str, Any], next_agent: str) -> Dict[str, Any]:
        """Prepare arguments for the next agent based on its type."""
        args = payload.copy()
        
        # Merge in agent_args from payload/context if present (e.g., from Discord interface)
        agent_args_from_context = None
        if "context" in payload and isinstance(payload["context"], dict):
            agent_args_from_context = payload["context"].get("agent_args")
        if agent_args_from_context and isinstance(agent_args_from_context, dict):
            args.update(agent_args_from_context)

        # Add command context
        args["original_command"] = command
        args["routed_at"] = datetime.utcnow().isoformat()
        
        # Agent-specific argument preparation
        if next_agent == "qa":
            args["query"] = command
        elif next_agent == "stats":
            # Extract time range from command if present
            if "last week" in command.lower():
                args["granularity"] = "week"
            elif "last month" in command.lower():
                args["granularity"] = "month"
            else:
                args["granularity"] = "day"
        elif next_agent == "digest":
            # Extract period from command and set appropriate date ranges
            now = datetime.utcnow()
            
            if "week" in command.lower():
                args["period"] = "week"
                args["start"] = (now - timedelta(days=7)).isoformat()
                args["end"] = now.isoformat()
            elif "month" in command.lower():
                args["period"] = "month"
                args["start"] = (now - timedelta(days=30)).isoformat()
                args["end"] = now.isoformat()
            elif "today" in command.lower():
                args["period"] = "day"
                # Set start to beginning of today (UTC)
                start_of_day = now.replace(hour=0, minute=0, second=0, microsecond=0)
                args["start"] = start_of_day.isoformat()
                args["end"] = now.isoformat()
            elif "yesterday" in command.lower():
                args["period"] = "day"
                # Set start to beginning of yesterday (UTC)
                start_of_yesterday = (now - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
                end_of_yesterday = start_of_yesterday.replace(hour=23, minute=59, second=59, microsecond=999999)
                args["start"] = start_of_yesterday.isoformat()
                args["end"] = end_of_yesterday.isoformat()
            else:
                args["period"] = "all_time"  # Default to all time instead of day
        elif next_agent == "trend":
            # Extract number of topics
            args["k"] = 5  # Default to 5 topics
        
        return args
    
    async def process(self, state: AgentState) -> AgentState:
        """Process state through the router agent."""
        command = state.get("user_context", {}).get("query", "")
        payload = {
            "user_id": state.get("user_context", {}).get("user_id"),
            "context": state.get("user_context", {})
        }
        
        result = await self.run(command=command, payload=payload)
        
        # Update state with routing result
        state["routing_result"] = result
        state["next_agent"] = result["next_agent"]
        state["agent_args"] = result["args"]
        
        return state
    
    def can_handle(self, task) -> bool:
        """Check if this agent can handle the given task."""
        return task.task_type == "routing" or "route" in task.description.lower() 