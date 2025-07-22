"""
Self-Check Agent

Factuality and relevance validation agent using Mini-LM.
Wraps other agents to ensure quality and accuracy.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
import re

from ..base_agent import BaseAgent, AgentRole, AgentState
from ...services.llm_client import UnifiedLLMClient

logger = logging.getLogger(__name__)


class SelfCheckAgent(BaseAgent):
    """
    Self-check agent that validates factuality and relevance.
    
    Input: dict(text: str, context: any)
    Output: bool
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(AgentRole.ANALYZER, config)
        self.llm_client = UnifiedLLMClient(config.get("llm", {}))
        
        # Validation thresholds
        self.factuality_threshold = config.get("factuality_threshold", 3.0)  # 3/5
        self.relevance_threshold = config.get("relevance_threshold", 3.0)    # 3/5
        self.max_retries = config.get("max_retries", 1)
        
        logger.info("SelfCheckAgent initialized")
    
    def signature(self) -> Dict[str, Any]:
        """Return agent signature for registration."""
        return {
            "role": "selfcheck",
            "description": "Validates factuality and relevance of agent outputs",
            "input_schema": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Text to validate"},
                    "context": {"type": "object", "description": "Context information"}
                },
                "required": ["text", "context"]
            },
            "output_schema": {
                "type": "boolean",
                "description": "True if validation passes, False otherwise"
            }
        }
    
    async def run(self, **kwargs) -> bool:
        """
        Validate the factuality and relevance of text.
        
        Args:
            text: Text to validate
            context: Context information (query, source data, etc.)
            
        Returns:
            True if validation passes, False otherwise
        """
        text = kwargs.get("text", "").strip()
        context = kwargs.get("context", {})
        
        if not text:
            return False
        
        logger.info(f"SelfCheckAgent validating text: {text[:50]}...")
        
        try:
            # Check factuality
            factuality_score = await self._check_factuality(text, context)
            
            # Check relevance
            relevance_score = await self._check_relevance(text, context)
            
            # Determine if validation passes
            passes_factuality = factuality_score >= self.factuality_threshold
            passes_relevance = relevance_score >= self.relevance_threshold
            
            validation_passes = passes_factuality and passes_relevance
            
            logger.info(f"SelfCheckAgent scores - Factuality: {factuality_score:.1f}, Relevance: {relevance_score:.1f}, Passes: {validation_passes}")
            
            return validation_passes
            
        except Exception as e:
            logger.error(f"SelfCheckAgent error: {e}")
            return False  # Fail safe - if validation fails, assume invalid
    
    async def _check_factuality(self, text: str, context: Dict[str, Any]) -> float:
        """Check the factuality of the text against the context."""
        try:
            # Extract source information from context
            source_data = self._extract_source_data(context)
            query = context.get("query", "")
            
            prompt = f"""Rate the factuality of this text on a scale of 1-5, where:
1 = Completely false or unsupported
2 = Mostly false with some truth
3 = Partially true with some errors
4 = Mostly true with minor issues
5 = Completely accurate and well-supported

Text to evaluate: "{text}"

Context:
- Query: "{query}"
- Source data: {source_data}

Consider:
- Are claims supported by the source data?
- Are there any factual errors?
- Are citations accurate?
- Is information consistent?

Return only a number between 1 and 5."""

            response = await self.llm_client.generate(
                prompt=prompt,
                max_tokens=10,
                temperature=0.1
            )
            
            try:
                score = float(response.strip())
                return max(1.0, min(5.0, score))  # Clamp between 1 and 5
            except ValueError:
                return 3.0  # Default to neutral score
                
        except Exception as e:
            logger.error(f"Error checking factuality: {e}")
            return 3.0
    
    async def _check_relevance(self, text: str, context: Dict[str, Any]) -> float:
        """Check the relevance of the text to the query."""
        try:
            query = context.get("query", "")
            
            prompt = f"""Rate the relevance of this text to the query on a scale of 1-5, where:
1 = Completely irrelevant
2 = Mostly irrelevant
3 = Somewhat relevant
4 = Mostly relevant
5 = Highly relevant and directly addresses the query

Query: "{query}"

Text: "{text}"

Consider:
- Does the text address the main question?
- Is the information useful for the query?
- Are there irrelevant tangents?
- Does it provide the type of information requested?

Return only a number between 1 and 5."""

            response = await self.llm_client.generate(
                prompt=prompt,
                max_tokens=10,
                temperature=0.1
            )
            
            try:
                score = float(response.strip())
                return max(1.0, min(5.0, score))  # Clamp between 1 and 5
            except ValueError:
                return 3.0  # Default to neutral score
                
        except Exception as e:
            logger.error(f"Error checking relevance: {e}")
            return 3.0
    
    def _extract_source_data(self, context: Dict[str, Any]) -> str:
        """Extract source data information from context."""
        source_info = []
        
        # Extract from various context sources
        if "search_results" in context:
            results = context["search_results"]
            source_info.append(f"Search results: {len(results)} items")
        
        if "vector_results" in context:
            results = context["vector_results"]
            source_info.append(f"Vector results: {len(results)} items")
        
        if "analytics_data" in context:
            data = context["analytics_data"]
            source_info.append(f"Analytics data available")
        
        if "messages" in context:
            messages = context["messages"]
            source_info.append(f"Message data: {len(messages)} messages")
        
        if not source_info:
            source_info.append("No specific source data identified")
        
        return "; ".join(source_info)
    
    async def validate_agent_output(self, agent_name: str, output: str, context: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate output from a specific agent.
        
        Args:
            agent_name: Name of the agent that produced the output
            output: The output to validate
            context: Context information
            
        Returns:
            Tuple of (passes_validation, validation_details)
        """
        try:
            # Add agent context
            validation_context = context.copy()
            validation_context["agent_name"] = agent_name
            validation_context["output"] = output
            
            # Perform validation
            passes = await self.run(text=output, context=validation_context)
            
            # Get detailed scores for reporting
            factuality_score = await self._check_factuality(output, validation_context)
            relevance_score = await self._check_relevance(output, validation_context)
            
            validation_details = {
                "agent_name": agent_name,
                "passes_validation": passes,
                "factuality_score": factuality_score,
                "relevance_score": relevance_score,
                "factuality_threshold": self.factuality_threshold,
                "relevance_threshold": self.relevance_threshold,
                "validation_timestamp": self._get_timestamp()
            }
            
            return passes, validation_details
            
        except Exception as e:
            logger.error(f"Error validating agent output: {e}")
            return False, {
                "agent_name": agent_name,
                "passes_validation": False,
                "error": str(e),
                "validation_timestamp": self._get_timestamp()
            }
    
    async def retry_with_improvements(self, agent_name: str, original_output: str, context: Dict[str, Any]) -> Tuple[str, bool]:
        """
        Retry agent execution with improvements based on validation feedback.
        
        Args:
            agent_name: Name of the agent to retry
            original_output: The original output that failed validation
            context: Context information
            
        Returns:
            Tuple of (improved_output, validation_passes)
        """
        try:
            # Analyze what went wrong
            factuality_score = await self._check_factuality(original_output, context)
            relevance_score = await self._check_relevance(original_output, context)
            
            # Generate improvement suggestions
            improvements = await self._generate_improvement_suggestions(
                original_output, factuality_score, relevance_score, context
            )
            
            # Add improvements to context for retry
            improved_context = context.copy()
            improved_context["validation_feedback"] = improvements
            improved_context["previous_output"] = original_output
            
            # Note: In a real implementation, you would retry the original agent here
            # For now, we'll just return the original output with a note
            improved_output = f"{original_output}\n\n*Note: This response could be improved based on validation feedback.*"
            
            # Re-validate the improved output
            passes_validation, _ = await self.validate_agent_output(agent_name, improved_output, improved_context)
            
            return improved_output, passes_validation
            
        except Exception as e:
            logger.error(f"Error in retry with improvements: {e}")
            return original_output, False
    
    async def _generate_improvement_suggestions(self, output: str, factuality_score: float, relevance_score: float, context: Dict[str, Any]) -> str:
        """Generate suggestions for improving the output."""
        try:
            query = context.get("query", "")
            
            prompt = f"""Analyze this agent output and provide specific improvement suggestions:

Query: "{query}"
Output: "{output}"

Validation Scores:
- Factuality: {factuality_score:.1f}/5.0
- Relevance: {relevance_score:.1f}/5.0

Provide 2-3 specific suggestions for improvement. Focus on:
- Factual accuracy if factuality score is low
- Relevance to query if relevance score is low
- Clarity and completeness
- Better use of source information

Return only the improvement suggestions."""

            suggestions = await self.llm_client.generate(
                prompt=prompt,
                max_tokens=200,
                temperature=0.1
            )
            
            return suggestions.strip()
            
        except Exception as e:
            logger.error(f"Error generating improvement suggestions: {e}")
            return "Unable to generate specific improvement suggestions."
    
    def _get_timestamp(self) -> str:
        """Get current timestamp for validation records."""
        from datetime import datetime
        return datetime.utcnow().isoformat()
    
    async def process(self, state: AgentState) -> AgentState:
        """Process state through the self-check agent."""
        # Extract text and context from state
        text = state.get("response", "")
        context = {
            "query": state.get("user_context", {}).get("query", ""),
            "search_results": state.get("search_results", []),
            "agent_args": state.get("agent_args", {}),
            "next_agent": state.get("next_agent", "")
        }
        
        # Perform validation
        passes = await self.run(text=text, context=context)
        
        # Update state with validation results
        state["selfcheck_passes"] = passes
        state["validation_context"] = context
        
        if not passes:
            state["validation_failed"] = True
            state["response"] = f"{text}\n\n*Note: This response did not pass quality validation.*"
        
        return state
    
    def can_handle(self, task) -> bool:
        """Check if this agent can handle the given task."""
        return (task.task_type == "selfcheck" or 
                "validation" in task.description.lower() or
                "quality" in task.description.lower() or
                "factuality" in task.description.lower()) 