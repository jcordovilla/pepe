"""
QA Agent

RAG answer agent using MCP server and LLM client.
Provides answers based on Discord message context.
"""

import logging
from typing import Dict, Any, List, Optional
import re

from ..base_agent import BaseAgent, AgentRole, AgentState
from ...services.llm_client import UnifiedLLMClient
from ...mcp import MCPServer
from ...utils.k_value_calculator import KValueCalculator

logger = logging.getLogger(__name__)


class QAAgent(BaseAgent):
    """
    QA agent that provides RAG answers using Discord message context.
    
    Input: dict(query: str)
    Output: str (markdown)
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(AgentRole.ANALYZER, config)
        self.llm_client = UnifiedLLMClient(config.get("llm", {}))
        
        # Initialize MCP server (replaces ChromaDB vector store)
        mcp_config = {
            "sqlite": {
                "db_path": "data/discord_messages.db"
            },
            "llm": config.get("llm", {})
        }
        self.mcp_server = MCPServer(mcp_config)
        
        # Initialize dynamic k-value calculator
        self.k_calculator = KValueCalculator(config)
        
        # Default search parameters (legacy, kept for backward compatibility)
        self.default_k = config.get("default_k", 5)
        self.max_context_length = config.get("max_context_length", 4000)
        
        logger.info("QAAgent initialized with MCP server")
    
    def signature(self) -> Dict[str, Any]:
        """Return agent signature for registration."""
        return {
            "role": "qa",
            "description": "Provides RAG answers using Discord message context",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Question to answer"}
                },
                "required": ["query"]
            },
            "output_schema": {
                "type": "string",
                "description": "Markdown formatted answer"
            }
        }
    
    async def process(self, input_data: Dict[str, Any]) -> str:
        """
        Process a query and return an answer based on Discord message context.
        
        Args:
            input_data: Dictionary containing the query
            
        Returns:
            Markdown formatted answer
        """
        try:
            query = input_data.get("query", "")
            if not query:
                return "❌ No query provided"
            
            # Calculate appropriate k value based on query
            k_value = self.k_calculator.calculate_k_value(query)
            logger.info(f"QA Agent processing query with k={k_value}: {query[:50]}...")
            
            # Search for relevant messages using MCP server
            relevant_messages = await self.mcp_server.search_messages(
                query=query,
                limit=k_value
            )
            
            if not relevant_messages:
                return f"❌ No relevant messages found for: {query}"
            
            # Build context from relevant messages
            context = self._build_context(relevant_messages)
            
            # Generate answer using LLM
            answer = await self._generate_answer(query, context)
            
            return answer
            
        except Exception as e:
            logger.error(f"Error in QA agent: {e}")
            return f"❌ Error processing query: {str(e)}"
    
    def _build_context(self, messages: List[Dict[str, Any]]) -> str:
        """Build context string from relevant messages."""
        context_parts = []
        
        for i, message in enumerate(messages[:self.default_k], 1):
            content = message.get("content", "")
            author = message.get("author_username", "Unknown")
            timestamp = message.get("timestamp", "")
            channel = message.get("channel_name", "Unknown")
            
            # Format message for context
            message_context = f"{i}. **{author}** in #{channel} ({timestamp}):\n{content}\n"
            context_parts.append(message_context)
        
        context = "\n".join(context_parts)
        
        # Truncate if too long
        if len(context) > self.max_context_length:
            context = context[:self.max_context_length] + "..."
        
        return context
    
    async def _generate_answer(self, query: str, context: str) -> str:
        """Generate answer using LLM with context."""
        system_prompt = """
You are a helpful assistant that answers questions about Discord server conversations.
Use the provided context from Discord messages to answer the user's question.

Guidelines:
1. Base your answer on the provided Discord message context
2. If the context doesn't contain enough information, say so
3. Use markdown formatting for better readability
4. Be concise but informative
5. Reference specific messages when relevant
6. Maintain a helpful and friendly tone

Context from Discord messages:
{context}

User question: {query}

Please provide a helpful answer based on the context above.
"""

        prompt = system_prompt.format(context=context, query=query)
        
        try:
            answer = await self.llm_client.generate(
                prompt=prompt,
                max_tokens=1000,
                temperature=0.3
            )
            
            return answer.strip()
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return f"❌ Error generating answer: {str(e)}"
    
    async def health_check(self) -> Dict[str, Any]:
        """Check agent health."""
        try:
            # Test MCP server
            mcp_health = await self.mcp_server.health_check()
            
            # Test LLM client
            llm_test = await self.llm_client.generate(
                prompt="Test",
                max_tokens=10
            )
            
            return {
                "status": "healthy",
                "mcp_server": mcp_health,
                "llm_client": "healthy" if llm_test else "unhealthy"
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    def can_handle(self, task) -> bool:
        """Check if this agent can handle the given task."""
        return (task.task_type == "qa" or 
                "question" in task.description.lower() or
                "answer" in task.description.lower() or
                "query" in task.description.lower()) 