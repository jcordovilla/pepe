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
        self.default_k = config.get("default_k", 50)  # Increased from 5 to 50
        self.max_context_length = config.get("max_context_length", 8000)  # Increased from 4000 to 8000
        
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
    
    async def process(self, state: AgentState) -> AgentState:
        """
        Process a query and return an answer based on Discord message context.
        
        Args:
            state: Agent state containing the query
            
        Returns:
            Updated agent state with response
        """
        try:
            # Extract query from state
            query = state.get("user_context", {}).get("query", "")
            if not query:
                state["response"] = "❌ No query provided"
                return state
            
            # Calculate appropriate k value based on query
            k_result = self.k_calculator.calculate_k_value(query)
            k_value = k_result.get("k_value", 10)  # Extract the actual k value from the result
            logger.info(f"QA Agent processing query with k={k_value}: {query[:50]}...")
            
            # Search for relevant messages using MCP server
            # Use natural language query for complex queries, text search for simple ones
            if len(query.split()) > 3:  # Complex query - use natural language
                relevant_messages = await self.mcp_server.query_messages(query)
            else:  # Simple query - use text search
                relevant_messages = await self.mcp_server.search_messages(
                    query=query,
                    filters=None,
                    limit=k_value
                )
            
            if not relevant_messages:
                state["response"] = f"❌ No relevant messages found for: {query}"
                return state
            
            # Build context from relevant messages
            context = self._build_context(relevant_messages)
            
            # Generate answer using LLM
            answer = await self._generate_answer(query, context, relevant_messages)
            
            # Update state with response
            state["response"] = answer
            state["analysis_results"] = {
                "relevant_messages": len(relevant_messages),
                "k_value": k_value,
                "context_length": len(context)
            }
            
            return state
            
        except Exception as e:
            logger.error(f"Error in QA agent: {e}")
            state["response"] = f"❌ Error processing query: {str(e)}"
            state["errors"].append(str(e))
            return state
    
    def _build_context(self, messages: List[Dict[str, Any]]) -> str:
        """Build context string from relevant messages."""
        context_parts = []
        
        for i, message in enumerate(messages, 1):  # Remove limit to show all results
            content = message.get("content", "")
            author_username = message.get("author_username", "Unknown")
            author_display = message.get("author_display_name", author_username)
            timestamp = message.get("timestamp", "")
            channel = message.get("channel_name", "Unknown")
            jump_url = message.get("jump_url", "")
            message_id = message.get("message_id", "")
            
            # Use display name if available, fallback to username
            author_name = author_display if author_display and author_display != "None" else author_username
            
            # Format message for context with jump URL
            message_context = f"{i}. **{author_name}** in #{channel} ({timestamp}):\n{content}\n"
            if jump_url:
                message_context += f"🔗 [View Message]({jump_url})\n"
            context_parts.append(message_context)
        
        context = "\n".join(context_parts)
        
        # Don't truncate - let the LLM handle all the information
        return context
    
    async def _generate_answer(self, query: str, context: str, messages: List[Dict[str, Any]]) -> str:
        """Generate answer using LLM with context."""
        system_prompt = """
You are a helpful assistant that answers questions about Discord server conversations.
Use the provided context from Discord messages to answer the user's question.

**CRITICAL REQUIREMENT: You MUST include the jump URL links (🔗 [View Message](url)) for each user you mention.**

**EXPERIENCE QUERY FILTERING RULES:**
When asked about users with experience in a field, ONLY include users who have EXPLICITLY STATED THEIR OWN experience, not users who just discussed the topic.

**INCLUDE users who:**
- Use first-person statements: "I have", "I am", "I work", "I'm certified", "my experience"
- Make professional declarations: "certified", "years of experience", "worked as", "specialize in"
- Have self-introductions: "Hi, I'm", "My name is", "About me", "Areas of Expertise"
- Explicitly claim personal expertise or experience in the field

**EXCLUDE users who:**
- Only discuss the topic without claiming personal experience
- Ask questions about the field
- Share opinions or general knowledge
- Mention the topic in passing without personal claims
- Have others mention their experience (unless they confirm it themselves)

Guidelines:
1. Base your answer on the provided Discord message context
2. If the context doesn't contain enough information, say so
3. Use markdown formatting for better readability
4. Be comprehensive and include ALL relevant users found
5. **ALWAYS include the jump URL link (🔗 [View Message](url)) for each user you mention**
6. Maintain a helpful and friendly tone
7. For user experience/skills queries, carefully extract and list ALL specific users and their mentioned experience
8. Look for user introductions, self-descriptions, and mentions of skills/expertise
9. Pay attention to user display names (preferred) and usernames, and their stated areas of expertise
10. If users mention specific roles, certifications, or years of experience, include those details
11. **MANDATORY: Copy the exact jump URL links from the context for each user you reference**
12. Do NOT limit the number of users - show ALL users found in the context
13. Use display names when available, fallback to usernames if display name is not available
14. **CRITICAL: Only include users who explicitly claim their own experience, not those who just discuss topics**

**FORMAT EXAMPLE:**
- **User Name**: Description of their experience 🔗 [View Message](https://discord.com/...)

Context from Discord messages:
{context}

User question: {query}

Please provide a comprehensive answer based on the context above. **CRITICAL: For each user you mention, you MUST include their jump URL link from the context above. Only include users who explicitly state their own experience, not those who just discuss topics.**
"""

        prompt = system_prompt.format(context=context, query=query)
        
        try:
            answer = await self.llm_client.generate(
                prompt=prompt,
                max_tokens=1000,
                temperature=0.3
            )
            
            # Post-process to ensure jump URLs are included
            answer = self._ensure_jump_urls_included(answer, messages)
            
            return answer.strip()
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return f"❌ Error generating answer: {str(e)}"
    
    def _ensure_jump_urls_included(self, answer: str, messages: List[Dict[str, Any]]) -> str:
        """Post-process answer to ensure jump URLs are included for mentioned users."""
        try:
            # Create a mapping of usernames to jump URLs
            user_jump_urls = {}
            for message in messages:
                author_username = message.get("author_username", "")
                author_display = message.get("author_display_name", "")
                jump_url = message.get("jump_url", "")
                
                if jump_url:
                    # Map both username and display name to jump URL
                    if author_username:
                        user_jump_urls[author_username.lower()] = jump_url
                    if author_display and author_display != "None":
                        user_jump_urls[author_display.lower()] = jump_url
            
            # Check if any users mentioned in the answer are missing jump URLs
            import re
            
            # Find user mentions in the answer (look for bold names)
            user_mentions = re.findall(r'\*\*(.*?)\*\*', answer)
            
            for mention in user_mentions:
                mention_lower = mention.lower()
                if mention_lower in user_jump_urls:
                    jump_url = user_jump_urls[mention_lower]
                    # Check if this user already has a jump URL in the answer
                    if jump_url not in answer:
                        # Add jump URL after the user mention
                        jump_link = f"🔗 [View Message]({jump_url})"
                        # Find the line with this user and add the jump URL
                        lines = answer.split('\n')
                        for i, line in enumerate(lines):
                            if f"**{mention}**" in line and jump_link not in line:
                                lines[i] = line.rstrip() + f" {jump_link}"
                                break
                        answer = '\n'.join(lines)
            
            return answer
            
        except Exception as e:
            logger.error(f"Error ensuring jump URLs: {e}")
            return answer
    
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