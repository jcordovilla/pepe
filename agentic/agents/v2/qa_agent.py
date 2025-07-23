"""
QA Agent

RAG answer agent using Chroma retrieval and LLM client.
Provides answers based on Discord message context.
"""

import logging
from typing import Dict, Any, List, Optional
import re

from ..base_agent import BaseAgent, AgentRole, AgentState
from ...services.llm_client import UnifiedLLMClient
from ...vectorstore.persistent_store import PersistentVectorStore

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
        self.vector_store = PersistentVectorStore(config.get("vector_config", {}))
        
        # Default search parameters
        self.default_k = config.get("default_k", 5)
        self.max_context_length = config.get("max_context_length", 4000)
        
        logger.info("QAAgent initialized")
    
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
    
    async def run(self, **kwargs) -> str:
        """
        Generate a RAG answer for the given query.
        
        Args:
            query: The question to answer
            
        Returns:
            Markdown formatted answer
        """
        query = kwargs.get("query", "").strip()
        
        if not query:
            return "I need a question to answer. Please provide a query."
        
        logger.info(f"QAAgent processing query: {query[:50]}...")
        
        try:
            # Check if this is a question about the bot's capabilities
            if self._is_bot_capability_question(query):
                return self._get_bot_capabilities_response()
            
            # Retrieve relevant context
            context_docs = await self._retrieve_context(query)
            
            if not context_docs:
                return "I don't know based on server data. No relevant information found in the Discord messages."
            
            # Generate answer using context
            answer = await self._generate_answer(query, context_docs)
            
            return answer
            
        except Exception as e:
            logger.error(f"QAAgent error: {e}")
            return f"Sorry, I encountered an error while processing your question: {str(e)}"
    
    async def _retrieve_context(self, query: str) -> List[Dict[str, Any]]:
        """Retrieve relevant context from vector store."""
        try:
            # For user experience queries, increase search coverage
            search_k = self.default_k
            if any(keyword in query.lower() for keyword in ["users", "people", "who", "list", "experience", "skills", "expertise"]):
                search_k = 20  # Increase search coverage for user-related queries
            
            # Perform similarity search
            results = await self.vector_store.similarity_search(
                query=query,
                k=search_k,
                filters=None
            )
            
            # Filter and format results
            context_docs = []
            for result in results:
                # Use similarity score from vector store (0-1, higher is better)
                similarity_score = result.get("similarity", 0)
                # Lower threshold for msmarco-distilbert-base-v4 model which produces larger distances
                if similarity_score >= 0.0:  # Accept any result for now
                    metadata = result.get("metadata", {})
                    
                    # Extract author information properly
                    author_display_name = metadata.get("author_display_name", "")
                    author_username = metadata.get("author_username", "")
                    author_id = metadata.get("author_id", "")
                    
                    # Prefer display_name over username, fallback to username, then ID
                    author_name = author_display_name if author_display_name else author_username if author_username else f"User-{author_id[-6:]}" if author_id else "Unknown"
                    
                    context_docs.append({
                        "content": result.get("content", ""),
                        "metadata": {
                            **metadata,
                            "author_name": author_name,  # Add processed author name
                            "author_display_name": author_display_name,
                            "author_username": author_username,
                            "author_id": author_id
                        },
                        "score": similarity_score,
                        "permalink": metadata.get("jump_url", "")
                    })
            
            return context_docs
            
        except Exception as e:
            logger.error(f"Context retrieval error: {e}")
            return []
    
    def _is_bot_capability_question(self, query: str) -> bool:
        """Check if the query is asking about the bot's capabilities."""
        query_lower = query.lower()
        
        # Keywords that indicate questions about bot capabilities
        capability_keywords = [
            "what can you do",
            "what do you do",
            "what are your capabilities",
            "what are you capable of",
            "what type of tasks",
            "what tasks can you",
            "how can you help",
            "what are your features",
            "what are your functions",
            "what can you help with",
            "what are you able to do",
            "what are your skills",
            "what are your abilities",
            "what can you perform",
            "what are you designed to do",
            "what is your purpose",
            "what do you specialize in",
            "what are your strengths",
            "what can you assist with",
            "what are your tools"
        ]
        
        return any(keyword in query_lower for keyword in capability_keywords)
    
    def _get_bot_capabilities_response(self) -> str:
        """Provide a comprehensive response about the bot's capabilities."""
        return """# 🤖 Discord Bot Capabilities

I'm an intelligent Discord bot designed to help you analyze and interact with your server's content. Here are my main capabilities:

## 🔍 **Search & Analysis**
- **Semantic Search**: Find relevant messages and discussions across your Discord server
- **Content Analysis**: Analyze conversation patterns, topics, and trends
- **User Activity**: Track user engagement and participation patterns
- **Channel Insights**: Understand activity levels and discussion topics in different channels

## 📊 **Digest & Summarization**
- **Weekly Digests**: Generate comprehensive summaries of server activity
- **Channel Summaries**: Create focused summaries for specific channels
- **Topic Analysis**: Identify key discussion themes and trends
- **Activity Reports**: Track engagement metrics and participation

## 🧠 **Intelligent Q&A**
- **Context-Aware Answers**: Answer questions based on your server's message history
- **Conversation Memory**: Remember previous interactions for better context
- **Multi-Agent System**: Use specialized agents for different types of queries
- **Real-time Processing**: Provide instant responses to your questions

## 🎯 **Specialized Features**
- **Resource Detection**: Automatically identify and catalog valuable links and resources
- **Trend Analysis**: Spot emerging topics and discussion patterns
- **Community Insights**: Help understand your community's interests and engagement
- **Performance Monitoring**: Track bot performance and usage statistics

## 💬 **How to Use Me**
- Ask me questions about your server's content: *"What are people discussing about AI?"*
- Request summaries: *"Give me a weekly digest of #general"*
- Search for specific topics: *"Find discussions about machine learning"*
- Get insights: *"What are the most active channels?"*

I'm constantly learning from your server's content to provide more relevant and helpful responses! 🚀"""
    
    async def _generate_answer(self, query: str, context_docs: List[Dict[str, Any]]) -> str:
        """Generate answer using LLM and context."""
        # Prepare context string
        context_str = self._format_context(context_docs)
        
        system_prompt = """You are a helpful Discord bot assistant. Answer questions using ONLY the provided context from Discord messages.

IMPORTANT RULES:
1. Answer ONLY using information from the provided CONTEXT
2. Cite Discord permalinks for every fact you mention
3. If the context doesn't contain enough information to answer the question, say "I don't know based on server data"
4. Be concise but thorough
5. Use markdown formatting for better readability
6. Always include relevant message links when citing information
7. When listing users, use their display names (not user IDs like <@123456>)
8. For user experience queries, list all relevant users found in the context

CONTEXT:
{context}

QUESTION: {query}

ANSWER:"""

        prompt = system_prompt.format(
            context=context_str,
            query=query
        )
        
        try:
            answer = await self.llm_client.generate(
                prompt=prompt,
                max_tokens=1000,
                temperature=0.1
            )
            
            # Post-process the answer to ensure proper formatting
            answer = self._post_process_answer(answer, context_docs)
            
            return answer
            
        except Exception as e:
            logger.error(f"Answer generation error: {e}")
            return "I encountered an error while generating the answer. Please try again."
    
    def _format_context(self, context_docs: List[Dict[str, Any]]) -> str:
        """Format context documents for LLM consumption."""
        if not context_docs:
            return "No relevant context found."
        
        formatted_contexts = []
        for i, doc in enumerate(context_docs, 1):
            content = doc.get("content", "")
            permalink = doc.get("permalink", "")
            score = doc.get("score", 0)
            metadata = doc.get("metadata", {})
            
            # Get author information
            author_name = metadata.get("author_name", "Unknown")
            channel_name = metadata.get("channel_name", "Unknown")
            timestamp = metadata.get("timestamp", "")
            
            # Truncate content if too long
            if len(content) > 500:
                content = content[:500] + "..."
            
            formatted_contexts.append(
                f"Message {i} (relevance: {score:.2f}) by {author_name} in #{channel_name}:\n{content}\n"
                f"Link: {permalink}\n"
            )
        
        return "\n".join(formatted_contexts)
    
    def _post_process_answer(self, answer: str, context_docs: List[Dict[str, Any]]) -> str:
        """Post-process the answer to ensure proper formatting and citations."""
        # Ensure the answer starts properly
        if not answer.strip():
            return "I don't know based on server data."
        
        # Clean up any remaining Discord user ID references (<@123456>)
        # Replace with a note about using display names
        answer = re.sub(r'<@\d+>', '[User mentioned]', answer)
        
        # Add a note if no citations were found
        if not re.search(r'https?://', answer):
            answer += "\n\n*Note: No specific message links were found in the context.*"
        
        return answer.strip()
    
    async def process(self, state: AgentState) -> AgentState:
        """Process state through the QA agent."""
        query = state.get("agent_args", {}).get("query", "")
        
        if not query:
            query = state.get("user_context", {}).get("query", "")
        
        answer = await self.run(query=query)
        
        # Update state with answer
        state["qa_answer"] = answer
        state["response"] = answer
        
        return state
    
    def can_handle(self, task) -> bool:
        """Check if this agent can handle the given task."""
        return (task.task_type == "qa" or 
                "question" in task.description.lower() or
                "search" in task.description.lower() or
                "find" in task.description.lower()) 