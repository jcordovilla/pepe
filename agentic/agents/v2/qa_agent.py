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
            # Perform similarity search
            results = await self.vector_store.similarity_search(
                query=query,
                k=self.default_k,
                filters=None
            )
            
            # Filter and format results
            context_docs = []
            for result in results:
                if result.get("score", 0) > 0.3:  # Minimum relevance threshold
                    context_docs.append({
                        "content": result.get("content", ""),
                        "metadata": result.get("metadata", {}),
                        "score": result.get("score", 0),
                        "permalink": result.get("metadata", {}).get("permalink", "")
                    })
            
            return context_docs
            
        except Exception as e:
            logger.error(f"Context retrieval error: {e}")
            return []
    
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
            
            # Truncate content if too long
            if len(content) > 500:
                content = content[:500] + "..."
            
            formatted_contexts.append(
                f"Message {i} (relevance: {score:.2f}):\n{content}\n"
                f"Link: {permalink}\n"
            )
        
        return "\n".join(formatted_contexts)
    
    def _post_process_answer(self, answer: str, context_docs: List[Dict[str, Any]]) -> str:
        """Post-process the answer to ensure proper formatting and citations."""
        # Ensure the answer starts properly
        if not answer.strip():
            return "I don't know based on server data."
        
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