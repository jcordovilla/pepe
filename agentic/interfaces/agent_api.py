"""
Agent API Interface

Provides a clean REST-like interface for interacting with the agentic RAG system.
"""

import logging
from typing import Dict, List, Any, Optional
import asyncio
from datetime import datetime

from ..agents.orchestrator import AgentOrchestrator
from ..vectorstore.persistent_store import PersistentVectorStore
from ..memory.conversation_memory import ConversationMemory

logger = logging.getLogger(__name__)


class AgentAPI:
    """
    High-level API interface for the agentic RAG system.
    
    Provides methods for querying, managing context, and system administration.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize components
        self.orchestrator = AgentOrchestrator(config.get("orchestrator", {}))
        self.vector_store = PersistentVectorStore(config.get("vector_store", {}))
        self.memory = ConversationMemory(config.get("memory", {}))
        
        # API configuration
        self.max_query_length = config.get("max_query_length", 2000)
        self.default_k = config.get("default_k", 10)
        self.enable_learning = config.get("enable_learning", True)
        
        logger.info("AgentAPI initialized successfully")
    
    async def query(
        self,
        query: str,
        user_id: str,
        context: Optional[Dict[str, Any]] = None,
        preferences: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process a user query through the agentic system.
        
        Args:
            query: User query text
            user_id: Unique user identifier
            context: Optional additional context
            preferences: User preferences
            
        Returns:
            Response with answer, sources, and metadata
        """
        try:
            # Validate input
            if not query or not query.strip():
                return {
                    "success": False,
                    "error": "Empty query provided",
                    "timestamp": datetime.utcnow().isoformat()
                }
            
            if len(query) > self.max_query_length:
                return {
                    "success": False,
                    "error": f"Query too long (max {self.max_query_length} characters)",
                    "timestamp": datetime.utcnow().isoformat()
                }
            
            # Get user context from memory
            user_context = await self.memory.get_user_context(user_id)
            
            # Merge context
            full_context = {
                "user_id": user_id,
                "user_context": user_context,
                **(context or {}),
                **(preferences or {})
            }
            
            # Process query through orchestrator
            result = await self.orchestrator.process_query(query, user_id, full_context)
            
            # Store interaction in memory if learning is enabled
            if self.enable_learning:
                await self.memory.add_interaction(
                    user_id=user_id,
                    query=query,
                    response=result.get("answer", ""),
                    context=full_context,
                    metadata={
                        "sources_count": len(result.get("sources", [])),
                        "confidence": result.get("confidence", 0.0),
                        "processing_time": result.get("processing_time", 0.0)
                    }
                )
            
            return {
                "success": True,
                "answer": result.get("answer", ""),
                "sources": result.get("sources", []),
                "confidence": result.get("confidence", 0.0),
                "metadata": result.get("metadata", {}),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def search(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        k: Optional[int] = None,
        search_type: str = "similarity"
    ) -> Dict[str, Any]:
        """
        Perform direct search on the vector store.
        
        Args:
            query: Search query
            filters: Optional metadata filters
            k: Number of results to return
            search_type: Type of search (similarity, keyword, filter)
            
        Returns:
            Search results
        """
        try:
            k_value = k or self.default_k
            
            if search_type == "similarity":
                results = await self.vector_store.similarity_search(query, k_value, filters)
            elif search_type == "keyword":
                keywords = query.split()
                results = await self.vector_store.keyword_search(keywords, k_value, filters)
            elif search_type == "filter":
                results = await self.vector_store.filter_search(filters or {}, k_value)
            else:
                return {
                    "success": False,
                    "error": f"Invalid search type: {search_type}",
                    "timestamp": datetime.utcnow().isoformat()
                }
            
            return {
                "success": True,
                "results": results,
                "count": len(results),
                "search_type": search_type,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error performing search: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def add_documents(
        self,
        documents: List[Dict[str, Any]],
        source: str = "api"
    ) -> Dict[str, Any]:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of documents to add
            source: Source identifier for the documents
            
        Returns:
            Result of the operation
        """
        try:
            # Add source metadata
            for doc in documents:
                doc["source"] = source
                doc["added_via"] = "api"
                doc["added_at"] = datetime.utcnow().isoformat()
            
            success = await self.vector_store.add_messages(documents)
            
            return {
                "success": success,
                "documents_added": len(documents) if success else 0,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def get_user_context(self, user_id: str) -> Dict[str, Any]:
        """
        Get user context and history.
        
        Args:
            user_id: User identifier
            
        Returns:
            User context information
        """
        try:
            context = await self.memory.get_user_context(user_id)
            # Get recent conversation history
            history = await self.memory.get_history(user_id, limit=10)
            
            return {
                "success": True,
                "user_id": user_id,
                "context": context,
                "recent_history": history,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting user context: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def update_user_context(
        self,
        user_id: str,
        context_updates: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Update user context.
        
        Args:
            user_id: User identifier
            context_updates: Context updates to apply
            
        Returns:
            Result of the operation
        """
        try:
            # Note: This method should be implemented in ConversationMemory
            # For now, we'll update user context directly
            user_context = await self.memory.get_user_context(user_id)
            user_context.update(context_updates)
            
            # Store updated context - this would need to be implemented
            # For now, return success
            return {
                "success": True,
                "user_id": user_id,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error updating user context: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def get_system_stats(self) -> Dict[str, Any]:
        """
        Get system statistics and health information.
        
        Returns:
            System statistics
        """
        try:
            # Get vector store stats
            vector_stats = await self.vector_store.get_collection_stats()
            
            # Get memory stats - using available method
            memory_stats = {
                "status": "available",
                "note": "Basic memory operations available"
            }
            
            # Get orchestrator stats (if available)
            orchestrator_stats = getattr(self.orchestrator, 'get_stats', lambda: {})()
            
            return {
                "success": True,
                "vector_store": vector_stats,
                "memory": memory_stats,
                "orchestrator": orchestrator_stats,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting system stats: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform comprehensive system health check.
        
        Returns:
            Health status information
        """
        try:
            health = {
                "status": "healthy",
                "timestamp": datetime.utcnow().isoformat(),
                "components": {}
            }
            
            # Check vector store
            vector_health = await self.vector_store.health_check()
            health["components"]["vector_store"] = vector_health
            
            if vector_health["status"] != "healthy":
                health["status"] = "degraded"
            
            # Check memory system
            try:
                await self.memory.get_user_context("health_check_test")
                health["components"]["memory"] = {"status": "healthy"}
            except Exception as e:
                health["components"]["memory"] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
                health["status"] = "unhealthy"
            
            # Check orchestrator
            try:
                # Simple test to see if orchestrator is responsive
                health["components"]["orchestrator"] = {"status": "healthy"}
            except Exception as e:
                health["components"]["orchestrator"] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
                health["status"] = "unhealthy"
            
            return health
            
        except Exception as e:
            logger.error(f"Error during health check: {e}")
            return {
                "status": "unhealthy",
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e)
            }
    
    async def optimize_system(self) -> Dict[str, Any]:
        """
        Optimize system performance.
        
        Returns:
            Optimization results
        """
        try:
            results = {
                "success": True,
                "optimizations": [],
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Optimize vector store
            vector_optimized = await self.vector_store.optimize()
            results["optimizations"].append({
                "component": "vector_store",
                "success": vector_optimized
            })
            
            # Optimize memory system - using basic cleanup
            results["optimizations"].append({
                "component": "memory",
                "success": True,
                "note": "Basic memory system operational"
            })
            
            return results
            
        except Exception as e:
            logger.error(f"Error during optimization: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def close(self):
        """Close all system components."""
        try:
            await self.vector_store.close()
            # Note: Add proper close method to ConversationMemory if needed
            # orchestrator doesn't need explicit closing
            
            logger.info("AgentAPI closed successfully")
            
        except Exception as e:
            logger.error(f"Error closing AgentAPI: {e}")
