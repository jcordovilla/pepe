"""
MCP Client Implementation

Client for communicating with the MCP server.
"""

import asyncio
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class MCPClient:
    """
    Client for communicating with the MCP server.
    
    Provides a simple interface for all MCP operations.
    """
    
    def __init__(self, server: Optional[Any] = None):
        """
        Initialize MCP client.
        
        Args:
            server: MCP server instance (if None, will be created)
        """
        self.server = server
        
        # Performance tracking
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "response_times": []
        }
        
        logger.info("MCP Client initialized")
    
    async def generate_embedding(self, text: str, model: Optional[str] = None) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            model: Model to use (optional)
            
        Returns:
            List of floats representing the embedding
        """
        start_time = datetime.utcnow()
        
        try:
            if not self.server:
                raise RuntimeError("MCP server not initialized")
            
            embedding = await self.server.generate_embedding(text, model)
            
            self._record_success(start_time)
            return embedding
            
        except Exception as e:
            self._record_failure(start_time)
            logger.error(f"Error generating embedding: {e}")
            raise
    
    async def batch_embed(self, texts: List[str], model: Optional[str] = None) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in batch.
        
        Args:
            texts: List of texts to embed
            model: Model to use (optional)
            
        Returns:
            List of embeddings
        """
        start_time = datetime.utcnow()
        
        try:
            if not self.server:
                raise RuntimeError("MCP server not initialized")
            
            embeddings = await self.server.batch_embed(texts, model)
            
            self._record_success(start_time)
            return embeddings
            
        except Exception as e:
            self._record_failure(start_time)
            logger.error(f"Error in batch embedding: {e}")
            raise
    
    async def similarity_search(
        self,
        query: str,
        k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        min_score: float = 0.0,
        model: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform semantic similarity search.
        
        Args:
            query: Search query
            k: Number of results to return
            filters: Optional metadata filters
            min_score: Minimum similarity score
            model: Embedding model to use
            
        Returns:
            List of search results with metadata
        """
        start_time = datetime.utcnow()
        
        try:
            if not self.server:
                raise RuntimeError("MCP server not initialized")
            
            results = await self.server.similarity_search(
                query=query,
                k=k,
                filters=filters,
                min_score=min_score,
                model=model
            )
            
            self._record_success(start_time)
            return results
            
        except Exception as e:
            self._record_failure(start_time)
            logger.error(f"Error in similarity search: {e}")
            raise
    
    async def filter_search(
        self,
        filters: Dict[str, Any],
        k: int = 10,
        sort_by: str = "timestamp"
    ) -> List[Dict[str, Any]]:
        """
        Search messages using only filters (no text query).
        
        Args:
            filters: Metadata filters
            k: Number of results to return
            sort_by: Field to sort by
            
        Returns:
            List of filtered results
        """
        start_time = datetime.utcnow()
        
        try:
            if not self.server:
                raise RuntimeError("MCP server not initialized")
            
            results = await self.server.filter_search(
                filters=filters,
                k=k,
                sort_by=sort_by
            )
            
            self._record_success(start_time)
            return results
            
        except Exception as e:
            self._record_failure(start_time)
            logger.error(f"Error in filter search: {e}")
            raise
    
    async def add_message_embedding(
        self,
        message_id: str,
        content: str,
        model: Optional[str] = None
    ) -> bool:
        """
        Add or update embedding for a message.
        
        Args:
            message_id: Unique message identifier
            content: Message content to embed
            model: Model to use (optional)
            
        Returns:
            True if successful
        """
        start_time = datetime.utcnow()
        
        try:
            if not self.server:
                raise RuntimeError("MCP server not initialized")
            
            success = await self.server.add_message_embedding(message_id, content, model)
            
            if success:
                self._record_success(start_time)
            else:
                self._record_failure(start_time)
            
            return success
            
        except Exception as e:
            self._record_failure(start_time)
            logger.error(f"Error adding message embedding: {e}")
            return False
    
    async def batch_add_embeddings(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Add embeddings for multiple messages in batch.
        
        Args:
            messages: List of message dictionaries with 'message_id' and 'content'
            model: Model to use (optional)
            
        Returns:
            Dictionary with success count and errors
        """
        start_time = datetime.utcnow()
        
        try:
            if not self.server:
                raise RuntimeError("MCP server not initialized")
            
            result = await self.server.batch_add_embeddings(messages, model)
            
            if result.get("error_count", 0) == 0:
                self._record_success(start_time)
            else:
                self._record_failure(start_time)
            
            return result
            
        except Exception as e:
            self._record_failure(start_time)
            logger.error(f"Error in batch add embeddings: {e}")
            return {"success_count": 0, "error_count": len(messages), "errors": [str(e)]}
    
    async def delete_message_embedding(self, message_id: str) -> bool:
        """
        Delete embedding for a message.
        
        Args:
            message_id: Message identifier to delete
            
        Returns:
            True if successful
        """
        start_time = datetime.utcnow()
        
        try:
            if not self.server:
                raise RuntimeError("MCP server not initialized")
            
            success = await self.server.delete_message_embedding(message_id)
            
            if success:
                self._record_success(start_time)
            else:
                self._record_failure(start_time)
            
            return success
            
        except Exception as e:
            self._record_failure(start_time)
            logger.error(f"Error deleting message embedding: {e}")
            return False
    
    async def get_embedding_stats(self) -> Dict[str, Any]:
        """Get embedding statistics."""
        try:
            if not self.server:
                raise RuntimeError("MCP server not initialized")
            
            return await self.server.get_embedding_stats()
            
        except Exception as e:
            logger.error(f"Error getting embedding stats: {e}")
            return {"total_embeddings": 0, "embeddings_by_model": {}}
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on MCP server."""
        try:
            if not self.server:
                raise RuntimeError("MCP server not initialized")
            
            return await self.server.health_check()
            
        except Exception as e:
            logger.error(f"Error in health check: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def _record_success(self, start_time: datetime):
        """Record successful request."""
        self.stats["total_requests"] += 1
        self.stats["successful_requests"] += 1
        
        response_time = (datetime.utcnow() - start_time).total_seconds()
        self.stats["response_times"].append(response_time)
        
        # Keep only last 100 response times
        if len(self.stats["response_times"]) > 100:
            self.stats["response_times"] = self.stats["response_times"][-100:]
    
    def _record_failure(self, start_time: datetime):
        """Record failed request."""
        self.stats["total_requests"] += 1
        self.stats["failed_requests"] += 1
        
        response_time = (datetime.utcnow() - start_time).total_seconds()
        self.stats["response_times"].append(response_time)
        
        # Keep only last 100 response times
        if len(self.stats["response_times"]) > 100:
            self.stats["response_times"] = self.stats["response_times"][-100:]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get client statistics."""
        response_times = self.stats["response_times"]
        
        return {
            **self.stats,
            "avg_response_time": sum(response_times) / len(response_times) if response_times else 0,
            "min_response_time": min(response_times) if response_times else 0,
            "max_response_time": max(response_times) if response_times else 0,
            "success_rate": (
                self.stats["successful_requests"] / self.stats["total_requests"]
                if self.stats["total_requests"] > 0 else 0
            )
        }
    
    async def close(self):
        """Close the MCP client."""
        try:
            if self.server:
                await self.server.close()
            
            logger.info("MCP Client closed successfully")
            
        except Exception as e:
            logger.error(f"Error closing MCP client: {e}") 