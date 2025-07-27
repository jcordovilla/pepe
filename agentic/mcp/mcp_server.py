"""
MCP Server Implementation

Main server that coordinates embedding generation and semantic search operations.
"""

import asyncio
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from .embedding_service import EmbeddingService
from .search_service import SearchService

logger = logging.getLogger(__name__)


class MCPServer:
    """
    Main MCP server that coordinates embedding and search operations.
    
    Provides a unified interface for:
    - Embedding generation
    - Semantic search
    - Batch operations
    - Health monitoring
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Initialize services
        self.embedding_service = EmbeddingService(config.get("embedding", {}))
        self.search_service = SearchService(config.get("search", {}))
        
        # Performance tracking
        self.stats = {
            "total_requests": 0,
            "embedding_requests": 0,
            "search_requests": 0,
            "batch_requests": 0,
            "errors": 0,
            "start_time": datetime.utcnow().isoformat()
        }
        
        logger.info("MCP Server initialized")
    
    async def generate_embedding(self, text: str, model: Optional[str] = None) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            model: Model to use (optional)
            
        Returns:
            List of floats representing the embedding
        """
        try:
            self.stats["total_requests"] += 1
            self.stats["embedding_requests"] += 1
            
            embedding = await self.embedding_service.generate_embedding(text, model)
            
            logger.debug(f"Generated embedding for text (length: {len(text)})")
            return embedding
            
        except Exception as e:
            self.stats["errors"] += 1
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
        try:
            self.stats["total_requests"] += 1
            self.stats["batch_requests"] += 1
            
            embeddings = await self.embedding_service.batch_embed(texts, model)
            
            logger.info(f"Generated {len(embeddings)} embeddings in batch")
            return embeddings
            
        except Exception as e:
            self.stats["errors"] += 1
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
        try:
            self.stats["total_requests"] += 1
            self.stats["search_requests"] += 1
            
            results = await self.search_service.similarity_search(
                query=query,
                k=k,
                filters=filters,
                min_score=min_score,
                model=model
            )
            
            logger.info(f"Similarity search returned {len(results)} results")
            return results
            
        except Exception as e:
            self.stats["errors"] += 1
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
        try:
            self.stats["total_requests"] += 1
            self.stats["search_requests"] += 1
            
            results = await self.search_service.filter_search(
                filters=filters,
                k=k,
                sort_by=sort_by
            )
            
            logger.info(f"Filter search returned {len(results)} results")
            return results
            
        except Exception as e:
            self.stats["errors"] += 1
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
        try:
            # Generate embedding
            embedding = await self.embedding_service.generate_embedding(content, model)
            
            # Store in database
            await self._store_embedding(message_id, embedding, model or self.embedding_service.default_model)
            
            logger.debug(f"Added embedding for message {message_id}")
            return True
            
        except Exception as e:
            self.stats["errors"] += 1
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
        try:
            if not messages:
                return {"success_count": 0, "error_count": 0, "errors": []}
            
            # Extract content for batch embedding
            texts = [msg.get("content", "") for msg in messages]
            message_ids = [msg.get("message_id", "") for msg in messages]
            
            # Generate embeddings in batch
            embeddings = await self.embedding_service.batch_embed(texts, model)
            
            # Store embeddings
            success_count = 0
            error_count = 0
            errors = []
            
            model_name = model or self.embedding_service.default_model
            
            for i, (message_id, embedding) in enumerate(zip(message_ids, embeddings)):
                try:
                    await self._store_embedding(message_id, embedding, model_name)
                    success_count += 1
                except Exception as e:
                    error_count += 1
                    errors.append(f"Message {message_id}: {str(e)}")
            
            logger.info(f"Batch added {success_count} embeddings, {error_count} errors")
            return {
                "success_count": success_count,
                "error_count": error_count,
                "errors": errors
            }
            
        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"Error in batch add embeddings: {e}")
            return {"success_count": 0, "error_count": len(messages), "errors": [str(e)]}
    
    async def _store_embedding(self, message_id: str, embedding: List[float], model: str):
        """Store embedding in the database."""
        try:
            import sqlite3
            from .search_service import SearchService
            
            # Use the search service's database path
            embeddings_db_path = self.search_service.embeddings_db_path
            
            with sqlite3.connect(embeddings_db_path) as conn:
                cursor = conn.cursor()
                
                # Convert embedding to blob
                embedding_blob = self.search_service._list_to_blob(embedding)
                
                # Store or update embedding
                cursor.execute("""
                    INSERT OR REPLACE INTO message_embeddings 
                    (message_id, embedding_vector, embedding_model, updated_at)
                    VALUES (?, ?, ?, ?)
                """, (message_id, embedding_blob, model, datetime.utcnow().isoformat()))
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error storing embedding: {e}")
            raise
    
    async def delete_message_embedding(self, message_id: str) -> bool:
        """
        Delete embedding for a message.
        
        Args:
            message_id: Message identifier to delete
            
        Returns:
            True if successful
        """
        try:
            import sqlite3
            
            embeddings_db_path = self.search_service.embeddings_db_path
            
            with sqlite3.connect(embeddings_db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM message_embeddings WHERE message_id = ?", (message_id,))
                conn.commit()
                
                deleted = cursor.rowcount > 0
                if deleted:
                    logger.debug(f"Deleted embedding for message {message_id}")
                else:
                    logger.warning(f"No embedding found for message {message_id}")
                
                return deleted
                
        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"Error deleting message embedding: {e}")
            return False
    
    async def get_embedding_stats(self) -> Dict[str, Any]:
        """Get embedding statistics."""
        try:
            import sqlite3
            
            embeddings_db_path = self.search_service.embeddings_db_path
            
            with sqlite3.connect(embeddings_db_path) as conn:
                cursor = conn.cursor()
                
                # Get total embeddings
                cursor.execute("SELECT COUNT(*) FROM message_embeddings")
                total_embeddings = cursor.fetchone()[0]
                
                # Get embeddings by model
                cursor.execute("""
                    SELECT embedding_model, COUNT(*) 
                    FROM message_embeddings 
                    GROUP BY embedding_model
                """)
                embeddings_by_model = dict(cursor.fetchall())
                
                return {
                    "total_embeddings": total_embeddings,
                    "embeddings_by_model": embeddings_by_model
                }
                
        except Exception as e:
            logger.error(f"Error getting embedding stats: {e}")
            return {"total_embeddings": 0, "embeddings_by_model": {}}
    
    def get_stats(self) -> Dict[str, Any]:
        """Get MCP server statistics."""
        return {
            **self.stats,
            "embedding_service_stats": self.embedding_service.get_stats(),
            "search_service_stats": self.search_service.get_stats(),
            "uptime": (datetime.utcnow() - datetime.fromisoformat(self.stats["start_time"])).total_seconds()
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        health = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "services": {}
        }
        
        try:
            # Check embedding service
            embedding_health = await self.embedding_service.health_check()
            health["services"]["embedding"] = embedding_health
            
            # Check search service
            search_health = await self.search_service.health_check()
            health["services"]["search"] = search_health
            
            # Determine overall status
            if embedding_health["status"] != "healthy" or search_health["status"] != "healthy":
                health["status"] = "degraded"
            
            # Add server stats
            health["stats"] = self.get_stats()
            
        except Exception as e:
            health["status"] = "unhealthy"
            health["error"] = str(e)
        
        return health
    
    async def cleanup_cache(self) -> Dict[str, Any]:
        """Clean up expired cache entries."""
        try:
            import sqlite3
            from datetime import datetime
            
            embeddings_db_path = self.search_service.embeddings_db_path
            
            with sqlite3.connect(embeddings_db_path) as conn:
                cursor = conn.cursor()
                
                # Delete expired cache entries
                cursor.execute("""
                    DELETE FROM search_cache 
                    WHERE expires_at < ?
                """, (datetime.utcnow().isoformat(),))
                
                deleted_count = cursor.rowcount
                conn.commit()
                
                logger.info(f"Cleaned up {deleted_count} expired cache entries")
                
                return {
                    "deleted_entries": deleted_count,
                    "timestamp": datetime.utcnow().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Error cleaning up cache: {e}")
            return {"deleted_entries": 0, "error": str(e)}
    
    async def close(self):
        """Close the MCP server and cleanup resources."""
        try:
            # Clean up cache
            await self.cleanup_cache()
            
            logger.info("MCP Server closed successfully")
            
        except Exception as e:
            logger.error(f"Error closing MCP server: {e}") 