"""
Search Service for MCP Server

Handles semantic search operations using embeddings and SQLite metadata storage.
"""

import asyncio
import json
import logging
import sqlite3
import hashlib
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

from .embedding_service import EmbeddingService

logger = logging.getLogger(__name__)


class SearchService:
    """
    Service for semantic search operations.
    
    Uses embeddings for similarity search and SQLite for metadata storage.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Database configuration
        self.db_path = Path(self.config.get("db_path", "data/discord_messages.db"))
        self.embeddings_db_path = Path(self.config.get("embeddings_db_path", "data/embeddings.db"))
        
        # Initialize embedding service
        self.embedding_service = EmbeddingService(config.get("embedding", {}))
        
        # Cache configuration
        self.cache_ttl = self.config.get("cache_ttl", 1800)  # 30 minutes
        
        # Performance tracking
        self.stats = {
            "total_searches": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "embedding_searches": 0,
            "filter_searches": 0,
            "errors": 0
        }
        
        # Initialize database
        self._init_database()
        
        logger.info("SearchService initialized")
    
    def _init_database(self):
        """Initialize SQLite database with required tables."""
        try:
            # Ensure directory exists
            self.embeddings_db_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create embeddings database
            with sqlite3.connect(self.embeddings_db_path) as conn:
                cursor = conn.cursor()
                
                # Message embeddings table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS message_embeddings (
                        message_id TEXT PRIMARY KEY,
                        embedding_vector BLOB NOT NULL,
                        embedding_model TEXT NOT NULL,
                        content_hash TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Search cache table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS search_cache (
                        query_hash TEXT PRIMARY KEY,
                        results TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        expires_at TIMESTAMP NOT NULL,
                        cache_hits INTEGER DEFAULT 0
                    )
                """)
                
                # Embedding model registry
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS embedding_models (
                        model_name TEXT PRIMARY KEY,
                        model_type TEXT NOT NULL,
                        dimensions INTEGER NOT NULL,
                        is_active BOOLEAN DEFAULT TRUE,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create indices
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_embeddings_model ON message_embeddings(embedding_model)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_embeddings_content_hash ON message_embeddings(content_hash)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_cache_expires ON search_cache(expires_at)")
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
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
            # Check cache first
            cache_key = self._generate_cache_key(query, k, filters, model)
            cached_results = await self._get_cached_results(cache_key)
            
            if cached_results:
                self.stats["cache_hits"] += 1
                return cached_results
            
            # Generate query embedding
            query_embedding = await self.embedding_service.generate_embedding(query, model)
            
            # Get filtered message IDs if filters provided
            filtered_message_ids = None
            if filters:
                filtered_message_ids = await self._apply_filters(filters)
            
            # Perform similarity search
            results = await self._perform_similarity_search(
                query_embedding, k, filtered_message_ids, min_score, model
            )
            
            # Cache results
            await self._cache_results(cache_key, results)
            
            self.stats["total_searches"] += 1
            self.stats["embedding_searches"] += 1
            
            logger.info(f"Similarity search found {len(results)} results")
            return results
            
        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"Error in similarity search: {e}")
            return []
    
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
            # Check cache
            cache_key = self._generate_cache_key("", k, filters, None, "filter")
            cached_results = await self._get_cached_results(cache_key)
            
            if cached_results:
                self.stats["cache_hits"] += 1
                return cached_results
            
            # Apply filters and get results
            results = await self._apply_filters_and_sort(filters, k, sort_by)
            
            # Cache results
            await self._cache_results(cache_key, results)
            
            self.stats["total_searches"] += 1
            self.stats["filter_searches"] += 1
            
            logger.info(f"Filter search found {len(results)} results")
            return results
            
        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"Error in filter search: {e}")
            return []
    
    async def _perform_similarity_search(
        self,
        query_embedding: List[float],
        k: int,
        filtered_message_ids: Optional[List[str]],
        min_score: float,
        model: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Perform the actual similarity search using embeddings."""
        try:
            with sqlite3.connect(self.embeddings_db_path) as conn:
                cursor = conn.cursor()
                
                # Build query
                query = """
                    SELECT me.message_id, me.embedding_vector, me.embedding_model
                    FROM message_embeddings me
                    WHERE me.embedding_model = ?
                """
                params = [model or self.embedding_service.default_model]
                
                if filtered_message_ids:
                    placeholders = ','.join(['?' for _ in filtered_message_ids])
                    query += f" AND me.message_id IN ({placeholders})"
                    params.extend(filtered_message_ids)
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                if not rows:
                    return []
                
                # Calculate similarities
                similarities = []
                for row in rows:
                    message_id, embedding_blob, embedding_model = row
                    
                    # Convert blob to list
                    embedding = self._blob_to_list(embedding_blob)
                    
                    # Calculate cosine similarity
                    similarity = self._cosine_similarity(query_embedding, embedding)
                    
                    if similarity >= min_score:
                        similarities.append((message_id, similarity))
                
                # Sort by similarity and take top k
                similarities.sort(key=lambda x: x[1], reverse=True)
                top_k = similarities[:k]
                
                # Get metadata for top results
                results = []
                for message_id, score in top_k:
                    metadata = await self._get_message_metadata(message_id)
                    if metadata:
                        results.append({
                            "message_id": message_id,
                            "content": metadata.get("content", ""),
                            "score": score,
                            "author": {
                                "id": metadata.get("author_id", ""),
                                "username": metadata.get("author_username", ""),
                                "display_name": metadata.get("author_display_name", "")
                            },
                            **metadata
                        })
                
                return results
                
        except Exception as e:
            logger.error(f"Error in similarity search: {e}")
            raise
    
    async def _apply_filters(self, filters: Dict[str, Any]) -> List[str]:
        """Apply filters to get matching message IDs."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Build WHERE clause
                where_clause, params = self._build_where_clause(filters)
                
                query = f"SELECT message_id FROM messages WHERE {where_clause}"
                cursor.execute(query, params)
                
                return [row[0] for row in cursor.fetchall()]
                
        except Exception as e:
            logger.error(f"Error applying filters: {e}")
            return []
    
    async def _apply_filters_and_sort(
        self,
        filters: Dict[str, Any],
        k: int,
        sort_by: str
    ) -> List[Dict[str, Any]]:
        """Apply filters and sort to get results."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Build WHERE clause
                where_clause, params = self._build_where_clause(filters)
                
                # Build ORDER BY clause
                order_clause = self._build_order_clause(sort_by)
                
                query = f"""
                    SELECT * FROM messages 
                    WHERE {where_clause}
                    ORDER BY {order_clause}
                    LIMIT ?
                """
                params.append(k)
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                # Convert to dictionaries
                results = []
                for row in rows:
                    result = dict(zip([col[0] for col in cursor.description], row))
                    results.append(result)
                
                return results
                
        except Exception as e:
            logger.error(f"Error in filter search: {e}")
            return []
    
    async def _get_message_metadata(self, message_id: str) -> Optional[Dict[str, Any]]:
        """Get message metadata from the main database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM messages WHERE message_id = ?", (message_id,))
                row = cursor.fetchone()
                
                if row:
                    return dict(zip([col[0] for col in cursor.description], row))
                return None
                
        except Exception as e:
            logger.error(f"Error getting message metadata: {e}")
            return None
    
    def _build_where_clause(self, filters: Dict[str, Any]) -> Tuple[str, List[Any]]:
        """Build SQL WHERE clause from filters."""
        conditions = []
        params = []
        
        for key, value in filters.items():
            if value is None:
                continue
            
            if isinstance(value, list):
                placeholders = ','.join(['?' for _ in value])
                conditions.append(f"{key} IN ({placeholders})")
                params.extend(value)
            elif isinstance(value, dict):
                # Handle operators like $gt, $lt, etc.
                for op, op_value in value.items():
                    if op == "$gt":
                        conditions.append(f"{key} > ?")
                        params.append(op_value)
                    elif op == "$lt":
                        conditions.append(f"{key} < ?")
                        params.append(op_value)
                    elif op == "$gte":
                        conditions.append(f"{key} >= ?")
                        params.append(op_value)
                    elif op == "$lte":
                        conditions.append(f"{key} <= ?")
                        params.append(op_value)
            else:
                conditions.append(f"{key} = ?")
                params.append(value)
        
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        return where_clause, params
    
    def _build_order_clause(self, sort_by: str) -> str:
        """Build SQL ORDER BY clause."""
        valid_fields = {
            "timestamp": "timestamp",
            "content_length": "LENGTH(content)",
            "author_username": "author_username",
            "channel_name": "channel_name"
        }
        
        field = valid_fields.get(sort_by, "timestamp")
        return f"{field} DESC"
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        try:
            vec1_array = np.array(vec1)
            vec2_array = np.array(vec2)
            
            dot_product = np.dot(vec1_array, vec2_array)
            norm1 = np.linalg.norm(vec1_array)
            norm2 = np.linalg.norm(vec2_array)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return float(dot_product / (norm1 * norm2))
            
        except Exception as e:
            logger.error(f"Error calculating cosine similarity: {e}")
            return 0.0
    
    def _blob_to_list(self, blob: bytes) -> List[float]:
        """Convert SQLite blob to list of floats."""
        try:
            return np.frombuffer(blob, dtype=np.float32).tolist()
        except Exception as e:
            logger.error(f"Error converting blob to list: {e}")
            return []
    
    def _list_to_blob(self, float_list: List[float]) -> bytes:
        """Convert list of floats to SQLite blob."""
        try:
            return np.array(float_list, dtype=np.float32).tobytes()
        except Exception as e:
            logger.error(f"Error converting list to blob: {e}")
            return b""
    
    def _generate_cache_key(
        self,
        query: str,
        k: int,
        filters: Optional[Dict[str, Any]],
        model: Optional[str],
        search_type: str = "similarity"
    ) -> str:
        """Generate cache key for search results."""
        key_data = {
            "query": query,
            "k": k,
            "filters": filters,
            "model": model,
            "type": search_type
        }
        
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    async def _get_cached_results(self, cache_key: str) -> Optional[List[Dict[str, Any]]]:
        """Get cached search results."""
        try:
            with sqlite3.connect(self.embeddings_db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT results FROM search_cache 
                    WHERE query_hash = ? AND expires_at > ?
                """, (cache_key, datetime.utcnow().isoformat()))
                
                row = cursor.fetchone()
                if row:
                    # Update cache hits
                    cursor.execute("""
                        UPDATE search_cache 
                        SET cache_hits = cache_hits + 1 
                        WHERE query_hash = ?
                    """, (cache_key,))
                    conn.commit()
                    
                    return json.loads(row[0])
                return None
                
        except Exception as e:
            logger.error(f"Error getting cached results: {e}")
            return None
    
    async def _cache_results(self, cache_key: str, results: List[Dict[str, Any]]):
        """Cache search results."""
        try:
            expires_at = datetime.utcnow() + timedelta(seconds=self.cache_ttl)
            
            with sqlite3.connect(self.embeddings_db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT OR REPLACE INTO search_cache 
                    (query_hash, results, expires_at) 
                    VALUES (?, ?, ?)
                """, (cache_key, json.dumps(results), expires_at.isoformat()))
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error caching results: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get search service statistics."""
        return {
            **self.stats,
            "embedding_service_stats": self.embedding_service.get_stats()
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on search service."""
        health = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "checks": {}
        }
        
        try:
            # Check main database
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT COUNT(*) FROM messages")
                    count = cursor.fetchone()[0]
                    health["checks"]["main_database"] = {
                        "status": "healthy",
                        "message_count": count
                    }
            except Exception as e:
                health["checks"]["main_database"] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
                health["status"] = "degraded"
            
            # Check embeddings database
            try:
                with sqlite3.connect(self.embeddings_db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT COUNT(*) FROM message_embeddings")
                    count = cursor.fetchone()[0]
                    health["checks"]["embeddings_database"] = {
                        "status": "healthy",
                        "embedding_count": count
                    }
            except Exception as e:
                health["checks"]["embeddings_database"] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
                health["status"] = "degraded"
            
            # Check embedding service
            embedding_health = await self.embedding_service.health_check()
            health["checks"]["embedding_service"] = embedding_health
            
            if embedding_health["status"] != "healthy":
                health["status"] = "degraded"
                
        except Exception as e:
            health["status"] = "unhealthy"
            health["error"] = str(e)
        
        return health 