"""
Unified Data Access Layer
Consolidates multiple storage backends with legacy-proven patterns
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class DataStore(ABC):
    """Abstract base for all data operations"""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]: pass
    
    @abstractmethod
    async def set(self, key: str, value: Any) -> bool: pass
    
    @abstractmethod
    async def search(self, query: str, limit: int = 10) -> List[Dict]: pass
    
    @abstractmethod
    async def health_check(self) -> bool: pass

class UnifiedDataManager:
    """
    Single entry point for all data operations
    Integrates legacy-proven patterns with modern storage
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.stores = {}
        self.initialized = False
        
        logger.info("🏗️ Unified data manager initializing...")
    
    async def initialize(self):
        """Initialize all storage backends"""
        if self.initialized:
            return
        
        try:
            # Initialize vector store (ChromaDB)
            from ..vectorstore.persistent_store import PersistentVectorStore
            self.stores["vector"] = PersistentVectorStore(self.config.get("vector_config", {}))
            
            # Initialize memory store (SQLite)
            from ..memory.conversation_memory import ConversationMemory
            self.stores["memory"] = ConversationMemory(self.config.get("memory_config", {}))
            
            # Initialize cache store
            from ..cache.smart_cache import SmartCache
            self.stores["cache"] = SmartCache(self.config.get("cache_config", {}))
            
            # Initialize analytics store
            from ..analytics.performance_monitor import PerformanceMonitor
            self.stores["analytics"] = PerformanceMonitor(self.config.get("analytics_config", {}))
            
            self.initialized = True
            logger.info("✅ Unified data manager initialized")
            
        except Exception as e:
            logger.error(f"❌ Error initializing data manager: {e}")
            raise
    
    async def store_message(self, message: Dict[str, Any]) -> bool:
        """
        Store message across all relevant storage systems
        Uses legacy-proven batch processing patterns
        """
        if not self.initialized:
            await self.initialize()
        
        try:
            message_id = message.get("message_id")
            
            # Store in vector store for semantic search
            vector_result = await self.stores["vector"].add_messages([message])
            
            # Store in memory for conversation tracking
            memory_result = await self.stores["memory"].store_conversation_turn(
                user_id=message.get("author", {}).get("id"),
                message=message.get("content", ""),
                metadata=message
            )
            
            # Cache for quick access
            cache_key = f"message_{message_id}"
            cache_result = await self.stores["cache"].set(cache_key, message, ttl=3600)
            
            # Track in analytics
            await self.stores["analytics"].track_message_processed(message)
            
            logger.debug(f"📦 Stored message {message_id} across all stores")
            return vector_result and memory_result and cache_result
            
        except Exception as e:
            logger.error(f"❌ Error storing message: {e}")
            return False
    
    async def search_messages(self, query: str, filters: Optional[Dict] = None, limit: int = 10) -> List[Dict]:
        """
        Unified search across all relevant stores
        """
        if not self.initialized:
            await self.initialize()
        
        try:
            # Primary search through vector store
            vector_results = await self.stores["vector"].similarity_search(
                query=query,
                limit=limit,
                filters=filters
            )
            
            # Enhance with cached data and analytics
            enhanced_results = []
            for result in vector_results:
                message_id = result.get("message_id")
                
                # Try to get from cache first (faster)
                cached = await self.stores["cache"].get(f"message_{message_id}")
                if cached:
                    enhanced_results.append(cached)
                else:
                    enhanced_results.append(result)
            
            # Track search analytics
            await self.stores["analytics"].track_search_performed(query, len(enhanced_results))
            
            return enhanced_results
            
        except Exception as e:
            logger.error(f"❌ Error searching messages: {e}")
            return []
    
    async def health_check(self) -> Dict[str, bool]:
        """Check health of all storage systems"""
        if not self.initialized:
            await self.initialize()
        
        health_status = {}
        
        for store_name, store in self.stores.items():
            try:
                health_status[store_name] = await store.health_check()
            except Exception as e:
                logger.warning(f"⚠️ Health check failed for {store_name}: {e}")
                health_status[store_name] = False
        
        return health_status
    
    async def get_analytics_summary(self) -> Dict[str, Any]:
        """Get comprehensive analytics summary"""
        if not self.initialized:
            await self.initialize()
        
        try:
            return await self.stores["analytics"].get_summary()
        except Exception as e:
            logger.error(f"❌ Error getting analytics: {e}")
            return {}
