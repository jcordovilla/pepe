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
            # Initialize MCP server (replaces ChromaDB vector store)
            from ..mcp import MCPServer
            mcp_config = {
                "sqlite": {
                    "db_path": "data/discord_messages.db"
                },
                "llm": self.config.get("llm", {})
            }
            self.stores["mcp"] = MCPServer(mcp_config)
            
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
            logger.info("✅ Unified data manager initialized with MCP server")
            
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
            
            # Note: MCP server doesn't need explicit message storage
            # Messages are already in SQLite database and can be queried directly
            # This is a key benefit of the new architecture
            
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
            
            logger.debug(f"📦 Stored message {message_id} in memory and cache")
            return memory_result and cache_result
            
        except Exception as e:
            logger.error(f"❌ Error storing message: {e}")
            return False
    
    async def search_messages(self, query: str, filters: Optional[Dict] = None, limit: int = 10) -> List[Dict]:
        """
        Unified search using MCP server for direct SQLite queries
        """
        if not self.initialized:
            await self.initialize()
        
        try:
            # Use MCP server for natural language to SQL translation and search
            mcp_results = await self.stores["mcp"].search_messages(
                query=query,
                filters=filters,
                limit=limit
            )
            
            # Enhance with cached data and analytics
            enhanced_results = []
            for result in mcp_results:
                message_id = result.get("message_id")
                
                # Try to get from cache first (faster)
                cached = await self.stores["cache"].get(f"message_{message_id}")
                if cached:
                    enhanced_results.append(cached)
                else:
                    enhanced_results.append(result)
            
            # Track search analytics
            try:
                await self.stores["analytics"].track_search_performed(query, len(enhanced_results))
            except AttributeError:
                # PerformanceMonitor might not have this method
                logger.debug("PerformanceMonitor track_search_performed method not available")
            
            return enhanced_results
            
        except Exception as e:
            logger.error(f"❌ Error searching messages: {e}")
            return []
    
    async def query_messages(self, natural_language_query: str) -> List[Dict]:
        """
        Query messages using natural language through MCP server
        """
        if not self.initialized:
            await self.initialize()
        
        try:
            # Use MCP server for natural language to SQL translation
            results = await self.stores["mcp"].query_messages(natural_language_query)
            
            # Track query analytics
            try:
                await self.stores["analytics"].track_search_performed(natural_language_query, len(results))
            except AttributeError:
                # PerformanceMonitor might not have this method
                logger.debug("PerformanceMonitor track_search_performed method not available")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Error querying messages: {e}")
            return []
    
    async def get_message_stats(self, filters: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Get message statistics using MCP server
        """
        if not self.initialized:
            await self.initialize()
        
        try:
            return await self.stores["mcp"].get_message_stats(filters)
            
        except Exception as e:
            logger.error(f"❌ Error getting message stats: {e}")
            return {}
    
    async def get_user_activity(self, user_id: Optional[str] = None, username: Optional[str] = None, time_range: str = "7d") -> Dict[str, Any]:
        """
        Get user activity statistics using MCP server
        """
        if not self.initialized:
            await self.initialize()
        
        try:
            return await self.stores["mcp"].get_user_activity(user_id, username, time_range)
            
        except Exception as e:
            logger.error(f"❌ Error getting user activity: {e}")
            return {}
    
    async def get_channel_summary(self, channel_id: Optional[str] = None, channel_name: Optional[str] = None, time_range: str = "7d") -> Dict[str, Any]:
        """
        Get channel summary using MCP server
        """
        if not self.initialized:
            await self.initialize()
        
        try:
            return await self.stores["mcp"].get_channel_summary(channel_id, channel_name, time_range)
            
        except Exception as e:
            logger.error(f"❌ Error getting channel summary: {e}")
            return {}
    
    async def health_check(self) -> Dict[str, bool]:
        """Check health of all storage systems"""
        if not self.initialized:
            await self.initialize()
        
        health_status = {}
        
        for store_name, store in self.stores.items():
            try:
                if hasattr(store, 'health_check'):
                    health = await store.health_check()
                    health_status[store_name] = health.get("status") == "healthy"
                else:
                    health_status[store_name] = True  # Assume healthy if no health check method
            except Exception as e:
                logger.error(f"❌ Health check failed for {store_name}: {e}")
                health_status[store_name] = False
        
        return health_status
    
    async def get_analytics_summary(self) -> Dict[str, Any]:
        """Get analytics summary from all stores"""
        if not self.initialized:
            await self.initialize()
        
        try:
            summary = {}
            
            # Get MCP server stats
            if "mcp" in self.stores:
                summary["mcp_stats"] = self.stores["mcp"].get_stats()
            
            # Get analytics summary
            if "analytics" in self.stores:
                try:
                    summary["analytics"] = await self.stores["analytics"].get_summary()
                except AttributeError:
                    # PerformanceMonitor might not have this method
                    summary["analytics"] = {"status": "get_summary method not available"}
            
            return summary
            
        except Exception as e:
            logger.error(f"❌ Error getting analytics summary: {e}")
            return {}
