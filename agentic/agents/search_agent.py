"""
Search Agent

Specialized agent for search operations using MCP server and direct SQLite queries.
"""

from typing import Dict, List, Any, Optional, Union
import logging
from datetime import datetime
import asyncio
import dateutil.parser

from .base_agent import BaseAgent, AgentRole, AgentState, SubTask, TaskStatus, agent_registry
from ..mcp import MCPServer
from ..cache.smart_cache import SmartCache
from ..utils.k_value_calculator import KValueCalculator

logger = logging.getLogger(__name__)


class SearchAgent(BaseAgent):
    """
    Agent responsible for all search-related operations.
    
    This agent:
    - Performs natural language to SQL translation using MCP server
    - Handles direct SQLite queries for message retrieval
    - Manages search result ranking and filtering
    - Implements smart caching for performance
    - Uses dynamic k-value calculation based on query analysis
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(AgentRole.SEARCHER, config)
        
        # Initialize MCP server (replaces ChromaDB vector store)
        mcp_config = {
            "sqlite": {
                "db_path": "data/discord_messages.db"
            },
            "llm": config.get("llm", {})
        }
        self.mcp_server = MCPServer(mcp_config)
        self.cache = SmartCache(config.get("cache", {}))
        
        # Initialize dynamic k-value calculator
        self.k_calculator = KValueCalculator(config)
        
        # Legacy configuration (kept for backward compatibility)
        self.default_k = config.get("default_k", 10)
        self.max_k = config.get("max_k", 100)
        self.similarity_threshold = config.get("similarity_threshold", 0.7)
        self.use_reranking = config.get("use_reranking", True)
        
        # Performance tracking
        self.search_stats = {
            "total_searches": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "dynamic_k_used": 0,
            "large_k_searches": 0  # k > 50
        }
        
        logger.info(f"SearchAgent initialized with MCP server")
        
        # Register this agent
        agent_registry.register_agent(self)
    
    def signature(self) -> Dict[str, Any]:
        """Return agent signature for registration."""
        return {
            "role": "searcher",
            "description": "Performs search operations using MCP server and SQLite",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "filters": {"type": "object", "description": "Optional filters"},
                    "limit": {"type": "integer", "description": "Maximum number of results"}
                },
                "required": ["query"]
            },
            "output_schema": {
                "type": "array",
                "items": {"type": "object"},
                "description": "List of search results"
            }
        }
    
    async def process(self, input_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Process a search query and return relevant results.
        
        Args:
            input_data: Dictionary containing search parameters
            
        Returns:
            List of search results
        """
        try:
            query = input_data.get("query", "")
            filters = input_data.get("filters", {})
            limit = input_data.get("limit", self.default_k)
            
            if not query:
                return []
            
            # Calculate appropriate k value based on query
            k_value = self.k_calculator.calculate_k_value(query)
            actual_limit = min(limit, k_value, self.max_k)
            
            logger.info(f"SearchAgent processing query with k={actual_limit}: {query[:50]}...")
            
            # Check cache first
            cache_key = f"search:{hash(query)}:{hash(str(filters))}:{actual_limit}"
            cached_results = await self.cache.get(cache_key)
            
            if cached_results:
                self.search_stats["cache_hits"] += 1
                logger.info(f"Cache hit for search query: {query[:30]}...")
                return cached_results
            
            self.search_stats["cache_misses"] += 1
            
            # Perform search using MCP server
            results = await self.mcp_server.search_messages(
                query=query,
                filters=filters,
                limit=actual_limit
            )
            
            # Track search statistics
            self.search_stats["total_searches"] += 1
            if actual_limit > 50:
                self.search_stats["large_k_searches"] += 1
            if actual_limit != self.default_k:
                self.search_stats["dynamic_k_used"] += 1
            
            # Cache results
            await self.cache.set(cache_key, results, ttl=1800)  # 30 minutes
            
            logger.info(f"Search completed: {len(results)} results for '{query[:30]}...'")
            return results
            
        except Exception as e:
            logger.error(f"Error in SearchAgent: {e}")
            return []
    
    async def natural_language_query(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Execute a natural language query using MCP server.
        
        Args:
            query: Natural language query
            limit: Maximum number of results
            
        Returns:
            List of query results
        """
        try:
            logger.info(f"Natural language query: {query[:50]}...")
            
            # Use MCP server for natural language to SQL translation
            results = await self.mcp_server.query_messages(query)
            
            # Limit results
            return results[:limit]
            
        except Exception as e:
            logger.error(f"Error in natural language query: {e}")
            return []
    
    async def get_message_stats(self, filters: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Get message statistics using MCP server.
        
        Args:
            filters: Optional filters to apply
            
        Returns:
            Dictionary with statistics
        """
        try:
            return await self.mcp_server.get_message_stats(filters)
            
        except Exception as e:
            logger.error(f"Error getting message stats: {e}")
            return {}
    
    async def get_user_activity(self, user_id: Optional[str] = None, username: Optional[str] = None, time_range: str = "7d") -> Dict[str, Any]:
        """
        Get user activity statistics using MCP server.
        
        Args:
            user_id: User ID to analyze
            username: Username to analyze (alternative to user_id)
            time_range: Time range (e.g., "7d", "30d", "1y")
            
        Returns:
            User activity statistics
        """
        try:
            return await self.mcp_server.get_user_activity(user_id, username, time_range)
            
        except Exception as e:
            logger.error(f"Error getting user activity: {e}")
            return {}
    
    async def get_channel_summary(self, channel_id: Optional[str] = None, channel_name: Optional[str] = None, time_range: str = "7d") -> Dict[str, Any]:
        """
        Get channel summary using MCP server.
        
        Args:
            channel_id: Channel ID to analyze
            channel_name: Channel name to analyze (alternative to channel_id)
            time_range: Time range (e.g., "7d", "30d", "1y")
            
        Returns:
            Channel summary statistics
        """
        try:
            return await self.mcp_server.get_channel_summary(channel_id, channel_name, time_range)
            
        except Exception as e:
            logger.error(f"Error getting channel summary: {e}")
            return {}
    
    def get_search_stats(self) -> Dict[str, Any]:
        """Get search agent statistics."""
        return {
            **self.search_stats,
            "cache_hit_rate": (
                self.search_stats["cache_hits"] / (self.search_stats["cache_hits"] + self.search_stats["cache_misses"])
                if (self.search_stats["cache_hits"] + self.search_stats["cache_misses"]) > 0 else 0
            ),
            "dynamic_k_usage_rate": (
                self.search_stats["dynamic_k_used"] / self.search_stats["total_searches"]
                if self.search_stats["total_searches"] > 0 else 0
            )
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Check agent health."""
        try:
            # Test MCP server
            mcp_health = await self.mcp_server.health_check()
            
            # Test cache
            cache_test = await self.cache.set("health_test", "test", ttl=60)
            cache_working = cache_test is not None
            
            return {
                "status": "healthy" if mcp_health["status"] == "healthy" and cache_working else "degraded",
                "mcp_server": mcp_health,
                "cache": "healthy" if cache_working else "unhealthy",
                "search_stats": self.get_search_stats()
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    def can_handle(self, task) -> bool:
        """Check if this agent can handle the given task."""
        return (task.task_type == "search" or 
                "search" in task.description.lower() or
                "find" in task.description.lower() or
                "query" in task.description.lower())
