"""
MCP Server Implementation

Main server that provides direct SQLite access for Discord message analysis.
"""

import asyncio
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from .sqlite_query_service import SQLiteQueryService

logger = logging.getLogger(__name__)


class MCPServer:
    """
    Main MCP server that provides Discord message analysis through direct SQLite queries.
    
    Provides a unified interface for:
    - Natural language to SQL translation
    - Direct database queries
    - User and channel analysis
    - Message statistics
    - Health monitoring
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Initialize SQLite query service
        self.query_service = SQLiteQueryService(config.get("sqlite", {}))
        
        # Performance tracking
        self.stats = {
            "total_requests": 0,
            "query_requests": 0,
            "stats_requests": 0,
            "user_analysis_requests": 0,
            "channel_analysis_requests": 0,
            "errors": 0,
            "start_time": datetime.utcnow().isoformat()
        }
        
        logger.info("MCP Server initialized with SQLite query service")
    
    async def query_messages(self, natural_language_query: str) -> List[Dict[str, Any]]:
        """
        Query messages using natural language.
        
        Args:
            natural_language_query: Natural language query about Discord messages
            
        Returns:
            List of message results
        """
        try:
            self.stats["total_requests"] += 1
            self.stats["query_requests"] += 1
            
            results = await self.query_service.query_messages(natural_language_query)
            
            logger.info(f"Query executed: {natural_language_query[:50]}... -> {len(results)} results")
            return results
            
        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"Error in query_messages: {e}")
            raise
    
    async def get_message_stats(self, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Get comprehensive message statistics.
        
        Args:
            filters: Optional filters to apply
            
        Returns:
            Dictionary with various statistics
        """
        try:
            self.stats["total_requests"] += 1
            self.stats["stats_requests"] += 1
            
            stats = await self.query_service.get_message_stats(filters)
            
            logger.info(f"Message stats retrieved with filters: {filters}")
            return stats
            
        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"Error in get_message_stats: {e}")
            raise
    
    async def search_messages(
        self, 
        query: str, 
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Search messages using text search and filters.
        
        Args:
            query: Text to search for
            filters: Optional metadata filters
            limit: Maximum number of results
            
        Returns:
            List of matching messages
        """
        try:
            self.stats["total_requests"] += 1
            self.stats["query_requests"] += 1
            
            results = await self.query_service.search_messages(query, filters, limit)
            
            logger.info(f"Search executed: {query} -> {len(results)} results")
            return results
            
        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"Error in search_messages: {e}")
            raise
    
    async def get_user_activity(
        self, 
        user_id: Optional[str] = None,
        username: Optional[str] = None,
        time_range: str = "7d"
    ) -> Dict[str, Any]:
        """
        Get user activity statistics.
        
        Args:
            user_id: User ID to analyze
            username: Username to analyze (alternative to user_id)
            time_range: Time range (e.g., "7d", "30d", "1y")
            
        Returns:
            User activity statistics
        """
        try:
            self.stats["total_requests"] += 1
            self.stats["user_analysis_requests"] += 1
            
            activity = await self.query_service.get_user_activity(user_id, username, time_range)
            
            logger.info(f"User activity retrieved for {user_id or username} over {time_range}")
            return activity
            
        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"Error in get_user_activity: {e}")
            raise
    
    async def get_channel_summary(
        self, 
        channel_id: Optional[str] = None,
        channel_name: Optional[str] = None,
        time_range: str = "7d"
    ) -> Dict[str, Any]:
        """
        Get channel activity summary.
        
        Args:
            channel_id: Channel ID to analyze
            channel_name: Channel name to analyze (alternative to channel_id)
            time_range: Time range (e.g., "7d", "30d", "1y")
            
        Returns:
            Channel activity summary
        """
        try:
            self.stats["total_requests"] += 1
            self.stats["channel_analysis_requests"] += 1
            
            summary = await self.query_service.get_channel_summary(channel_id, channel_name, time_range)
            
            logger.info(f"Channel summary retrieved for {channel_id or channel_name} over {time_range}")
            return summary
            
        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"Error in get_channel_summary: {e}")
            raise
    
    async def get_database_info(self) -> Dict[str, Any]:
        """
        Get database information and statistics.
        
        Returns:
            Database information
        """
        try:
            # Get basic stats
            stats = await self.query_service.get_message_stats()
            
            # Get query service stats
            query_stats = self.query_service.get_stats()
            
            return {
                "database_stats": stats,
                "query_service_stats": query_stats,
                "server_stats": self.get_stats()
            }
            
        except Exception as e:
            logger.error(f"Error getting database info: {e}")
            return {"error": str(e)}
    
    def get_stats(self) -> Dict[str, Any]:
        """Get MCP server statistics."""
        return {
            **self.stats,
            "query_service_stats": self.query_service.get_stats(),
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
            # Check query service
            query_health = await self.query_service.health_check()
            health["services"]["query_service"] = query_health
            
            # Determine overall status
            if query_health["status"] != "healthy":
                health["status"] = "degraded"
            
            # Add server stats
            health["stats"] = self.get_stats()
            
        except Exception as e:
            health["status"] = "unhealthy"
            health["error"] = str(e)
        
        return health
    
    async def test_query(self, query: str = "show me 5 recent messages") -> Dict[str, Any]:
        """
        Test the query system with a sample query.
        
        Args:
            query: Test query to execute
            
        Returns:
            Test results
        """
        try:
            start_time = datetime.utcnow()
            
            # Execute test query
            results = await self.query_messages(query)
            
            end_time = datetime.utcnow()
            response_time = (end_time - start_time).total_seconds()
            
            return {
                "success": True,
                "query": query,
                "results_count": len(results),
                "response_time": response_time,
                "sample_results": results[:3] if results else []
            }
            
        except Exception as e:
            return {
                "success": False,
                "query": query,
                "error": str(e)
            }
    
    async def close(self):
        """Close the MCP server and cleanup resources."""
        try:
            logger.info("MCP Server closed successfully")
            
        except Exception as e:
            logger.error(f"Error closing MCP server: {e}") 