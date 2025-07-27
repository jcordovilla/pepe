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
    
    async def query_messages(self, natural_language_query: str) -> List[Dict[str, Any]]:
        """
        Query messages using natural language.
        
        Args:
            natural_language_query: Natural language query about Discord messages
            
        Returns:
            List of message results
        """
        start_time = datetime.utcnow()
        
        try:
            if not self.server:
                raise RuntimeError("MCP server not initialized")
            
            results = await self.server.query_messages(natural_language_query)
            
            self._record_success(start_time)
            return results
            
        except Exception as e:
            self._record_failure(start_time)
            logger.error(f"Error querying messages: {e}")
            raise
    
    async def get_message_stats(self, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Get comprehensive message statistics.
        
        Args:
            filters: Optional filters to apply
            
        Returns:
            Dictionary with various statistics
        """
        start_time = datetime.utcnow()
        
        try:
            if not self.server:
                raise RuntimeError("MCP server not initialized")
            
            stats = await self.server.get_message_stats(filters)
            
            self._record_success(start_time)
            return stats
            
        except Exception as e:
            self._record_failure(start_time)
            logger.error(f"Error getting message stats: {e}")
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
        start_time = datetime.utcnow()
        
        try:
            if not self.server:
                raise RuntimeError("MCP server not initialized")
            
            results = await self.server.search_messages(query, filters, limit)
            
            self._record_success(start_time)
            return results
            
        except Exception as e:
            self._record_failure(start_time)
            logger.error(f"Error searching messages: {e}")
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
        start_time = datetime.utcnow()
        
        try:
            if not self.server:
                raise RuntimeError("MCP server not initialized")
            
            activity = await self.server.get_user_activity(user_id, username, time_range)
            
            self._record_success(start_time)
            return activity
            
        except Exception as e:
            self._record_failure(start_time)
            logger.error(f"Error getting user activity: {e}")
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
        start_time = datetime.utcnow()
        
        try:
            if not self.server:
                raise RuntimeError("MCP server not initialized")
            
            summary = await self.server.get_channel_summary(channel_id, channel_name, time_range)
            
            self._record_success(start_time)
            return summary
            
        except Exception as e:
            self._record_failure(start_time)
            logger.error(f"Error getting channel summary: {e}")
            raise
    
    async def get_database_info(self) -> Dict[str, Any]:
        """
        Get database information and statistics.
        
        Returns:
            Database information
        """
        try:
            if not self.server:
                raise RuntimeError("MCP server not initialized")
            
            return await self.server.get_database_info()
            
        except Exception as e:
            logger.error(f"Error getting database info: {e}")
            return {"error": str(e)}
    
    async def test_query(self, query: str = "show me 5 recent messages") -> Dict[str, Any]:
        """
        Test the query system with a sample query.
        
        Args:
            query: Test query to execute
            
        Returns:
            Test results
        """
        try:
            if not self.server:
                raise RuntimeError("MCP server not initialized")
            
            return await self.server.test_query(query)
            
        except Exception as e:
            logger.error(f"Error in test query: {e}")
            return {
                "success": False,
                "query": query,
                "error": str(e)
            }
    
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