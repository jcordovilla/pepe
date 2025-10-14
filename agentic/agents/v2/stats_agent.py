"""
Stats Agent

Pure SQL analytics agent that queries the analytics database.
Provides metrics and statistics about Discord activity.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import sqlite3
import json

from ..base_agent import BaseAgent, AgentRole, AgentState
from ...config.modernized_config import get_modernized_config

logger = logging.getLogger(__name__)


class StatsAgent(BaseAgent):
    """
    Stats agent that provides analytics and metrics.
    
    Input: dict(start: dt, end: dt, gran: str)
    Output: dict (metrics JSON)
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(AgentRole.ANALYZER, config)
        
        # Get analytics database path from config
        analytics_config = get_modernized_config().get("data", {}).get("analytics_config", {})
        self.db_path = analytics_config.get("database_url", "sqlite:///data/analytics.db")
        
        # Convert SQLAlchemy URL to file path
        if self.db_path.startswith("sqlite:///"):
            self.db_path = self.db_path.replace("sqlite:///", "")
        elif self.db_path.startswith("sqlite://"):
            self.db_path = self.db_path.replace("sqlite://", "")
        
        logger.info(f"StatsAgent initialized with database: {self.db_path}")
    
    def signature(self) -> Dict[str, Any]:
        """Return agent signature for registration."""
        return {
            "role": "stats",
            "description": "Provides analytics and metrics from Discord activity",
            "input_schema": {
                "type": "object",
                "properties": {
                    "start": {"type": "string", "description": "Start date (ISO format)"},
                    "end": {"type": "string", "description": "End date (ISO format)"},
                    "granularity": {"type": "string", "description": "Time granularity (day, week, month)"}
                },
                "required": ["start", "end"]
            },
            "output_schema": {
                "type": "object",
                "description": "Metrics and statistics"
            }
        }
    
    async def run(self, **kwargs) -> Dict[str, Any]:
        """
        Generate statistics for the given time period.
        
        Args:
            start: Start date
            end: End date  
            granularity: Time granularity (day, week, month)
            
        Returns:
            Dict with metrics and statistics
        """
        start_str = kwargs.get("start")
        end_str = kwargs.get("end")
        granularity = kwargs.get("granularity", "day")
        
        # Parse dates
        try:
            start_date = datetime.fromisoformat(start_str) if start_str else datetime.now() - timedelta(days=7)
            end_date = datetime.fromisoformat(end_str) if end_str else datetime.now()
        except ValueError as e:
            logger.error(f"Date parsing error: {e}")
            return {"error": f"Invalid date format: {e}"}
        
        logger.info(f"StatsAgent processing stats from {start_date} to {end_date} ({granularity})")
        
        try:
            # Generate comprehensive statistics
            stats = {
                "period": {
                    "start": start_date.isoformat(),
                    "end": end_date.isoformat(),
                    "granularity": granularity
                },
                "message_stats": await self._get_message_stats(start_date, end_date, granularity),
                "user_stats": await self._get_user_stats(start_date, end_date),
                "channel_stats": await self._get_channel_stats(start_date, end_date),
                "engagement_stats": await self._get_engagement_stats(start_date, end_date),
                "activity_trends": await self._get_activity_trends(start_date, end_date, granularity)
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"StatsAgent error: {e}")
            return {"error": f"Failed to generate statistics: {str(e)}"}
    
    async def _get_message_stats(self, start_date: datetime, end_date: datetime, granularity: str) -> Dict[str, Any]:
        """Get message statistics."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Total messages
            cursor.execute("""
                SELECT COUNT(*) FROM messages 
                WHERE timestamp BETWEEN ? AND ?
            """, (start_date.isoformat(), end_date.isoformat()))
            total_messages = cursor.fetchone()[0]
            
            # Messages by day/week/month
            if granularity == "day":
                group_by = "date(timestamp)"
            elif granularity == "week":
                group_by = "strftime('%Y-%W', timestamp)"
            else:  # month
                group_by = "strftime('%Y-%m', timestamp)"
            
            cursor.execute(f"""
                SELECT {group_by} as period, COUNT(*) as count
                FROM messages 
                WHERE timestamp BETWEEN ? AND ?
                GROUP BY {group_by}
                ORDER BY period
            """, (start_date.isoformat(), end_date.isoformat()))
            
            messages_over_time = [{"period": row[0], "count": row[1]} for row in cursor.fetchall()]
            
            # Average message length
            cursor.execute("""
                SELECT AVG(LENGTH(content)) FROM messages 
                WHERE timestamp BETWEEN ? AND ?
            """, (start_date.isoformat(), end_date.isoformat()))
            avg_length = cursor.fetchone()[0] or 0
            
            conn.close()
            
            return {
                "total_messages": total_messages,
                "messages_over_time": messages_over_time,
                "average_length": round(avg_length, 1)
            }
            
        except Exception as e:
            logger.error(f"Message stats error: {e}")
            return {"error": str(e)}
    
    async def _get_user_stats(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Get user activity statistics."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Active users
            cursor.execute("""
                SELECT COUNT(DISTINCT author_id) FROM messages 
                WHERE timestamp BETWEEN ? AND ?
            """, (start_date.isoformat(), end_date.isoformat()))
            active_users = cursor.fetchone()[0]
            
            # Top users by message count
            cursor.execute("""
                SELECT author_id, author_name, COUNT(*) as message_count
                FROM messages 
                WHERE timestamp BETWEEN ? AND ?
                GROUP BY author_id
                ORDER BY message_count DESC
                LIMIT 10
            """, (start_date.isoformat(), end_date.isoformat()))
            
            top_users = [
                {"user_id": row[0], "username": row[1], "message_count": row[2]}
                for row in cursor.fetchall()
            ]
            
            conn.close()
            
            return {
                "active_users": active_users,
                "top_users": top_users
            }
            
        except Exception as e:
            logger.error(f"User stats error: {e}")
            return {"error": str(e)}
    
    async def _get_channel_stats(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Get channel activity statistics."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Active channels
            cursor.execute("""
                SELECT COUNT(DISTINCT channel_id) FROM messages 
                WHERE timestamp BETWEEN ? AND ?
            """, (start_date.isoformat(), end_date.isoformat()))
            active_channels = cursor.fetchone()[0]
            
            # Top channels by message count
            cursor.execute("""
                SELECT channel_id, channel_name, COUNT(*) as message_count
                FROM messages 
                WHERE timestamp BETWEEN ? AND ?
                GROUP BY channel_id
                ORDER BY message_count DESC
                LIMIT 10
            """, (start_date.isoformat(), end_date.isoformat()))
            
            top_channels = [
                {"channel_id": row[0], "channel_name": row[1], "message_count": row[2]}
                for row in cursor.fetchall()
            ]
            
            conn.close()
            
            return {
                "active_channels": active_channels,
                "top_channels": top_channels
            }
            
        except Exception as e:
            logger.error(f"Channel stats error: {e}")
            return {"error": str(e)}
    
    async def _get_engagement_stats(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Get engagement statistics."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Messages with reactions
            cursor.execute("""
                SELECT COUNT(*) FROM messages 
                WHERE timestamp BETWEEN ? AND ? AND reactions IS NOT NULL
            """, (start_date.isoformat(), end_date.isoformat()))
            messages_with_reactions = cursor.fetchone()[0]
            
            # Total reactions
            cursor.execute("""
                SELECT SUM(json_array_length(reactions)) FROM messages 
                WHERE timestamp BETWEEN ? AND ? AND reactions IS NOT NULL
            """, (start_date.isoformat(), end_date.isoformat()))
            total_reactions = cursor.fetchone()[0] or 0
            
            conn.close()
            
            return {
                "messages_with_reactions": messages_with_reactions,
                "total_reactions": total_reactions
            }
            
        except Exception as e:
            logger.error(f"Engagement stats error: {e}")
            return {"error": str(e)}
    
    async def _get_activity_trends(self, start_date: datetime, end_date: datetime, granularity: str) -> Dict[str, Any]:
        """Get activity trends over time."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Activity by hour of day
            cursor.execute("""
                SELECT strftime('%H', timestamp) as hour, COUNT(*) as count
                FROM messages 
                WHERE timestamp BETWEEN ? AND ?
                GROUP BY hour
                ORDER BY hour
            """, (start_date.isoformat(), end_date.isoformat()))
            
            hourly_activity = [{"hour": row[0], "count": row[1]} for row in cursor.fetchall()]
            
            # Activity by day of week
            cursor.execute("""
                SELECT strftime('%w', timestamp) as day, COUNT(*) as count
                FROM messages 
                WHERE timestamp BETWEEN ? AND ?
                GROUP BY day
                ORDER BY day
            """, (start_date.isoformat(), end_date.isoformat()))
            
            daily_activity = [{"day": row[0], "count": row[1]} for row in cursor.fetchall()]
            
            conn.close()
            
            return {
                "hourly_activity": hourly_activity,
                "daily_activity": daily_activity
            }
            
        except Exception as e:
            logger.error(f"Activity trends error: {e}")
            return {"error": str(e)}
    
    async def process(self, state: AgentState) -> AgentState:
        """Process state through the stats agent."""
        args = state.get("agent_args", {})
        
        # Extract date parameters
        start = args.get("start")
        end = args.get("end")
        granularity = args.get("granularity", "day")
        
        # If no dates provided, use defaults
        if not start:
            start = (datetime.now() - timedelta(days=7)).isoformat()
        if not end:
            end = datetime.now().isoformat()
        
        stats = await self.run(start=start, end=end, granularity=granularity)
        
        # Update state with statistics
        state["stats_result"] = stats
        state["response"] = self._format_stats_response(stats)
        
        return state
    
    def _format_stats_response(self, stats: Dict[str, Any]) -> str:
        """Format statistics as a readable response."""
        if "error" in stats:
            return f"Error generating statistics: {stats['error']}"
        
        response = "## 📊 Discord Activity Statistics\n\n"
        
        # Period info
        period = stats.get("period", {})
        response += f"**Period:** {period.get('start', 'N/A')} to {period.get('end', 'N/A')}\n\n"
        
        # Message stats
        msg_stats = stats.get("message_stats", {})
        if "error" not in msg_stats:
            response += f"**Total Messages:** {msg_stats.get('total_messages', 0):,}\n"
            response += f"**Average Length:** {msg_stats.get('average_length', 0)} characters\n\n"
        
        # User stats
        user_stats = stats.get("user_stats", {})
        if "error" not in user_stats:
            response += f"**Active Users:** {user_stats.get('active_users', 0):,}\n\n"
        
        # Channel stats
        channel_stats = stats.get("channel_stats", {})
        if "error" not in channel_stats:
            response += f"**Active Channels:** {channel_stats.get('active_channels', 0):,}\n\n"
        
        return response
    
    def can_handle(self, task) -> bool:
        """Check if this agent can handle the given task."""
        return (task.task_type == "stats" or 
                "statistics" in task.description.lower() or
                "metrics" in task.description.lower() or
                "analytics" in task.description.lower()) 