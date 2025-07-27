"""
SQLite Query Service for MCP Server

Handles natural language to SQL translation and direct database queries for Discord message analysis.
"""

import asyncio
import json
import logging
import sqlite3
import re
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path

from ..services.llm_client import UnifiedLLMClient

logger = logging.getLogger(__name__)


class SQLiteQueryService:
    """
    Service for translating natural language queries to SQL and executing them.
    
    Provides comprehensive Discord message analysis through direct SQL queries.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Database configuration
        self.db_path = Path(self.config.get("db_path", "data/discord_messages.db"))
        
        # Initialize LLM client for query translation
        self.llm_client = UnifiedLLMClient(config.get("llm", {}))
        
        # Performance tracking
        self.stats = {
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "avg_response_time": 0.0,
            "cache_hits": 0,
            "cache_misses": 0
        }
        
        # Initialize database indices
        self._ensure_indices()
        
        logger.info("SQLiteQueryService initialized")
    
    def _ensure_indices(self):
        """Ensure database has proper indices for performance."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create indices for common query patterns
                indices = [
                    "CREATE INDEX IF NOT EXISTS idx_messages_content_search ON messages(content)",
                    "CREATE INDEX IF NOT EXISTS idx_messages_timestamp_range ON messages(timestamp)",
                    "CREATE INDEX IF NOT EXISTS idx_messages_author_activity ON messages(author_id, timestamp)",
                    "CREATE INDEX IF NOT EXISTS idx_messages_channel_activity ON messages(channel_id, timestamp)",
                    "CREATE INDEX IF NOT EXISTS idx_messages_author_username ON messages(author_username)",
                    "CREATE INDEX IF NOT EXISTS idx_messages_channel_name ON messages(channel_name)",
                    "CREATE INDEX IF NOT EXISTS idx_messages_embeds_count ON messages(embeds_count)",
                    "CREATE INDEX IF NOT EXISTS idx_messages_pinned ON messages(pinned)"
                ]
                
                for index_sql in indices:
                    cursor.execute(index_sql)
                
                conn.commit()
                logger.info("Database indices ensured")
                
        except Exception as e:
            logger.error(f"Error ensuring indices: {e}")
    
    async def query_messages(self, natural_language_query: str) -> List[Dict[str, Any]]:
        """
        Translate natural language query to SQL and execute it.
        
        Args:
            natural_language_query: Natural language query about Discord messages
            
        Returns:
            List of message results
        """
        start_time = datetime.utcnow()
        
        try:
            # Translate natural language to SQL
            sql_query = await self._translate_to_sql(natural_language_query)
            
            # Execute SQL query
            results = await self._execute_sql_query(sql_query)
            
            self._record_success(start_time)
            logger.info(f"Query executed successfully: {natural_language_query[:50]}...")
            
            return results
            
        except Exception as e:
            self._record_failure(start_time)
            logger.error(f"Error in query_messages: {e}")
            return []
    
    async def get_message_stats(self, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Get comprehensive message statistics.
        
        Args:
            filters: Optional filters to apply
            
        Returns:
            Dictionary with various statistics
        """
        try:
            where_clause, params = self._build_where_clause(filters)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Total messages
                cursor.execute(f"SELECT COUNT(*) FROM messages WHERE {where_clause}", params)
                total_messages = cursor.fetchone()[0]
                
                # Unique users
                cursor.execute(f"SELECT COUNT(DISTINCT author_id) FROM messages WHERE {where_clause}", params)
                unique_users = cursor.fetchone()[0]
                
                # Unique channels
                cursor.execute(f"SELECT COUNT(DISTINCT channel_id) FROM messages WHERE {where_clause}", params)
                unique_channels = cursor.fetchone()[0]
                
                # Date range
                cursor.execute(f"SELECT MIN(timestamp), MAX(timestamp) FROM messages WHERE {where_clause}", params)
                min_date, max_date = cursor.fetchone()
                
                # Most active users
                cursor.execute(f"""
                    SELECT author_username, COUNT(*) as message_count 
                    FROM messages 
                    WHERE {where_clause}
                    GROUP BY author_id, author_username 
                    ORDER BY message_count DESC 
                    LIMIT 10
                """, params)
                top_users = [{"username": row[0], "count": row[1]} for row in cursor.fetchall()]
                
                # Most active channels
                cursor.execute(f"""
                    SELECT channel_name, COUNT(*) as message_count 
                    FROM messages 
                    WHERE {where_clause}
                    GROUP BY channel_id, channel_name 
                    ORDER BY message_count DESC 
                    LIMIT 10
                """, params)
                top_channels = [{"name": row[0], "count": row[1]} for row in cursor.fetchall()]
                
                return {
                    "total_messages": total_messages,
                    "unique_users": unique_users,
                    "unique_channels": unique_channels,
                    "date_range": {
                        "start": min_date,
                        "end": max_date
                    },
                    "top_users": top_users,
                    "top_channels": top_channels,
                    "filters_applied": filters
                }
                
        except Exception as e:
            logger.error(f"Error getting message stats: {e}")
            return {}
    
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
            where_clause, params = self._build_where_clause(filters)
            
            # Add text search condition
            search_condition = "content LIKE ?"
            params.append(f"%{query}%")
            
            final_where = f"{where_clause} AND {search_condition}"
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                sql = f"""
                    SELECT * FROM messages 
                    WHERE {final_where}
                    ORDER BY timestamp DESC
                    LIMIT ?
                """
                params.append(limit)
                
                cursor.execute(sql, params)
                rows = cursor.fetchall()
                
                # Convert to dictionaries
                results = []
                for row in rows:
                    result = dict(zip([col[0] for col in cursor.description], row))
                    results.append(result)
                
                return results
                
        except Exception as e:
            logger.error(f"Error searching messages: {e}")
            return []
    
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
            # Parse time range
            start_date = self._parse_time_range(time_range)
            
            # Build user filter
            user_filter = ""
            if user_id:
                user_filter = f"author_id = '{user_id}'"
            elif username:
                user_filter = f"author_username = '{username}'"
            else:
                return {"error": "Must provide user_id or username"}
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Total messages in time range
                cursor.execute(f"""
                    SELECT COUNT(*) FROM messages 
                    WHERE {user_filter} AND timestamp >= ?
                """, (start_date,))
                total_messages = cursor.fetchone()[0]
                
                # Messages by channel
                cursor.execute(f"""
                    SELECT channel_name, COUNT(*) as count 
                    FROM messages 
                    WHERE {user_filter} AND timestamp >= ?
                    GROUP BY channel_id, channel_name 
                    ORDER BY count DESC
                """, (start_date,))
                messages_by_channel = [{"channel": row[0], "count": row[1]} for row in cursor.fetchall()]
                
                # Daily activity
                cursor.execute(f"""
                    SELECT DATE(timestamp) as date, COUNT(*) as count 
                    FROM messages 
                    WHERE {user_filter} AND timestamp >= ?
                    GROUP BY DATE(timestamp) 
                    ORDER BY date DESC
                    LIMIT 30
                """, (start_date,))
                daily_activity = [{"date": row[0], "count": row[1]} for row in cursor.fetchall()]
                
                # Most active hours
                cursor.execute(f"""
                    SELECT strftime('%H', timestamp) as hour, COUNT(*) as count 
                    FROM messages 
                    WHERE {user_filter} AND timestamp >= ?
                    GROUP BY strftime('%H', timestamp) 
                    ORDER BY count DESC
                    LIMIT 5
                """, (start_date,))
                active_hours = [{"hour": row[0], "count": row[1]} for row in cursor.fetchall()]
                
                return {
                    "total_messages": total_messages,
                    "messages_by_channel": messages_by_channel,
                    "daily_activity": daily_activity,
                    "active_hours": active_hours,
                    "time_range": time_range,
                    "start_date": start_date
                }
                
        except Exception as e:
            logger.error(f"Error getting user activity: {e}")
            return {"error": str(e)}
    
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
            # Parse time range
            start_date = self._parse_time_range(time_range)
            
            # Build channel filter
            channel_filter = ""
            if channel_id:
                channel_filter = f"channel_id = '{channel_id}'"
            elif channel_name:
                channel_filter = f"channel_name = '{channel_name}'"
            else:
                return {"error": "Must provide channel_id or channel_name"}
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Total messages in time range
                cursor.execute(f"""
                    SELECT COUNT(*) FROM messages 
                    WHERE {channel_filter} AND timestamp >= ?
                """, (start_date,))
                total_messages = cursor.fetchone()[0]
                
                # Unique users
                cursor.execute(f"""
                    SELECT COUNT(DISTINCT author_id) FROM messages 
                    WHERE {channel_filter} AND timestamp >= ?
                """, (start_date,))
                unique_users = cursor.fetchone()[0]
                
                # Most active users
                cursor.execute(f"""
                    SELECT author_username, COUNT(*) as count 
                    FROM messages 
                    WHERE {channel_filter} AND timestamp >= ?
                    GROUP BY author_id, author_username 
                    ORDER BY count DESC 
                    LIMIT 10
                """, (start_date,))
                top_users = [{"username": row[0], "count": row[1]} for row in cursor.fetchall()]
                
                # Daily activity
                cursor.execute(f"""
                    SELECT DATE(timestamp) as date, COUNT(*) as count 
                    FROM messages 
                    WHERE {channel_filter} AND timestamp >= ?
                    GROUP BY DATE(timestamp) 
                    ORDER BY date DESC
                    LIMIT 30
                """, (start_date,))
                daily_activity = [{"date": row[0], "count": row[1]} for row in cursor.fetchall()]
                
                # Messages with attachments/embeds
                cursor.execute(f"""
                    SELECT COUNT(*) FROM messages 
                    WHERE {channel_filter} AND timestamp >= ? AND (embeds_count > 0 OR attachments IS NOT NULL)
                """, (start_date,))
                messages_with_media = cursor.fetchone()[0]
                
                return {
                    "total_messages": total_messages,
                    "unique_users": unique_users,
                    "top_users": top_users,
                    "daily_activity": daily_activity,
                    "messages_with_media": messages_with_media,
                    "time_range": time_range,
                    "start_date": start_date
                }
                
        except Exception as e:
            logger.error(f"Error getting channel summary: {e}")
            return {"error": str(e)}
    
    async def _translate_to_sql(self, natural_language_query: str) -> str:
        """Translate natural language query to SQL using LLM."""
        try:
            # Create system prompt for SQL translation
            system_prompt = """
You are a SQL expert specializing in Discord message analysis. Translate natural language queries into SQL queries for a SQLite database with the following schema:

Table: messages
- message_id (TEXT, PRIMARY KEY)
- channel_id (TEXT)
- channel_name (TEXT)
- guild_id (TEXT)
- guild_name (TEXT)
- author_id (TEXT)
- author_username (TEXT)
- author_display_name (TEXT)
- content (TEXT)
- timestamp (TEXT, ISO format)
- jump_url (TEXT)
- thread_id (TEXT)
- thread_name (TEXT)
- forum_channel_id (TEXT)
- forum_channel_name (TEXT)
- is_forum_thread (BOOLEAN)
- mentions (TEXT, JSON)
- reactions (TEXT, JSON)
- attachments (TEXT, JSON)
- embeds_count (INTEGER)
- pinned (BOOLEAN)
- message_type (TEXT)
- reference (TEXT, JSON)
- raw_data (TEXT, JSON)

Rules:
1. Always use LIMIT clause to prevent large result sets
2. Use appropriate date functions for timestamp filtering
3. Handle JSON fields carefully (mentions, reactions, attachments)
4. Return only the SQL query, no explanations
5. DO NOT use parameterized queries (?) - use direct values in the query
6. Focus on common Discord analysis patterns
7. For user experience/skills queries, search in content for relevant terms
8. Always include content, author_username, and channel_name in SELECT for user analysis
9. For "list users who have..." queries, use DISTINCT on author_username and search content
10. Use LIKE with wildcards for flexible text matching

**CRITICAL FOR EXPERIENCE QUERIES:**
- Focus on messages where users DECLARE THEIR OWN experience, not discuss topics
- TARGET SPECIFIC CHANNELS where users typically declare experience:
  * Channels with "find" in the name (e.g., "find-a-buddy", "find-mentor")
  * Channels with "onboarding" in the name
  * Introduction channels (e.g., "introductions", "👋introductions")
- Look for first-person statements: "I have", "I am", "I work", "I'm certified", "my experience"
- Look for self-introductions: "Hi, I'm", "My name is", "About me", "Areas of Expertise"
- Look for professional declarations: "certified", "years of experience", "worked as", "specialize in"
- AVOID messages that just discuss topics without personal experience claims
- EXCLUDE opinion-based statements: "I think", "I believe", "I heard", "I read", "I saw"
- EXCLUDE questions about the topic without claiming personal experience
- EXCLUDE general knowledge sharing without personal claims

**GENERIC QUERY PATTERN FOR ANY SKILL:**
For queries like "list users with experience in [SKILL]", use this pattern:
SELECT DISTINCT author_username, author_display_name, content, channel_name, jump_url, message_id 
FROM messages 
WHERE (channel_name LIKE '%find%' OR channel_name LIKE '%onboarding%' OR channel_name LIKE '%introduction%' OR channel_name LIKE '%👋%') 
AND (content LIKE '%[SKILL]%' OR content LIKE '%[SKILL_TERM1]%' OR content LIKE '%[SKILL_TERM2]%') 
AND (content LIKE '%I am%' OR content LIKE '%I have%' OR content LIKE '%I work%' OR content LIKE '%I''m certified%' OR content LIKE '%my experience%' OR content LIKE '%certified%' OR content LIKE '%years of experience%' OR content LIKE '%worked as%' OR content LIKE '%specialize in%' OR content LIKE '%Areas of Expertise%') 
AND NOT (content LIKE '%I think%' OR content LIKE '%I believe%' OR content LIKE '%I heard%' OR content LIKE '%I read%' OR content LIKE '%I saw%') 
ORDER BY timestamp DESC LIMIT 200

**EXAMPLES FOR DIFFERENT SKILL TYPES:**
- "list users with experience in [SKILL]" → Use the generic pattern above, replacing [SKILL] with the actual skill
- "find users who mentioned [TOPIC]" → Use the generic pattern above, replacing [TOPIC] with the actual topic
- "list users who have declared experience" → Use the generic pattern above, removing the skill-specific content filter

**IMPORTANT GUIDELINES:**
- Extract the skill/topic from the user's query and use it in the content LIKE clauses
- Include related terms for the skill (e.g., for "cloud computing" include "cloud", "computing", "aws", "azure", "gcp")
- Always maintain the channel targeting and experience declaration filters
- Keep the query generic and adaptable to any skill or topic
- Use proper SQL escaping for single quotes (I''m instead of I'm)

IMPORTANT: Always search in the 'content' field, not in 'mentions' or other fields. The 'content' field contains the actual message text where users describe their experience. For user experience queries, focus on messages that contain "Areas of Expertise" sections or user introductions. Always include author_display_name, jump_url, and message_id in SELECT for better user identification and message linking. Use a higher LIMIT (200) to capture more results and ORDER BY timestamp DESC to get the most recent introductions first. For forum channels, check both channel_name and forum_channel_name fields.

**CHANNEL TARGETING STRATEGY:**
- Focus on channels where users typically declare experience: find-*, onboarding-*, introduction channels
- These channels are more likely to contain user self-introductions and experience declarations
- Avoid general discussion channels where users might just discuss topics without claiming personal experience

**EXPERIENCE FILTERING PATTERNS:**
- First-person declarations: "I have", "I am", "I work", "I'm certified", "my experience"
- Professional statements: "certified", "years of experience", "worked as", "specialize in"
- Self-introductions: "Hi, I'm", "My name is", "About me", "Areas of Expertise"
- AVOID: General discussions, opinions, questions, or topic mentions without personal claims
"""

            # Generate SQL query
            sql_query = await self.llm_client.generate(
                prompt=natural_language_query,
                system_prompt=system_prompt,
                max_tokens=500,
                temperature=0.1
            )
            
            # Clean up the response
            sql_query = sql_query.strip()
            if sql_query.startswith("```sql"):
                sql_query = sql_query[6:]
            if sql_query.endswith("```"):
                sql_query = sql_query[:-3]
            sql_query = sql_query.strip()
            
            # Fix single quotes in LIKE clauses to prevent SQL syntax errors
            # Replace I'm with I''m (double single quotes for SQL escaping)
            sql_query = sql_query.replace("I'm", "I''m")
            sql_query = sql_query.replace("I'm", "I''m")  # Handle any remaining instances
            
            # More robust single quote handling for any patterns
            # This handles cases like 'word'word' and replace with 'word''word'
            sql_query = re.sub(r"(\w)'(\w)", r"\1''\2", sql_query)
            
            # Also handle standalone single quotes in LIKE clauses
            # Replace 'text' with ''text'' in LIKE patterns
            sql_query = re.sub(r"LIKE '%([^']*)'([^']*)'%'", r"LIKE '%\1''\2'%'", sql_query)
            
            logger.debug(f"Translated query: {natural_language_query} -> {sql_query}")
            return sql_query
            
        except Exception as e:
            logger.error(f"Error translating to SQL: {e}")
            # Fallback to simple text search
            return f"SELECT * FROM messages WHERE content LIKE '%{natural_language_query}%' LIMIT 50"
    
    async def _execute_sql_query(self, sql_query: str) -> List[Dict[str, Any]]:
        """Execute SQL query and return results."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Check if query has parameters (contains ?)
                if '?' in sql_query:
                    # For now, replace parameterized queries with direct values
                    # This is a simple fix - in production, we'd want proper parameter handling
                    logger.warning(f"Parameterized query detected, using direct execution: {sql_query}")
                    cursor.execute(sql_query)
                else:
                    cursor.execute(sql_query)
                
                rows = cursor.fetchall()
                
                # Convert to dictionaries
                results = []
                for row in rows:
                    result = dict(zip([col[0] for col in cursor.description], row))
                    results.append(result)
                
                return results
                
        except Exception as e:
            logger.error(f"Error executing SQL query: {e}")
            return []
    
    def _build_where_clause(self, filters: Optional[Dict[str, Any]]) -> Tuple[str, List[Any]]:
        """Build SQL WHERE clause from filters."""
        if filters is None:
            return "1=1", []
            
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
    
    def _parse_time_range(self, time_range: str) -> str:
        """Parse time range string to start date."""
        now = datetime.utcnow()
        
        if time_range.endswith('d'):
            days = int(time_range[:-1])
            start_date = now - timedelta(days=days)
        elif time_range.endswith('w'):
            weeks = int(time_range[:-1])
            start_date = now - timedelta(weeks=weeks)
        elif time_range.endswith('m'):
            months = int(time_range[:-1])
            start_date = now - timedelta(days=months * 30)
        elif time_range.endswith('y'):
            years = int(time_range[:-1])
            start_date = now - timedelta(days=years * 365)
        else:
            # Default to 7 days
            start_date = now - timedelta(days=7)
        
        return start_date.isoformat()
    
    def _record_success(self, start_time: datetime):
        """Record successful query."""
        self.stats["total_queries"] += 1
        self.stats["successful_queries"] += 1
        
        response_time = (datetime.utcnow() - start_time).total_seconds()
        self.stats["avg_response_time"] = (
            (self.stats["avg_response_time"] * (self.stats["successful_queries"] - 1) + response_time) 
            / self.stats["successful_queries"]
        )
    
    def _record_failure(self, start_time: datetime):
        """Record failed query."""
        self.stats["total_queries"] += 1
        self.stats["failed_queries"] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get query service statistics."""
        return {
            **self.stats,
            "success_rate": (
                self.stats["successful_queries"] / self.stats["total_queries"]
                if self.stats["total_queries"] > 0 else 0
            )
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on query service."""
        health = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "checks": {}
        }
        
        try:
            # Check database connection
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT COUNT(*) FROM messages")
                    count = cursor.fetchone()[0]
                    health["checks"]["database"] = {
                        "status": "healthy",
                        "message_count": count
                    }
            except Exception as e:
                health["checks"]["database"] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
                health["status"] = "degraded"
            
            # Check LLM client
            try:
                # Simple test query
                test_sql = await self._translate_to_sql("show me 5 recent messages")
                health["checks"]["llm_client"] = {
                    "status": "healthy",
                    "test_query": test_sql[:50] + "..." if len(test_sql) > 50 else test_sql
                }
            except Exception as e:
                health["checks"]["llm_client"] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
                health["status"] = "degraded"
                
        except Exception as e:
            health["status"] = "unhealthy"
            health["error"] = str(e)
        
        return health 