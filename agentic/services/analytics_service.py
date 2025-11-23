"""
Analytics Service

Unified analytics service for generating LLM-powered reports on user activity,
channel summaries, community digests, and trends using MCP SQLite and local LLM.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from collections import Counter
from pathlib import Path

logger = logging.getLogger(__name__)


class AnalyticsService:
    """
    Main analytics coordinator that combines MCP SQLite queries with LLM-powered
    report generation for comprehensive Discord community analytics.

    Features:
    - User activity reports
    - Channel summaries
    - Weekly/monthly digests
    - Trend analysis
    - Engagement metrics
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the analytics service.

        Args:
            config: Optional configuration override
        """
        self.config = config or {}
        self.database_path = self.config.get("database_path", "data/discord_messages.db")

        # Lazy-loaded components
        self._mcp_server = None
        self._llm_client = None
        self._report_generator = None

        # Cache settings
        self.cache_ttl = self.config.get("cache_ttl", 300)  # 5 minutes
        self._cache: Dict[str, Any] = {}
        self._cache_timestamps: Dict[str, datetime] = {}

        logger.info("AnalyticsService initialized")

    @property
    def mcp_server(self):
        """Lazy-load MCP SQLite server."""
        if self._mcp_server is None:
            from ..mcp import MCPSQLiteServer
            self._mcp_server = MCPSQLiteServer({
                "database_path": self.database_path,
                "enable_write": False
            })
        return self._mcp_server

    @property
    def llm_client(self):
        """Lazy-load LLM client."""
        if self._llm_client is None:
            from .llm_client import get_llm_client
            self._llm_client = get_llm_client()
        return self._llm_client

    @property
    def report_generator(self):
        """Lazy-load report generator."""
        if self._report_generator is None:
            from .report_generator import ReportGenerator
            self._report_generator = ReportGenerator(self.llm_client)
        return self._report_generator

    # =========================================================================
    # User Activity Reports
    # =========================================================================

    async def get_user_activity_report(
        self,
        user_id: Optional[str] = None,
        username: Optional[str] = None,
        days: int = 7
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive user activity report.

        Args:
            user_id: Discord user ID
            username: Discord username (alternative to user_id)
            days: Number of days to analyze

        Returns:
            User activity report with LLM-generated summary
        """
        try:
            # Get user messages
            user_data = await self._get_user_messages(user_id, username, days)

            if not user_data["messages"]:
                return {
                    "success": False,
                    "error": f"No messages found for user in the last {days} days",
                    "user": username or user_id
                }

            # Calculate metrics
            metrics = self._calculate_user_metrics(user_data["messages"])

            # Generate LLM summary
            summary = await self.report_generator.generate_user_report(
                username=user_data["display_name"],
                messages=user_data["messages"],
                metrics=metrics,
                days=days
            )

            return {
                "success": True,
                "user": user_data["display_name"],
                "user_id": user_id,
                "period_days": days,
                "metrics": metrics,
                "summary": summary,
                "generated_at": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Error generating user activity report: {e}")
            return {"success": False, "error": str(e)}

    async def _get_user_messages(
        self,
        user_id: Optional[str],
        username: Optional[str],
        days: int
    ) -> Dict[str, Any]:
        """Fetch user messages from database."""
        import aiosqlite

        start_date = (datetime.utcnow() - timedelta(days=days)).isoformat()

        async with aiosqlite.connect(self.database_path) as db:
            db.row_factory = aiosqlite.Row

            if user_id:
                query = """
                    SELECT message_id, channel_name, content, timestamp,
                           author_display_name, author_username, reactions
                    FROM messages
                    WHERE author_id = ? AND timestamp >= ?
                    ORDER BY timestamp DESC
                    LIMIT 500
                """
                async with db.execute(query, (user_id, start_date)) as cursor:
                    rows = await cursor.fetchall()
            else:
                # Search by username (case-insensitive)
                query = """
                    SELECT message_id, channel_name, content, timestamp,
                           author_display_name, author_username, reactions, author_id
                    FROM messages
                    WHERE (LOWER(author_username) = LOWER(?)
                           OR LOWER(author_display_name) = LOWER(?))
                    AND timestamp >= ?
                    ORDER BY timestamp DESC
                    LIMIT 500
                """
                async with db.execute(query, (username, username, start_date)) as cursor:
                    rows = await cursor.fetchall()

            messages = [dict(row) for row in rows]
            display_name = messages[0]["author_display_name"] if messages else username

            return {
                "messages": messages,
                "display_name": display_name
            }

    def _calculate_user_metrics(self, messages: List[Dict]) -> Dict[str, Any]:
        """Calculate user activity metrics."""
        if not messages:
            return {}

        # Channel distribution
        channels = Counter(m["channel_name"] for m in messages)

        # Activity by day of week
        day_activity = Counter()
        hour_activity = Counter()
        for m in messages:
            try:
                dt = datetime.fromisoformat(m["timestamp"].replace("Z", "+00:00"))
                day_activity[dt.strftime("%A")] += 1
                hour_activity[dt.hour] += 1
            except:
                pass

        # Message length stats
        lengths = [len(m.get("content", "") or "") for m in messages]
        avg_length = sum(lengths) / len(lengths) if lengths else 0

        # Reactions received
        total_reactions = 0
        for m in messages:
            reactions = m.get("reactions")
            if reactions and reactions not in ["[]", "null", ""]:
                try:
                    import json
                    reaction_list = json.loads(reactions) if isinstance(reactions, str) else reactions
                    total_reactions += sum(r.get("count", 0) for r in reaction_list)
                except:
                    pass

        return {
            "total_messages": len(messages),
            "channels_active": dict(channels.most_common(10)),
            "unique_channels": len(channels),
            "most_active_day": day_activity.most_common(1)[0] if day_activity else None,
            "most_active_hour": hour_activity.most_common(1)[0] if hour_activity else None,
            "avg_message_length": round(avg_length),
            "total_reactions_received": total_reactions
        }

    # =========================================================================
    # Channel Summaries
    # =========================================================================

    async def get_channel_summary(
        self,
        channel_name: str,
        days: int = 7
    ) -> Dict[str, Any]:
        """
        Generate a channel activity summary.

        Args:
            channel_name: Name of the Discord channel
            days: Number of days to analyze

        Returns:
            Channel summary with LLM-generated insights
        """
        try:
            # Get channel messages
            channel_data = await self._get_channel_messages(channel_name, days)

            if not channel_data["messages"]:
                return {
                    "success": False,
                    "error": f"No messages found in #{channel_name} in the last {days} days",
                    "channel": channel_name
                }

            # Calculate metrics
            metrics = self._calculate_channel_metrics(channel_data["messages"])

            # Generate LLM summary
            summary = await self.report_generator.generate_channel_summary(
                channel_name=channel_name,
                messages=channel_data["messages"],
                metrics=metrics,
                days=days
            )

            return {
                "success": True,
                "channel": channel_name,
                "period_days": days,
                "metrics": metrics,
                "summary": summary,
                "generated_at": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Error generating channel summary: {e}")
            return {"success": False, "error": str(e)}

    async def _get_channel_messages(
        self,
        channel_name: str,
        days: int
    ) -> Dict[str, Any]:
        """Fetch channel messages from database."""
        import aiosqlite

        start_date = (datetime.utcnow() - timedelta(days=days)).isoformat()

        # Clean channel name (remove # if present)
        channel_name = channel_name.lstrip("#")

        async with aiosqlite.connect(self.database_path) as db:
            db.row_factory = aiosqlite.Row

            # Search for channel (partial match, case-insensitive)
            query = """
                SELECT message_id, channel_name, content, timestamp,
                       author_display_name, author_username, author_id, reactions
                FROM messages
                WHERE LOWER(channel_name) LIKE LOWER(?)
                AND timestamp >= ?
                AND (raw_data IS NULL
                     OR json_extract(raw_data, '$.author.bot') IS NULL
                     OR json_extract(raw_data, '$.author.bot') = 0)
                ORDER BY timestamp DESC
                LIMIT 1000
            """
            async with db.execute(query, (f"%{channel_name}%", start_date)) as cursor:
                rows = await cursor.fetchall()

            return {"messages": [dict(row) for row in rows]}

    def _calculate_channel_metrics(self, messages: List[Dict]) -> Dict[str, Any]:
        """Calculate channel activity metrics."""
        if not messages:
            return {}

        # Top contributors
        contributors = Counter(
            m.get("author_display_name") or m.get("author_username", "Unknown")
            for m in messages
        )

        # Activity by day
        daily_activity = Counter()
        for m in messages:
            try:
                dt = datetime.fromisoformat(m["timestamp"].replace("Z", "+00:00"))
                daily_activity[dt.strftime("%Y-%m-%d")] += 1
            except:
                pass

        # Most reacted messages
        reacted_messages = []
        for m in messages:
            reactions = m.get("reactions")
            if reactions and reactions not in ["[]", "null", ""]:
                try:
                    import json
                    reaction_list = json.loads(reactions) if isinstance(reactions, str) else reactions
                    reaction_count = sum(r.get("count", 0) for r in reaction_list)
                    if reaction_count > 0:
                        reacted_messages.append({
                            "content": (m.get("content", "") or "")[:100],
                            "author": m.get("author_display_name") or m.get("author_username"),
                            "reactions": reaction_count
                        })
                except:
                    pass

        reacted_messages.sort(key=lambda x: x["reactions"], reverse=True)

        return {
            "total_messages": len(messages),
            "unique_contributors": len(contributors),
            "top_contributors": dict(contributors.most_common(10)),
            "daily_activity": dict(sorted(daily_activity.items())[-7:]),
            "avg_messages_per_day": round(len(messages) / max(len(daily_activity), 1), 1),
            "top_reacted_messages": reacted_messages[:5]
        }

    # =========================================================================
    # Community Digest
    # =========================================================================

    async def get_community_digest(
        self,
        period: str = "weekly"
    ) -> Dict[str, Any]:
        """
        Generate a community digest for the specified period.

        Args:
            period: "daily", "weekly", or "monthly"

        Returns:
            Community digest with highlights and LLM summary
        """
        try:
            days = {"daily": 1, "weekly": 7, "monthly": 30}.get(period, 7)

            # Get all messages for the period
            digest_data = await self._get_digest_data(days)

            if not digest_data["messages"]:
                return {
                    "success": False,
                    "error": f"No messages found for {period} digest",
                    "period": period
                }

            # Calculate digest metrics
            metrics = self._calculate_digest_metrics(digest_data["messages"])

            # Generate LLM digest
            summary = await self.report_generator.generate_digest(
                period=period,
                messages=digest_data["messages"],
                metrics=metrics
            )

            return {
                "success": True,
                "period": period,
                "period_days": days,
                "metrics": metrics,
                "digest": summary,
                "generated_at": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Error generating community digest: {e}")
            return {"success": False, "error": str(e)}

    async def _get_digest_data(self, days: int) -> Dict[str, Any]:
        """Fetch digest data from database."""
        import aiosqlite

        start_date = (datetime.utcnow() - timedelta(days=days)).isoformat()

        async with aiosqlite.connect(self.database_path) as db:
            db.row_factory = aiosqlite.Row

            query = """
                SELECT message_id, channel_name, content, timestamp,
                       author_display_name, author_username, author_id, reactions
                FROM messages
                WHERE timestamp >= ?
                AND (raw_data IS NULL
                     OR json_extract(raw_data, '$.author.bot') IS NULL
                     OR json_extract(raw_data, '$.author.bot') = 0)
                ORDER BY timestamp DESC
                LIMIT 2000
            """
            async with db.execute(query, (start_date,)) as cursor:
                rows = await cursor.fetchall()

            return {"messages": [dict(row) for row in rows]}

    def _calculate_digest_metrics(self, messages: List[Dict]) -> Dict[str, Any]:
        """Calculate digest metrics."""
        if not messages:
            return {}

        # Channel activity
        channels = Counter(m["channel_name"] for m in messages)

        # Top contributors
        contributors = Counter(
            m.get("author_display_name") or m.get("author_username", "Unknown")
            for m in messages
        )

        # Most engaging content (reactions)
        top_content = []
        for m in messages:
            reactions = m.get("reactions")
            if reactions and reactions not in ["[]", "null", ""]:
                try:
                    import json
                    reaction_list = json.loads(reactions) if isinstance(reactions, str) else reactions
                    reaction_count = sum(r.get("count", 0) for r in reaction_list)
                    if reaction_count >= 2:
                        top_content.append({
                            "channel": m["channel_name"],
                            "author": m.get("author_display_name") or m.get("author_username"),
                            "content": (m.get("content", "") or "")[:150],
                            "reactions": reaction_count
                        })
                except:
                    pass

        top_content.sort(key=lambda x: x["reactions"], reverse=True)

        return {
            "total_messages": len(messages),
            "active_channels": dict(channels.most_common(10)),
            "unique_channels": len(channels),
            "top_contributors": dict(contributors.most_common(10)),
            "unique_contributors": len(contributors),
            "top_engaging_content": top_content[:10]
        }

    # =========================================================================
    # Trends Analysis
    # =========================================================================

    async def get_trends(self, days: int = 7) -> Dict[str, Any]:
        """
        Analyze trending topics and activity patterns.

        Args:
            days: Number of days to analyze

        Returns:
            Trends analysis with LLM insights
        """
        try:
            # Get messages for trend analysis
            trend_data = await self._get_digest_data(days)

            if not trend_data["messages"]:
                return {
                    "success": False,
                    "error": f"No messages found for trend analysis",
                    "days": days
                }

            # Calculate trend metrics
            metrics = self._calculate_trend_metrics(trend_data["messages"])

            # Generate LLM trend analysis
            summary = await self.report_generator.generate_trends(
                messages=trend_data["messages"],
                metrics=metrics,
                days=days
            )

            return {
                "success": True,
                "period_days": days,
                "metrics": metrics,
                "analysis": summary,
                "generated_at": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Error analyzing trends: {e}")
            return {"success": False, "error": str(e)}

    def _calculate_trend_metrics(self, messages: List[Dict]) -> Dict[str, Any]:
        """Calculate trend metrics."""
        import re

        if not messages:
            return {}

        # Extract keywords/topics (simple word frequency)
        word_counts = Counter()
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "must", "shall",
            "can", "need", "dare", "ought", "used", "to", "of", "in",
            "for", "on", "with", "at", "by", "from", "as", "into", "through",
            "during", "before", "after", "above", "below", "between", "under",
            "again", "further", "then", "once", "here", "there", "when", "where",
            "why", "how", "all", "each", "few", "more", "most", "other", "some",
            "such", "no", "nor", "not", "only", "own", "same", "so", "than",
            "too", "very", "just", "and", "but", "if", "or", "because", "until",
            "while", "this", "that", "these", "those", "i", "you", "he", "she",
            "it", "we", "they", "what", "which", "who", "whom", "its", "his",
            "her", "their", "our", "your", "my", "me", "him", "them", "us",
            "about", "like", "also", "get", "got", "think", "know", "see",
            "want", "way", "look", "first", "new", "now", "even", "good",
            "come", "make", "said", "say", "one", "two", "people", "time",
            "really", "something", "going", "yeah", "yes", "well", "https",
            "http", "www", "com", "discord", "channel"
        }

        for m in messages:
            content = m.get("content", "") or ""
            # Extract words (4+ characters, alphanumeric)
            words = re.findall(r'\b[a-zA-Z]{4,}\b', content.lower())
            for word in words:
                if word not in stop_words:
                    word_counts[word] += 1

        # Activity trend (by day)
        daily_activity = Counter()
        for m in messages:
            try:
                dt = datetime.fromisoformat(m["timestamp"].replace("Z", "+00:00"))
                daily_activity[dt.strftime("%Y-%m-%d")] += 1
            except:
                pass

        # Channel momentum (new vs old)
        channel_activity = Counter(m["channel_name"] for m in messages)

        return {
            "trending_topics": dict(word_counts.most_common(20)),
            "daily_message_volume": dict(sorted(daily_activity.items())),
            "active_channels": dict(channel_activity.most_common(10)),
            "total_messages": len(messages)
        }

    # =========================================================================
    # Quick Stats
    # =========================================================================

    async def get_quick_stats(self) -> Dict[str, Any]:
        """
        Get quick server statistics.

        Returns:
            Quick stats summary
        """
        try:
            import aiosqlite

            async with aiosqlite.connect(self.database_path) as db:
                # Total messages
                async with db.execute("SELECT COUNT(*) FROM messages") as cursor:
                    total_messages = (await cursor.fetchone())[0]

                # Unique users
                async with db.execute("SELECT COUNT(DISTINCT author_id) FROM messages") as cursor:
                    unique_users = (await cursor.fetchone())[0]

                # Unique channels
                async with db.execute("SELECT COUNT(DISTINCT channel_name) FROM messages") as cursor:
                    unique_channels = (await cursor.fetchone())[0]

                # Messages today
                today = datetime.utcnow().strftime("%Y-%m-%d")
                async with db.execute(
                    "SELECT COUNT(*) FROM messages WHERE timestamp >= ?",
                    (today,)
                ) as cursor:
                    messages_today = (await cursor.fetchone())[0]

                # Messages this week
                week_ago = (datetime.utcnow() - timedelta(days=7)).isoformat()
                async with db.execute(
                    "SELECT COUNT(*) FROM messages WHERE timestamp >= ?",
                    (week_ago,)
                ) as cursor:
                    messages_week = (await cursor.fetchone())[0]

                # Most active channel this week
                async with db.execute(
                    """SELECT channel_name, COUNT(*) as cnt
                       FROM messages WHERE timestamp >= ?
                       GROUP BY channel_name ORDER BY cnt DESC LIMIT 1""",
                    (week_ago,)
                ) as cursor:
                    row = await cursor.fetchone()
                    top_channel = row[0] if row else "N/A"

                # Most active user this week
                async with db.execute(
                    """SELECT author_display_name, COUNT(*) as cnt
                       FROM messages WHERE timestamp >= ?
                       AND (raw_data IS NULL
                            OR json_extract(raw_data, '$.author.bot') IS NULL
                            OR json_extract(raw_data, '$.author.bot') = 0)
                       GROUP BY author_id ORDER BY cnt DESC LIMIT 1""",
                    (week_ago,)
                ) as cursor:
                    row = await cursor.fetchone()
                    top_user = row[0] if row else "N/A"

            return {
                "success": True,
                "stats": {
                    "total_messages": total_messages,
                    "unique_users": unique_users,
                    "unique_channels": unique_channels,
                    "messages_today": messages_today,
                    "messages_this_week": messages_week,
                    "top_channel_this_week": top_channel,
                    "top_user_this_week": top_user
                },
                "generated_at": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Error getting quick stats: {e}")
            return {"success": False, "error": str(e)}


# Global instance
_analytics_service = None

def get_analytics_service() -> AnalyticsService:
    """Get the global analytics service instance."""
    global _analytics_service
    if _analytics_service is None:
        _analytics_service = AnalyticsService()
    return _analytics_service
