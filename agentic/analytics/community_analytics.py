"""
Community Analytics

Provides community-focused analytics metrics for Discord servers.
Works in conjunction with the AnalyticsService for comprehensive reporting.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter, defaultdict
import json
import re

logger = logging.getLogger(__name__)


class CommunityAnalytics:
    """
    Community-focused analytics for Discord servers.

    Provides metrics on:
    - Member engagement and activity
    - Channel health and activity
    - Content themes and topics
    - Engagement patterns
    """

    def __init__(self, database_path: str = "data/discord_messages.db"):
        """
        Initialize community analytics.

        Args:
            database_path: Path to the Discord messages database
        """
        self.database_path = database_path
        logger.info("CommunityAnalytics initialized")

    async def get_member_engagement_score(
        self,
        user_id: str,
        days: int = 30
    ) -> Dict[str, Any]:
        """
        Calculate an engagement score for a member.

        Factors:
        - Message frequency
        - Channel diversity
        - Reactions received
        - Response rate (replies to others)

        Args:
            user_id: Discord user ID
            days: Days to analyze

        Returns:
            Engagement metrics and score
        """
        import aiosqlite

        start_date = (datetime.utcnow() - timedelta(days=days)).isoformat()

        async with aiosqlite.connect(self.database_path) as db:
            db.row_factory = aiosqlite.Row

            # Get user messages
            query = """
                SELECT channel_name, content, reactions, timestamp
                FROM messages
                WHERE author_id = ? AND timestamp >= ?
            """
            async with db.execute(query, (user_id, start_date)) as cursor:
                messages = [dict(row) for row in await cursor.fetchall()]

            if not messages:
                return {"user_id": user_id, "score": 0, "reason": "No activity"}

            # Calculate metrics
            message_count = len(messages)
            unique_channels = len(set(m["channel_name"] for m in messages))

            # Reactions received
            total_reactions = 0
            for m in messages:
                reactions = m.get("reactions")
                if reactions and reactions not in ["[]", "null", ""]:
                    try:
                        r_list = json.loads(reactions) if isinstance(reactions, str) else reactions
                        total_reactions += sum(r.get("count", 0) for r in r_list)
                    except:
                        pass

            # Calculate score (0-100)
            # Weights: frequency (40%), diversity (30%), engagement (30%)
            freq_score = min(40, (message_count / days) * 10)  # ~4 msgs/day = max
            diversity_score = min(30, unique_channels * 5)  # 6+ channels = max
            engagement_score = min(30, total_reactions * 2)  # 15+ reactions = max

            total_score = freq_score + diversity_score + engagement_score

            return {
                "user_id": user_id,
                "score": round(total_score, 1),
                "grade": self._score_to_grade(total_score),
                "metrics": {
                    "message_count": message_count,
                    "unique_channels": unique_channels,
                    "reactions_received": total_reactions,
                    "messages_per_day": round(message_count / days, 2)
                },
                "breakdown": {
                    "frequency_score": round(freq_score, 1),
                    "diversity_score": round(diversity_score, 1),
                    "engagement_score": round(engagement_score, 1)
                },
                "period_days": days
            }

    def _score_to_grade(self, score: float) -> str:
        """Convert score to letter grade."""
        if score >= 90:
            return "A+"
        elif score >= 80:
            return "A"
        elif score >= 70:
            return "B"
        elif score >= 60:
            return "C"
        elif score >= 50:
            return "D"
        else:
            return "F"

    async def get_channel_health(
        self,
        channel_name: str,
        days: int = 30
    ) -> Dict[str, Any]:
        """
        Analyze the health of a channel.

        Metrics:
        - Activity level
        - Contributor diversity
        - Engagement (reactions)
        - Conversation depth (message length, threads)

        Args:
            channel_name: Channel name (partial match)
            days: Days to analyze

        Returns:
            Channel health metrics
        """
        import aiosqlite

        start_date = (datetime.utcnow() - timedelta(days=days)).isoformat()
        channel_name = channel_name.lstrip("#")

        async with aiosqlite.connect(self.database_path) as db:
            db.row_factory = aiosqlite.Row

            query = """
                SELECT content, author_id, author_display_name, reactions, timestamp
                FROM messages
                WHERE LOWER(channel_name) LIKE LOWER(?)
                AND timestamp >= ?
                AND (raw_data IS NULL
                     OR json_extract(raw_data, '$.author.bot') IS NULL
                     OR json_extract(raw_data, '$.author.bot') = 0)
            """
            async with db.execute(query, (f"%{channel_name}%", start_date)) as cursor:
                messages = [dict(row) for row in await cursor.fetchall()]

            if not messages:
                return {"channel": channel_name, "health": "inactive", "reason": "No activity"}

            # Activity metrics
            message_count = len(messages)
            unique_contributors = len(set(m["author_id"] for m in messages))
            msgs_per_day = message_count / days

            # Engagement
            total_reactions = 0
            for m in messages:
                reactions = m.get("reactions")
                if reactions and reactions not in ["[]", "null", ""]:
                    try:
                        r_list = json.loads(reactions) if isinstance(reactions, str) else reactions
                        total_reactions += sum(r.get("count", 0) for r in r_list)
                    except:
                        pass

            # Content quality (avg message length)
            lengths = [len(m.get("content", "") or "") for m in messages]
            avg_length = sum(lengths) / len(lengths) if lengths else 0

            # Health score (0-100)
            activity_score = min(40, msgs_per_day * 4)  # 10 msgs/day = max
            diversity_score = min(30, unique_contributors * 3)  # 10+ contributors = max
            engagement_score = min(30, (total_reactions / max(message_count, 1)) * 30)  # reactions/msg ratio

            health_score = activity_score + diversity_score + engagement_score

            # Determine health status
            if health_score >= 70:
                health_status = "thriving"
            elif health_score >= 50:
                health_status = "healthy"
            elif health_score >= 30:
                health_status = "moderate"
            else:
                health_status = "needs_attention"

            return {
                "channel": channel_name,
                "health": health_status,
                "score": round(health_score, 1),
                "metrics": {
                    "message_count": message_count,
                    "unique_contributors": unique_contributors,
                    "messages_per_day": round(msgs_per_day, 2),
                    "total_reactions": total_reactions,
                    "avg_message_length": round(avg_length)
                },
                "breakdown": {
                    "activity_score": round(activity_score, 1),
                    "diversity_score": round(diversity_score, 1),
                    "engagement_score": round(engagement_score, 1)
                },
                "period_days": days
            }

    async def get_active_hours(self, days: int = 7) -> Dict[str, Any]:
        """
        Analyze when the community is most active.

        Args:
            days: Days to analyze

        Returns:
            Activity by hour and day
        """
        import aiosqlite

        start_date = (datetime.utcnow() - timedelta(days=days)).isoformat()

        async with aiosqlite.connect(self.database_path) as db:
            db.row_factory = aiosqlite.Row

            query = """
                SELECT timestamp FROM messages
                WHERE timestamp >= ?
            """
            async with db.execute(query, (start_date,)) as cursor:
                rows = await cursor.fetchall()

            hour_counts = Counter()
            day_counts = Counter()
            hourday_counts = Counter()

            for row in rows:
                try:
                    dt = datetime.fromisoformat(row["timestamp"].replace("Z", "+00:00"))
                    hour_counts[dt.hour] += 1
                    day_counts[dt.strftime("%A")] += 1
                    hourday_counts[(dt.strftime("%A"), dt.hour)] += 1
                except:
                    pass

            # Find peak times
            peak_hour = hour_counts.most_common(1)[0] if hour_counts else (0, 0)
            peak_day = day_counts.most_common(1)[0] if day_counts else ("Unknown", 0)

            return {
                "peak_hour": peak_hour[0],
                "peak_hour_count": peak_hour[1],
                "peak_day": peak_day[0],
                "peak_day_count": peak_day[1],
                "hourly_distribution": dict(sorted(hour_counts.items())),
                "daily_distribution": dict(day_counts),
                "period_days": days
            }

    async def get_topic_clusters(
        self,
        days: int = 7,
        min_mentions: int = 5
    ) -> Dict[str, Any]:
        """
        Identify topic clusters from message content.

        Args:
            days: Days to analyze
            min_mentions: Minimum mentions to include

        Returns:
            Topic clusters and frequencies
        """
        import aiosqlite

        start_date = (datetime.utcnow() - timedelta(days=days)).isoformat()

        # Topic categories and their keywords
        topic_keywords = {
            "ai_ml": ["ai", "machine learning", "llm", "gpt", "claude", "model", "neural", "training", "inference"],
            "programming": ["python", "javascript", "code", "programming", "developer", "api", "function", "bug"],
            "tools": ["docker", "kubernetes", "git", "github", "vscode", "cursor", "terminal", "cli"],
            "cloud": ["aws", "azure", "gcp", "cloud", "server", "deploy", "hosting"],
            "data": ["database", "sql", "data", "analytics", "visualization", "dashboard"],
            "learning": ["course", "tutorial", "learn", "study", "certification", "workshop"],
            "career": ["job", "career", "interview", "resume", "hire", "salary", "work"],
            "community": ["help", "question", "thanks", "welcome", "introduce", "share"]
        }

        async with aiosqlite.connect(self.database_path) as db:
            db.row_factory = aiosqlite.Row

            query = """
                SELECT content FROM messages
                WHERE timestamp >= ?
                AND content IS NOT NULL AND content != ''
                AND (raw_data IS NULL
                     OR json_extract(raw_data, '$.author.bot') IS NULL
                     OR json_extract(raw_data, '$.author.bot') = 0)
            """
            async with db.execute(query, (start_date,)) as cursor:
                messages = await cursor.fetchall()

            # Count topic mentions
            topic_counts = Counter()
            for row in messages:
                content = (row["content"] or "").lower()
                for topic, keywords in topic_keywords.items():
                    if any(kw in content for kw in keywords):
                        topic_counts[topic] += 1

            # Filter by minimum mentions
            filtered_topics = {k: v for k, v in topic_counts.items() if v >= min_mentions}

            # Sort by count
            sorted_topics = dict(sorted(filtered_topics.items(), key=lambda x: x[1], reverse=True))

            return {
                "topics": sorted_topics,
                "total_messages_analyzed": len(messages),
                "period_days": days,
                "keywords_used": topic_keywords
            }

    async def get_contributor_leaderboard(
        self,
        days: int = 7,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Get a contributor leaderboard.

        Args:
            days: Days to analyze
            limit: Number of top contributors

        Returns:
            List of top contributors with metrics
        """
        import aiosqlite

        start_date = (datetime.utcnow() - timedelta(days=days)).isoformat()

        async with aiosqlite.connect(self.database_path) as db:
            db.row_factory = aiosqlite.Row

            query = """
                SELECT author_id, author_display_name, author_username,
                       COUNT(*) as message_count,
                       COUNT(DISTINCT channel_name) as channels_active
                FROM messages
                WHERE timestamp >= ?
                AND (raw_data IS NULL
                     OR json_extract(raw_data, '$.author.bot') IS NULL
                     OR json_extract(raw_data, '$.author.bot') = 0)
                GROUP BY author_id
                ORDER BY message_count DESC
                LIMIT ?
            """
            async with db.execute(query, (start_date, limit)) as cursor:
                rows = await cursor.fetchall()

            leaderboard = []
            for i, row in enumerate(rows, 1):
                leaderboard.append({
                    "rank": i,
                    "user_id": row["author_id"],
                    "display_name": row["author_display_name"] or row["author_username"],
                    "message_count": row["message_count"],
                    "channels_active": row["channels_active"]
                })

            return leaderboard

    async def compare_periods(
        self,
        days: int = 7
    ) -> Dict[str, Any]:
        """
        Compare current period with previous period.

        Args:
            days: Period length to compare

        Returns:
            Comparison metrics
        """
        import aiosqlite

        now = datetime.utcnow()
        current_start = (now - timedelta(days=days)).isoformat()
        previous_start = (now - timedelta(days=days * 2)).isoformat()
        previous_end = (now - timedelta(days=days)).isoformat()

        async with aiosqlite.connect(self.database_path) as db:
            # Current period
            async with db.execute(
                "SELECT COUNT(*) FROM messages WHERE timestamp >= ?",
                (current_start,)
            ) as cursor:
                current_messages = (await cursor.fetchone())[0]

            async with db.execute(
                "SELECT COUNT(DISTINCT author_id) FROM messages WHERE timestamp >= ?",
                (current_start,)
            ) as cursor:
                current_users = (await cursor.fetchone())[0]

            # Previous period
            async with db.execute(
                "SELECT COUNT(*) FROM messages WHERE timestamp >= ? AND timestamp < ?",
                (previous_start, previous_end)
            ) as cursor:
                previous_messages = (await cursor.fetchone())[0]

            async with db.execute(
                "SELECT COUNT(DISTINCT author_id) FROM messages WHERE timestamp >= ? AND timestamp < ?",
                (previous_start, previous_end)
            ) as cursor:
                previous_users = (await cursor.fetchone())[0]

        # Calculate changes
        msg_change = ((current_messages - previous_messages) / max(previous_messages, 1)) * 100
        user_change = ((current_users - previous_users) / max(previous_users, 1)) * 100

        return {
            "period_days": days,
            "current": {
                "messages": current_messages,
                "unique_users": current_users
            },
            "previous": {
                "messages": previous_messages,
                "unique_users": previous_users
            },
            "changes": {
                "messages_percent": round(msg_change, 1),
                "users_percent": round(user_change, 1)
            },
            "trend": "growing" if msg_change > 10 else "declining" if msg_change < -10 else "stable"
        }


# Global instance
_community_analytics = None

def get_community_analytics() -> CommunityAnalytics:
    """Get the global community analytics instance."""
    global _community_analytics
    if _community_analytics is None:
        _community_analytics = CommunityAnalytics()
    return _community_analytics
