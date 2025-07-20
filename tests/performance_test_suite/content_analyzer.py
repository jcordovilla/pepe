"""
Content Analyzer

Systematically analyzes Discord server content to understand structure, patterns, and content types.
Provides insights for generating realistic test queries and expected responses.
"""

import asyncio
import json
import logging
import random
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter, defaultdict
import sqlite3
import re

logger = logging.getLogger(__name__)


class ContentAnalyzer:
    """
    Analyzes Discord server content to understand structure and patterns.
    
    Provides insights for:
    - Server structure and dynamics
    - Content types and patterns
    - User activity patterns
    - Channel characteristics
    - Topic distribution
    """
    
    def __init__(self, db_path: str = "data/discord_messages.db"):
        self.db_path = db_path
        self.sample_size = 0.15  # 15% minimum sample
        self.analysis_results = {}
        
        logger.info(f"ContentAnalyzer initialized with database: {db_path}")
    
    async def analyze_server_content(self) -> Dict[str, Any]:
        """
        Perform comprehensive analysis of Discord server content.
        
        Returns:
            Analysis results with server structure and content insights
        """
        try:
            logger.info("Starting comprehensive server content analysis...")
            
            # Get total message count
            total_messages = await self._get_total_message_count()
            sample_size = max(int(total_messages * self.sample_size), 1000)
            
            logger.info(f"Total messages: {total_messages}, Sample size: {sample_size}")
            
            # Perform systematic analysis
            analysis = {
                "metadata": {
                    "analysis_timestamp": datetime.utcnow().isoformat(),
                    "total_messages": total_messages,
                    "sample_size": sample_size,
                    "sample_percentage": (sample_size / total_messages) * 100
                },
                "server_structure": await self._analyze_server_structure(),
                "content_patterns": await self._analyze_content_patterns(sample_size),
                "user_activity": await self._analyze_user_activity(sample_size),
                "channel_characteristics": await self._analyze_channel_characteristics(),
                "topic_distribution": await self._analyze_topic_distribution(sample_size),
                "engagement_patterns": await self._analyze_engagement_patterns(sample_size),
                "temporal_patterns": await self._analyze_temporal_patterns(sample_size)
            }
            
            self.analysis_results = analysis
            
            # Save analysis results
            await self._save_analysis_results(analysis)
            
            logger.info("Server content analysis completed successfully")
            return analysis
            
        except Exception as e:
            logger.error(f"Error in server content analysis: {e}")
            raise
    
    async def _get_total_message_count(self) -> int:
        """Get total number of messages in the database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM messages")
            count = cursor.fetchone()[0]
            
            conn.close()
            return count
            
        except Exception as e:
            logger.error(f"Error getting message count: {e}")
            return 0
    
    async def _analyze_server_structure(self) -> Dict[str, Any]:
        """Analyze server structure including channels and categories."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get channel information
            cursor.execute("""
                SELECT DISTINCT channel_id, channel_name, 
                       COUNT(*) as message_count,
                       MIN(timestamp) as first_message,
                       MAX(timestamp) as last_message
                FROM messages 
                GROUP BY channel_id, channel_name
                ORDER BY message_count DESC
            """)
            
            channels = []
            for row in cursor.fetchall():
                channels.append({
                    "channel_id": row[0],
                    "channel_name": row[1],
                    "message_count": row[2],
                    "first_message": row[3],
                    "last_message": row[4]
                })
            
            # Analyze channel types
            channel_types = self._classify_channels(channels)
            
            conn.close()
            
            return {
                "total_channels": len(channels),
                "channels": channels,
                "channel_types": channel_types,
                "channel_distribution": {
                    "high_activity": len([c for c in channels if c["message_count"] > 1000]),
                    "medium_activity": len([c for c in channels if 100 <= c["message_count"] <= 1000]),
                    "low_activity": len([c for c in channels if c["message_count"] < 100])
                }
            }
            
        except Exception as e:
            logger.error(f"Error analyzing server structure: {e}")
            return {}
    
    async def _analyze_content_patterns(self, sample_size: int) -> Dict[str, Any]:
        """Analyze content patterns in messages."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get random sample of messages
            cursor.execute(f"""
                SELECT content, channel_name, user_id, username, timestamp, reactions
                FROM messages 
                ORDER BY RANDOM() 
                LIMIT {sample_size}
            """)
            
            messages = cursor.fetchall()
            conn.close()
            
            # Analyze content patterns
            content_analysis = {
                "message_lengths": [],
                "content_types": Counter(),
                "url_patterns": Counter(),
                "mention_patterns": Counter(),
                "code_blocks": 0,
                "emojis": Counter(),
                "hashtags": Counter()
            }
            
            for message in messages:
                content = message[0]
                
                # Message length
                content_analysis["message_lengths"].append(len(content))
                
                # Content types
                content_type = self._classify_content_type(content)
                content_analysis["content_types"][content_type] += 1
                
                # URLs
                urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', content)
                for url in urls:
                    domain = re.search(r'https?://([^/]+)', url)
                    if domain:
                        content_analysis["url_patterns"][domain.group(1)] += 1
                
                # Mentions
                mentions = re.findall(r'<@!?(\d+)>', content)
                content_analysis["mention_patterns"]["total_mentions"] += len(mentions)
                
                # Code blocks
                if '```' in content:
                    content_analysis["code_blocks"] += 1
                
                # Emojis
                emojis = re.findall(r'<a?:[^:]+:\d+>', content)
                content_analysis["emojis"]["total_custom_emojis"] += len(emojis)
                
                # Hashtags
                hashtags = re.findall(r'#\w+', content)
                for hashtag in hashtags:
                    content_analysis["hashtags"][hashtag] += 1
            
            # Calculate statistics
            lengths = content_analysis["message_lengths"]
            content_analysis["length_stats"] = {
                "mean": sum(lengths) / len(lengths) if lengths else 0,
                "median": sorted(lengths)[len(lengths)//2] if lengths else 0,
                "min": min(lengths) if lengths else 0,
                "max": max(lengths) if lengths else 0
            }
            
            return content_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing content patterns: {e}")
            return {}
    
    async def _analyze_user_activity(self, sample_size: int) -> Dict[str, Any]:
        """Analyze user activity patterns."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get user activity data
            cursor.execute(f"""
                SELECT user_id, username, 
                       COUNT(*) as message_count,
                       COUNT(DISTINCT channel_id) as channels_used,
                       MIN(timestamp) as first_message,
                       MAX(timestamp) as last_message,
                       AVG(LENGTH(content)) as avg_message_length
                FROM messages 
                GROUP BY user_id, username
                ORDER BY message_count DESC
            """)
            
            users = []
            for row in cursor.fetchall():
                users.append({
                    "user_id": row[0],
                    "username": row[1],
                    "message_count": row[2],
                    "channels_used": row[3],
                    "first_message": row[4],
                    "last_message": row[5],
                    "avg_message_length": row[6]
                })
            
            # Analyze user types
            user_types = self._classify_users(users)
            
            conn.close()
            
            return {
                "total_users": len(users),
                "users": users[:50],  # Top 50 users
                "user_types": user_types,
                "activity_distribution": {
                    "super_active": len([u for u in users if u["message_count"] > 500]),
                    "active": len([u for u in users if 100 <= u["message_count"] <= 500]),
                    "moderate": len([u for u in users if 10 <= u["message_count"] < 100]),
                    "inactive": len([u for u in users if u["message_count"] < 10])
                }
            }
            
        except Exception as e:
            logger.error(f"Error analyzing user activity: {e}")
            return {}
    
    async def _analyze_channel_characteristics(self) -> Dict[str, Any]:
        """Analyze characteristics of different channels."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get channel characteristics
            cursor.execute("""
                SELECT channel_id, channel_name,
                       COUNT(*) as message_count,
                       COUNT(DISTINCT user_id) as unique_users,
                       AVG(LENGTH(content)) as avg_message_length,
                       COUNT(CASE WHEN reactions IS NOT NULL AND reactions != '' THEN 1 END) as messages_with_reactions
                FROM messages 
                GROUP BY channel_id, channel_name
                ORDER BY message_count DESC
            """)
            
            channels = []
            for row in cursor.fetchall():
                channels.append({
                    "channel_id": row[0],
                    "channel_name": row[1],
                    "message_count": row[2],
                    "unique_users": row[3],
                    "avg_message_length": row[4],
                    "messages_with_reactions": row[5],
                    "engagement_rate": (row[5] / row[2]) * 100 if row[2] > 0 else 0
                })
            
            conn.close()
            
            return {
                "channels": channels,
                "channel_insights": {
                    "most_active": channels[0] if channels else None,
                    "most_engaged": max(channels, key=lambda x: x["engagement_rate"]) if channels else None,
                    "longest_messages": max(channels, key=lambda x: x["avg_message_length"]) if channels else None
                }
            }
            
        except Exception as e:
            logger.error(f"Error analyzing channel characteristics: {e}")
            return {}
    
    async def _analyze_topic_distribution(self, sample_size: int) -> Dict[str, Any]:
        """Analyze topic distribution in messages."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get sample messages for topic analysis
            cursor.execute(f"""
                SELECT content, channel_name
                FROM messages 
                ORDER BY RANDOM() 
                LIMIT {sample_size}
            """)
            
            messages = cursor.fetchall()
            conn.close()
            
            # Define topic keywords
            topics = {
                "ai_ml": ["ai", "machine learning", "neural", "model", "training", "gpt", "llm"],
                "programming": ["python", "javascript", "code", "programming", "development", "bug", "api"],
                "discussion": ["think", "opinion", "believe", "discuss", "consider", "thoughts"],
                "questions": ["how", "what", "why", "when", "where", "can", "could", "should"],
                "resources": ["link", "url", "documentation", "tutorial", "guide", "article"],
                "technical": ["docker", "kubernetes", "deployment", "ci/cd", "database", "server"],
                "community": ["community", "help", "support", "welcome", "introduction"]
            }
            
            topic_counts = defaultdict(int)
            channel_topics = defaultdict(lambda: defaultdict(int))
            
            for message in messages:
                content = message[0].lower()
                channel = message[1]
                
                for topic, keywords in topics.items():
                    if any(keyword in content for keyword in keywords):
                        topic_counts[topic] += 1
                        channel_topics[channel][topic] += 1
            
            return {
                "topic_distribution": dict(topic_counts),
                "channel_topics": dict(channel_topics),
                "top_topics": sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            }
            
        except Exception as e:
            logger.error(f"Error analyzing topic distribution: {e}")
            return {}
    
    async def _analyze_engagement_patterns(self, sample_size: int) -> Dict[str, Any]:
        """Analyze engagement patterns (reactions, replies)."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get messages with reactions
            cursor.execute(f"""
                SELECT reactions, content, channel_name, user_id
                FROM messages 
                WHERE reactions IS NOT NULL AND reactions != ''
                ORDER BY RANDOM() 
                LIMIT {sample_size}
            """)
            
            messages = cursor.fetchall()
            conn.close()
            
            reaction_analysis = {
                "total_messages_with_reactions": len(messages),
                "reaction_types": Counter(),
                "high_engagement_messages": [],
                "engagement_by_channel": defaultdict(int),
                "engagement_by_user": defaultdict(int)
            }
            
            for message in messages:
                reactions = message[0]
                content = message[1]
                channel = message[2]
                user_id = message[3]
                
                # Parse reactions (assuming JSON format)
                try:
                    reaction_data = json.loads(reactions) if reactions else {}
                    total_reactions = sum(reaction_data.values())
                    
                    if total_reactions > 5:  # High engagement threshold
                        reaction_analysis["high_engagement_messages"].append({
                            "content": content[:100] + "..." if len(content) > 100 else content,
                            "reactions": reaction_data,
                            "total_reactions": total_reactions,
                            "channel": channel
                        })
                    
                    reaction_analysis["engagement_by_channel"][channel] += total_reactions
                    reaction_analysis["engagement_by_user"][user_id] += total_reactions
                    
                    for reaction_type in reaction_data.keys():
                        reaction_analysis["reaction_types"][reaction_type] += reaction_data[reaction_type]
                        
                except (json.JSONDecodeError, TypeError):
                    continue
            
            return reaction_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing engagement patterns: {e}")
            return {}
    
    async def _analyze_temporal_patterns(self, sample_size: int) -> Dict[str, Any]:
        """Analyze temporal patterns in message activity."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get temporal data
            cursor.execute(f"""
                SELECT timestamp, channel_name, user_id
                FROM messages 
                ORDER BY RANDOM() 
                LIMIT {sample_size}
            """)
            
            messages = cursor.fetchall()
            conn.close()
            
            temporal_analysis = {
                "hourly_activity": defaultdict(int),
                "daily_activity": defaultdict(int),
                "weekly_activity": defaultdict(int),
                "monthly_activity": defaultdict(int),
                "peak_hours": [],
                "peak_days": []
            }
            
            for message in messages:
                timestamp = message[0]
                try:
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    
                    # Hourly activity
                    temporal_analysis["hourly_activity"][dt.hour] += 1
                    
                    # Daily activity
                    temporal_analysis["daily_activity"][dt.strftime('%Y-%m-%d')] += 1
                    
                    # Weekly activity
                    temporal_analysis["weekly_activity"][dt.strftime('%Y-W%U')] += 1
                    
                    # Monthly activity
                    temporal_analysis["monthly_activity"][dt.strftime('%Y-%m')] += 1
                    
                except (ValueError, AttributeError):
                    continue
            
            # Find peak times
            if temporal_analysis["hourly_activity"]:
                peak_hour = max(temporal_analysis["hourly_activity"].items(), key=lambda x: x[1])
                temporal_analysis["peak_hours"] = [peak_hour[0]]
            
            if temporal_analysis["daily_activity"]:
                peak_day = max(temporal_analysis["daily_activity"].items(), key=lambda x: x[1])
                temporal_analysis["peak_days"] = [peak_day[0]]
            
            return temporal_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing temporal patterns: {e}")
            return {}
    
    def _classify_channels(self, channels: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Classify channels by type based on name and activity patterns."""
        channel_types = {
            "general": [],
            "technical": [],
            "community": [],
            "announcements": [],
            "help_support": [],
            "off_topic": []
        }
        
        for channel in channels:
            name = channel["channel_name"].lower()
            
            if "general" in name or "main" in name:
                channel_types["general"].append(channel["channel_name"])
            elif any(tech in name for tech in ["dev", "tech", "programming", "ai", "ml"]):
                channel_types["technical"].append(channel["channel_name"])
            elif any(comm in name for comm in ["community", "chat", "discussion"]):
                channel_types["community"].append(channel["channel_name"])
            elif any(announce in name for announce in ["announce", "news", "updates"]):
                channel_types["announcements"].append(channel["channel_name"])
            elif any(help in name for help in ["help", "support", "questions"]):
                channel_types["help_support"].append(channel["channel_name"])
            else:
                channel_types["off_topic"].append(channel["channel_name"])
        
        return channel_types
    
    def _classify_users(self, users: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Classify users by activity level."""
        user_types = {
            "super_active": [],
            "active": [],
            "moderate": [],
            "inactive": []
        }
        
        for user in users:
            if user["message_count"] > 500:
                user_types["super_active"].append(user["username"])
            elif user["message_count"] > 100:
                user_types["active"].append(user["username"])
            elif user["message_count"] > 10:
                user_types["moderate"].append(user["username"])
            else:
                user_types["inactive"].append(user["username"])
        
        return user_types
    
    def _classify_content_type(self, content: str) -> str:
        """Classify message content by type."""
        content_lower = content.lower()
        
        if '```' in content:
            return "code_block"
        elif re.search(r'http[s]?://', content):
            return "link_share"
        elif re.search(r'<@!?\d+>', content):
            return "mention"
        elif re.search(r'#\w+', content):
            return "hashtag"
        elif len(content) > 200:
            return "long_text"
        elif len(content) < 50:
            return "short_text"
        else:
            return "normal_text"
    
    async def _save_analysis_results(self, analysis: Dict[str, Any]) -> None:
        """Save analysis results to file."""
        try:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filename = f"tests/performance_test_suite/data/content_analysis_{timestamp}.json"
            
            # Ensure directory exists
            import os
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            
            with open(filename, 'w') as f:
                json.dump(analysis, f, indent=2, default=str)
            
            logger.info(f"Analysis results saved to: {filename}")
            
        except Exception as e:
            logger.error(f"Error saving analysis results: {e}")
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """Get a summary of the analysis results."""
        if not self.analysis_results:
            return {}
        
        analysis = self.analysis_results
        
        return {
            "server_overview": {
                "total_messages": analysis["metadata"]["total_messages"],
                "total_channels": analysis["server_structure"]["total_channels"],
                "total_users": analysis["user_activity"]["total_users"],
                "sample_analyzed": analysis["metadata"]["sample_size"]
            },
            "key_insights": {
                "most_active_channel": analysis["server_structure"]["channels"][0]["channel_name"] if analysis["server_structure"]["channels"] else None,
                "top_topics": [topic[0] for topic in analysis["topic_distribution"]["top_topics"][:5]],
                "peak_activity_hour": analysis["temporal_patterns"]["peak_hours"][0] if analysis["temporal_patterns"]["peak_hours"] else None,
                "engagement_rate": len(analysis["engagement_patterns"]["high_engagement_messages"])
            },
            "content_characteristics": {
                "avg_message_length": analysis["content_patterns"]["length_stats"]["mean"],
                "code_blocks_percentage": (analysis["content_patterns"]["code_blocks"] / analysis["metadata"]["sample_size"]) * 100,
                "url_sharing_percentage": sum(analysis["content_patterns"]["url_patterns"].values()) / analysis["metadata"]["sample_size"] * 100
            }
        } 