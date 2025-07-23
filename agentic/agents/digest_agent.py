"""
Digest Agent

Specialized agent for generating weekly, monthly, and custom time-period digests
with engagement analysis and content summarization.
"""

from typing import Dict, List, Any, Optional
import logging
from datetime import datetime, timedelta
from collections import Counter
import asyncio

from .base_agent import BaseAgent, AgentRole, AgentState, SubTask, TaskStatus, agent_registry

logger = logging.getLogger(__name__)


class DigestAgent(BaseAgent):
    """
    Agent responsible for generating content digests and summaries.
    
    This agent:
    - Generates weekly, monthly, and custom period digests
    - Analyzes user engagement and activity patterns
    - Identifies trending content and high-engagement messages
    - Provides structured summaries with metadata
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(AgentRole.DIGESTER, config)
        
        # Digest configuration
        self.default_digest_period = config.get("default_digest_period", "weekly")
        self.max_digest_messages = config.get("max_digest_messages", 500)
        self.trending_threshold = config.get("trending_threshold", 5)
        
        logger.info(f"DigestAgent initialized with period={self.default_digest_period}")
        
        # Register this agent
        agent_registry.register_agent(self)
    
    async def process(self, state: AgentState) -> AgentState:
        """
        Process digest-related subtasks.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with digest results
        """
        try:
            # Handle current subtask (orchestrator mode)
            if "current_subtask" in state and state["current_subtask"] is not None:
                subtask = state["current_subtask"]
                if self.can_handle(subtask):
                    logger.info(f"Processing digest subtask: {subtask.task_type}")
                    
                    if subtask.task_type in ["digest", "weekly_digest", "monthly_digest"]:
                        result = await self._generate_digest(subtask, state)
                    elif subtask.task_type == "engagement_analysis":
                        result = await self._analyze_engagement(subtask, state)
                    elif subtask.task_type == "trending_content":
                        result = await self._identify_trending_content(subtask, state)
                    else:
                        logger.warning(f"Unknown digest task type: {subtask.task_type}")
                        result = {"error": f"Unknown task type: {subtask.task_type}"}
                    
                    # Update subtask and state
                    subtask.status = TaskStatus.COMPLETED
                    subtask.result = result
                    state["digest_results"] = result
                    
                    logger.info(f"Digest generation completed successfully")
                    return state
            
            # Fallback: process all digest subtasks
            subtasks = state.get("subtasks", [])
            digest_subtasks = [task for task in subtasks if self.can_handle(task)]
            
            if not digest_subtasks:
                logger.warning("No digest subtasks found")
                return state
            
            digest_results = {}
            
            for subtask in digest_subtasks:
                logger.info(f"Processing digest subtask: {subtask.task_type}")
                
                if subtask.task_type in ["digest", "weekly_digest", "monthly_digest"]:
                    result = await self._generate_digest(subtask, state)
                elif subtask.task_type == "engagement_analysis":
                    result = await self._analyze_engagement(subtask, state)
                elif subtask.task_type == "trending_content":
                    result = await self._identify_trending_content(subtask, state)
                else:
                    logger.warning(f"Unknown digest task type: {subtask.task_type}")
                    continue
                
                digest_results[subtask.task_type] = result
                subtask.status = TaskStatus.COMPLETED
                subtask.result = result
            
            # Update state
            state["digest_results"] = digest_results
            state["metadata"]["digest_agent"] = {
                "digest_time": datetime.utcnow().isoformat(),
                "subtasks_processed": len(digest_subtasks),
                "results_count": len(digest_results)
            }
            
            logger.info(f"Digest processing completed: {len(digest_results)} results generated")
            return state
            
        except Exception as e:
            logger.error(f"Error in digest agent: {e}")
            state["errors"] = state.get("errors", [])
            state["errors"].append(f"Digest error: {str(e)}")
            return state
    
    def can_handle(self, task: SubTask) -> bool:
        """
        Determine if this agent can handle the given task.
        
        Args:
            task: Task to evaluate
            
        Returns:
            True if task is digest-related
        """
        digest_types = [
            "digest", "weekly_digest", "monthly_digest", "daily_digest",
            "engagement_analysis", "trending_content", "summary", "recap"
        ]
        return any(digest_type in task.task_type.lower() for digest_type in digest_types)
    
    async def _generate_digest(self, subtask: SubTask, state: AgentState) -> Dict[str, Any]:
        """
        Generate a comprehensive digest for the specified time period.
        
        Args:
            subtask: Digest generation subtask
            state: Current agent state
            
        Returns:
            Digest results
        """
        try:
            # Extract parameters
            period = subtask.parameters.get("period", self.default_digest_period)
            channel_filter = subtask.parameters.get("channel_filter")
            start_date = subtask.parameters.get("start_date")
            end_date = subtask.parameters.get("end_date")
            
            # Get search results (messages)
            search_results = state.get("search_results", [])
            
            if not search_results:
                return {
                    "digest": "No messages found for the specified period.",
                    "period": period,
                    "message_count": 0,
                    "user_count": 0,
                    "channels": [],
                    "engagement_summary": {}
                }
            
            # Calculate actual time period if not provided
            if not start_date or not end_date:
                start_date, end_date = self._calculate_time_period(period, search_results)
            
            # Generate digest components
            digest_data = {
                "period": period,
                "start_date": start_date,
                "end_date": end_date,
                "total_messages": len(search_results),
                "unique_users": len(set(msg.get("author", {}).get("username", "") for msg in search_results)),
                "channels_summary": await self._summarize_channels(search_results),
                "user_activity": await self._analyze_user_activity(search_results),
                "engagement_metrics": await self._calculate_engagement_metrics(search_results),
                "trending_content": await self._identify_trending_messages(search_results),
                "key_discussions": await self._extract_key_discussions(search_results),
                "time_patterns": await self._analyze_time_patterns(search_results)
            }
            
            # Format digest text
            digest_text = await self._format_digest_text(digest_data)
            
            return {
                "digest": digest_text,
                "metadata": digest_data,
                "generated_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating digest: {e}")
            return {"error": f"Failed to generate digest: {str(e)}"}
    
    def _calculate_time_period(self, period: str, messages: List[Dict[str, Any]]) -> tuple:
        """Calculate start and end dates for the digest period."""
        now = datetime.utcnow()
        
        if period == "daily":
            start_date = now - timedelta(days=1)
        elif period == "weekly":
            start_date = now - timedelta(weeks=1)
        elif period == "monthly":
            start_date = now - timedelta(days=30)
        else:
            # Use the actual date range from messages
            timestamps = [msg.get("timestamp") for msg in messages if msg.get("timestamp")]
            if timestamps:
                dates = [datetime.fromisoformat(ts.replace("Z", "+00:00")) for ts in timestamps]
                start_date = min(dates)
                now = max(dates)
            else:
                start_date = now - timedelta(weeks=1)
        
        return start_date.isoformat(), now.isoformat()
    
    async def _summarize_channels(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Summarize activity by channel."""
        channel_counts = Counter()
        channel_users = {}
        
        for msg in messages:
            channel = msg.get("channel_name", "unknown")
            user = msg.get("author", {}).get("username", "unknown")
            
            channel_counts[channel] += 1
            if channel not in channel_users:
                channel_users[channel] = set()
            channel_users[channel].add(user)
        
        return {
            "total_channels": len(channel_counts),
            "most_active": dict(channel_counts.most_common(5)),
            "channel_user_counts": {ch: len(users) for ch, users in channel_users.items()}
        }
    
    async def _analyze_user_activity(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze user activity patterns."""
        user_counts = Counter()
        user_channels = {}
        
        for msg in messages:
            user = msg.get("author", {}).get("username", "unknown")
            channel = msg.get("channel_name", "unknown")
            
            user_counts[user] += 1
            if user not in user_channels:
                user_channels[user] = set()
            user_channels[user].add(channel)
        
        return {
            "total_users": len(user_counts),
            "most_active_users": dict(user_counts.most_common(10)),
            "average_messages_per_user": sum(user_counts.values()) / len(user_counts) if user_counts else 0,
            "cross_channel_users": sum(1 for channels in user_channels.values() if len(channels) > 1)
        }
    
    async def _calculate_engagement_metrics(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate engagement metrics."""
        total_reactions = 0
        total_attachments = 0
        messages_with_engagement = 0
        
        for msg in messages:
            reactions = msg.get("reactions", [])
            attachments = msg.get("attachments", [])
            
            reaction_count = sum(r.get("count", 0) for r in reactions if isinstance(r, dict))
            total_reactions += reaction_count
            total_attachments += len(attachments)
            
            if reaction_count > 0 or len(attachments) > 0:
                messages_with_engagement += 1
        
        return {
            "total_reactions": total_reactions,
            "total_attachments": total_attachments,
            "engagement_rate": messages_with_engagement / len(messages) if messages else 0,
            "average_reactions_per_message": total_reactions / len(messages) if messages else 0
        }
    
    async def _identify_trending_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify trending/high-engagement messages only for trending section."""
        trending = []
        for msg in messages:
            reactions = msg.get("reactions", [])
            reaction_count = sum(r.get("count", 0) for r in reactions if isinstance(r, dict))
            attachment_count = len(msg.get("attachments", []))
            engagement_score = reaction_count + (attachment_count * 2)
            if engagement_score >= self.trending_threshold:
                trending.append({
                    "content": msg.get("content", "")[:200],
                    "author": msg.get("author", {}).get("username", "unknown"),
                    "channel": msg.get("channel_name", "unknown"),
                    "timestamp": msg.get("timestamp", ""),
                    "engagement_score": engagement_score,
                    "reactions": reaction_count,
                    "attachments": attachment_count
                })
        trending.sort(key=lambda x: x["engagement_score"], reverse=True)
        return trending[:10]  # Top 10
    
    async def _extract_key_discussions(self, messages: List[Dict[str, Any]]) -> List[str]:
        """Extract key discussion topics using simple keyword analysis."""
        # Simple implementation - could be enhanced with NLP
        all_content = " ".join(msg.get("content", "") for msg in messages)
        words = all_content.lower().split()
        
        # Filter out common words and short words
        stop_words = {"the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "is", "are", "was", "were", "be", "been", "have", "has", "had", "do", "does", "did", "will", "would", "could", "should", "may", "might", "can", "a", "an", "this", "that", "these", "those", "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them"}
        filtered_words = [word for word in words if len(word) > 3 and word not in stop_words]
        
        # Get most common meaningful words
        word_counts = Counter(filtered_words)
        key_topics = [word for word, count in word_counts.most_common(10) if count > 2]
        
        return key_topics
    
    async def _analyze_time_patterns(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze temporal patterns in the messages."""
        hour_counts = Counter()
        day_counts = Counter()
        
        for msg in messages:
            timestamp = msg.get("timestamp")
            if timestamp:
                try:
                    dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                    hour_counts[dt.hour] += 1
                    day_counts[dt.strftime("%A")] += 1
                except:
                    continue
        
        return {
            "peak_hours": dict(hour_counts.most_common(3)),
            "peak_days": dict(day_counts.most_common(3)),
            "activity_distribution": {
                "morning": sum(hour_counts[h] for h in range(6, 12)),
                "afternoon": sum(hour_counts[h] for h in range(12, 18)),
                "evening": sum(hour_counts[h] for h in range(18, 24)),
                "night": sum(hour_counts[h] for h in range(0, 6))
            }
        }
    
    async def _format_digest_text(self, digest_data: Dict[str, Any]) -> str:
        """Format the digest data into a readable text summary."""
        period = digest_data["period"].title()
        start_date = digest_data["start_date"][:10]  # Just the date part
        end_date = digest_data["end_date"][:10]
        
        digest_text = f"""# 📊 {period} Digest
**Period**: {start_date} to {end_date}
**Total Messages**: {digest_data["total_messages"]}
**Active Users**: {digest_data["unique_users"]}

## 👥 Most Active Users
"""
        
        for user, count in list(digest_data["user_activity"]["most_active_users"].items())[:5]:
            digest_text += f"• **{user}**: {count} messages\n"
        
        digest_text += "\n## 📋 Channel Activity\n"
        for channel, count in list(digest_data["channels_summary"]["most_active"].items())[:5]:
            digest_text += f"• **#{channel}**: {count} messages\n"
        
        if digest_data["trending_content"]:
            digest_text += "\n## 🔥 High Engagement Content\n"
            for item in digest_data["trending_content"][:3]:
                digest_text += f"• **{item['author']}** in **#{item['channel']}**: {item['content'][:100]}...\n"
                digest_text += f"  *{item['reactions']} reactions, {item['attachments']} attachments*\n\n"
        
        if digest_data["key_discussions"]:
            digest_text += f"## 🗣️ Key Discussion Topics\n"
            digest_text += f"{', '.join(digest_data['key_discussions'][:8])}\n\n"
        
        engagement_rate = digest_data["engagement_metrics"]["engagement_rate"]
        digest_text += f"## 📈 Engagement Summary\n"
        digest_text += f"• **Engagement Rate**: {engagement_rate:.1%}\n"
        digest_text += f"• **Total Reactions**: {digest_data['engagement_metrics']['total_reactions']}\n"
        digest_text += f"• **Attachments Shared**: {digest_data['engagement_metrics']['total_attachments']}\n"
        
        return digest_text
    
    async def _analyze_engagement(self, subtask: SubTask, state: AgentState) -> Dict[str, Any]:
        """Analyze user engagement patterns."""
        search_results = state.get("search_results", [])
        return await self._calculate_engagement_metrics(search_results)
    
    async def _identify_trending_content(self, subtask: SubTask, state: AgentState) -> List[Dict[str, Any]]:
        """Identify trending content."""
        search_results = state.get("search_results", [])
        return await self._identify_trending_messages(search_results) 