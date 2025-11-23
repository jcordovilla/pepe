"""
Report Generator

LLM-powered report generation for analytics using local Ollama models.
Generates natural language summaries for user activity, channel summaries,
community digests, and trend analysis.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    Generates LLM-powered reports for Discord analytics.

    Uses local Ollama models (deepseek-r1:8b or llama3.2:3b) to create
    natural language summaries and insights from message data.
    """

    def __init__(self, llm_client):
        """
        Initialize the report generator.

        Args:
            llm_client: UnifiedLLMClient instance for LLM calls
        """
        self.llm_client = llm_client

        # Report settings
        self.max_messages_for_context = 50  # Limit messages sent to LLM
        self.max_content_length = 200  # Truncate long messages

        logger.info("ReportGenerator initialized")

    # =========================================================================
    # User Activity Report
    # =========================================================================

    async def generate_user_report(
        self,
        username: str,
        messages: List[Dict],
        metrics: Dict[str, Any],
        days: int
    ) -> str:
        """
        Generate a user activity report.

        Args:
            username: Display name of the user
            messages: User's messages
            metrics: Calculated metrics
            days: Number of days analyzed

        Returns:
            LLM-generated report string
        """
        try:
            # Prepare message samples for context
            sample_messages = self._prepare_message_samples(messages)

            # Build prompt
            prompt = f"""Analyze this Discord user's activity and write a brief, insightful report.

**User:** {username}
**Period:** Last {days} days

**Activity Metrics:**
- Total messages: {metrics.get('total_messages', 0)}
- Active in {metrics.get('unique_channels', 0)} channels
- Top channels: {self._format_dict(metrics.get('channels_active', {}))}
- Most active day: {metrics.get('most_active_day', 'N/A')}
- Most active hour: {metrics.get('most_active_hour', 'N/A')}
- Average message length: {metrics.get('avg_message_length', 0)} characters
- Total reactions received: {metrics.get('total_reactions_received', 0)}

**Sample Messages:**
{sample_messages}

Write a 2-3 paragraph report that:
1. Summarizes what topics/themes this user discusses
2. Notes their activity level relative to the period (e.g., "26 messages over 30 days = ~1/day")
3. Lists their most active channels and what they discuss there

Be factual and specific. Avoid words like "actively engaged" or "valuable contributor" - just describe what they did.
If activity is low or concentrated in few channels, note that objectively."""

            system_prompt = """You are an objective analytics assistant generating reports about Discord community members.
Focus on factual observations about participation patterns - what topics they discuss, where they're active.
Be specific and data-driven. Avoid flattery or generic positive statements.
If someone has low activity, note it. If their contributions are limited to certain areas, say so.
Describe their actual behavior, not an idealized version."""

            response = await self.llm_client.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                max_tokens=500,
                temperature=0.3
            )

            return response.strip()

        except Exception as e:
            logger.error(f"Error generating user report: {e}")
            return f"Unable to generate detailed report. Basic stats: {metrics.get('total_messages', 0)} messages across {metrics.get('unique_channels', 0)} channels."

    # =========================================================================
    # Channel Summary
    # =========================================================================

    async def generate_channel_summary(
        self,
        channel_name: str,
        messages: List[Dict],
        metrics: Dict[str, Any],
        days: int
    ) -> str:
        """
        Generate a channel activity summary.

        Args:
            channel_name: Name of the channel
            messages: Channel messages
            metrics: Calculated metrics
            days: Number of days analyzed

        Returns:
            LLM-generated summary string
        """
        try:
            # Prepare message samples
            sample_messages = self._prepare_message_samples(messages)

            # Format top contributors
            top_contributors = self._format_dict(metrics.get('top_contributors', {}))

            # Format popular messages
            popular = metrics.get('top_reacted_messages', [])
            popular_str = ""
            for p in popular[:3]:
                popular_str += f"- \"{p['content'][:80]}...\" by {p['author']} ({p['reactions']} reactions)\n"

            prompt = f"""Summarize the activity in this Discord channel.

**Channel:** #{channel_name}
**Period:** Last {days} days

**Activity Metrics:**
- Total messages: {metrics.get('total_messages', 0)}
- Unique contributors: {metrics.get('unique_contributors', 0)}
- Average messages per day: {metrics.get('avg_messages_per_day', 0)}
- Top contributors: {top_contributors}

**Most Popular Messages:**
{popular_str if popular_str else "No highly-reacted messages"}

**Sample Discussions:**
{sample_messages}

Write a 2-3 paragraph summary that:
1. Describes the main topics and themes discussed
2. Notes the engagement level and community dynamics
3. Highlights any notable discussions or decisions

Keep it concise and informative."""

            system_prompt = """You are an objective analytics assistant summarizing Discord channel activity.
Focus on factual observations: what topics were discussed, who participated, what was decided.
Be honest about activity levels - if a channel is quiet, say so. If only a few people dominate, note it.
Avoid generic positive language. Report what happened, not what you wish happened."""

            response = await self.llm_client.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                max_tokens=500,
                temperature=0.3
            )

            return response.strip()

        except Exception as e:
            logger.error(f"Error generating channel summary: {e}")
            return f"Unable to generate detailed summary. Basic stats: {metrics.get('total_messages', 0)} messages from {metrics.get('unique_contributors', 0)} contributors."

    # =========================================================================
    # Community Digest
    # =========================================================================

    async def generate_digest(
        self,
        period: str,
        messages: List[Dict],
        metrics: Dict[str, Any]
    ) -> str:
        """
        Generate a community digest.

        Args:
            period: "daily", "weekly", or "monthly"
            messages: All messages for the period
            metrics: Calculated metrics

        Returns:
            LLM-generated digest string
        """
        try:
            # Group messages by channel for context
            channel_samples = self._group_messages_by_channel(messages)

            # Format top engaging content
            top_content = metrics.get('top_engaging_content', [])
            highlights_str = ""
            for c in top_content[:5]:
                highlights_str += f"- [{c['channel']}] {c['author']}: \"{c['content'][:100]}...\" ({c['reactions']} reactions)\n"

            # Calculate context metrics for calibration
            total_members = 926  # Could be made dynamic
            active_pct = round((metrics.get('unique_contributors', 0) / total_members) * 100, 1)
            msgs_per_member = round(metrics.get('total_messages', 0) / max(metrics.get('unique_contributors', 1), 1), 1)

            prompt = f"""Create a {period} digest for this Discord community.

**Period:** {period.capitalize()}

**Activity Metrics:**
- Total messages: {metrics.get('total_messages', 0)}
- Active channels: {metrics.get('unique_channels', 0)}
- Active members: {metrics.get('unique_contributors', 0)} out of {total_members} total ({active_pct}% participation)
- Average messages per active member: {msgs_per_member}

**Most Active Channels:**
{self._format_dict(metrics.get('active_channels', {}))}

**Top Contributors:**
{self._format_dict(metrics.get('top_contributors', {}))}

**Highlights (Most Engaging Content):**
{highlights_str if highlights_str else "No highlighted content"}

**Sample Discussions by Channel:**
{channel_samples}

Write an OBJECTIVE digest that:
1. **Overview**: Factual summary of activity levels - be honest about whether activity is high, moderate, or low
2. **Highlights**: Key discussions or notable content (if any)
3. **Active Areas**: Where conversations happened
4. **Observations**: Factual notes about participation patterns

Be realistic - if only 5% of members are active, note that. If activity is low, say so.
Do NOT use phrases like "busy week", "thriving community", or "highly engaged" unless the data supports it.
Format with headers. Be concise and factual."""

            system_prompt = """You are an objective analytics reporter for a Discord community.
Your job is to accurately describe activity levels without exaggeration or spin.
If activity is low, say it's low. If participation is limited, note that.
Avoid cheerleading language - be professional and factual like a business report.
Focus on what actually happened, not on making things sound better than they are."""

            response = await self.llm_client.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                max_tokens=700,
                temperature=0.4
            )

            return response.strip()

        except Exception as e:
            logger.error(f"Error generating digest: {e}")
            return f"Unable to generate detailed digest. This {period}: {metrics.get('total_messages', 0)} messages from {metrics.get('unique_contributors', 0)} members across {metrics.get('unique_channels', 0)} channels."

    # =========================================================================
    # Trends Analysis
    # =========================================================================

    async def generate_trends(
        self,
        messages: List[Dict],
        metrics: Dict[str, Any],
        days: int
    ) -> str:
        """
        Generate a trends analysis.

        Args:
            messages: Messages for trend analysis
            metrics: Calculated metrics
            days: Number of days analyzed

        Returns:
            LLM-generated trends analysis
        """
        try:
            # Format trending topics
            topics = metrics.get('trending_topics', {})
            topics_str = ", ".join([f"{word} ({count})" for word, count in list(topics.items())[:15]])

            # Format daily volume
            daily = metrics.get('daily_message_volume', {})
            volume_trend = "increasing" if len(daily) > 1 and list(daily.values())[-1] > list(daily.values())[0] else "stable"

            # Get sample messages around trending topics
            top_topics = list(topics.keys())[:5]
            topic_samples = self._get_topic_samples(messages, top_topics)

            prompt = f"""Analyze trends in this Discord community.

**Period:** Last {days} days
**Total Messages:** {metrics.get('total_messages', 0)}

**Trending Keywords:**
{topics_str}

**Message Volume Trend:** {volume_trend}
**Daily Activity:** {self._format_dict(daily)}

**Active Channels:**
{self._format_dict(metrics.get('active_channels', {}))}

**Sample Messages Around Trending Topics:**
{topic_samples}

Write a trends analysis that:
1. Identifies the main topics driving conversation
2. Notes any emerging themes or increasing interest areas
3. Highlights channel-specific trends if notable
4. Provides insight on community focus areas

Be specific and data-driven. Avoid generic observations."""

            system_prompt = """You are an objective community trends analyst.
Identify what topics appeared frequently - but be honest about sample sizes.
If trends are based on small numbers (e.g., 20 mentions), note that the data is limited.
Avoid overstating significance. Report observations, not hype.
Be specific and factual - avoid phrases like "strong interest" unless data clearly supports it."""

            response = await self.llm_client.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                max_tokens=500,
                temperature=0.3
            )

            return response.strip()

        except Exception as e:
            logger.error(f"Error generating trends: {e}")
            topics = list(metrics.get('trending_topics', {}).keys())[:5]
            return f"Unable to generate detailed analysis. Top topics: {', '.join(topics)}"

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _prepare_message_samples(self, messages: List[Dict], limit: int = None) -> str:
        """Prepare message samples for LLM context."""
        limit = limit or self.max_messages_for_context
        samples = []

        for m in messages[:limit]:
            content = (m.get("content", "") or "")[:self.max_content_length]
            if content.strip():
                author = m.get("author_display_name") or m.get("author_username", "Unknown")
                channel = m.get("channel_name", "")
                samples.append(f"[{channel}] {author}: {content}")

        return "\n".join(samples[:30])  # Limit total samples

    def _group_messages_by_channel(self, messages: List[Dict]) -> str:
        """Group messages by channel for context."""
        from collections import defaultdict

        by_channel = defaultdict(list)
        for m in messages[:100]:
            channel = m.get("channel_name", "unknown")
            content = (m.get("content", "") or "")[:150]
            if content.strip():
                author = m.get("author_display_name") or m.get("author_username", "Unknown")
                by_channel[channel].append(f"{author}: {content}")

        result = []
        for channel, msgs in list(by_channel.items())[:5]:
            result.append(f"**#{channel}:**")
            for msg in msgs[:3]:
                result.append(f"  - {msg}")
            result.append("")

        return "\n".join(result)

    def _get_topic_samples(self, messages: List[Dict], topics: List[str]) -> str:
        """Get message samples containing trending topics."""
        samples = []

        for m in messages[:200]:
            content = (m.get("content", "") or "").lower()
            for topic in topics:
                if topic.lower() in content:
                    author = m.get("author_display_name") or m.get("author_username", "Unknown")
                    channel = m.get("channel_name", "")
                    samples.append(f"[{channel}] {author}: {m.get('content', '')[:150]}")
                    break

            if len(samples) >= 10:
                break

        return "\n".join(samples) if samples else "No specific samples available"

    def _format_dict(self, d: Dict, limit: int = 5) -> str:
        """Format a dictionary for display."""
        if not d:
            return "N/A"

        items = list(d.items())[:limit]
        return ", ".join([f"{k}: {v}" for k, v in items])
