"""
Digest Agent

Map-reduce summarization agent for high-engagement messages.
Creates periodic digests and summaries of Discord activity.
"""

import logging
import sqlite3
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import re

from ..base_agent import BaseAgent, AgentRole, AgentState
from ...services.llm_client import UnifiedLLMClient
from ...vectorstore.persistent_store import PersistentVectorStore

logger = logging.getLogger(__name__)


class DigestAgent(BaseAgent):
    """
    Digest agent that creates summaries of high-engagement messages.
    
    Input: dict(start: dt, end: dt, period: str)
    Output: str (markdown)
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(AgentRole.DIGESTER, config)
        self.llm_client = UnifiedLLMClient(config.get("llm", {}))
        self.vector_store = PersistentVectorStore(config.get("vector_config", {}))
        
        # Digest configuration - adaptive limits based on time period
        self.base_max_messages = config.get("max_messages", 25)
        self.max_summary_length = config.get("max_summary_length", 250)
        
        # Adaptive message limits for different time periods
        self.period_limits = {
            "day": 25,
            "week": 100,  # Increased for weekly digests
            "month": 200,  # Increased for monthly digests
            "all_time": 150
        }
        
        logger.info("DigestAgent initialized")
    
    def signature(self) -> Dict[str, Any]:
        """Return agent signature for registration."""
        return {
            "role": "digest",
            "description": "Creates summaries of high-engagement Discord messages",
            "input_schema": {
                "type": "object",
                "properties": {
                    "start": {"type": "string", "description": "Start date (ISO format)"},
                    "end": {"type": "string", "description": "End date (ISO format)"},
                    "period": {"type": "string", "description": "Period type (day, week, month)"}
                },
                "required": ["start", "end"]
            },
            "output_schema": {
                "type": "string",
                "description": "Markdown formatted digest"
            }
        }
    
    async def run(self, **kwargs) -> str:
        """
        Generate a digest for the given time period.
        
        Args:
            start: Start date
            end: End date
            period: Period type (day, week, month)
            
        Returns:
            Markdown formatted digest
        """
        import time
        start_time = time.time()
        
        start_str = kwargs.get("start")
        end_str = kwargs.get("end")
        period = kwargs.get("period", "day")
        channel_id = kwargs.get("channel_id")
        channel = kwargs.get("channel")
        
        # Parse dates - handle None for "all time" queries
        try:
            if start_str is None and end_str is None:
                # "All time" query - no date filtering
                start_date = None
                end_date = None
                logger.info(f"DigestAgent processing digest for all time ({period})")
            else:
                # Specific time period query
                start_date = datetime.fromisoformat(start_str) if start_str else datetime.now() - timedelta(days=1)
                end_date = datetime.fromisoformat(end_str) if end_str else datetime.now()
                logger.info(f"DigestAgent processing digest from {start_date} to {end_date} ({period})")
        except ValueError as e:
            logger.error(f"Date parsing error: {e}")
            return f"Error: Invalid date format - {e}"
        
        try:
            # Get all messages in the period (optionally filter by channel_id or channel_name)
            logger.info(f"[DigestAgent] Starting message retrieval at {time.time() - start_time:.2f}s")
            messages = await self._get_all_messages(start_date, end_date, channel_id, channel, period)
            logger.info(f"[DigestAgent] Message retrieval completed at {time.time() - start_time:.2f}s")
            
            # Get resources for the period
            logger.info(f"[DigestAgent] Starting resource retrieval at {time.time() - start_time:.2f}s")
            resources = await self._get_resources_for_period(start_date, end_date, channel_id, channel)
            logger.info(f"[DigestAgent] Resource retrieval completed at {time.time() - start_time:.2f}s")
            
            # Track whether resources were filtered by channel
            # Check if we have channel_name or can map channel_id to channel_name
            resolved_channel_name = channel
            if not channel and channel_id:
                resolved_channel_name = await self._get_channel_name_from_id(channel_id)
            channel_filtered = bool(resolved_channel_name)
            
            if not messages and not resources:
                return f"No messages or resources found for the {period} period."
            
            # Generate individual summaries
            logger.info(f"[DigestAgent] Starting summary generation at {time.time() - start_time:.2f}s")
            summaries = await self._generate_message_summaries(messages, period)
            logger.info(f"[DigestAgent] Summary generation completed at {time.time() - start_time:.2f}s")
            
            # Combine summaries and resources into final digest
            logger.info(f"[DigestAgent] Starting digest combination at {time.time() - start_time:.2f}s")
            digest = await self._combine_summaries_with_resources(summaries, resources, period, channel_filtered)
            logger.info(f"[DigestAgent] Digest combination completed at {time.time() - start_time:.2f}s")
            
            total_time = time.time() - start_time
            logger.info(f"[DigestAgent] Total digest generation time: {total_time:.2f}s")
            
            return digest
            
        except Exception as e:
            logger.error(f"DigestAgent error: {e}")
            return f"Error generating digest: {str(e)}"
    
    async def _get_all_messages(self, start_date: Optional[datetime], end_date: Optional[datetime], channel_id: Optional[str] = None, channel: Optional[str] = None, period: str = "day") -> List[Dict[str, Any]]:
        """Get all messages from the vector store for the period, optionally filtered by channel_id/forum_channel_id or channel_name."""
        try:
            # Determine adaptive message limit based on period
            max_messages = self.period_limits.get(period, self.base_max_messages)
            
            # Use timestamp_unix (float) for ChromaDB filtering
            start_ts = start_date.timestamp() if start_date else None
            end_ts = end_date.timestamp() if end_date else None
            filters_and = []
            
            # Add timestamp filters only if dates are provided
            if start_ts is not None:
                filters_and.append({"timestamp_unix": {"$gte": start_ts}})
            if end_ts is not None:
                filters_and.append({"timestamp_unix": {"$lte": end_ts}})
            
            # Always include channel/forum filter if channel_id is provided
            if channel_id:
                filters_and.append({"$or": [
                    {"channel_id": channel_id},
                    {"forum_channel_id": channel_id}
                ]})
            elif channel:
                filters_and.append({"channel_name": channel})
            
            # Filter out bot messages for digest queries
            filters_and.append({"author_bot": False})
            
            # Construct final filter
            if len(filters_and) == 0:
                # No filters - get all messages
                filters = {}
            elif len(filters_and) == 1:
                # Single filter
                filters = filters_and[0]
            else:
                # Multiple filters - use $and
                filters = {"$and": filters_and}
                
            logger.info(f"[DigestAgent] FINAL filter_search filters: {filters}")
            
            # For cross-server digests (no channel filter), use smarter retrieval
            if not channel_id and not channel and period in ["week", "month", "all_time"]:
                messages = await self._get_cross_server_messages(filters, max_messages, period)
            else:
                # Single channel or daily digest - use standard approach
                results = await self.vector_store.filter_search(
                    filters=filters,
                    k=max_messages,
                    sort_by="timestamp"
                )
                messages = self._process_results(results)
            
            logger.info(f"[DigestAgent] Retrieved {len(messages)} messages from vector store for {period} period.")
            for i, message in enumerate(messages[:3]):
                logger.info(f"[DigestAgent] Message {i+1}: channel={message.get('channel')}, content={message.get('content', '')[:60]}")
            
            return messages
        except Exception as e:
            logger.error(f"Error getting messages: {e}")
            return []
    
    async def _get_cross_server_messages(self, filters: Dict[str, Any], max_messages: int, period: str) -> List[Dict[str, Any]]:
        """Get representative messages from multiple channels for cross-server digests."""
        try:
            # Strategy: Get high-engagement messages first, then fill with recent messages from different channels
            
            # Step 1: Get high-engagement messages (prioritize reactions)
            high_engagement_results = await self.vector_store.filter_search(
                filters=filters,
                k=max_messages // 2,  # Half from high engagement
                sort_by="total_reactions"
            )
            
            # Step 2: Get recent messages from different channels
            recent_results = await self.vector_store.filter_search(
                filters=filters,
                k=max_messages,  # Get more to ensure channel diversity
                sort_by="timestamp"
            )
            
            # Step 3: Combine and deduplicate, ensuring channel diversity
            all_messages = self._process_results(high_engagement_results + recent_results)
            
            # Step 4: Ensure channel diversity by sampling from different channels
            diverse_messages = self._ensure_channel_diversity(all_messages, max_messages, period)
            
            return diverse_messages
            
        except Exception as e:
            logger.error(f"Error in cross-server message retrieval: {e}")
            # Fallback to standard retrieval
            results = await self.vector_store.filter_search(
                filters=filters,
                k=max_messages,
                sort_by="timestamp"
            )
            return self._process_results(results)
    
    async def _get_channel_name_from_id(self, channel_id: str) -> Optional[str]:
        """Get channel name from channel ID using the discord_messages database."""
        try:
            db_path = "data/discord_messages.db"
            if not Path(db_path).exists():
                logger.warning("Discord messages database not found for channel lookup")
                return None
            
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Query for channel name
            cursor.execute("SELECT DISTINCT channel_name FROM messages WHERE channel_id = ? LIMIT 1", (channel_id,))
            result = cursor.fetchone()
            
            conn.close()
            
            if result:
                channel_name = result[0]
                logger.info(f"[DigestAgent] Mapped channel_id {channel_id} to channel_name: {channel_name}")
                return channel_name
            else:
                logger.warning(f"[DigestAgent] No channel name found for channel_id: {channel_id}")
                return None
                
        except Exception as e:
            logger.error(f"Error looking up channel name for channel_id {channel_id}: {e}")
            return None
    
    async def _get_resources_for_period(self, start_date: Optional[datetime], end_date: Optional[datetime], channel_id: Optional[str] = None, channel_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get resources from the enhanced resources database for the given time period and channel."""
        try:
            db_path = "data/enhanced_resources.db"
            if not Path(db_path).exists():
                logger.warning("Enhanced resources database not found")
                return []
            
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Build query conditions
            conditions = []
            params = []
            
            # Date filtering
            if start_date:
                conditions.append("timestamp >= ?")
                params.append(start_date.strftime("%Y-%m-%d"))
            if end_date:
                conditions.append("timestamp <= ?")
                params.append(end_date.strftime("%Y-%m-%d"))
            
            # Channel filtering - prioritize channel_name, but map channel_id if needed
            resolved_channel_name = channel_name
            if not channel_name and channel_id:
                # Try to map channel_id to channel_name
                resolved_channel_name = await self._get_channel_name_from_id(channel_id)
            
            if resolved_channel_name:
                conditions.append("channel_name = ?")
                params.append(resolved_channel_name)
                logger.info(f"[DigestAgent] Filtering resources by channel_name: {resolved_channel_name}")
            elif channel_id:
                logger.warning(f"[DigestAgent] Could not map channel_id {channel_id} to channel_name - getting all resources for the time period")
            
            # Build the query
            query = "SELECT * FROM resources"
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            query += " ORDER BY quality_score DESC, timestamp DESC"
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            # Get column names
            columns = [description[0] for description in cursor.description]
            
            # Convert to list of dictionaries
            resources = []
            for row in rows:
                resource = dict(zip(columns, row))
                resources.append(resource)
            
            conn.close()
            
            logger.info(f"[DigestAgent] Retrieved {len(resources)} resources from database")
            return resources
            
        except Exception as e:
            logger.error(f"Error getting resources from database: {e}")
            return []
    
    def _process_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process vector store results into message format."""
        messages = []
        for result in results:
            # Prefer display_name over username for better user identification
            author_display_name = result.get("author_display_name", "")
            author_username = result.get("author_username", "")
            author = author_display_name if author_display_name else author_username if author_username else "Unknown"
            
            messages.append({
                "content": result.get("content", ""),
                "metadata": result,
                "permalink": result.get("jump_url", ""),
                "reactions": result.get("total_reactions", 0),
                "author": author,
                "channel": result.get("channel_name", "Unknown"),
                "channel_id": result.get("channel_id", "")
            })
        return messages
    
    def _ensure_channel_diversity(self, messages: List[Dict[str, Any]], max_messages: int, period: str) -> List[Dict[str, Any]]:
        """Ensure messages come from diverse channels for better representation."""
        if not messages:
            return messages
        
        # Group messages by channel
        channels = {}
        for message in messages:
            channel_id = message.get("channel_id", "")
            if channel_id not in channels:
                channels[channel_id] = []
            channels[channel_id].append(message)
        
        # Calculate how many messages to take from each channel
        num_channels = len(channels)
        if num_channels == 0:
            return messages[:max_messages]
        
        # For weekly/monthly digests, ensure we get messages from multiple channels
        if period in ["week", "month"]:
            messages_per_channel = max(3, max_messages // num_channels)  # At least 3 per channel
        else:
            messages_per_channel = max_messages // num_channels
        
        # Sample from each channel
        diverse_messages = []
        for channel_id, channel_messages in channels.items():
            # Sort by reactions first, then by recency
            sorted_messages = sorted(channel_messages, 
                                   key=lambda x: (x.get("reactions", 0), x.get("metadata", {}).get("timestamp", "")), 
                                   reverse=True)
            diverse_messages.extend(sorted_messages[:messages_per_channel])
        
        # Sort final list by reactions and recency, then limit
        diverse_messages.sort(key=lambda x: (x.get("reactions", 0), x.get("metadata", {}).get("timestamp", "")), reverse=True)
        
        return diverse_messages[:max_messages]
    
    async def _get_high_engagement_messages(self, start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
        """Get high-engagement messages for trending only."""
        try:
            results = await self.vector_store.reaction_search(
                reaction="",
                k=self.max_messages,
                filters={
                    "$and": [
                        {
                            "timestamp": {
                                "$gte": start_date.isoformat(),
                                "$lte": end_date.isoformat()
                            }
                        },
                        {
                            "author_bot": False
                        }
                    ]
                },
                sort_by="total_reactions"
            )
            high_engagement = []
            for result in results:
                reactions = result.get("metadata", {}).get("reactions", [])
                total_reactions = sum(reaction.get("count", 0) for reaction in reactions)
                if total_reactions >= 3:  # Only for trending
                    # Prefer display_name over username for better user identification
                    metadata = result.get("metadata", {})
                    author_display_name = metadata.get("author_display_name", "")
                    author_username = metadata.get("author_username", "")
                    author = author_display_name if author_display_name else author_username if author_username else "Unknown"
                    
                    high_engagement.append({
                        "content": result.get("content", ""),
                        "metadata": metadata,
                        "permalink": metadata.get("permalink", ""),
                        "reactions": total_reactions,
                        "author": author,
                        "channel": metadata.get("channel_name", "Unknown")
                    })
            return high_engagement[:self.max_messages]
        except Exception as e:
            logger.error(f"Error getting high-engagement messages: {e}")
            return []
    
    async def _generate_message_summaries(self, messages: List[Dict[str, Any]], period: str = "day") -> List[Dict[str, Any]]:
        """Generate summaries for messages with aggressive optimization for large sets."""
        if not messages:
            return []
        
        # For very large numbers of messages, use ultra-simplified approach
        if len(messages) > 20:
            return await self._ultra_simplified_summaries(messages, period)
        # For medium numbers, use batch approach
        elif len(messages) > 10:
            return await self._batch_summarize_messages(messages)
        else:
            return await self._individual_summarize_messages(messages)
    
    async def _ultra_simplified_summaries(self, messages: List[Dict[str, Any]], period: str = "day") -> List[Dict[str, Any]]:
        """Generate ultra-simplified summaries for large message sets."""
        # Take only the first 20 messages to avoid overwhelming the LLM
        sample_messages = messages[:20]
        
        # Create a simple summary in one LLM call
        summary_texts = []
        for i, message in enumerate(sample_messages):
            content = message.get("content", "")
            author = message.get("author", "Unknown")
            channel = message.get("channel", "Unknown")
            
            # Truncate content
            if len(content) > 100:
                content = content[:100] + "..."
            
            summary_texts.append(f"{i+1}. [{author}]: {content}")
        
        # Adjust prompt based on period
        if period == "all_time":
            time_context = "throughout the channel's history"
            period_description = "all time"
            focus_instruction = "Focus on recurring themes, key discussions, important topics that have emerged, and the overall community dynamics. Make it comprehensive and informative, reflecting the depth and breadth of discussions that have occurred over time."
        else:
            time_context = f"in the {period} period"
            period_description = period
            focus_instruction = "Focus on the main topics and key discussions that occurred during this time period. Highlight notable conversations and community engagement."
        
        prompt = f"""Summarize these Discord messages in a comprehensive paragraph. Focus on the main topics and key discussions. This is a "{period_description}" summary covering {time_context}, so avoid references to specific dates, times, or "today/yesterday" unless relevant to the {period_description} context:

{chr(10).join(summary_texts)}

Write a detailed, natural paragraph that summarizes the main topics discussed {time_context}. {focus_instruction} Do not include phrases like "Here is a summary" or "This paragraph summarizes" - just write the summary directly."""

        try:
            response = await self.llm_client.generate(
                prompt=prompt,
                max_tokens=500,  # Increased from 200 for more elaborate content
                temperature=0.1
            )
            
            # Create a single summary entry for all messages
            return [{
                "original": {"content": "Multiple messages", "author": "Various users"},
                "summary": response.strip()
            }]
            
        except Exception as e:
            logger.error(f"Error in ultra-simplified summary: {e}")
            # Fallback: create basic summary
            return [{
                "original": {"content": "Multiple messages", "author": "Various users"},
                "summary": f"Generated summary of {len(messages)} messages from various users covering {time_context}."
            }]
    
    async def _ultra_simplified_summaries_no_header(self, messages: List[Dict[str, Any]], period: str = "day") -> str:
        """Generate ultra-simplified summary without header (for use in combined digests)."""
        if not messages:
            return "No messages found for this period."
        
        # Take only the first 20 messages to avoid overwhelming the LLM
        sample_messages = messages[:20]
        
        # Create a simple summary in one LLM call
        summary_texts = []
        for i, message in enumerate(sample_messages):
            content = message.get("content", "")
            author = message.get("author", "Unknown")
            channel = message.get("channel", "Unknown")
            
            # Truncate content
            if len(content) > 100:
                content = content[:100] + "..."
            
            summary_texts.append(f"{i+1}. [{author}]: {content}")
        
        # Adjust prompt based on period
        if period == "all_time":
            time_context = "throughout the channel's history"
            period_description = "all time"
            focus_instruction = "Focus on recurring themes, key discussions, important topics that have emerged, and the overall community dynamics. Make it comprehensive and informative, reflecting the depth and breadth of discussions that have occurred over time."
        else:
            time_context = f"in the {period} period"
            period_description = period
            focus_instruction = "Focus on the main topics and key discussions that occurred during this time period. Highlight notable conversations and community engagement."
        
        prompt = f"""Summarize these Discord messages in a comprehensive paragraph. Focus on the main topics and key discussions. This is a "{period_description}" summary covering {time_context}, so avoid references to specific dates, times, or "today/yesterday" unless relevant to the {period_description} context:

{chr(10).join(summary_texts)}

Write a detailed, natural paragraph that summarizes the main topics discussed {time_context}. {focus_instruction} Do not include phrases like "Here is a summary" or "This paragraph summarizes" - just write the summary directly."""

        try:
            response = await self.llm_client.generate(
                prompt=prompt,
                max_tokens=800,  # Increased for more elaborate content
                temperature=0.1
            )
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Error in ultra-simplified summary: {e}")
            # Fallback: create basic summary
            return f"Generated summary of {len(messages)} messages from various users covering {time_context}."
    
    async def _batch_summarize_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Summarize messages in batches to reduce LLM calls."""
        summaries = []
        
        # Process messages in batches of 10
        batch_size = 10
        for i in range(0, len(messages), batch_size):
            batch = messages[i:i + batch_size]
            batch_summaries = await self._summarize_batch(batch, i + 1)
            summaries.extend(batch_summaries)
        
        return summaries
    
    async def _summarize_batch(self, messages: List[Dict[str, Any]], batch_num: int) -> List[Dict[str, Any]]:
        """Summarize a batch of messages in a single LLM call."""
        # Prepare batch content
        batch_content = []
        for i, message in enumerate(messages):
            content = message.get("content", "")
            author = message.get("author", "Unknown")
            channel = message.get("channel", "Unknown")
            reactions = message.get("reactions", 0)
            
            # Truncate content if too long
            if len(content) > 200:
                content = content[:200] + "..."
            
            batch_content.append(f"{i+1}. [{author} in #{channel}] ({reactions} reactions): {content}")
        
        prompt = f"""Summarize these Discord messages in a concise way. For each message, provide a 1-2 sentence summary focusing on the key topic or main point:

{chr(10).join(batch_content)}

Provide summaries in this format:
1. [summary of message 1]
2. [summary of message 2]
...etc"""

        try:
            response = await self.llm_client.generate(
                prompt=prompt,
                max_tokens=500,
                temperature=0.1
            )
            
            # Parse the response into individual summaries
            lines = response.strip().split('\n')
            batch_summaries = []
            
            for i, line in enumerate(lines):
                if i < len(messages):
                    # Extract summary from numbered line
                    summary = line.strip()
                    if summary and summary[0].isdigit():
                        # Remove the number and dot
                        summary = summary.split('.', 1)[1].strip() if '.' in summary else summary
                    
                    batch_summaries.append({
                        "original": messages[i],
                        "summary": summary or f"Message by {messages[i].get('author', 'Unknown')}"
                    })
                else:
                    break
            
            # Ensure we have summaries for all messages
            while len(batch_summaries) < len(messages):
                msg = messages[len(batch_summaries)]
                batch_summaries.append({
                    "original": msg,
                    "summary": f"Message by {msg.get('author', 'Unknown')} in {msg.get('channel', 'Unknown')}"
                })
            
            return batch_summaries
            
        except Exception as e:
            logger.error(f"Error summarizing batch: {e}")
            # Fallback: create basic summaries
            return [{
                "original": msg,
                "summary": f"Message by {msg.get('author', 'Unknown')} in {msg.get('channel', 'Unknown')}"
            } for msg in messages]
    
    async def _individual_summarize_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate individual summaries for each message (for small batches)."""
        summaries = []
        
        for message in messages:
            try:
                summary = await self._summarize_message(message)
                summaries.append({
                    "original": message,
                    "summary": summary
                })
            except Exception as e:
                logger.error(f"Error summarizing message: {e}")
                continue
        
        return summaries

    async def _summarize_message(self, message: Dict[str, Any]) -> str:
        """Generate a summary for a single message."""
        content = message.get("content", "")
        author = message.get("author", "Unknown")
        channel = message.get("channel", "Unknown")
        reactions = message.get("reactions", 0)
        
        # Truncate content if too long
        if len(content) > 500:
            content = content[:500] + "..."
        
        prompt = f"""Summarize this Discord message in 1-2 sentences:

Message: "{content}"
Author: {author}
Channel: {channel}
Reactions: {reactions}

Focus on the key points and main topic. Be concise but informative."""

        try:
            summary = await self.llm_client.generate(
                prompt=prompt,
                max_tokens=100,
                temperature=0.1
            )
            return summary.strip()
        except Exception as e:
            logger.error(f"Error generating message summary: {e}")
            return f"Message by {author} in {channel} ({reactions} reactions)"
    
    async def _combine_summaries(self, summaries: List[Dict[str, Any]], period: str) -> str:
        """Combine individual summaries into a final digest."""
        if not summaries:
            return f"No content to summarize for the {period} period."
        
        # For ultra-simplified summaries (large message sets), return directly
        if len(summaries) == 1 and summaries[0]["original"].get("content") == "Multiple messages":
            summary = summaries[0]["summary"]
            header = self._generate_descriptive_header(period, "digest")
            return header + summary
        
        # For large numbers of summaries, use a simplified approach
        if len(summaries) > 20:
            return await self._generate_simplified_digest(summaries, period)
        else:
            # Group summaries by theme
            grouped_summaries = await self._group_by_theme(summaries)
            # Generate final digest
            digest = await self._generate_final_digest(grouped_summaries, period)
            return digest
    
    def _generate_descriptive_header(self, period: str, query_type: str = "digest") -> str:
        """Generate a descriptive header based on the period and query type."""
        if period == "all_time":
            period_display = "All Time"
        elif period == "week":
            period_display = "Weekly"
        elif period == "month":
            period_display = "Monthly"
        else:
            period_display = "Daily"
        
        return f"# 📋 {period_display} {query_type.title()}\n\n"
    
    async def _generate_simplified_digest(self, summaries: List[Dict[str, Any]], period: str) -> str:
        """Generate a simplified digest for large numbers of messages."""
        # Take a sample of summaries for the digest
        sample_size = min(15, len(summaries))
        sample_summaries = summaries[:sample_size]
        
        # Create a simple digest without complex grouping
        summary_texts = []
        for i, summary_data in enumerate(sample_summaries):
            summary = summary_data["summary"]
            author = summary_data["original"].get("author", "Unknown")
            summary_texts.append(f"{i+1}. {summary} (by {author})")
        
        time_context = "throughout the channel's history" if period == "all_time" else f"in the {period} period"
        
        # Adjust prompt based on period for more elaborate content
        if period == "all_time":
            prompt = f"""Create a comprehensive digest of these Discord messages {time_context}. Focus on the main topics and key discussions. Since this covers {time_context}, avoid references to specific dates, times, or "today/yesterday":

{chr(10).join(summary_texts)}

Write a detailed, flowing summary covering:
1. Main topics and themes discussed {time_context}
2. Key insights or notable discussions that have emerged
3. Overall activity patterns and community engagement
4. Recurring themes and long-term trends

Write this as a natural, comprehensive paragraph without meta-instructions. Make it detailed and informative, reflecting the depth of discussion that has occurred {time_context}."""
        else:
            prompt = f"""Create a comprehensive digest of these Discord messages {time_context}. Focus on the main topics and key discussions. Since this covers {time_context}, avoid references to specific dates, times, or "today/yesterday":

{chr(10).join(summary_texts)}

Write a detailed, flowing summary covering:
1. Main topics and themes discussed {time_context}
2. Key insights or notable discussions that have emerged
3. Overall activity patterns and community engagement
4. Notable contributions and community highlights
5. Emerging trends or patterns in the discussions

Write this as a natural, comprehensive paragraph without meta-instructions like "Here is a summary" or "This covers". Just write the content directly. Make it detailed and engaging, capturing the full scope of activity."""

        try:
            digest = await self.llm_client.generate(
                prompt=prompt,
                max_tokens=800 if period == "all_time" else 600,  # More tokens for more elaborate content
                temperature=0.1
            )
            
            # Add descriptive header
            header = self._generate_descriptive_header(period, "digest")
            
            return header + digest.strip()
            
        except Exception as e:
            logger.error(f"Error generating simplified digest: {e}")
            # Fallback: create basic digest
            header = self._generate_descriptive_header(period, "digest")
            return f"{header}Generated digest with {len(summaries)} messages covering {time_context}. Main topics covered: {', '.join([s['summary'][:50] + '...' for s in sample_summaries[:5]])}"
    
    async def _generate_simplified_digest_no_header(self, summaries: List[Dict[str, Any]], period: str) -> str:
        """Generate a simplified digest without header (for use in combined digests)."""
        # Take a sample of summaries for the digest
        sample_size = min(15, len(summaries))
        sample_summaries = summaries[:sample_size]
        
        # Create a simple digest without complex grouping
        summary_texts = []
        for i, summary_data in enumerate(sample_summaries):
            summary = summary_data["summary"]
            author = summary_data["original"].get("author", "Unknown")
            summary_texts.append(f"{i+1}. {summary} (by {author})")
        
        time_context = "throughout the channel's history" if period == "all_time" else f"in the {period} period"
        
        # Adjust prompt based on period for more elaborate content
        if period == "all_time":
            prompt = f"""Create a comprehensive digest of these Discord messages {time_context}. Focus on the main topics and key discussions. Since this covers {time_context}, avoid references to specific dates, times, or "today/yesterday":

{chr(10).join(summary_texts)}

Write a detailed, flowing summary covering:
1. Main topics and themes discussed {time_context}
2. Key insights or notable discussions that have emerged
3. Overall activity patterns and community engagement
4. Recurring themes and long-term trends

Write this as a natural, comprehensive paragraph without meta-instructions. Make it detailed and informative, reflecting the depth of discussion that has occurred {time_context}."""
        else:
            prompt = f"""Create a comprehensive digest of these Discord messages {time_context}. Focus on the main topics and key discussions. Since this covers {time_context}, avoid references to specific dates, times, or "today/yesterday":

{chr(10).join(summary_texts)}

Write a detailed, flowing summary covering:
1. Main topics and themes discussed {time_context}
2. Key insights or notable discussions that have emerged
3. Overall activity patterns and community engagement
4. Notable contributions and community highlights
5. Emerging trends or patterns in the discussions

Write this as a natural, comprehensive paragraph without meta-instructions like "Here is a summary" or "This covers". Just write the content directly. Make it detailed and engaging, capturing the full scope of activity."""

        try:
            digest = await self.llm_client.generate(
                prompt=prompt,
                max_tokens=800 if period == "all_time" else 600,  # More tokens for more elaborate content
                temperature=0.1
            )
            
            return digest.strip()
            
        except Exception as e:
            logger.error(f"Error generating simplified digest: {e}")
            # Fallback: create basic digest
            time_context = "throughout the channel's history" if period == "all_time" else f"in the {period} period"
            return f"Generated digest with {len(summaries)} messages covering {time_context}. Main topics covered: {', '.join([s['summary'][:50] + '...' for s in sample_summaries[:5]])}"
    
    async def _group_by_theme(self, summaries: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Group summaries by theme using LLM."""
        if len(summaries) <= 3:
            # For small numbers, just group as "General Discussion"
            return {"General Discussion": summaries}
        
        # Use LLM to group by theme
        summary_texts = []
        for i, summary_data in enumerate(summaries):
            summary = summary_data["summary"]
            summary_texts.append(f"{i+1}. {summary}")
        
        prompt = f"""Group these message summaries by theme/topic. Return a JSON object where keys are theme names and values are lists of summary numbers (1, 2, 3, etc.):

Summaries:
{chr(10).join(summary_texts)}

Group them into 3-5 themes. Return only valid JSON."""

        try:
            response = await self.llm_client.generate_json(prompt=prompt)
            
            # Parse the grouping
            grouped = {}
            for theme, indices in response.items():
                theme_summaries = []
                for idx in indices:
                    if 1 <= idx <= len(summaries):
                        theme_summaries.append(summaries[idx-1])
                if theme_summaries:
                    grouped[theme] = theme_summaries
            
            return grouped if grouped else {"General Discussion": summaries}
            
        except Exception as e:
            logger.error(f"Error grouping by theme: {e}")
            return {"General Discussion": summaries}
    
    async def _generate_final_digest(self, grouped_summaries: Dict[str, List[Dict[str, Any]]], period: str) -> str:
        """Generate the final digest from grouped summaries."""
        # Prepare input for final digest
        themes_text = []
        for theme, summaries in grouped_summaries.items():
            theme_content = f"## {theme}\n"
            for summary_data in summaries:
                original = summary_data["original"]
                summary = summary_data["summary"]
                permalink = original.get("permalink", "")
                author = original.get("author", "Unknown")
                
                theme_content += f"- **{author}**: {summary} [Link]({permalink})\n"
            themes_text.append(theme_content)
        
        themes_combined = "\n".join(themes_text)
        
        prompt = f"""Combine the summaries below into a {self.max_summary_length}-word bullet-point digest grouped by theme. Each bullet must cite at least one message link.

{themes_combined}

Create a concise, well-organized digest that captures the key discussions and highlights."""

        try:
            digest = await self.llm_client.generate(
                prompt=prompt,
                max_tokens=500,
                temperature=0.1
            )
            
            # Add header
            header = self._generate_descriptive_header(period, "digest")
            
            return header + digest.strip()
            
        except Exception as e:
            logger.error(f"Error generating final digest: {e}")
            header = self._generate_descriptive_header(period, "digest")
            return f"{header}Error generating digest: {str(e)}"
    
    async def _combine_summaries_with_resources(self, summaries: List[Dict[str, Any]], resources: List[Dict[str, Any]], period: str, channel_filtered: bool = True) -> str:
        """Combine message summaries and resources into a comprehensive digest."""
        try:
            # Generate header
            header = self._generate_descriptive_header(period, "digest")
            
            # Start building the digest
            digest_parts = []
            
            # Add message summaries if available
            if summaries:
                if len(summaries) <= 15:
                    # Use simplified digest for smaller sets - but remove the header to avoid duplication
                    message_digest = await self._generate_simplified_digest_no_header(summaries, period)
                    digest_parts.append(message_digest)
                else:
                    # Use ultra-simplified for large sets - but remove the header to avoid duplication
                    message_digest = await self._ultra_simplified_summaries_no_header(summaries, period)
                    digest_parts.append(message_digest)
            
            # Add resources section if available
            if resources:
                resources_section = await self._format_resources_section(resources, period, channel_filtered)
                digest_parts.append(resources_section)
            
            # Combine all parts with header at the top
            full_digest = header + "\n\n".join(digest_parts)
            
            return full_digest
            
        except Exception as e:
            logger.error(f"Error combining summaries with resources: {e}")
            return f"Error generating digest: {str(e)}"
    
    async def _format_resources_section(self, resources: List[Dict[str, Any]], period: str, channel_filtered: bool = True) -> str:
        """Format resources into a digest section."""
        try:
            if not resources:
                return ""
            
            # Sort resources by quality score
            sorted_resources = sorted(resources, key=lambda x: x.get('quality_score', 0), reverse=True)
            
            # Limit to top resources
            top_resources = sorted_resources[:5]  # Show top 5 resources
            
            # Group by category
            categories = {}
            for resource in top_resources:
                category = resource.get('category', 'Other')
                if category not in categories:
                    categories[category] = []
                categories[category].append(resource)
            
            # Build resources section
            resources_text = ["## 📚 Shared Resources\n"]
            
            # Add note if resources weren't filtered by channel
            if not channel_filtered:
                resources_text.append("*Note: Resources shown are from all channels for this time period*\n\n")
            
            for category, category_resources in categories.items():
                resources_text.append(f"### {category}")
                
                for resource in category_resources:
                    url = resource.get('url', '')
                    title = resource.get('description', '')[:100] + "..." if len(resource.get('description', '')) > 100 else resource.get('description', '')
                    author = resource.get('author', 'Unknown')
                    quality = resource.get('quality_score', 0)
                    channel = resource.get('channel_name', 'Unknown')
                    
                    # Format quality as stars
                    quality_stars = "⭐" * int(quality * 5) if quality > 0 else "⭐"
                    
                    resources_text.append(f"• **{title}**")
                    resources_text.append(f"  📍 {channel} | 👤 {author} | {quality_stars}")
                    resources_text.append(f"  🔗 {url}\n")
            
            return "\n".join(resources_text)
            
        except Exception as e:
            logger.error(f"Error formatting resources section: {e}")
            return "## 📚 Shared Resources\n*Error formatting resources*"
    
    async def process(self, state: AgentState) -> AgentState:
        """Process state through the digest agent."""
        args = state.get("agent_args", {})
        
        # Extract date parameters
        start = args.get("start")
        end = args.get("end")
        period = args.get("period", "all_time")
        
        # If no dates provided, use "all time" instead of defaulting to 24 hours
        if not start:
            start = None  # Will be handled in _get_all_messages to get all messages
        if not end:
            end = None  # Will be handled in _get_all_messages to get all messages
        
        digest = await self.run(start=start, end=end, period=period)
        
        # Update state with digest
        state["digest_result"] = digest
        state["response"] = digest
        
        return state
    
    def can_handle(self, task) -> bool:
        """Check if this agent can handle the given task."""
        return (task.task_type == "digest" or 
                "summary" in task.description.lower() or
                "summarize" in task.description.lower() or
                "digest" in task.description.lower()) 