"""
Digest Agent

Map-reduce summarization agent for high-engagement messages.
Creates periodic digests and summaries of Discord activity.
"""

import logging
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
        
        # Digest configuration
        self.max_messages = config.get("max_messages", 50)
        self.max_summary_length = config.get("max_summary_length", 250)
        
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
        start_str = kwargs.get("start")
        end_str = kwargs.get("end")
        period = kwargs.get("period", "day")
        channel_id = kwargs.get("channel_id")
        channel = kwargs.get("channel")
        
        # Parse dates
        try:
            start_date = datetime.fromisoformat(start_str) if start_str else datetime.now() - timedelta(days=1)
            end_date = datetime.fromisoformat(end_str) if end_str else datetime.now()
        except ValueError as e:
            logger.error(f"Date parsing error: {e}")
            return f"Error: Invalid date format - {e}"
        
        logger.info(f"DigestAgent processing digest from {start_date} to {end_date} ({period})")
        
        try:
            # Get all messages in the period (optionally filter by channel_id or channel_name)
            messages = await self._get_all_messages(start_date, end_date, channel_id, channel)
            if not messages:
                return f"No messages found for the {period} period."
            
            # Generate individual summaries
            summaries = await self._generate_message_summaries(messages)
            
            # Combine summaries into final digest
            digest = await self._combine_summaries(summaries, period)
            
            return digest
            
        except Exception as e:
            logger.error(f"DigestAgent error: {e}")
            return f"Error generating digest: {str(e)}"
    
    async def _get_all_messages(self, start_date: datetime, end_date: datetime, channel_id: Optional[str] = None, channel: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all messages from the vector store for the period, optionally filtered by channel_id/forum_channel_id or channel_name."""
        try:
            # Use timestamp_unix (float) for ChromaDB filtering
            start_ts = start_date.timestamp() if start_date else None
            end_ts = end_date.timestamp() if end_date else None
            filters_and = []
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
            # If channel_id is provided but no timestamp, still use $and for channel/forum filter
            if channel_id and not (start_ts or end_ts):
                filters = {"$and": [
                    {"$or": [
                        {"channel_id": channel_id},
                        {"forum_channel_id": channel_id}
                    ]}
                ]}
            else:
                filters = {"$and": filters_and} if len(filters_and) > 1 else (filters_and[0] if filters_and else {})
            logger.info(f"[DigestAgent] FINAL filter_search filters: {filters}")
            results = await self.vector_store.filter_search(
                filters=filters,
                k=self.max_messages,
                sort_by="timestamp"
            )
            logger.info(f"[DigestAgent] Retrieved {len(results)} messages from vector store.")
            for i, result in enumerate(results[:3]):
                logger.info(f"[DigestAgent] Message {i+1}: channel_id={result.get('channel_id')}, forum_channel_id={result.get('forum_channel_id')}, content={result.get('content', '')[:60]}")
            messages = []
            for result in results:
                messages.append({
                    "content": result.get("content", ""),
                    "metadata": result,
                    "permalink": result.get("jump_url", ""),
                    "reactions": result.get("total_reactions", 0),
                    "author": result.get("author_username", "Unknown"),
                    "channel": result.get("channel_name", "Unknown")
                })
            return messages[:self.max_messages]
        except Exception as e:
            logger.error(f"Error getting messages: {e}")
            return []
    
    async def _get_high_engagement_messages(self, start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
        """Get high-engagement messages for trending only."""
        try:
            results = await self.vector_store.reaction_search(
                reaction="",
                k=self.max_messages,
                filters={
                    "timestamp": {
                        "$gte": start_date.isoformat(),
                        "$lte": end_date.isoformat()
                    }
                },
                sort_by="total_reactions"
            )
            high_engagement = []
            for result in results:
                reactions = result.get("metadata", {}).get("reactions", [])
                total_reactions = sum(reaction.get("count", 0) for reaction in reactions)
                if total_reactions >= 3:  # Only for trending
                    high_engagement.append({
                        "content": result.get("content", ""),
                        "metadata": result.get("metadata", {}),
                        "permalink": result.get("metadata", {}).get("permalink", ""),
                        "reactions": total_reactions,
                        "author": result.get("metadata", {}).get("author_name", "Unknown"),
                        "channel": result.get("metadata", {}).get("channel_name", "Unknown")
                    })
            return high_engagement[:self.max_messages]
        except Exception as e:
            logger.error(f"Error getting high-engagement messages: {e}")
            return []
    
    async def _generate_message_summaries(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate individual summaries for each message."""
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
        
        # Group summaries by theme
        grouped_summaries = await self._group_by_theme(summaries)
        
        # Generate final digest
        digest = await self._generate_final_digest(grouped_summaries, period)
        
        return digest
    
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
            period_display = period.capitalize()
            header = f"# 📋 {period_display} Digest\n\n"
            
            return header + digest.strip()
            
        except Exception as e:
            logger.error(f"Error generating final digest: {e}")
            return f"# 📋 {period.capitalize()} Digest\n\nError generating digest: {str(e)}"
    
    async def process(self, state: AgentState) -> AgentState:
        """Process state through the digest agent."""
        args = state.get("agent_args", {})
        
        # Extract date parameters
        start = args.get("start")
        end = args.get("end")
        period = args.get("period", "day")
        
        # If no dates provided, use defaults
        if not start:
            start = (datetime.now() - timedelta(days=1)).isoformat()
        if not end:
            end = datetime.now().isoformat()
        
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