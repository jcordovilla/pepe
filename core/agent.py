"""
Simplified agent system for Discord bot queries.
Removes LangChain complexity while maintaining functionality.
"""
import logging
from typing import Any
from core.ai_client import get_ai_client
from core.rag_engine import get_answer, get_agent_answer as rag_get_agent_answer
from tools.tools import (
    search_messages,
    summarize_messages,
    validate_data_availability,
    get_channels,
    resolve_channel_name
)

logger = logging.getLogger(__name__)

def get_agent_answer(query: str) -> str:
    """
    Simplified agent that routes queries to appropriate handlers.
    
    Args:
        query: User query string
        
    Returns:
        Response string
    """
    if not query.strip():
        return "Please provide a valid query."
    
    query_lower = query.lower().strip()
    
    # Data availability queries
    if any(kw in query_lower for kw in [
        "what data", "data available", "how many messages", 
        "database status", "message count", "how much data"
    ]):
        try:
            data = validate_data_availability()
            if data.get("status") == "ok":
                channels = ", ".join([f"{name} ({count})" for name, count in data["channels"].items()])
                return (
                    f"📊 **Data Status**: {data['count']} messages across {len(data['channels'])} channels\n"
                    f"📅 **Date Range**: {data['date_range']['oldest']} to {data['date_range']['newest']}\n"
                    f"📋 **Channels**: {channels}"
                )
            else:
                return f"⚠️ {data.get('message', 'Unable to check data availability')}"
        except Exception as e:
            logger.error(f"Data availability check failed: {e}")
            return "❌ Unable to check data availability at this time."
    
    # Channel list queries
    if any(kw in query_lower for kw in ["list channels", "show channels", "what channels", "available channels"]):
        try:
            channels = get_channels()
            if channels:
                channel_list = "\n".join([f"- **{ch['name']}** (ID: {ch['id']}, Messages: {ch.get('message_count', 'N/A')})" 
                                        for ch in channels[:20]])
                return f"📋 **Available Channels**:\n{channel_list}"
            else:
                return "No channels found."
        except Exception as e:
            logger.error(f"Error getting channels: {e}")
            return "❌ Error retrieving channel list."
    
    # Time-based summary queries
    if any(kw in query_lower for kw in ["summary", "summarize", "what happened", "activity"]):
        try:
            return rag_get_agent_answer(query)
        except Exception as e:
            logger.error(f"Summary query failed: {e}")
            return "❌ Error generating summary."
    
    # Default: semantic search
    try:
        return get_answer(query, k=5, return_matches=False)
    except Exception as e:
        logger.error(f"Search query failed: {e}")
        return "❌ Error searching messages. Please try rephrasing your query."

def process_query(query: str, channel_id: int = None) -> str:
    """
    Process a query with optional channel scoping.
    
    Args:
        query: User query
        channel_id: Optional channel ID to scope search
        
    Returns:
        Response string
    """
    try:
        # If channel_id provided, scope the search
        if channel_id:
            return get_answer(query, k=5, channel_id=channel_id, return_matches=False)
        else:
            return get_agent_answer(query)
    except Exception as e:
        logger.error(f"Query processing failed: {e}")
        return f"❌ Error processing query: {str(e)}"

# Legacy compatibility function for existing imports
def execute_agent_query(query: str) -> str:
    """Legacy wrapper for backward compatibility."""
    return get_agent_answer(query)
