"""
Enhanced agent system for Discord bot queries with 768D embedding architecture.

Features:
- Intelligent query routing based on query type and keywords
- Hybrid search combining Discord messages and curated resources
- Resource-specific search for documentation and guides
- Channel-scoped search capabilities
- Temporal summary generation
- Compatible with msmarco-distilbert-base-v4 (768D) embedding model
- Enhanced semantic search with upgraded RAG engine
- Query analysis and debugging capabilities

Architecture:
- Uses upgraded 768D embedding model for better semantic understanding
- Integrates with enhanced RAG engine supporting multiple search strategies
- Fallback mechanisms ensure robust query handling
- Comprehensive error handling and logging
"""
import logging
from typing import Any, Dict, Optional
from core.ai_client import get_ai_client
from core.rag_engine import (
    get_answer, 
    get_agent_answer as rag_get_agent_answer,
    get_resource_answer,
    get_hybrid_answer
)
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
    
    # Resource-specific queries (documentation, guides, links, tutorials)
    if any(kw in query_lower for kw in [
        "documentation", "docs", "guide", "tutorial", "resource", "link", "article",
        "official", "reference", "manual", "how to", "setup", "install", "configure"
    ]):
        try:
            return get_resource_answer(query, k=5, return_matches=False)
        except Exception as e:
            logger.error(f"Resource query failed: {e}")
            # Fallback to regular search
            try:
                return get_answer(query, k=5, return_matches=False)
            except Exception as fallback_e:
                logger.error(f"Fallback search also failed: {fallback_e}")
                return "❌ Error searching resources and messages. Please try rephrasing your query."
    
    # Complex queries that benefit from hybrid search (both messages and resources)
    if any(kw in query_lower for kw in [
        "best practices", "recommend", "comparison", "vs", "difference", "which",
        "should i", "how do i", "getting started", "beginner", "advanced",
        "problem", "issue", "troubleshoot", "error", "fix", "solution"
    ]):
        try:
            return get_hybrid_answer(query, k_messages=3, k_resources=3, return_matches=False)
        except Exception as e:
            logger.error(f"Hybrid query failed: {e}")
            # Fallback to regular search
            try:
                return get_answer(query, k=5, return_matches=False)
            except Exception as fallback_e:
                logger.error(f"Fallback search also failed: {fallback_e}")
                return "❌ Error performing hybrid search. Please try rephrasing your query."
    
    # Time-based summary queries
    if any(kw in query_lower for kw in ["summary", "summarize", "what happened", "activity"]):
        try:
            return rag_get_agent_answer(query)
        except Exception as e:
            logger.error(f"Summary query failed: {e}")
            return "❌ Error generating summary."
    
    # Default: semantic search with enhanced context
    try:
        return get_answer(query, k=5, return_matches=False)
    except Exception as e:
        logger.error(f"Search query failed: {e}")
        return "❌ Error searching messages. Please try rephrasing your query."

def process_query(query: str, channel_id: int = None, enable_hybrid: bool = True) -> str:
    """
    Process a query with optional channel scoping and hybrid search capabilities.
    
    Args:
        query: User query
        channel_id: Optional channel ID to scope search
        enable_hybrid: Whether to use hybrid search for complex queries
        
    Returns:
        Response string
    """
    try:
        # If channel_id provided, scope the search
        if channel_id:
            # For channel-scoped queries, check if hybrid search would be beneficial
            if enable_hybrid and any(kw in query.lower() for kw in [
                "best practices", "documentation", "guide", "tutorial", "how to",
                "recommend", "solution", "troubleshoot", "getting started"
            ]):
                try:
                    return get_hybrid_answer(
                        query, 
                        k_messages=3, 
                        k_resources=2, 
                        channel_id=channel_id,
                        return_matches=False
                    )
                except Exception as e:
                    logger.error(f"Channel-scoped hybrid query failed: {e}")
                    # Fallback to regular channel search
            
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

def analyze_query_type(query: str) -> Dict[str, Any]:
    """
    Analyze query to determine the best search strategy.
    Useful for debugging and transparency.
    
    Args:
        query: User query string
        
    Returns:
        Dictionary with query analysis results
    """
    if not query or not query.strip():
        return {
            "query_type": "invalid",
            "strategy": "none",
            "confidence": 0.0,
            "keywords": []
        }
    
    query_lower = query.lower().strip()
    analysis = {
        "query_type": "semantic_search",
        "strategy": "messages_only",
        "confidence": 0.5,
        "keywords": [],
        "reasoning": "Default semantic search"
    }
    
    # Data availability queries
    data_keywords = ["what data", "data available", "how many messages", "database status", "message count"]
    if any(kw in query_lower for kw in data_keywords):
        analysis.update({
            "query_type": "meta",
            "strategy": "data_status",
            "confidence": 0.9,
            "keywords": [kw for kw in data_keywords if kw in query_lower],
            "reasoning": "Query about data availability and status"
        })
        return analysis
    
    # Channel list queries
    channel_keywords = ["list channels", "show channels", "what channels", "available channels"]
    if any(kw in query_lower for kw in channel_keywords):
        analysis.update({
            "query_type": "meta",
            "strategy": "channel_list",
            "confidence": 0.9,
            "keywords": [kw for kw in channel_keywords if kw in query_lower],
            "reasoning": "Query about available channels"
        })
        return analysis
    
    # Resource-specific queries
    resource_keywords = ["documentation", "docs", "guide", "tutorial", "resource", "link", "article", "official", "reference"]
    resource_matches = [kw for kw in resource_keywords if kw in query_lower]
    if resource_matches:
        analysis.update({
            "query_type": "resource_search",
            "strategy": "resources_only",
            "confidence": 0.8,
            "keywords": resource_matches,
            "reasoning": "Query focused on documentation and resources"
        })
        return analysis
    
    # Hybrid search queries
    hybrid_keywords = ["best practices", "recommend", "comparison", "vs", "difference", "which", "should i", "how do i", "getting started", "problem", "troubleshoot"]
    hybrid_matches = [kw for kw in hybrid_keywords if kw in query_lower]
    if hybrid_matches:
        analysis.update({
            "query_type": "complex_query",
            "strategy": "hybrid_search",
            "confidence": 0.85,
            "keywords": hybrid_matches,
            "reasoning": "Complex query benefiting from both messages and resources"
        })
        return analysis
    
    # Summary queries
    summary_keywords = ["summary", "summarize", "what happened", "activity"]
    summary_matches = [kw for kw in summary_keywords if kw in query_lower]
    if summary_matches:
        analysis.update({
            "query_type": "temporal_summary",
            "strategy": "agent_summary",
            "confidence": 0.8,
            "keywords": summary_matches,
            "reasoning": "Time-based summary query"
        })
        return analysis
    
    # Enhanced confidence for semantic search based on query characteristics
    if len(query.split()) > 5:
        analysis["confidence"] = 0.7
        analysis["reasoning"] = "Detailed semantic search query"
    elif any(char in query for char in ["?", "how", "what", "why", "when", "where"]):
        analysis["confidence"] = 0.75
        analysis["reasoning"] = "Question-based semantic search"
    
    return analysis
