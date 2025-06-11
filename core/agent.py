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
import re
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
    resolve_channel_name,
    get_buddy_group_analysis
)

logger = logging.getLogger(__name__)

def _determine_optimal_k(
    query: str, 
    channel_id: Optional[int] = None, 
    channel_name: Optional[str] = None, 
    guild_id: Optional[int] = None
) -> int:
    """
    Determine optimal k parameter using enhanced database-driven analysis.
    
    Args:
        query: User query string
        channel_id: Optional channel ID for context
        channel_name: Optional channel name for context
        guild_id: Optional guild ID for context
        
    Returns:
        Optimal k value for retrieval
    """
    try:
        from .enhanced_k_determination import determine_optimal_k
        return determine_optimal_k(query, channel_id, channel_name, guild_id, use_enhanced_analysis=True)
    except Exception as e:
        logger.warning(f"Enhanced k determination failed, using fallback: {e}")
        return _determine_optimal_k_fallback(query)


def _determine_optimal_k_fallback(query: str) -> int:
    """
    Fallback k determination using original logic.
    Used when enhanced analysis fails.
    """
    query_lower = query.lower().strip()
    
    # Comprehensive queries that need more context
    comprehensive_keywords = [
        "digest", "summary", "overview", "highlights", "weekly", "monthly", 
        "trending", "popular", "most", "best", "top", "all", "everything",
        "comprehensive", "complete", "full", "entire", "total", "overall"
    ]
    
    # Broad scope indicators
    broad_scope_keywords = [
        "community", "server", "discord", "channels", "everyone", "users",
        "discussions", "conversations", "activity", "what happened", "latest"
    ]
    
    # Specific/narrow queries
    specific_keywords = [
        "specific", "particular", "exact", "precise", "certain", "single",
        "one", "individual", "unique"
    ]
    
    # Question complexity indicators
    complex_indicators = [
        "compare", "analysis", "analyze", "explain", "detailed", "in-depth",
        "comprehensive", "thorough", "extensive"
    ]
    
    # Count matches
    comprehensive_matches = sum(1 for kw in comprehensive_keywords if kw in query_lower)
    broad_matches = sum(1 for kw in broad_scope_keywords if kw in query_lower)
    specific_matches = sum(1 for kw in specific_keywords if kw in query_lower)
    complex_matches = sum(1 for kw in complex_indicators if kw in query_lower)
    
    # Query length factor
    word_count = len(query.split())
    
    # Determine k based on analysis
    if comprehensive_matches >= 2 or "weekly digest" in query_lower or "monthly digest" in query_lower:
        # High-level digest queries need extensive context
        return 30
    elif comprehensive_matches >= 1 or broad_matches >= 2:
        # Broad queries need good coverage
        return 20
    elif complex_matches >= 2 or word_count >= 8:
        # Complex analytical queries need substantial context
        return 15
    elif broad_matches >= 1 or word_count >= 5:
        # Medium scope queries
        return 10
    elif specific_matches >= 1:
        # Very targeted queries can use fewer results
        return 3
    else:
        # Default for simple queries
        return 5

def preprocess_discord_mentions(query: str) -> tuple[str, Optional[int]]:
    """
    Preprocess a query to extract Discord channel mentions and convert them to usable format.
    
    Args:
        query: Raw query string potentially containing Discord mentions
        
    Returns:
        tuple: (cleaned_query, channel_id)
            - cleaned_query: Query with Discord mentions replaced with channel names
            - channel_id: Extracted channel ID if found, None otherwise
    """
    import re
    
    # Pattern to match Discord channel mentions
    discord_mention_pattern = r'<#(\d+)>'
    matches = re.findall(discord_mention_pattern, query)
    
    if not matches:
        return query, None
    
    # Take the first channel mention found
    channel_id = int(matches[0])
    
    # Try to resolve channel ID to name for better query readability
    try:
        from db.db import SessionLocal, Message
        session = SessionLocal()
        try:
            result = session.query(Message.channel_name).filter(Message.channel_id == channel_id).first()
            if result:
                channel_name = result[0]
                # Replace Discord mention with readable channel name
                cleaned_query = re.sub(discord_mention_pattern, f"#{channel_name}", query)
                logger.info(f"Converted Discord mention <#{channel_id}> to #{channel_name}")
                return cleaned_query, channel_id
        finally:
            session.close()
    except Exception as e:
        logger.warning(f"Failed to resolve channel ID {channel_id} to name: {e}")
    
    # Fallback: remove the Discord mention entirely
    cleaned_query = re.sub(discord_mention_pattern, "", query).strip()
    logger.info(f"Removed Discord mention <#{channel_id}> from query")
    return cleaned_query, channel_id


def get_agent_answer(query: str) -> str:
    """
    Simplified agent that routes queries to appropriate handlers.
    
    Args:
        query: User query string
        
    Returns:
        Response string
        
    Raises:
        ValueError: If query is empty or invalid
    """
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")
    
    query = query.strip()
    
    # Preprocess Discord mentions before any other processing
    processed_query, extracted_channel_id = preprocess_discord_mentions(query)
    if extracted_channel_id:
        logger.info(f"Extracted channel ID {extracted_channel_id} from Discord mention")
    
    # Use processed query for all subsequent analysis
    query_lower = processed_query.lower().strip()
    
    # Check for ambiguous queries that need clarification
    ambiguous_patterns = [
        r'^tell me something interesting\.?$',
        r'^something interesting\.?$',
        r'^what.*interesting\??$',
        r'^show me.*interesting\??$',
        r'^give me.*interesting\??$'
    ]
    
    if any(re.match(pattern, query_lower) for pattern in ambiguous_patterns):
        return "I'd be happy to help! Could you please specify which channel, timeframe, or keyword you're interested in? For example: 'Tell me something interesting from #general-chat last week' or 'Show me interesting discussions about AI'."
    
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
            k = min(_determine_optimal_k(processed_query), 10)  # Cap at 10 for resources
            return get_resource_answer(processed_query, k=k, return_matches=False)
        except Exception as e:
            logger.error(f"Resource query failed: {e}")
            # Fallback to regular search
            try:
                k = _determine_optimal_k(processed_query)
                return get_answer(processed_query, k=k, return_matches=False)
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
            k = _determine_optimal_k(processed_query)
            k_messages = min(k // 2, 10)  # Split k between messages and resources
            k_resources = min(k - k_messages, 5)  # Fewer resources needed
            return get_hybrid_answer(processed_query, k_messages=k_messages, k_resources=k_resources, return_matches=False)
        except Exception as e:
            logger.error(f"Hybrid query failed: {e}")
            # Fallback to regular search
            try:
                return get_answer(query, k=5, return_matches=False)
            except Exception as fallback_e:
                logger.error(f"Fallback search also failed: {fallback_e}")
                return "❌ Error performing hybrid search. Please try rephrasing your query."
    
    # Enhanced analysis queries (buddy groups, specific channel patterns, etc.)
    analysis_patterns = [
        "analyze", "analysis", "patterns", "compare", "comparison", "activity patterns",
        "engagement patterns", "distribution", "breakdown", "statistics"
    ]
    
    if any(pattern in query_lower for pattern in analysis_patterns):
        # Check for specific channel groups or patterns
        if "buddy group" in query_lower or "buddy-group" in query_lower:
            try:
                return get_buddy_group_analysis(processed_query)
            except Exception as e:
                logger.error(f"Buddy group analysis failed: {e}")
                # Fallback to regular analysis
                return rag_get_agent_answer(processed_query)
        
        # Check for specific channel analysis
        elif "#" in processed_query or "channel" in query_lower or extracted_channel_id:
            try:
                # Use processed query and pass channel ID if available
                if extracted_channel_id:
                    # For channel-specific queries, use get_answer with channel_id parameter
                    k = _determine_optimal_k(processed_query)
                    return get_answer(processed_query, k=k, channel_id=extracted_channel_id, return_matches=False)
                else:
                    return rag_get_agent_answer(processed_query)
            except Exception as e:
                logger.error(f"Channel analysis failed: {e}")
                return "❌ Error performing channel analysis."

    # Time-based summary queries
    if any(kw in query_lower for kw in ["summary", "summarize", "digest", "what happened", "activity", "highlights", "highlight"]):
        try:
            return rag_get_agent_answer(processed_query)
        except ValueError as ve:
            # Check for specific error types
            error_msg = str(ve).lower()
            if "end time must be after start time" in error_msg:
                return "End time must be after start time"
            elif "unknown channel" in error_msg:
                return "Unknown channel"
            else:
                logger.error(f"Summary query failed: {ve}")
                return "❌ Error generating summary."
        except Exception as e:
            logger.error(f"Summary query failed: {e}")
            return "❌ Error generating summary."
    
    # Default: semantic search with enhanced context
    try:
        # Use adaptive k parameter based on query scope
        k = _determine_optimal_k(processed_query)
        logger.info(f"Using adaptive k={k} for query: {processed_query}")
        return get_answer(processed_query, k=k, return_matches=False)
    except ValueError as ve:
        # Check for specific error types
        error_msg = str(ve).lower()
        if "end time must be after start time" in error_msg:
            return "End time must be after start time"
        elif "unknown channel" in error_msg:
            return "Unknown channel"
        else:
            logger.error(f"Search query failed: {ve}")
            # Use enhanced fallback system for better error handling
            try:
                from core.enhanced_fallback_system import EnhancedFallbackSystem
                fallback_system = EnhancedFallbackSystem()
                
                # Determine capability for better fallback
                capability = "general"
                if any(kw in query_lower for kw in ["analyze", "analysis", "patterns", "data"]):
                    capability = "server_data_analysis"
                elif any(kw in query_lower for kw in ["feedback", "summary", "summarize"]):
                    capability = "feedback_summarization"
                elif any(kw in query_lower for kw in ["trending", "popular", "topics"]):
                    capability = "trending_topics"
                elif any(kw in query_lower for kw in ["question", "answer", "qa"]):
                    capability = "qa_concepts"
                elif any(kw in query_lower for kw in ["statistics", "stats", "metrics"]):
                    capability = "statistics_generation"
                elif any(kw in query_lower for kw in ["channel", "structure", "organization"]):
                    capability = "server_structure_analysis"
                
                fallback_response = fallback_system.generate_intelligent_fallback(
                    query=query,
                    capability=capability
                )
                
                return fallback_response["response"]
                
            except Exception as fallback_error:
                logger.error(f"Fallback system failed: {fallback_error}")
                return "❌ Error searching messages. Please try rephrasing your query."
    except Exception as e:
        logger.error(f"Search query failed: {e}")
        # Use enhanced fallback for unexpected errors too
        try:
            from core.enhanced_fallback_system import EnhancedFallbackSystem
            fallback_system = EnhancedFallbackSystem()
            
            fallback_response = fallback_system.generate_intelligent_fallback(
                query=query,
                capability="general"
            )
            
            return fallback_response["response"]
            
        except Exception as fallback_error:
            logger.error(f"Fallback system failed: {fallback_error}")
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
    
    # Enhanced analysis queries (buddy groups, specific channel patterns, etc.)
    analysis_patterns = [
        "analyze", "analysis", "patterns", "compare", "comparison", "activity patterns",
        "engagement patterns", "distribution", "breakdown", "statistics"
    ]
    
    if any(pattern in query_lower for pattern in analysis_patterns):
        analysis["query_type"] = "analysis_query"
        analysis["strategy"] = "dynamic_analysis"
        analysis["confidence"] = 0.9
        analysis["keywords"] = [pattern for pattern in analysis_patterns if pattern in query_lower]
        analysis["reasoning"] = "Query requesting detailed analysis or comparison"
        return analysis

    # Enhanced confidence for semantic search based on query characteristics
    if len(query.split()) > 5:
        analysis["confidence"] = 0.7
        analysis["reasoning"] = "Detailed semantic search query"
    elif any(char in query for char in ["?", "how", "what", "why", "when", "where"]):
        analysis["confidence"] = 0.75
        analysis["reasoning"] = "Question-based semantic search"
    
    return analysis
