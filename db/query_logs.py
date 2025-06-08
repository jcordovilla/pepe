"""
Query and Response Logging System for PEPE Discord Bot

Provides comprehensive logging of user queries and agent responses with detailed metadata.
Supports both database storage and JSON file logging for analytics and debugging.
"""

from sqlalchemy import Column, Integer, String, Text, DateTime, JSON, Float, Boolean
from sqlalchemy.orm import declarative_base
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List, Union
import json
import logging
import os
from pathlib import Path
from db.db import Base, get_db_session, with_retry
from core.config import get_config

logger = logging.getLogger(__name__)

class QueryLog(Base):
    """Database model for storing user queries and agent responses."""
    __tablename__ = "query_logs"
    
    # Primary identification
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, index=True, nullable=True)  # For grouping related queries
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False, index=True)
    
    # User context
    user_id = Column(String, index=True, nullable=False)  # Discord user ID
    username = Column(String, nullable=False)  # Discord username
    guild_id = Column(String, index=True, nullable=True)  # Discord guild ID
    channel_id = Column(String, index=True, nullable=True)  # Discord channel ID
    channel_name = Column(String, nullable=True)  # Discord channel name
    
    # Query information
    query_text = Column(Text, nullable=False)  # User's original query
    query_length = Column(Integer, nullable=False)  # Query character count
    query_hash = Column(String, index=True, nullable=True)  # For deduplication
    
    # Agent analysis metadata
    query_type = Column(String, nullable=True)  # e.g., 'meta', 'resource_search', 'complex_query'
    routing_strategy = Column(String, nullable=True)  # e.g., 'data_status', 'hybrid_search', 'messages_only'
    confidence_score = Column(Float, nullable=True)  # Agent's confidence in routing decision
    extracted_keywords = Column(JSON, nullable=True)  # Keywords identified by agent
    reasoning = Column(Text, nullable=True)  # Agent's reasoning for strategy selection
    
    # Response information
    response_text = Column(Text, nullable=True)  # Agent's response
    response_length = Column(Integer, nullable=True)  # Response character count
    response_type = Column(String, nullable=True)  # Type of response (string, list, dict, etc.)
    response_status = Column(String, nullable=False, default='success')  # success, error, timeout
    
    # Performance metrics
    processing_time_ms = Column(Integer, nullable=True)  # Total processing time in milliseconds
    search_results_count = Column(Integer, nullable=True)  # Number of search results found
    faiss_search_time_ms = Column(Integer, nullable=True)  # FAISS search time
    llm_generation_time_ms = Column(Integer, nullable=True)  # LLM response generation time
    
    # Enhanced metadata
    temporal_references = Column(JSON, nullable=True)  # Time references extracted from query
    channel_references = Column(JSON, nullable=True)  # Channel references in query
    mention_references = Column(JSON, nullable=True)  # User mentions in query
    detected_intent = Column(String, nullable=True)  # Detected user intent
    language = Column(String, default='en', nullable=False)  # Query language
    
    # Search context
    search_scope = Column(String, nullable=True)  # e.g., 'global', 'channel_specific', 'timeframe'
    k_parameter = Column(Integer, nullable=True)  # Number of results requested
    filters_applied = Column(JSON, nullable=True)  # Search filters used
    index_used = Column(String, nullable=True)  # Which FAISS index was used
    
    # Response quality indicators
    user_feedback = Column(String, nullable=True)  # User feedback if available
    follow_up_query = Column(Boolean, default=False)  # Whether user asked follow-up
    error_message = Column(Text, nullable=True)  # Error details if any
    fallback_used = Column(Boolean, default=False)  # Whether fallback strategy was used
    
    # Analytical flags
    is_duplicate = Column(Boolean, default=False)  # Similar to recent query
    is_complex = Column(Boolean, default=False)  # Multi-part or complex query
    is_successful = Column(Boolean, default=True)  # Whether query was handled successfully
    requires_review = Column(Boolean, default=False)  # Flag for manual review


class QueryLogManager:
    """Manager class for handling query and response logging."""
    
    def __init__(self):
        self.config = get_config()
        self.json_log_dir = Path("logs/query_logs")
        self.json_log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create daily JSON log file
        today = datetime.now().strftime('%Y%m%d')
        self.json_log_file = self.json_log_dir / f"queries_{today}.jsonl"
        
    @with_retry(max_retries=3)
    def log_query_start(
        self,
        user_id: str,
        username: str,
        query_text: str,
        guild_id: Optional[str] = None,
        channel_id: Optional[str] = None,
        channel_name: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> int:
        """
        Log the start of a query processing session.
        Returns the query log ID for updating later.
        """
        try:
            query_hash = self._generate_query_hash(query_text)
            
            with get_db_session() as session:
                query_log = QueryLog(
                    session_id=session_id,
                    user_id=str(user_id),
                    username=username,
                    guild_id=str(guild_id) if guild_id else None,
                    channel_id=str(channel_id) if channel_id else None,
                    channel_name=channel_name,
                    query_text=query_text,
                    query_length=len(query_text),
                    query_hash=query_hash
                )
                
                session.add(query_log)
                session.flush()  # Get the ID without committing
                query_log_id = query_log.id
                session.commit()
                
                logger.info(f"Query logged with ID: {query_log_id}")
                return query_log_id
                
        except Exception as e:
            logger.error(f"Failed to log query start: {e}")
            return -1
    
    @with_retry(max_retries=3)
    def update_query_analysis(
        self,
        query_log_id: int,
        analysis: Dict[str, Any],
        processing_time_ms: Optional[int] = None
    ) -> bool:
        """Update query log with agent analysis results."""
        try:
            with get_db_session() as session:
                query_log = session.query(QueryLog).filter(QueryLog.id == query_log_id).first()
                if not query_log:
                    logger.warning(f"Query log ID {query_log_id} not found")
                    return False
                
                # Update with analysis data
                query_log.query_type = analysis.get('query_type')
                query_log.routing_strategy = analysis.get('strategy')
                query_log.confidence_score = analysis.get('confidence')
                query_log.extracted_keywords = analysis.get('keywords', [])
                query_log.reasoning = analysis.get('reasoning')
                query_log.detected_intent = analysis.get('intent')
                
                if processing_time_ms:
                    query_log.processing_time_ms = processing_time_ms
                
                session.commit()
                logger.debug(f"Updated query analysis for ID: {query_log_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to update query analysis: {e}")
            return False
    
    @with_retry(max_retries=3)
    def log_query_completion(
        self,
        query_log_id: int,
        response_text: Optional[str],
        response_type: str,
        response_status: str = 'success',
        processing_time_ms: Optional[int] = None,
        search_results_count: Optional[int] = None,
        performance_metrics: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None,
        fallback_used: bool = False
    ) -> bool:
        """Complete the query log with response information."""
        try:
            with get_db_session() as session:
                query_log = session.query(QueryLog).filter(QueryLog.id == query_log_id).first()
                if not query_log:
                    logger.warning(f"Query log ID {query_log_id} not found")
                    return False
                
                # Update response information
                query_log.response_text = response_text
                query_log.response_length = len(response_text) if response_text else 0
                query_log.response_type = response_type
                query_log.response_status = response_status
                query_log.error_message = error_message
                query_log.fallback_used = fallback_used
                query_log.is_successful = (response_status == 'success')
                
                if processing_time_ms:
                    query_log.processing_time_ms = processing_time_ms
                
                if search_results_count is not None:
                    query_log.search_results_count = search_results_count
                
                # Update performance metrics if provided
                if performance_metrics:
                    query_log.faiss_search_time_ms = performance_metrics.get('faiss_search_time_ms')
                    query_log.llm_generation_time_ms = performance_metrics.get('llm_generation_time_ms')
                    query_log.k_parameter = performance_metrics.get('k_parameter')
                    query_log.index_used = performance_metrics.get('index_used')
                    query_log.filters_applied = performance_metrics.get('filters_applied')
                
                session.commit()
                
                # Also log to JSON file for analytics
                self._log_to_json(query_log)
                
                logger.info(f"Query completion logged for ID: {query_log_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to log query completion: {e}")
            return False
    
    def _log_to_json(self, query_log: QueryLog):
        """Log query data to JSON file for analytics."""
        try:
            log_data = {
                'id': query_log.id,
                'timestamp': query_log.timestamp.isoformat() if query_log.timestamp else None,
                'user_id': query_log.user_id,
                'username': query_log.username,
                'guild_id': query_log.guild_id,
                'channel_id': query_log.channel_id,
                'channel_name': query_log.channel_name,
                'query_text': query_log.query_text,
                'query_length': query_log.query_length,
                'query_type': query_log.query_type,
                'routing_strategy': query_log.routing_strategy,
                'confidence_score': query_log.confidence_score,
                'response_length': query_log.response_length,
                'response_status': query_log.response_status,
                'processing_time_ms': query_log.processing_time_ms,
                'search_results_count': query_log.search_results_count,
                'is_successful': query_log.is_successful,
                'fallback_used': query_log.fallback_used
            }
            
            with open(self.json_log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_data, ensure_ascii=False) + '\n')
                
        except Exception as e:
            logger.error(f"Failed to write JSON log: {e}")
    
    def _generate_query_hash(self, query_text: str) -> str:
        """Generate a hash for the query to detect duplicates."""
        import hashlib
        # Normalize query text (lowercase, strip whitespace)
        normalized = query_text.lower().strip()
        return hashlib.md5(normalized.encode('utf-8')).hexdigest()
    
    def get_recent_queries(
        self,
        user_id: Optional[str] = None,
        hours: int = 24,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get recent queries for analysis."""
        try:
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
            
            with get_db_session() as session:
                query = session.query(QueryLog).filter(
                    QueryLog.timestamp >= cutoff_time
                ).order_by(QueryLog.timestamp.desc()).limit(limit)
                
                if user_id:
                    query = query.filter(QueryLog.user_id == str(user_id))
                
                results = []
                for log in query.all():
                    results.append({
                        'id': log.id,
                        'timestamp': log.timestamp.isoformat(),
                        'user_id': log.user_id,
                        'username': log.username,
                        'query_text': log.query_text,
                        'query_type': log.query_type,
                        'routing_strategy': log.routing_strategy,
                        'confidence_score': log.confidence_score,
                        'response_status': log.response_status,
                        'processing_time_ms': log.processing_time_ms,
                        'is_successful': log.is_successful
                    })
                
                return results
                
        except Exception as e:
            logger.error(f"Failed to get recent queries: {e}")
            return []
    
    def get_query_analytics(self, days: int = 7) -> Dict[str, Any]:
        """Get analytics about query patterns and performance."""
        try:
            cutoff_time = datetime.now(timezone.utc) - timedelta(days=days)
            
            with get_db_session() as session:
                # Basic statistics
                total_queries = session.query(QueryLog).filter(
                    QueryLog.timestamp >= cutoff_time
                ).count()
                
                successful_queries = session.query(QueryLog).filter(
                    QueryLog.timestamp >= cutoff_time,
                    QueryLog.is_successful == True
                ).count()
                
                # Strategy distribution - fixed query
                from sqlalchemy import func
                strategy_results = session.query(
                    QueryLog.routing_strategy,
                    func.count(QueryLog.id).label('count')
                ).filter(
                    QueryLog.timestamp >= cutoff_time,
                    QueryLog.routing_strategy.isnot(None)
                ).group_by(QueryLog.routing_strategy).all()
                
                strategy_distribution = {result[0]: result[1] for result in strategy_results if result[0]}
                
                # Average processing time
                avg_processing_time = session.query(QueryLog.processing_time_ms).filter(
                    QueryLog.timestamp >= cutoff_time,
                    QueryLog.processing_time_ms.isnot(None)
                ).all()
                
                avg_time = sum(t[0] for t in avg_processing_time) / len(avg_processing_time) if avg_processing_time else 0
                
                return {
                    'period_days': days,
                    'total_queries': total_queries,
                    'successful_queries': successful_queries,
                    'success_rate': successful_queries / total_queries if total_queries > 0 else 0,
                    'strategy_distribution': strategy_distribution,
                    'average_processing_time_ms': round(avg_time, 2),
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }
                
        except Exception as e:
            logger.error(f"Failed to get query analytics: {e}")
            return {}


# Global instance for easy import
query_log_manager = QueryLogManager()

# Convenience functions
def log_query_start(*args, **kwargs) -> int:
    """Convenience function to log query start."""
    return query_log_manager.log_query_start(*args, **kwargs)

def update_query_analysis(*args, **kwargs) -> bool:
    """Convenience function to update query analysis."""
    return query_log_manager.update_query_analysis(*args, **kwargs)

def log_query_completion(*args, **kwargs) -> bool:
    """Convenience function to log query completion."""
    return query_log_manager.log_query_completion(*args, **kwargs)

def get_recent_queries(*args, **kwargs) -> List[Dict[str, Any]]:
    """Convenience function to get recent queries."""
    return query_log_manager.get_recent_queries(*args, **kwargs)

def get_query_analytics(*args, **kwargs) -> Dict[str, Any]:
    """Convenience function to get query analytics."""
    return query_log_manager.get_query_analytics(*args, **kwargs)

# Add a simple text log for easy searching
def log_simple_query(
    user_id: str,
    username: str,
    query_text: str,
    response_text: str,
    interface: str = "discord",  # "discord" or "streamlit"
    guild_id: Optional[str] = None,
    channel_name: Optional[str] = None
):
    """
    Log query and response to a simple text file for easy searching.
    This creates human-readable logs in addition to the database logging.
    """
    try:
        # Create simple logs directory
        simple_log_dir = Path("logs/simple_logs")
        simple_log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create daily simple log file
        today = datetime.now().strftime('%Y%m%d')
        simple_log_file = simple_log_dir / f"queries_simple_{today}.txt"
        
        # Format the log entry
        timestamp = datetime.now().isoformat()
        log_entry = f"""
=== QUERY LOG ENTRY ===
Timestamp: {timestamp}
Interface: {interface.upper()}
User: {username} (ID: {user_id})
Guild ID: {guild_id or 'N/A'}
Channel: {channel_name or 'N/A'}

QUERY:
{query_text}

RESPONSE:
{response_text}

{'='*50}

"""
        
        # Append to the log file
        with open(simple_log_file, 'a', encoding='utf-8') as f:
            f.write(log_entry)
            
        logger.debug(f"Simple log entry written to {simple_log_file}")
        
    except Exception as e:
        logger.error(f"Failed to write simple log: {e}")
