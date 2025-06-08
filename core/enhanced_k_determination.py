"""
Enhanced K Parameter Determination System

This module provides intelligent k parameter calculation for the Discord bot's RAG system
that leverages the rich preprocessed database data including temporal patterns, 
community metadata, and content analysis to make data-driven k sizing decisions.

Features:
- Temporal query analysis using actual message counts and activity patterns
- Database-driven k calculation based on available data volume
- Community engagement metrics integration for better k sizing
- Enhanced content analysis using preprocessed fields (topics, keywords, sentiment)
- Fallback mechanisms for when preprocessed data is unavailable
- Intelligent scaling based on query complexity and scope
"""

import os
import sys
import re
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
from sqlalchemy import func, and_, or_
from sqlalchemy.orm import Session

# Add project root to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from db import SessionLocal, Message, get_db_session
from tools.statistical_analyzer import DiscordStatisticalAnalyzer

logger = logging.getLogger(__name__)

class EnhancedKDetermination:
    """
    Enhanced k parameter determination system that leverages rich database metadata
    and temporal analysis for intelligent retrieval sizing with context window awareness.
    """
    
    def __init__(self):
        self.analyzer = DiscordStatisticalAnalyzer()
        # Context window configuration - now with 128K tokens!
        self.max_context_tokens = int(os.getenv('CHAT_MAX_TOKENS', '128000'))
        self.system_prompt_tokens = 200  # Estimated tokens for system prompt
        self.response_buffer_tokens = 1000  # Larger buffer for complex responses
        self.available_context_tokens = self.max_context_tokens - self.system_prompt_tokens - self.response_buffer_tokens
        
        # Token estimation ratios (approximate)
        self.words_per_token = 0.75  # English: ~1.33 tokens per word, so ~0.75 words per token
        self.metadata_tokens_per_message = 25  # Estimated tokens for author, timestamp, channel info per message
        
        # Query classification patterns
        self.temporal_patterns = self._build_temporal_patterns()
        self.complexity_indicators = self._build_complexity_indicators()
        self.scope_indicators = self._build_scope_indicators()
        
        logger.info(f"Enhanced K Determination initialized with LARGE context window: {self.max_context_tokens} tokens, "
                   f"available for context: {self.available_context_tokens} tokens - Monthly digests fully supported!")

    def _estimate_message_tokens(self, content_length: float) -> int:
        """Estimate tokens needed for a message including content and metadata."""
        content_tokens = int(content_length / self.words_per_token)
        return content_tokens + self.metadata_tokens_per_message

    def _estimate_total_tokens_for_k(self, k: int, avg_content_length: float) -> int:
        """Estimate total tokens needed for k messages."""
        avg_tokens_per_message = self._estimate_message_tokens(avg_content_length)
        return k * avg_tokens_per_message

    def _calculate_max_viable_k(self, avg_content_length: float) -> int:
        """Calculate maximum k that fits within context window."""
        if avg_content_length <= 0:
            avg_content_length = 50  # Default assumption: 50 words per message
        
        avg_tokens_per_message = self._estimate_message_tokens(avg_content_length)
        max_k = int(self.available_context_tokens / avg_tokens_per_message)
        
        # Ensure minimum viable k
        return max(max_k, 3)

    def _apply_context_window_constraints(self, k: int, db_stats: Dict[str, Any]) -> int:
        """Apply context window constraints - now much more lenient with 128K context!"""
        avg_content_length = db_stats.get('avg_content_length', 50)
        max_viable_k = self._calculate_max_viable_k(avg_content_length)
        
        # With 128K context window, we rarely need to constrain k for normal queries
        if k > max_viable_k:
            logger.info(f"Context window constraint: reducing k from {k} to {max_viable_k} "
                       f"(avg_content_length: {avg_content_length} words, "
                       f"estimated tokens per message: {self._estimate_message_tokens(avg_content_length)}) "
                       f"- Large context window allows much higher k values!")
            return max_viable_k
        
        # Ensure minimum viable k even if not constrained
        return max(k, 3)

    def _build_temporal_patterns(self) -> Dict[str, Dict]:
        """Build patterns for detecting temporal queries."""
        return {
            'recent': {
                'patterns': [
                    r'\brecent\b', r'\blast\s+\d+\s+hours?\b', r'\btoday\b', 
                    r'\byesterday\b', r'\blast\s+24\s+hours?\b', r'\bcurrent\b',
                    r'\blately\b', r'\bnowadays\b', r'\bpast\s+few\s+days\b', 
                    r'\blast\s+few\s+days\b'
                ],
                'base_days': 1,
                'min_k': 10
            },
            'daily': {
                'patterns': [
                    r'\bdaily\b', r'\beach\s+day\b', r'\bper\s+day\b',
                    r'\blast\s+\d+\s+days?\b', r'\bthis\s+week\b',
                    r'\bdaily\s+digest\b', r'\bpast\s+day\b', r'\bdaily\s+summary\b'
                ],
                'base_days': 3,
                'min_k': 15
            },
            'weekly': {
                'patterns': [
                    r'\bweekly\b', r'\beach\s+week\b', r'\bper\s+week\b',
                    r'\blast\s+week\b', r'\bthis\s+week\b', r'\bweek\s+digest\b',
                    r'\bweekly\s+digest\b', r'\bpast\s+week\b', r'\bweekly\s+summary\b'
                ],
                'base_days': 7,
                'min_k': 25
            },
            'monthly': {
                'patterns': [
                    r'\bmonthly\b', r'\beach\s+month\b', r'\bper\s+month\b',
                    r'\blast\s+month\b', r'\bthis\s+month\b', r'\bmonth\s+digest\b',
                    r'\bmonthly\s+digest\b', r'\bpast\s+month\b', r'\bmonthly\s+summary\b',
                    r'\bdigest\s+for\s+.*month\b', r'\bmonth.*summary\b', r'\bmonth.*overview\b'
                ],
                'base_days': 30,
                'min_k': 50,
                'high_volume_likely': True  # Flag for queries likely to hit context limits
            },
            'quarterly': {
                'patterns': [
                    r'\bquarterly\b', r'\beach\s+quarter\b', r'\bper\s+quarter\b',
                    r'\blast\s+quarter\b', r'\bthis\s+quarter\b', r'\bquarter\s+digest\b',
                    r'\blast\s+3\s+months?\b', r'\bquarterly\s+summary\b',
                    r'\bdigest\s+for\s+.*quarter\b', r'\bquarter.*summary\b'
                ],
                'base_days': 90,
                'min_k': 100,
                'high_volume_likely': True
            },
            'yearly': {
                'patterns': [
                    r'\byearly\b', r'\bannual\b', r'\beach\s+year\b', r'\bper\s+year\b',
                    r'\blast\s+year\b', r'\bthis\s+year\b', r'\byear\s+digest\b',
                    r'\blast\s+12\s+months?\b', r'\bannual\s+summary\b',
                    r'\bdigest\s+for\s+.*year\b', r'\byear.*summary\b', r'\byear.*overview\b'
                ],
                'base_days': 365,
                'min_k': 200,
                'high_volume_likely': True
            },
            'all_time': {
                'patterns': [
                    r'\ball\s+time\b', r'\bever\b', r'\bsince\s+beginning\b', r'\bserver\s+beginnings?\b',
                    r'\bentire\s+history\b', r'\bcomplete\s+history\b', r'\bfrom\s+start\b',
                    r'\bsince\s+inception\b', r'\beverything\s+ever\b', r'\btotal\s+summary\b',
                    r'\bfull\s+archive\b', r'\bcomprehensive\s+digest\b', r'\ball\s+messages\b',
                    r'\bentire\s+server\b', r'\bcomplete\s+archive\b'
                ],
                'base_days': 999999,  # Effectively unlimited
                'min_k': 500,
                'high_volume_likely': True
            }
        }
    
    def _build_complexity_indicators(self) -> Dict[str, List[str]]:
        """Build indicators for query complexity."""
        return {
            'high_complexity': [
                'analyze', 'analysis', 'compare', 'comparison', 'detailed', 'comprehensive',
                'in-depth', 'thorough', 'extensive', 'breakdown', 'explain', 'examine'
            ],
            'medium_complexity': [
                'overview', 'summary', 'highlights', 'trends', 'patterns', 'insights'
            ],
            'low_complexity': [
                'find', 'show', 'list', 'get', 'search', 'mention'
            ]
        }
    
    def _build_scope_indicators(self) -> Dict[str, List[str]]:
        """Build indicators for query scope."""
        return {
            'broad_scope': [
                'all', 'everything', 'entire', 'complete', 'full', 'total',
                'community', 'server', 'discord', 'channels', 'everyone', 'users',
                'discussions', 'conversations', 'activity', 'what happened', 'latest'
            ],
            'medium_scope': [
                'main', 'key', 'important', 'significant', 'notable', 'major',
                'popular', 'trending', 'top', 'best', 'most'
            ],
            'narrow_scope': [
                'specific', 'particular', 'exact', 'precise', 'certain', 'single',
                'one', 'individual', 'unique', 'this', 'that'
            ]
        }
    
    def determine_optimal_k(
        self,
        query: str,
        channel_id: Optional[int] = None,
        channel_name: Optional[str] = None,
        guild_id: Optional[int] = None,
        use_enhanced_analysis: bool = True
    ) -> int:
        """
        Determine optimal k parameter using enhanced database-driven analysis.
        
        Args:
            query: User query string
            channel_id: Optional channel ID for context
            channel_name: Optional channel name for context
            guild_id: Optional guild ID for context
            use_enhanced_analysis: Whether to use enhanced analysis (fallback to basic if False)
            
        Returns:
            Optimal k value for retrieval
        """
        try:
            if not query or not query.strip():
                return 5  # Default for empty queries
            
            if use_enhanced_analysis:
                return self._enhanced_k_determination(query, channel_id, channel_name, guild_id)
            else:
                return self._basic_k_determination(query)
                
        except Exception as e:
            logger.warning(f"Error in k determination, falling back to basic: {e}")
            return self._basic_k_determination(query)
    
    def _enhanced_k_determination(
        self,
        query: str,
        channel_id: Optional[int] = None,
        channel_name: Optional[str] = None,
        guild_id: Optional[int] = None
    ) -> int:
        """
        Enhanced k determination using database analysis and temporal patterns.
        """
        query_lower = query.lower().strip()
        
        # 1. Detect temporal query patterns
        temporal_info = self._analyze_temporal_query(query_lower)
        
        # 2. Get database statistics for context
        db_stats = self._get_database_context_stats(channel_id, channel_name, guild_id, temporal_info)
        
        # 3. Analyze query complexity and scope
        complexity_score = self._calculate_complexity_score(query_lower)
        scope_score = self._calculate_scope_score(query_lower)
        
        # 4. Calculate base k using temporal analysis
        if temporal_info['is_temporal']:
            base_k = self._calculate_temporal_k(temporal_info, db_stats)
        else:
            base_k = self._calculate_standard_k(complexity_score, scope_score)
        
        # 5. Apply database-driven adjustments
        adjusted_k = self._apply_database_adjustments(base_k, db_stats, complexity_score, scope_score)
        
        # 6. Apply final bounds and validation
        final_k = self._apply_bounds_and_validation(adjusted_k, temporal_info)
        
        # 7. Apply context window constraints to prevent token overflow
        context_constrained_k = self._apply_context_window_constraints(final_k, db_stats)
        
        # Calculate estimated token usage for logging
        estimated_tokens = self._estimate_total_tokens_for_k(context_constrained_k, db_stats.get('avg_content_length', 50))
        
        logger.info(f"Enhanced k determination: query='{query[:50]}...', temporal={temporal_info['is_temporal']}, "
                   f"base_k={base_k}, adjusted_k={adjusted_k}, final_k={final_k}, "
                   f"context_constrained_k={context_constrained_k}, "
                   f"estimated_tokens={estimated_tokens}/{self.available_context_tokens}")
        
        return context_constrained_k
    
    def _analyze_temporal_query(self, query_lower: str) -> Dict[str, Any]:
        """Analyze query for temporal patterns."""
        temporal_info = {
            'is_temporal': False,
            'temporal_type': None,
            'time_period_days': None,
            'confidence': 0.0
        }
        
        for temporal_type, config in self.temporal_patterns.items():
            for pattern in config['patterns']:
                if re.search(pattern, query_lower):
                    temporal_info.update({
                        'is_temporal': True,
                        'temporal_type': temporal_type,
                        'time_period_days': config['base_days'],
                        'confidence': 0.9,
                        'min_k': config['min_k']
                    })
                    break
            if temporal_info['is_temporal']:
                break
        
        return temporal_info
    
    def _get_database_context_stats(
        self,
        channel_id: Optional[int],
        channel_name: Optional[str],
        guild_id: Optional[int],
        temporal_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get relevant database statistics for context."""
        try:
            with get_db_session() as session:
                # Build base query
                query = session.query(Message)
                
                if guild_id:
                    query = query.filter(Message.guild_id == guild_id)
                if channel_id:
                    query = query.filter(Message.channel_id == channel_id)
                elif channel_name:
                    query = query.filter(Message.channel_name.ilike(f"%{channel_name}%"))
                
                # Apply temporal filter if relevant
                if temporal_info.get('is_temporal') and temporal_info.get('time_period_days'):
                    cutoff_date = datetime.utcnow() - timedelta(days=temporal_info['time_period_days'])
                    query = query.filter(Message.timestamp >= cutoff_date)
                else:
                    # For non-temporal queries, look at recent activity (30 days)
                    cutoff_date = datetime.utcnow() - timedelta(days=30)
                    query = query.filter(Message.timestamp >= cutoff_date)
                
                # Get message counts and engagement stats
                total_messages = query.count()
                
                messages_sample = query.limit(1000).all()  # Sample for detailed analysis
                
                stats = {
                    'total_messages': total_messages,
                    'has_enhanced_content': 0,
                    'has_topics': 0,
                    'has_keywords': 0,
                    'engagement_messages': 0,
                    'technical_messages': 0,
                    'avg_content_length': 0.0,
                    'unique_authors': 0,  # Initialize as integer, not set
                    'channel_diversity': 0,
                    'density_score': 0.0
                }
                
                if messages_sample:
                    content_lengths = []
                    unique_authors_set = set()  # Use separate variable for set operations
                    
                    for msg in messages_sample:
                        # Analyze enhanced fields
                        if msg.enhanced_content:
                            stats['has_enhanced_content'] += 1
                        if msg.topics:
                            stats['has_topics'] += 1
                        if msg.keywords:
                            stats['has_keywords'] += 1
                        if msg.reactions or msg.reference:
                            stats['engagement_messages'] += 1
                        if msg.mentioned_technologies:
                            stats['technical_messages'] += 1
                        
                        # Content analysis - calculate word count not character count
                        content = msg.content or ''
                        word_count = len(content.split()) if content.strip() else 0
                        content_lengths.append(word_count)
                        if msg.author:
                            unique_authors_set.add(msg.author.get('username', 'unknown'))
                    
                    stats['avg_content_length'] = sum(content_lengths) / len(content_lengths) if content_lengths else 0.0
                    stats['unique_authors'] = len(unique_authors_set)  # Convert set to count
                    
                    # Calculate density score (messages per day)
                    if temporal_info.get('time_period_days'):
                        stats['density_score'] = total_messages / max(temporal_info['time_period_days'], 1)
                    else:
                        stats['density_score'] = total_messages / 30.0
                
                return stats
                
        except Exception as e:
            logger.warning(f"Error getting database stats: {e}")
            return {
                'total_messages': 0,
                'has_enhanced_content': 0,
                'has_topics': 0,
                'has_keywords': 0,
                'engagement_messages': 0,
                'technical_messages': 0,
                'avg_content_length': 0.0,
                'unique_authors': 0,
                'channel_diversity': 0,
                'density_score': 0.0
            }
    
    def _calculate_complexity_score(self, query_lower: str) -> float:
        """Calculate query complexity score (0.0 to 1.0)."""
        high_matches = sum(1 for indicator in self.complexity_indicators['high_complexity'] 
                          if indicator in query_lower)
        medium_matches = sum(1 for indicator in self.complexity_indicators['medium_complexity'] 
                            if indicator in query_lower)
        low_matches = sum(1 for indicator in self.complexity_indicators['low_complexity'] 
                         if indicator in query_lower)
        
        # Weight scores
        total_score = (high_matches * 1.0) + (medium_matches * 0.6) + (low_matches * 0.2)
        word_count = len(query_lower.split())
        
        # Factor in query length
        length_factor = min(word_count / 10.0, 1.0)  # Normalize to 0-1
        
        return min(total_score + length_factor, 1.0)
    
    def _calculate_scope_score(self, query_lower: str) -> float:
        """Calculate query scope score (0.0 to 1.0)."""
        broad_matches = sum(1 for indicator in self.scope_indicators['broad_scope'] 
                           if indicator in query_lower)
        medium_matches = sum(1 for indicator in self.scope_indicators['medium_scope'] 
                            if indicator in query_lower)
        narrow_matches = sum(1 for indicator in self.scope_indicators['narrow_scope'] 
                            if indicator in query_lower)
        
        # Weight scores (broad scope = higher k)
        total_score = (broad_matches * 1.0) + (medium_matches * 0.6) + (narrow_matches * 0.2)
        
        return min(total_score, 1.0)
    
    def _calculate_temporal_k(self, temporal_info: Dict[str, Any], db_stats: Dict[str, Any]) -> int:
        """Calculate k for temporal queries based on actual data availability - NO CAPS."""
        base_min_k = temporal_info.get('min_k', 10)
        total_messages = db_stats['total_messages']
        density_score = db_stats['density_score']
        temporal_type = temporal_info.get('temporal_type', 'recent')
        
        # For very long-term queries, we need much higher k values
        if total_messages == 0:
            return base_min_k
        
        # Dynamic scaling based on message availability and temporal scope
        if temporal_type == 'all_time':
            # For all-time queries, use a significant portion of available messages
            # Scale with total messages but ensure comprehensive coverage
            if total_messages < 100:
                return max(total_messages, base_min_k)
            elif total_messages < 1000:
                return int(total_messages * 0.8)  # 80% for small datasets
            elif total_messages < 10000:
                return int(total_messages * 0.3)  # 30% for medium datasets
            else:
                return int(total_messages * 0.1)  # 10% for large datasets, still hundreds/thousands
        
        elif temporal_type in ['yearly', 'quarterly']:
            # For long-term queries, scale significantly with message count
            if total_messages < 50:
                return max(total_messages, base_min_k)
            elif total_messages < 500:
                return int(total_messages * 0.6)  # 60% coverage
            elif total_messages < 2000:
                return int(total_messages * 0.4)  # 40% coverage
            else:
                return int(total_messages * 0.2)  # 20% coverage for very large datasets
        
        elif temporal_type == 'monthly':
            # Monthly queries need substantial coverage
            if total_messages < 20:
                return max(total_messages, base_min_k)
            elif total_messages < 200:
                return int(total_messages * 0.8)  # High coverage for smaller datasets
            elif total_messages < 1000:
                return int(total_messages * 0.5)  # Medium coverage
            else:
                return int(total_messages * 0.3)  # Lower coverage for very active channels
        
        elif temporal_type == 'weekly':
            # Weekly queries
            if total_messages < 10:
                return max(total_messages, base_min_k // 2)
            elif total_messages < 100:
                return int(total_messages * 0.7)
            else:
                return int(total_messages * 0.4)
        
        else:  # daily, recent
            # Shorter term queries can use most/all available messages
            if total_messages < 5:
                return max(total_messages, 3)
            elif total_messages < 50:
                return total_messages  # Use all available for small datasets
            else:
                return int(total_messages * 0.8)  # Use most for larger datasets
    
    def _calculate_standard_k(self, complexity_score: float, scope_score: float) -> int:
        """Calculate k for non-temporal queries."""
        # Base k calculation
        base_k = 5  # Minimum base
        
        # Add for complexity
        complexity_addition = int(complexity_score * 10)
        
        # Add for scope
        scope_addition = int(scope_score * 8)
        
        return base_k + complexity_addition + scope_addition
    
    def _apply_database_adjustments(
        self,
        base_k: int,
        db_stats: Dict[str, Any],
        complexity_score: float,
        scope_score: float
    ) -> int:
        """Apply database-driven adjustments to base k."""
        adjusted_k = base_k
        
        # Adjust based on data richness
        if db_stats['has_enhanced_content'] > 0:
            # Boost for enhanced content availability
            enhancement_ratio = db_stats['has_enhanced_content'] / max(db_stats['total_messages'], 1)
            if enhancement_ratio > 0.5:
                adjusted_k = int(adjusted_k * 1.2)
        
        # Adjust based on engagement patterns
        if db_stats['engagement_messages'] > 0:
            engagement_ratio = db_stats['engagement_messages'] / max(db_stats['total_messages'], 1)
            if engagement_ratio > 0.3:  # High engagement channel
                adjusted_k = int(adjusted_k * 1.1)
        
        # Adjust based on technical content
        if db_stats['technical_messages'] > 0:
            tech_ratio = db_stats['technical_messages'] / max(db_stats['total_messages'], 1)
            if tech_ratio > 0.4 and complexity_score > 0.6:  # Technical query in tech channel
                adjusted_k = int(adjusted_k * 1.15)
        
        # Adjust based on author diversity
        if db_stats['unique_authors'] > 10:  # High diversity
            adjusted_k = int(adjusted_k * 1.1)
        elif db_stats['unique_authors'] < 3:  # Low diversity
            adjusted_k = int(adjusted_k * 0.9)
        
        # Adjust based on message density
        if db_stats['density_score'] > 20:  # Very active
            adjusted_k = int(adjusted_k * 1.2)
        elif db_stats['density_score'] < 2:  # Very quiet
            adjusted_k = int(adjusted_k * 0.8)
        
        return adjusted_k
    
    def _apply_bounds_and_validation(self, k: int, temporal_info: Dict[str, Any]) -> int:
        """Apply minimal validation to k value - NO CAPS, fully dynamic."""
        # Only ensure minimum viable k, no maximum limits
        min_k = max(temporal_info.get('min_k', 3), 3)
        
        # For all-time queries, ensure substantial minimum
        if temporal_info.get('temporal_type') == 'all_time':
            min_k = max(min_k, 100)
        elif temporal_info.get('temporal_type') in ['yearly', 'quarterly']:
            min_k = max(min_k, 50)
        
        return max(k, min_k)
    
    def _basic_k_determination(self, query: str) -> int:
        """
        Fallback basic k determination (original logic).
        Used when enhanced analysis fails or is disabled.
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
        
        # All-time/historical query indicators
        all_time_keywords = [
            "all time", "ever", "since beginning", "server beginnings", "entire history",
            "complete history", "from start", "since inception", "everything ever",
            "total summary", "full archive", "comprehensive digest", "all messages",
            "entire server", "complete archive"
        ]
        
        # Determine k based on analysis
        # Check for all-time queries first
        if any(kw in query_lower for kw in all_time_keywords):
            return 200  # High fallback for all-time queries
        elif "yearly" in query_lower or "annual" in query_lower:
            return 100  # High fallback for yearly queries
        elif "quarterly" in query_lower or "quarter" in query_lower:
            return 75   # High fallback for quarterly queries
        elif comprehensive_matches >= 2 or "weekly digest" in query_lower or "monthly digest" in query_lower:
            return 30
        elif comprehensive_matches >= 1 or broad_matches >= 2:
            return 20
        elif complex_matches >= 2 or word_count >= 8:
            return 15
        elif broad_matches >= 1 or word_count >= 5:
            return 10
        elif specific_matches >= 1:
            return 3
        else:
            return 5


# Global instance for easy access
enhanced_k_determiner = EnhancedKDetermination()


def determine_optimal_k(
    query: str,
    channel_id: Optional[int] = None,
    channel_name: Optional[str] = None,
    guild_id: Optional[int] = None,
    use_enhanced_analysis: bool = True
) -> int:
    """
    Public function for determining optimal k parameter.
    
    Args:
        query: User query string
        channel_id: Optional channel ID for context
        channel_name: Optional channel name for context
        guild_id: Optional guild ID for context
        use_enhanced_analysis: Whether to use enhanced analysis
        
    Returns:
        Optimal k value for retrieval
    """
    return enhanced_k_determiner.determine_optimal_k(
        query, channel_id, channel_name, guild_id, use_enhanced_analysis
    )


def get_k_determination_info(
    query: str,
    channel_id: Optional[int] = None,
    channel_name: Optional[str] = None,
    guild_id: Optional[int] = None
) -> Dict[str, Any]:
    """
    Get detailed information about k determination process for debugging/analysis.
    
    Args:
        query: User query string
        channel_id: Optional channel ID for context
        channel_name: Optional channel name for context
        guild_id: Optional guild ID for context
        
    Returns:
        Dictionary with detailed k determination information
    """
    try:
        determiner = enhanced_k_determiner
        query_lower = query.lower().strip()
        
        # Analyze temporal patterns
        temporal_info = determiner._analyze_temporal_query(query_lower)
        
        # Get database stats
        db_stats = determiner._get_database_context_stats(channel_id, channel_name, guild_id, temporal_info)
        
        # Calculate scores
        complexity_score = determiner._calculate_complexity_score(query_lower)
        scope_score = determiner._calculate_scope_score(query_lower)
        
        # Calculate k values at each step
        if temporal_info['is_temporal']:
            base_k = determiner._calculate_temporal_k(temporal_info, db_stats)
        else:
            base_k = determiner._calculate_standard_k(complexity_score, scope_score)
        
        adjusted_k = determiner._apply_database_adjustments(base_k, db_stats, complexity_score, scope_score)
        final_k = determiner._apply_bounds_and_validation(adjusted_k, temporal_info)
        
        # Fallback k for comparison
        fallback_k = determiner._basic_k_determination(query)
        
        return {
            'query': query,
            'temporal_info': temporal_info,
            'database_stats': db_stats,
            'complexity_score': complexity_score,
            'scope_score': scope_score,
            'k_calculation_steps': {
                'base_k': base_k,
                'adjusted_k': adjusted_k,
                'final_k': final_k,
                'fallback_k': fallback_k
            },
            'enhancement_benefit': final_k - fallback_k
        }
        
    except Exception as e:
        logger.error(f"Error getting k determination info: {e}")
        return {
            'error': str(e),
            'fallback_k': enhanced_k_determiner._basic_k_determination(query)
        }
