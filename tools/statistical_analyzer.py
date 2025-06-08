"""
Enhanced Statistical Analysis Tools for Discord Bot Agent

This module provides comprehensive statistical analysis capabilities for the Discord bot,
allowing the agent to perform complex data analytics on message data, user activity,
channel metrics, and community engagement patterns.

Features:
- Message statistics (counts, averages, min/max, percentiles)
- User activity analytics (engagement scores, patterns, trends)
- Channel performance metrics (activity levels, response rates)
- Temporal analysis (time series, trends, patterns)
- Community insights (expertise distribution, help patterns)
- Content analysis (length statistics, type distributions)
- Comparative analysis (channel comparisons, user comparisons)
"""

import os
import sys
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import json
import numpy as np
from sqlalchemy import func, and_, or_, text
from sqlalchemy.orm import Session

# Add project root to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from db import SessionLocal, Message
import logging

logger = logging.getLogger(__name__)

class DiscordStatisticalAnalyzer:
    """
    Comprehensive statistical analysis for Discord message data.
    Provides advanced analytics capabilities for the agent system.
    """
    
    def __init__(self):
        self.session = SessionLocal()
    
    def __del__(self):
        if hasattr(self, 'session'):
            self.session.close()
    
    # =====================================
    # MESSAGE STATISTICS
    # =====================================
    
    def get_message_statistics(
        self,
        channel_id: Optional[int] = None,
        channel_name: Optional[str] = None,
        guild_id: Optional[int] = None,
        author_name: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Get comprehensive message statistics with filtering options.
        
        Returns counts, averages, min/max, percentiles, and distribution metrics.
        """
        try:
            query = self.session.query(Message)
            
            # Apply filters
            query = self._apply_filters(query, channel_id, channel_name, guild_id, author_name, start_date, end_date)
            
            # Get basic counts
            total_messages = query.count()
            
            if total_messages == 0:
                return {"error": "No messages found with the specified filters", "count": 0}
            
            # Content length statistics
            content_lengths = [
                len(msg.content or '') for msg in query.all()
            ]
            
            # Calculate statistics
            stats = {
                "basic_counts": {
                    "total_messages": total_messages,
                    "unique_authors": self._count_unique_authors(query),
                    "unique_channels": self._count_unique_channels(query),
                    "messages_with_content": sum(1 for length in content_lengths if length > 0),
                    "empty_messages": sum(1 for length in content_lengths if length == 0)
                },
                "content_length_stats": {
                    "average_length": np.mean(content_lengths),
                    "median_length": np.median(content_lengths),
                    "min_length": np.min(content_lengths),
                    "max_length": np.max(content_lengths),
                    "std_dev": np.std(content_lengths),
                    "percentiles": {
                        "25th": np.percentile(content_lengths, 25),
                        "75th": np.percentile(content_lengths, 75),
                        "90th": np.percentile(content_lengths, 90),
                        "95th": np.percentile(content_lengths, 95),
                        "99th": np.percentile(content_lengths, 99)
                    }
                },
                "content_distribution": {
                    "very_short": sum(1 for l in content_lengths if 0 < l <= 10),
                    "short": sum(1 for l in content_lengths if 10 < l <= 50),
                    "medium": sum(1 for l in content_lengths if 50 < l <= 200),
                    "long": sum(1 for l in content_lengths if 200 < l <= 500),
                    "very_long": sum(1 for l in content_lengths if l > 500)
                },
                "engagement_stats": self._calculate_engagement_stats(query),
                "temporal_stats": self._calculate_temporal_stats(query)
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error calculating message statistics: {e}")
            return {"error": str(e)}
    
    # =====================================
    # USER ACTIVITY ANALYTICS
    # =====================================
    
    def analyze_user_activity(
        self,
        time_period_days: int = 30,
        channel_name: Optional[str] = None,
        min_messages: int = 1
    ) -> Dict[str, Any]:
        """
        Comprehensive user activity analysis with engagement scoring.
        """
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=time_period_days)
            
            query = self.session.query(Message).filter(Message.timestamp >= cutoff_date)
            
            if channel_name:
                query = query.filter(Message.channel_name.ilike(f"%{channel_name}%"))
            
            messages = query.all()
            
            user_analytics = defaultdict(lambda: {
                'total_messages': 0,
                'channels_active': set(),
                'avg_message_length': 0,
                'total_chars': 0,
                'messages_with_attachments': 0,
                'messages_with_embeds': 0,
                'messages_with_reactions': 0,
                'reply_messages': 0,
                'help_seeking_indicators': 0,
                'help_providing_indicators': 0,
                'technical_content_score': 0,
                'activity_by_day': defaultdict(int),
                'activity_by_hour': defaultdict(int),
                'first_message_date': None,
                'last_message_date': None
            })
            
            # Process all messages
            for msg in messages:
                if not msg.author:
                    continue
                    
                username = msg.author.get('username', 'unknown')
                user_data = user_analytics[username]
                
                user_data['total_messages'] += 1
                user_data['channels_active'].add(msg.channel_name)
                
                content_length = len(msg.content or '')
                user_data['total_chars'] += content_length
                
                if msg.attachments:
                    user_data['messages_with_attachments'] += 1
                if msg.embeds:
                    user_data['messages_with_embeds'] += 1
                if msg.reactions:
                    user_data['messages_with_reactions'] += 1
                if msg.reference:
                    user_data['reply_messages'] += 1
                
                # Analyze content for help patterns
                content = (msg.content or '').lower()
                if any(word in content for word in ['help', 'how to', 'question', '?', 'please', 'can someone']):
                    user_data['help_seeking_indicators'] += 1
                if any(word in content for word in ['here is', 'try this', 'solution', 'answer', 'you can']):
                    user_data['help_providing_indicators'] += 1
                
                # Technical content scoring
                if any(word in content for word in ['code', 'function', 'api', 'database', 'algorithm', 'python', 'javascript']):
                    user_data['technical_content_score'] += 1
                
                # Temporal patterns
                day_key = msg.timestamp.strftime('%Y-%m-%d')
                hour_key = msg.timestamp.hour
                user_data['activity_by_day'][day_key] += 1
                user_data['activity_by_hour'][hour_key] += 1
                
                # Date tracking
                if not user_data['first_message_date'] or msg.timestamp < user_data['first_message_date']:
                    user_data['first_message_date'] = msg.timestamp
                if not user_data['last_message_date'] or msg.timestamp > user_data['last_message_date']:
                    user_data['last_message_date'] = msg.timestamp
            
            # Calculate derived metrics and filter
            processed_analytics = {}
            
            for username, data in user_analytics.items():
                if data['total_messages'] < min_messages:
                    continue
                
                # Calculate averages and scores
                data['avg_message_length'] = data['total_chars'] / data['total_messages'] if data['total_messages'] > 0 else 0
                data['channels_active'] = len(data['channels_active'])
                data['engagement_score'] = self._calculate_user_engagement_score(data)
                data['help_ratio'] = data['help_providing_indicators'] / max(data['help_seeking_indicators'], 1)
                data['technical_ratio'] = data['technical_content_score'] / data['total_messages']
                data['multimedia_ratio'] = (data['messages_with_attachments'] + data['messages_with_embeds']) / data['total_messages']
                data['interaction_ratio'] = (data['reply_messages'] + data['messages_with_reactions']) / data['total_messages']
                
                # Convert datetime objects for JSON serialization
                data['first_message_date'] = data['first_message_date'].isoformat() if data['first_message_date'] else None
                data['last_message_date'] = data['last_message_date'].isoformat() if data['last_message_date'] else None
                
                # Convert defaultdicts to regular dicts
                data['activity_by_day'] = dict(data['activity_by_day'])
                data['activity_by_hour'] = dict(data['activity_by_hour'])
                
                processed_analytics[username] = data
            
            # Generate summary statistics
            summary_stats = self._generate_user_summary_stats(processed_analytics)
            
            return {
                "summary": summary_stats,
                "user_analytics": processed_analytics,
                "analysis_period": {
                    "days": time_period_days,
                    "start_date": cutoff_date.isoformat(),
                    "end_date": datetime.utcnow().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Error analyzing user activity: {e}")
            return {"error": str(e)}
    
    # =====================================
    # CHANNEL PERFORMANCE METRICS
    # =====================================
    
    def analyze_channel_performance(
        self,
        time_period_days: int = 30,
        include_empty: bool = False
    ) -> Dict[str, Any]:
        """
        Comprehensive channel performance analysis.
        """
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=time_period_days)
            
            messages = self.session.query(Message).filter(Message.timestamp >= cutoff_date).all()
            
            channel_analytics = defaultdict(lambda: {
                'total_messages': 0,
                'unique_authors': set(),
                'avg_message_length': 0,
                'total_chars': 0,
                'messages_with_attachments': 0,
                'messages_with_embeds': 0,
                'messages_with_reactions': 0,
                'reply_chains': 0,
                'help_seeking_messages': 0,
                'help_providing_messages': 0,
                'technical_discussions': 0,
                'activity_by_day': defaultdict(int),
                'peak_activity_hour': defaultdict(int),
                'author_diversity_score': 0,
                'engagement_quality_score': 0
            })
            
            # Process messages
            for msg in messages:
                channel = msg.channel_name
                if not channel:
                    continue
                
                data = channel_analytics[channel]
                data['total_messages'] += 1
                
                if msg.author:
                    data['unique_authors'].add(msg.author.get('username', 'unknown'))
                
                content_length = len(msg.content or '')
                data['total_chars'] += content_length
                
                if msg.attachments:
                    data['messages_with_attachments'] += 1
                if msg.embeds:
                    data['messages_with_embeds'] += 1
                if msg.reactions:
                    data['messages_with_reactions'] += 1
                if msg.reference:
                    data['reply_chains'] += 1
                
                # Content analysis
                content = (msg.content or '').lower()
                if any(word in content for word in ['help', 'question', '?', 'how to']):
                    data['help_seeking_messages'] += 1
                if any(word in content for word in ['answer', 'solution', 'here is', 'try this']):
                    data['help_providing_messages'] += 1
                if any(word in content for word in ['code', 'api', 'technical', 'algorithm']):
                    data['technical_discussions'] += 1
                
                # Temporal analysis
                day_key = msg.timestamp.strftime('%Y-%m-%d')
                hour_key = msg.timestamp.hour
                data['activity_by_day'][day_key] += 1
                data['peak_activity_hour'][hour_key] += 1
            
            # Calculate derived metrics
            processed_channels = {}
            
            for channel, data in channel_analytics.items():
                if not include_empty and data['total_messages'] == 0:
                    continue
                
                data['unique_authors'] = len(data['unique_authors'])
                data['avg_message_length'] = data['total_chars'] / data['total_messages'] if data['total_messages'] > 0 else 0
                data['author_diversity_score'] = data['unique_authors'] / max(data['total_messages'], 1)
                data['help_response_ratio'] = data['help_providing_messages'] / max(data['help_seeking_messages'], 1)
                data['technical_ratio'] = data['technical_discussions'] / data['total_messages'] if data['total_messages'] > 0 else 0
                data['engagement_ratio'] = (data['messages_with_reactions'] + data['reply_chains']) / data['total_messages'] if data['total_messages'] > 0 else 0
                
                # Find peak activity hour
                if data['peak_activity_hour']:
                    data['peak_hour'] = max(data['peak_activity_hour'].items(), key=lambda x: x[1])[0]
                else:
                    data['peak_hour'] = None
                
                # Convert defaultdicts to regular dicts
                data['activity_by_day'] = dict(data['activity_by_day'])
                data['peak_activity_hour'] = dict(data['peak_activity_hour'])
                
                processed_channels[channel] = data
            
            # Generate summary and rankings
            summary_stats = self._generate_channel_summary_stats(processed_channels)
            
            return {
                "summary": summary_stats,
                "channel_analytics": processed_channels,
                "rankings": {
                    "most_active": self._rank_channels_by_activity(processed_channels),
                    "highest_engagement": self._rank_channels_by_engagement(processed_channels),
                    "most_technical": self._rank_channels_by_technical_content(processed_channels),
                    "best_help_ratio": self._rank_channels_by_help_ratio(processed_channels)
                },
                "analysis_period": {
                    "days": time_period_days,
                    "start_date": cutoff_date.isoformat(),
                    "end_date": datetime.utcnow().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Error analyzing channel performance: {e}")
            return {"error": str(e)}
    
    # =====================================
    # TEMPORAL ANALYSIS
    # =====================================
    
    def analyze_temporal_patterns(
        self,
        time_period_days: int = 90,
        granularity: str = "daily"  # daily, hourly, weekly
    ) -> Dict[str, Any]:
        """
        Analyze temporal patterns and trends in message activity.
        """
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=time_period_days)
            
            messages = self.session.query(Message).filter(Message.timestamp >= cutoff_date).all()
            
            temporal_data = {
                'activity_timeline': defaultdict(int),
                'hourly_patterns': defaultdict(int),
                'daily_patterns': defaultdict(int),
                'weekly_patterns': defaultdict(int),
                'trend_analysis': {},
                'peak_times': {},
                'quiet_periods': {}
            }
            
            # Process messages for temporal patterns
            for msg in messages:
                timestamp = msg.timestamp
                
                # Generate different granularity keys
                if granularity == "hourly":
                    key = timestamp.strftime('%Y-%m-%d %H:00')
                elif granularity == "weekly":
                    key = timestamp.strftime('%Y-W%U')
                else:  # daily
                    key = timestamp.strftime('%Y-%m-%d')
                
                temporal_data['activity_timeline'][key] += 1
                temporal_data['hourly_patterns'][timestamp.hour] += 1
                temporal_data['daily_patterns'][timestamp.strftime('%A')] += 1
                temporal_data['weekly_patterns'][timestamp.strftime('%Y-W%U')] += 1
            
            # Convert to regular dicts and analyze trends
            temporal_data['activity_timeline'] = dict(temporal_data['activity_timeline'])
            temporal_data['hourly_patterns'] = dict(temporal_data['hourly_patterns'])
            temporal_data['daily_patterns'] = dict(temporal_data['daily_patterns'])
            temporal_data['weekly_patterns'] = dict(temporal_data['weekly_patterns'])
            
            # Calculate trends and patterns
            temporal_data['trend_analysis'] = self._calculate_temporal_trends(temporal_data['activity_timeline'])
            temporal_data['peak_times'] = self._identify_peak_times(temporal_data)
            temporal_data['quiet_periods'] = self._identify_quiet_periods(temporal_data)
            
            return temporal_data
            
        except Exception as e:
            logger.error(f"Error analyzing temporal patterns: {e}")
            return {"error": str(e)}
    
    # =====================================
    # COMPARATIVE ANALYSIS
    # =====================================
    
    def compare_channels(
        self,
        channel_names: List[str],
        time_period_days: int = 30
    ) -> Dict[str, Any]:
        """
        Compare multiple channels across key metrics.
        """
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=time_period_days)
            
            comparison_data = {}
            
            for channel_name in channel_names:
                messages = self.session.query(Message).filter(
                    and_(
                        Message.timestamp >= cutoff_date,
                        Message.channel_name.ilike(f"%{channel_name}%")
                    )
                ).all()
                
                if messages:
                    stats = self._calculate_channel_comparison_stats(messages)
                    comparison_data[channel_name] = stats
                else:
                    comparison_data[channel_name] = {"error": "No messages found", "count": 0}
            
            # Generate comparative insights
            insights = self._generate_comparative_insights(comparison_data)
            
            return {
                "channel_comparisons": comparison_data,
                "comparative_insights": insights,
                "analysis_period": {
                    "days": time_period_days,
                    "start_date": cutoff_date.isoformat(),
                    "end_date": datetime.utcnow().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Error comparing channels: {e}")
            return {"error": str(e)}
    
    def compare_users(
        self,
        usernames: List[str],
        time_period_days: int = 30
    ) -> Dict[str, Any]:
        """
        Compare multiple users across key metrics.
        """
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=time_period_days)
            
            comparison_data = {}
            
            for username in usernames:
                messages = self.session.query(Message).filter(
                    and_(
                        Message.timestamp >= cutoff_date,
                        func.json_extract(Message.author, '$.username') == username
                    )
                ).all()
                
                if messages:
                    stats = self._calculate_user_comparison_stats(messages)
                    comparison_data[username] = stats
                else:
                    comparison_data[username] = {"error": "No messages found", "count": 0}
            
            # Generate comparative insights
            insights = self._generate_user_comparative_insights(comparison_data)
            
            return {
                "user_comparisons": comparison_data,
                "comparative_insights": insights,
                "analysis_period": {
                    "days": time_period_days,
                    "start_date": cutoff_date.isoformat(),
                    "end_date": datetime.utcnow().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Error comparing users: {e}")
            return {"error": str(e)}
    
    # =====================================
    # HELPER METHODS
    # =====================================
    
    def _apply_filters(self, query, channel_id, channel_name, guild_id, author_name, start_date, end_date):
        """Apply common filters to queries."""
        if channel_id:
            query = query.filter(Message.channel_id == channel_id)
        if channel_name:
            query = query.filter(Message.channel_name.ilike(f"%{channel_name}%"))
        if guild_id:
            query = query.filter(Message.guild_id == guild_id)
        if author_name:
            query = query.filter(func.json_extract(Message.author, '$.username') == author_name)
        if start_date:
            query = query.filter(Message.timestamp >= start_date)
        if end_date:
            query = query.filter(Message.timestamp <= end_date)
        return query
    
    def _count_unique_authors(self, query):
        """Count unique authors in query results."""
        return query.with_entities(func.json_extract(Message.author, '$.username')).distinct().count()
    
    def _count_unique_channels(self, query):
        """Count unique channels in query results."""
        return query.with_entities(Message.channel_name).distinct().count()
    
    def _calculate_engagement_stats(self, query):
        """Calculate engagement statistics from messages."""
        messages = query.all()
        
        engagement_stats = {
            "messages_with_reactions": sum(1 for msg in messages if msg.reactions),
            "messages_with_attachments": sum(1 for msg in messages if msg.attachments),
            "messages_with_embeds": sum(1 for msg in messages if msg.embeds),
            "reply_messages": sum(1 for msg in messages if msg.reference),
            "pinned_messages": sum(1 for msg in messages if msg.pinned),
            "total_reactions": sum(len(msg.reactions or []) for msg in messages)
        }
        
        total_messages = len(messages)
        if total_messages > 0:
            engagement_stats["reaction_rate"] = engagement_stats["messages_with_reactions"] / total_messages
            engagement_stats["attachment_rate"] = engagement_stats["messages_with_attachments"] / total_messages
            engagement_stats["reply_rate"] = engagement_stats["reply_messages"] / total_messages
        
        return engagement_stats
    
    def _calculate_temporal_stats(self, query):
        """Calculate temporal statistics from messages."""
        messages = query.all()
        
        if not messages:
            return {}
        
        timestamps = [msg.timestamp for msg in messages]
        earliest = min(timestamps)
        latest = max(timestamps)
        
        return {
            "earliest_message": earliest.isoformat(),
            "latest_message": latest.isoformat(),
            "time_span_days": (latest - earliest).days,
            "messages_per_day": len(messages) / max((latest - earliest).days, 1)
        }
    
    def _calculate_user_engagement_score(self, user_data):
        """Calculate a composite engagement score for a user."""
        # Normalize factors (0-1 scale)
        message_factor = min(user_data['total_messages'] / 100, 1.0)  # Cap at 100 messages
        channel_factor = min(user_data['channels_active'] / 10, 1.0)  # Cap at 10 channels
        interaction_factor = min((user_data['reply_messages'] + user_data['messages_with_reactions']) / user_data['total_messages'], 1.0)
        help_factor = min(user_data['help_providing_indicators'] / max(user_data['total_messages'], 1), 1.0)
        
        # Weighted composite score
        engagement_score = (
            message_factor * 0.3 +
            channel_factor * 0.2 +
            interaction_factor * 0.3 +
            help_factor * 0.2
        )
        
        return round(engagement_score, 3)
    
    def _generate_user_summary_stats(self, user_analytics):
        """Generate summary statistics for user analytics."""
        if not user_analytics:
            return {}
        
        total_users = len(user_analytics)
        total_messages = sum(data['total_messages'] for data in user_analytics.values())
        
        engagement_scores = [data['engagement_score'] for data in user_analytics.values()]
        
        return {
            "total_active_users": total_users,
            "total_messages_analyzed": total_messages,
            "avg_messages_per_user": total_messages / total_users,
            "avg_engagement_score": np.mean(engagement_scores),
            "top_contributors": sorted(
                [(user, data['total_messages']) for user, data in user_analytics.items()],
                key=lambda x: x[1],
                reverse=True
            )[:10]
        }
    
    def _generate_channel_summary_stats(self, channel_analytics):
        """Generate summary statistics for channel analytics."""
        if not channel_analytics:
            return {}
        
        total_channels = len(channel_analytics)
        total_messages = sum(data['total_messages'] for data in channel_analytics.values())
        
        return {
            "total_active_channels": total_channels,
            "total_messages_analyzed": total_messages,
            "avg_messages_per_channel": total_messages / total_channels,
            "most_active_channel": max(
                channel_analytics.items(),
                key=lambda x: x[1]['total_messages']
            )[0] if channel_analytics else None
        }
    
    def _rank_channels_by_activity(self, channels):
        """Rank channels by total message activity."""
        return sorted(
            [(channel, data['total_messages']) for channel, data in channels.items()],
            key=lambda x: x[1],
            reverse=True
        )[:10]
    
    def _rank_channels_by_engagement(self, channels):
        """Rank channels by engagement ratio."""
        return sorted(
            [(channel, data.get('engagement_ratio', 0)) for channel, data in channels.items()],
            key=lambda x: x[1],
            reverse=True
        )[:10]
    
    def _rank_channels_by_technical_content(self, channels):
        """Rank channels by technical content ratio."""
        return sorted(
            [(channel, data.get('technical_ratio', 0)) for channel, data in channels.items()],
            key=lambda x: x[1],
            reverse=True
        )[:10]
    
    def _rank_channels_by_help_ratio(self, channels):
        """Rank channels by help response ratio."""
        return sorted(
            [(channel, data.get('help_response_ratio', 0)) for channel, data in channels.items()],
            key=lambda x: x[1],
            reverse=True
        )[:10]
    
    def _calculate_temporal_trends(self, timeline_data):
        """Calculate trends from timeline data."""
        if len(timeline_data) < 2:
            return {"trend": "insufficient_data"}
        
        values = list(timeline_data.values())
        dates = sorted(timeline_data.keys())
        
        # Simple linear trend calculation
        x = np.arange(len(values))
        y = np.array([timeline_data[date] for date in dates])
        
        if len(y) > 1:
            slope = np.polyfit(x, y, 1)[0]
            
            return {
                "trend_direction": "increasing" if slope > 0 else "decreasing" if slope < 0 else "stable",
                "trend_strength": abs(slope),
                "avg_daily_activity": np.mean(y),
                "peak_activity": np.max(y),
                "peak_date": dates[np.argmax(y)]
            }
        
        return {"trend": "insufficient_data"}
    
    def _identify_peak_times(self, temporal_data):
        """Identify peak activity times."""
        hourly = temporal_data['hourly_patterns']
        daily = temporal_data['daily_patterns']
        
        peak_hour = max(hourly.items(), key=lambda x: x[1])[0] if hourly else None
        peak_day = max(daily.items(), key=lambda x: x[1])[0] if daily else None
        
        return {
            "peak_hour": peak_hour,
            "peak_day": peak_day,
            "peak_hour_activity": hourly.get(peak_hour, 0) if peak_hour else 0,
            "peak_day_activity": daily.get(peak_day, 0) if peak_day else 0
        }
    
    def _identify_quiet_periods(self, temporal_data):
        """Identify quiet activity periods."""
        hourly = temporal_data['hourly_patterns']
        daily = temporal_data['daily_patterns']
        
        quiet_hour = min(hourly.items(), key=lambda x: x[1])[0] if hourly else None
        quiet_day = min(daily.items(), key=lambda x: x[1])[0] if daily else None
        
        return {
            "quiet_hour": quiet_hour,
            "quiet_day": quiet_day,
            "quiet_hour_activity": hourly.get(quiet_hour, 0) if quiet_hour else 0,
            "quiet_day_activity": daily.get(quiet_day, 0) if quiet_day else 0
        }
    
    def _calculate_channel_comparison_stats(self, messages):
        """Calculate comparison statistics for a channel."""
        if not messages:
            return {"count": 0}
        
        total_messages = len(messages)
        content_lengths = [len(msg.content or '') for msg in messages]
        unique_authors = len(set(msg.author.get('username') for msg in messages if msg.author))
        
        return {
            "total_messages": total_messages,
            "unique_authors": unique_authors,
            "avg_message_length": np.mean(content_lengths),
            "messages_with_reactions": sum(1 for msg in messages if msg.reactions),
            "messages_with_attachments": sum(1 for msg in messages if msg.attachments),
            "reply_messages": sum(1 for msg in messages if msg.reference),
            "author_diversity": unique_authors / total_messages,
            "engagement_rate": sum(1 for msg in messages if msg.reactions or msg.reference) / total_messages
        }
    
    def _calculate_user_comparison_stats(self, messages):
        """Calculate comparison statistics for a user."""
        if not messages:
            return {"count": 0}
        
        total_messages = len(messages)
        content_lengths = [len(msg.content or '') for msg in messages]
        unique_channels = len(set(msg.channel_name for msg in messages if msg.channel_name))
        
        return {
            "total_messages": total_messages,
            "unique_channels": unique_channels,
            "avg_message_length": np.mean(content_lengths),
            "messages_with_reactions": sum(1 for msg in messages if msg.reactions),
            "messages_with_attachments": sum(1 for msg in messages if msg.attachments),
            "reply_messages": sum(1 for msg in messages if msg.reference),
            "channel_diversity": unique_channels / total_messages,
            "interaction_rate": sum(1 for msg in messages if msg.reactions or msg.reference) / total_messages
        }
    
    def _generate_comparative_insights(self, comparison_data):
        """Generate insights from channel comparison data."""
        insights = {}
        
        valid_channels = {k: v for k, v in comparison_data.items() if 'error' not in v}
        
        if len(valid_channels) < 2:
            return {"error": "Need at least 2 channels with data for comparison"}
        
        # Find extremes
        most_active = max(valid_channels.items(), key=lambda x: x[1]['total_messages'])
        highest_engagement = max(valid_channels.items(), key=lambda x: x[1].get('engagement_rate', 0))
        most_diverse = max(valid_channels.items(), key=lambda x: x[1].get('author_diversity', 0))
        
        insights = {
            "most_active_channel": {"name": most_active[0], "messages": most_active[1]['total_messages']},
            "highest_engagement_channel": {"name": highest_engagement[0], "rate": highest_engagement[1].get('engagement_rate', 0)},
            "most_diverse_channel": {"name": most_diverse[0], "diversity": most_diverse[1].get('author_diversity', 0)},
            "total_channels_compared": len(valid_channels)
        }
        
        return insights
    
    def _generate_user_comparative_insights(self, comparison_data):
        """Generate insights from user comparison data."""
        valid_users = {k: v for k, v in comparison_data.items() if 'error' not in v}
        
        if len(valid_users) < 2:
            return {"error": "Need at least 2 users with data for comparison"}
        
        # Find extremes
        most_active = max(valid_users.items(), key=lambda x: x[1]['total_messages'])
        highest_interaction = max(valid_users.items(), key=lambda x: x[1].get('interaction_rate', 0))
        most_diverse = max(valid_users.items(), key=lambda x: x[1].get('channel_diversity', 0))
        
        insights = {
            "most_active_user": {"name": most_active[0], "messages": most_active[1]['total_messages']},
            "highest_interaction_user": {"name": highest_interaction[0], "rate": highest_interaction[1].get('interaction_rate', 0)},
            "most_diverse_user": {"name": most_diverse[0], "diversity": most_diverse[1].get('channel_diversity', 0)},
            "total_users_compared": len(valid_users)
        }
        
        return insights


# Global instance for tool integration
statistical_analyzer = DiscordStatisticalAnalyzer()


# =====================================
# TOOL FUNCTIONS FOR AGENT INTEGRATION
# =====================================

def get_message_statistics(
    channel_name: Optional[str] = None,
    author_name: Optional[str] = None,
    days_back: int = 30
) -> Dict[str, Any]:
    """
    Get comprehensive message statistics.
    
    Args:
        channel_name: Filter by channel name (partial match)
        author_name: Filter by author username
        days_back: Number of days to analyze (default: 30)
    
    Returns:
        Dictionary with comprehensive statistics including counts, averages, min/max, percentiles
    """
    start_date = datetime.utcnow() - timedelta(days=days_back)
    
    return statistical_analyzer.get_message_statistics(
        channel_name=channel_name,
        author_name=author_name,
        start_date=start_date
    )

def analyze_user_activity_stats(
    time_period_days: int = 30,
    channel_name: Optional[str] = None,
    min_messages: int = 1
) -> Dict[str, Any]:
    """
    Analyze user activity with engagement scoring and behavioral patterns.
    
    Args:
        time_period_days: Number of days to analyze
        channel_name: Filter by channel name
        min_messages: Minimum messages required for inclusion
    
    Returns:
        Dictionary with user analytics and engagement scores
    """
    return statistical_analyzer.analyze_user_activity(
        time_period_days=time_period_days,
        channel_name=channel_name,
        min_messages=min_messages
    )

def analyze_channel_performance_stats(
    time_period_days: int = 30,
    include_empty: bool = False
) -> Dict[str, Any]:
    """
    Analyze channel performance metrics and rankings.
    
    Args:
        time_period_days: Number of days to analyze
        include_empty: Whether to include channels with no activity
    
    Returns:
        Dictionary with channel analytics and performance rankings
    """
    return statistical_analyzer.analyze_channel_performance(
        time_period_days=time_period_days,
        include_empty=include_empty
    )

def analyze_temporal_patterns_stats(
    time_period_days: int = 90,
    granularity: str = "daily"
) -> Dict[str, Any]:
    """
    Analyze temporal patterns and trends in activity.
    
    Args:
        time_period_days: Number of days to analyze
        granularity: Time granularity - "daily", "hourly", or "weekly"
    
    Returns:
        Dictionary with temporal patterns and trend analysis
    """
    return statistical_analyzer.analyze_temporal_patterns(
        time_period_days=time_period_days,
        granularity=granularity
    )

def compare_channels_stats(
    channel_names: List[str],
    time_period_days: int = 30
) -> Dict[str, Any]:
    """
    Compare multiple channels across key metrics.
    
    Args:
        channel_names: List of channel names to compare
        time_period_days: Number of days to analyze
    
    Returns:
        Dictionary with comparative analysis of channels
    """
    return statistical_analyzer.compare_channels(
        channel_names=channel_names,
        time_period_days=time_period_days
    )

def compare_users_stats(
    usernames: List[str],
    time_period_days: int = 30
) -> Dict[str, Any]:
    """
    Compare multiple users across key metrics.
    
    Args:
        usernames: List of usernames to compare
        time_period_days: Number of days to analyze
    
    Returns:
        Dictionary with comparative analysis of users
    """
    return statistical_analyzer.compare_users(
        usernames=usernames,
        time_period_days=time_period_days
    )
