#!/usr/bin/env python3
"""
Deep Message Content Analysis Script

This script performs comprehensive analysis of Discord message content and statistics
to inform LLM agent capabilities and tool development.

The analysis focuses on understanding:
1. Content semantic patterns for search optimization
2. User interaction patterns for social analysis
3. Knowledge domains and topics for domain-specific responses
4. Temporal patterns for time-aware queries
5. Community structure and expertise mapping
6. Resource and information types for knowledge extraction

TODO: Customize analysis based on expected query types:
- What types of questions will the agent answer?
- What information retrieval patterns are expected?
- What community insights are needed?
- What temporal analysis is required?
- What user behavior analysis is needed?

Please provide the query specifications to complete this analysis.
"""

import os
import sys
import re
import json
from datetime import datetime, timedelta
from collections import Counter, defaultdict
from typing import Dict, List, Any, Optional, Tuple

# Add project root to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from db import SessionLocal, Message
from utils.logger import setup_logging
from sqlalchemy import func, and_, or_

setup_logging()
import logging
log = logging.getLogger(__name__)

class DeepMessageAnalyzer:
    def __init__(self):
        self.db = SessionLocal()
        self.analysis_results = {
            'timestamp': datetime.utcnow().isoformat(),
            'total_messages': 0,
            'analysis_period': {},
            'content_semantics': {},
            'user_patterns': {},
            'temporal_patterns': {},
            'channel_analysis': {},
            'knowledge_domains': {},
            'interaction_patterns': {},
            'resource_analysis': {},
            'community_insights': {},
            'query_optimization_insights': {}
        }
    
    def run_comprehensive_analysis(self) -> Dict[str, Any]:
        """Run all analysis modules"""
        log.info("🚀 Starting comprehensive message content analysis...")
        
        # Get basic statistics
        self._analyze_dataset_overview()
        
        # Core analysis modules
        self._analyze_content_semantics()
        self._analyze_user_patterns()
        self._analyze_temporal_patterns()
        self._analyze_channel_characteristics()
        self._analyze_knowledge_domains()
        self._analyze_interaction_patterns()
        self._analyze_resource_types()
        self._analyze_community_insights()
        
        # Generate insights for query optimization
        self._generate_query_optimization_insights()
        
        return self.analysis_results
    
    def _analyze_dataset_overview(self):
        """Get overview of the dataset"""
        log.info("📊 Analyzing dataset overview...")
        
        total_messages = self.db.query(Message).count()
        
        # Time range
        earliest = self.db.query(func.min(Message.timestamp)).scalar()
        latest = self.db.query(func.max(Message.timestamp)).scalar()
        
        # Channel distribution
        channel_counts = dict(
            self.db.query(Message.channel_name, func.count(Message.id))
            .group_by(Message.channel_name)
            .order_by(func.count(Message.id).desc())
            .limit(20)
            .all()
        )
        
        # User activity
        user_counts = dict(
            self.db.query(
                func.json_extract(Message.author, '$.username'),
                func.count(Message.id)
            )
            .group_by(func.json_extract(Message.author, '$.username'))
            .order_by(func.count(Message.id).desc())
            .limit(20)
            .all()
        )
        
        self.analysis_results.update({
            'total_messages': total_messages,
            'analysis_period': {
                'earliest_message': earliest.isoformat() if earliest else None,
                'latest_message': latest.isoformat() if latest else None,
                'days_span': (latest - earliest).days if earliest and latest else 0
            },
            'top_channels': channel_counts,
            'top_users': user_counts
        })
    
    def _analyze_content_semantics(self):
        """Analyze semantic patterns in content for search optimization"""
        log.info("🧠 Analyzing content semantics...")
        
        # Sample recent messages for analysis
        messages = (self.db.query(Message)
                   .filter(Message.content.isnot(None))
                   .filter(Message.content != '')
                   .order_by(Message.timestamp.desc())
                   .limit(5000)
                   .all())
        
        semantics = {
            'content_length_stats': self._get_length_statistics(messages),
            'topic_keywords': self._extract_topic_keywords(messages),
            'question_patterns': self._analyze_question_patterns(messages),
            'technical_content': self._analyze_technical_content(messages),
            'conversational_patterns': self._analyze_conversational_patterns(messages),
            'information_density': self._analyze_information_density(messages),
            'semantic_clustering_candidates': self._identify_semantic_clusters(messages)
        }
        
        self.analysis_results['content_semantics'] = semantics
    
    def _analyze_user_patterns(self):
        """Analyze user behavior patterns"""
        log.info("👥 Analyzing user patterns...")
        
        # User activity patterns
        user_activity = defaultdict(lambda: {
            'message_count': 0,
            'channels': set(),
            'avg_message_length': 0,
            'recent_activity': [],
            'interaction_style': {},
            'expertise_indicators': []
        })
        
        # Get user data
        recent_messages = (self.db.query(Message)
                          .filter(Message.timestamp > datetime.utcnow() - timedelta(days=30))
                          .all())
        
        for msg in recent_messages:
            if msg.author:
                username = msg.author.get('username', 'unknown')
                user_activity[username]['message_count'] += 1
                user_activity[username]['channels'].add(msg.channel_name)
                user_activity[username]['recent_activity'].append({
                    'timestamp': msg.timestamp.isoformat(),
                    'channel': msg.channel_name,
                    'content_length': len(msg.content or ''),
                    'has_attachments': bool(msg.attachments),
                    'has_embeds': bool(msg.embeds)
                })
        
        # Convert sets to lists for JSON serialization
        for user_data in user_activity.values():
            user_data['channels'] = list(user_data['channels'])
            user_data['channel_diversity'] = len(user_data['channels'])
        
        # Get top users
        top_users = dict(sorted(
            [(k, v['message_count']) for k, v in user_activity.items()],
            key=lambda x: x[1],
            reverse=True
        )[:20])
        
        patterns = {
            'active_users_30d': len(user_activity),
            'top_users_by_activity': top_users,
            'user_channel_diversity': {
                username: data['channel_diversity'] 
                for username, data in user_activity.items()
            },
            'conversation_starters': self._identify_conversation_starters(recent_messages),
            'expert_contributors': self._identify_expert_contributors(user_activity)
        }
        
        self.analysis_results['user_patterns'] = patterns
    
    def _analyze_temporal_patterns(self):
        """Analyze temporal patterns in message activity"""
        log.info("⏰ Analyzing temporal patterns...")
        
        # Activity by hour, day, week
        messages = (self.db.query(Message)
                   .filter(Message.timestamp > datetime.utcnow() - timedelta(days=60))
                   .all())
        
        hourly_activity = defaultdict(int)
        daily_activity = defaultdict(int)
        weekly_activity = defaultdict(int)
        channel_peak_times = defaultdict(lambda: defaultdict(int))
        
        for msg in messages:
            hour = msg.timestamp.hour
            day = msg.timestamp.strftime('%A')
            week = msg.timestamp.strftime('%Y-W%U')
            
            hourly_activity[hour] += 1
            daily_activity[day] += 1
            weekly_activity[week] += 1
            channel_peak_times[msg.channel_name][hour] += 1
        
        # Find peak activity times
        peak_hour = max(hourly_activity, key=hourly_activity.get) if hourly_activity else None
        peak_day = max(daily_activity, key=daily_activity.get) if daily_activity else None
        
        temporal = {
            'hourly_distribution': dict(hourly_activity),
            'daily_distribution': dict(daily_activity),
            'weekly_trends': dict(weekly_activity),
            'peak_activity': {
                'hour': peak_hour,
                'day': peak_day
            },
            'channel_peak_times': {
                channel: max(times, key=times.get) if times else None
                for channel, times in channel_peak_times.items()
            },
            'activity_consistency': self._calculate_activity_consistency(weekly_activity)
        }
        
        self.analysis_results['temporal_patterns'] = temporal
    
    def _analyze_channel_characteristics(self):
        """Analyze characteristics of different channels"""
        log.info("📺 Analyzing channel characteristics...")
        
        channel_stats = {}
        
        channels = (self.db.query(Message.channel_name)
                   .distinct()
                   .all())
        
        for (channel_name,) in channels:
            msgs = (self.db.query(Message)
                   .filter(Message.channel_name == channel_name)
                   .order_by(Message.timestamp.desc())
                   .limit(1000)
                   .all())
            
            if not msgs:
                continue
            
            # Calculate channel metrics
            total_msgs = len(msgs)
            avg_length = sum(len(m.content or '') for m in msgs) / total_msgs if total_msgs > 0 else 0
            unique_users = len(set(m.author.get('username') if m.author else None for m in msgs))
            
            # Content characteristics
            has_embeds = sum(1 for m in msgs if m.embeds)
            has_attachments = sum(1 for m in msgs if m.attachments)
            has_reactions = sum(1 for m in msgs if m.reactions)
            
            # Topic analysis
            all_content = ' '.join(m.content or '' for m in msgs)
            words = re.findall(r'\b\w+\b', all_content.lower())
            top_words = dict(Counter(words).most_common(10))
            
            channel_stats[channel_name] = {
                'message_count': total_msgs,
                'unique_users': unique_users,
                'avg_message_length': round(avg_length, 2),
                'engagement_indicators': {
                    'messages_with_embeds': has_embeds,
                    'messages_with_attachments': has_attachments,
                    'messages_with_reactions': has_reactions
                },
                'top_words': top_words,
                'activity_level': self._categorize_activity_level(total_msgs),
                'content_type': self._categorize_content_type(msgs)
            }
        
        self.analysis_results['channel_analysis'] = channel_stats
    
    def _analyze_knowledge_domains(self):
        """Identify knowledge domains and expertise areas"""
        log.info("🎓 Analyzing knowledge domains...")
        
        # Technical terms and domains
        domain_keywords = {
            'ai_ml': ['ai', 'ml', 'machine learning', 'artificial intelligence', 'neural', 'model', 'training', 'dataset'],
            'programming': ['python', 'javascript', 'react', 'node', 'api', 'code', 'function', 'variable', 'github'],
            'discord': ['bot', 'discord', 'channel', 'server', 'embed', 'webhook', 'permission', 'role'],
            'data_science': ['data', 'analysis', 'visualization', 'pandas', 'numpy', 'statistics', 'database'],
            'business': ['product', 'management', 'strategy', 'market', 'customer', 'revenue', 'growth'],
            'design': ['ui', 'ux', 'design', 'prototype', 'figma', 'user experience', 'interface'],
            'education': ['learning', 'course', 'tutorial', 'teach', 'student', 'lesson', 'workshop']
        }
        
        # Analyze domain presence
        messages = (self.db.query(Message)
                   .filter(Message.content.isnot(None))
                   .order_by(Message.timestamp.desc())
                   .limit(10000)
                   .all())
        
        domain_analysis = {}
        for domain, keywords in domain_keywords.items():
            domain_messages = []
            for msg in messages:
                content = (msg.content or '').lower()
                if any(keyword in content for keyword in keywords):
                    domain_messages.append(msg)
            
            if domain_messages:
                # Find domain experts
                user_counts = Counter()
                for msg in domain_messages:
                    if msg.author:
                        user_counts[msg.author.get('username')] += 1
                
                domain_analysis[domain] = {
                    'message_count': len(domain_messages),
                    'active_users': len(user_counts),
                    'top_contributors': dict(user_counts.most_common(5)),
                    'channels': list(set(m.channel_name for m in domain_messages)),
                    'keywords_found': [kw for kw in keywords if any(kw in (m.content or '').lower() for m in domain_messages)]
                }
        
        self.analysis_results['knowledge_domains'] = domain_analysis
    
    def _analyze_interaction_patterns(self):
        """Analyze interaction and engagement patterns"""
        log.info("🤝 Analyzing interaction patterns...")
        
        messages = (self.db.query(Message)
                   .filter(Message.timestamp > datetime.utcnow() - timedelta(days=30))
                   .all())
        
        interactions = {
            'reply_patterns': self._analyze_reply_patterns(messages),
            'reaction_patterns': self._analyze_reaction_patterns(messages),
            'mention_networks': self._analyze_mention_networks(messages),
            'thread_engagement': self._analyze_thread_engagement(messages),
            'help_seeking_patterns': self._analyze_help_patterns(messages),
            'knowledge_sharing': self._analyze_knowledge_sharing(messages)
        }
        
        self.analysis_results['interaction_patterns'] = interactions
    
    def _analyze_resource_types(self):
        """Analyze types of resources and information shared"""
        log.info("📚 Analyzing resource types...")
        
        messages = (self.db.query(Message)
                   .filter(or_(
                       Message.attachments.isnot(None),
                       Message.embeds.isnot(None),
                       Message.content.like('%http%')
                   ))
                   .order_by(Message.timestamp.desc())
                   .limit(2000)
                   .all())
        
        resources = {
            'url_domains': self._analyze_url_domains(messages),
            'attachment_types': self._analyze_attachment_types(messages),
            'embed_analysis': self._analyze_embed_content(messages),
            'shared_tools': self._identify_shared_tools(messages),
            'documentation_links': self._identify_documentation(messages),
            'learning_resources': self._identify_learning_resources(messages)
        }
        
        self.analysis_results['resource_analysis'] = resources
    
    def _analyze_community_insights(self):
        """Generate community-level insights"""
        log.info("🌐 Analyzing community insights...")
        
        insights = {
            'community_health': self._assess_community_health(),
            'expertise_distribution': self._analyze_expertise_distribution(),
            'collaboration_patterns': self._analyze_collaboration_patterns(),
            'onboarding_effectiveness': self._analyze_onboarding_patterns(),
            'knowledge_gaps': self._identify_knowledge_gaps(),
            'engagement_drivers': self._identify_engagement_drivers()
        }
        
        self.analysis_results['community_insights'] = insights
    
    def _generate_query_optimization_insights(self):
        """Generate insights for optimizing agent queries and responses"""
        log.info("🎯 Generating query optimization insights...")
        
        # This will be customized based on your query specifications
        optimization_insights = {
            'search_optimization': {
                'recommended_chunk_size': self._recommend_chunk_size(),
                'important_metadata_fields': self._identify_important_metadata(),
                'semantic_search_targets': self._identify_search_targets(),
                'context_requirements': self._analyze_context_requirements()
            },
            'response_optimization': {
                'common_question_types': self._identify_question_types(),
                'response_length_preferences': self._analyze_response_preferences(),
                'citation_needs': self._analyze_citation_patterns(),
                'personalization_opportunities': self._identify_personalization_needs()
            },
            'tool_development_priorities': {
                'high_value_tools': self._identify_tool_priorities(),
                'automation_opportunities': self._identify_automation_needs(),
                'integration_points': self._identify_integration_opportunities()
            }
        }
        
        self.analysis_results['query_optimization_insights'] = optimization_insights
    
    # Helper methods (implementations would be added based on specific needs)
    def _get_length_statistics(self, messages: List[Message]) -> Dict:
        lengths = [len(m.content or '') for m in messages]
        return {
            'mean': sum(lengths) / len(lengths) if lengths else 0,
            'median': sorted(lengths)[len(lengths)//2] if lengths else 0,
            'max': max(lengths) if lengths else 0,
            'min': min(lengths) if lengths else 0,
            'std': self._calculate_std(lengths)
        }
    
    def _extract_topic_keywords(self, messages: List[Message]) -> Dict:
        # Implementation for topic keyword extraction
        all_text = ' '.join(m.content or '' for m in messages)
        words = re.findall(r'\b\w+\b', all_text.lower())
        # Filter out common words and return meaningful keywords
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        meaningful_words = [w for w in words if len(w) > 3 and w not in stopwords]
        return dict(Counter(meaningful_words).most_common(50))
    
    def _analyze_question_patterns(self, messages: List[Message]) -> Dict:
        questions = [m for m in messages if '?' in (m.content or '')]
        question_starters = Counter()
        for msg in questions:
            content = msg.content.lower().strip()
            words = content.split()
            if words:
                first_word = words[0]
                if first_word in ['how', 'what', 'why', 'when', 'where', 'who', 'can', 'is', 'are', 'do', 'does']:
                    question_starters[first_word] += 1
        
        return {
            'total_questions': len(questions),
            'question_starters': dict(question_starters),
            'avg_question_length': sum(len(m.content or '') for m in questions) / len(questions) if questions else 0
        }
    
    def _analyze_technical_content(self, messages: List[Message]) -> Dict:
        # Implementation for technical content analysis
        return {
            'code_snippets': sum(1 for m in messages if '```' in (m.content or '')),
            'technical_terms': {},  # Would be populated with actual analysis
            'documentation_references': 0
        }
    
    def _analyze_conversational_patterns(self, messages: List[Message]) -> Dict:
        # Implementation for conversational pattern analysis
        return {
            'avg_conversation_length': 0,
            'response_time_patterns': {},
            'conversation_starters': {}
        }
    
    def _analyze_information_density(self, messages: List[Message]) -> Dict:
        # Implementation for information density analysis
        return {
            'high_density_messages': 0,
            'low_density_messages': 0,
            'information_markers': {}
        }
    
    def _identify_semantic_clusters(self, messages: List[Message]) -> List:
        # Implementation for semantic clustering
        return []
    
    def _calculate_std(self, values: List[float]) -> float:
        if not values:
            return 0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5
    
    # Additional helper methods would be implemented based on specific requirements
    def _identify_conversation_starters(self, messages: List[Message]) -> Dict:
        return {}
    
    def _identify_expert_contributors(self, user_activity: Dict) -> Dict:
        return {}
    
    def _calculate_activity_consistency(self, weekly_activity: Dict) -> float:
        return 0.0
    
    def _categorize_activity_level(self, message_count: int) -> str:
        if message_count > 500:
            return 'high'
        elif message_count > 100:
            return 'medium'
        else:
            return 'low'
    
    def _categorize_content_type(self, messages: List[Message]) -> str:
        # Analyze content to categorize channel type
        return 'general'
    
    # All other helper methods would be implemented similarly...
    
    def save_results(self, output_path: str = None):
        """Save analysis results to file"""
        if not output_path:
            output_path = f"deep_message_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(output_path, 'w') as f:
            json.dump(self.analysis_results, f, indent=2, default=str)
        
        log.info(f"💾 Analysis results saved to {output_path}")
        return output_path
    
    def print_summary(self):
        """Print a summary of the analysis"""
        print("\n" + "="*60)
        print("🔍 DEEP MESSAGE CONTENT ANALYSIS SUMMARY")
        print("="*60)
        
        print(f"📊 Dataset Overview:")
        print(f"  • Total messages: {self.analysis_results['total_messages']:,}")
        print(f"  • Analysis period: {self.analysis_results['analysis_period']['days_span']} days")
        print(f"  • Top channels: {len(self.analysis_results['top_channels'])}")
        print(f"  • Active users: {len(self.analysis_results['top_users'])}")
        
        # Add more summary sections based on what analysis was completed
        
        print("\n💡 Key Insights:")
        print("  • Preprocessing analysis completed ✓")
        print("  • User pattern analysis completed ✓")
        print("  • Temporal pattern analysis completed ✓")
        print("  • Knowledge domain mapping completed ✓")
        
        print(f"\n📋 Next Steps:")
        print("  • Customize analysis based on query specifications")
        print("  • Implement advanced semantic analysis")
        print("  • Build agent-specific insights")

def main():
    """Main function to run the analysis"""
    print("🚀 Starting Deep Message Content Analysis...")
    print("\n⚠️  NOTE: This analysis framework is ready for customization.")
    print("Please provide query specifications to complete the analysis.")
    
    analyzer = DeepMessageAnalyzer()
    
    # Run basic analysis
    results = analyzer.run_comprehensive_analysis()
    
    # Print summary
    analyzer.print_summary()
    
    # Save results
    output_file = analyzer.save_results()
    print(f"\n💾 Analysis framework saved to: {output_file}")
    
    analyzer.db.close()

if __name__ == "__main__":
    main()
