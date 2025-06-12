#!/usr/bin/env python3
"""
Content Preprocessing Analysis Script

This script analyzes Discord message content to determine optimal preprocessing
strategies for embedding generation and semantic search.

Analysis includes:
- Content quality assessment
- Noise detection (bot messages, system messages, etc.)
- Language patterns and cleaning needs
- Content length distribution
- Special format detection (URLs, mentions, code blocks, etc.)
- Semantic value assessment
"""

import os
import sys
import re
import json
from datetime import datetime
from collections import Counter, defaultdict
from typing import Dict, List, Any, Optional

# Add project root to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from db import SessionLocal, Message
from utils.logger import setup_logging

setup_logging()
import logging
log = logging.getLogger(__name__)

class ContentPreprocessingAnalyzer:
    def __init__(self):
        self.db = SessionLocal()
        self.analysis_results = {
            'timestamp': datetime.utcnow().isoformat(),
            'total_messages': 0,
            'content_quality': {},
            'noise_patterns': {},
            'language_patterns': {},
            'content_types': {},
            'preprocessing_recommendations': []
        }
    
    def analyze_sample(self, sample_size: int = 1000) -> Dict[str, Any]:
        """Analyze a representative sample of messages for preprocessing insights"""
        log.info(f"🔍 Analyzing {sample_size} messages for preprocessing insights...")
        
        # Get stratified sample across different channels and time periods
        messages = (self.db.query(Message)
                   .order_by(Message.timestamp.desc())
                   .limit(sample_size)
                   .all())
        
        self.analysis_results['total_messages'] = len(messages)
        
        # Analyze different aspects
        self._analyze_content_quality(messages)
        self._analyze_noise_patterns(messages)
        self._analyze_language_patterns(messages)
        self._analyze_content_types(messages)
        self._analyze_special_formats(messages)
        self._generate_preprocessing_recommendations()
        
        return self.analysis_results
    
    def _analyze_content_quality(self, messages: List[Message]):
        """Analyze content quality metrics"""
        log.info("📊 Analyzing content quality...")
        
        quality_metrics = {
            'empty_content': 0,
            'very_short': 0,  # < 10 chars
            'short': 0,       # 10-50 chars
            'medium': 0,      # 50-200 chars
            'long': 0,        # 200-500 chars
            'very_long': 0,   # > 500 chars
            'has_substance': 0,  # meaningful content
            'mostly_symbols': 0,
            'mostly_mentions': 0,
            'content_length_dist': []
        }
        
        for msg in messages:
            content = msg.content or ""
            clean_content = msg.clean_content or ""
            length = len(content)
            clean_length = len(clean_content)
            
            quality_metrics['content_length_dist'].append(length)
            
            if length == 0:
                quality_metrics['empty_content'] += 1
            elif length < 10:
                quality_metrics['very_short'] += 1
            elif length < 50:
                quality_metrics['short'] += 1
            elif length < 200:
                quality_metrics['medium'] += 1
            elif length < 500:
                quality_metrics['long'] += 1
            else:
                quality_metrics['very_long'] += 1
            
            # Check for substance
            if self._has_semantic_substance(content, clean_content):
                quality_metrics['has_substance'] += 1
            
            # Check for symbol-heavy content
            if self._is_mostly_symbols(content):
                quality_metrics['mostly_symbols'] += 1
            
            # Check for mention-heavy content
            if self._is_mostly_mentions(content, msg.mention_ids):
                quality_metrics['mostly_mentions'] += 1
        
        self.analysis_results['content_quality'] = quality_metrics
    
    def _analyze_noise_patterns(self, messages: List[Message]):
        """Identify common noise patterns that should be filtered or cleaned"""
        log.info("🔇 Analyzing noise patterns...")
        
        noise_patterns = {
            'bot_messages': 0,
            'system_messages': 0,
            'webhook_messages': 0,
            'auto_moderation': 0,
            'join_leave_messages': 0,
            'emoji_only': 0,
            'repeated_content': Counter(),
            'spam_indicators': 0,
            'thread_starters': 0,
            'poll_messages': 0
        }
        
        for msg in messages:
            # Bot detection
            if msg.author and isinstance(msg.author, dict) and msg.author.get('bot', False):
                noise_patterns['bot_messages'] += 1
            
            # System messages
            if msg.type and 'system' in str(msg.type).lower():
                noise_patterns['system_messages'] += 1
            
            # Webhook messages
            if msg.webhook_id:
                noise_patterns['webhook_messages'] += 1
            
            # Thread-related
            if msg.thread:
                noise_patterns['thread_starters'] += 1
            
            # Poll messages
            if msg.poll:
                noise_patterns['poll_messages'] += 1
            
            # Emoji-only messages
            if self._is_emoji_only(msg.content):
                noise_patterns['emoji_only'] += 1
            
            # Repeated content detection
            content_hash = self._get_content_hash(msg.content)
            if content_hash:
                noise_patterns['repeated_content'][content_hash] += 1
            
            # Spam indicators
            if self._has_spam_indicators(msg):
                noise_patterns['spam_indicators'] += 1
        
        # Convert repeated content to readable format
        noise_patterns['repeated_content'] = dict(
            noise_patterns['repeated_content'].most_common(10)
        )
        
        self.analysis_results['noise_patterns'] = noise_patterns
    
    def _analyze_language_patterns(self, messages: List[Message]):
        """Analyze language patterns for text processing"""
        log.info("🌐 Analyzing language patterns...")
        
        language_patterns = {
            'languages_detected': Counter(),
            'code_blocks': 0,
            'inline_code': 0,
            'urls': 0,
            'mentions': 0,
            'hashtags': 0,
            'special_chars': Counter(),
            'common_words': Counter(),
            'technical_terms': Counter()
        }
        
        # Technical terms to look for
        tech_terms = ['ai', 'ml', 'api', 'bot', 'discord', 'python', 'javascript', 
                     'react', 'node', 'database', 'server', 'client', 'token',
                     'embed', 'webhook', 'oauth', 'json', 'rest', 'graphql']
        
        for msg in messages:
            content = msg.content or ""
            
            # Code block detection
            if '```' in content:
                language_patterns['code_blocks'] += 1
            if '`' in content and '```' not in content:
                language_patterns['inline_code'] += 1
            
            # URL detection
            url_pattern = r'https?://[^\s]+'
            urls = re.findall(url_pattern, content)
            language_patterns['urls'] += len(urls)
            
            # Mention counting
            language_patterns['mentions'] += len(msg.mention_ids or [])
            
            # Hashtag detection
            hashtags = re.findall(r'#\w+', content)
            language_patterns['hashtags'] += len(hashtags)
            
            # Word analysis
            words = re.findall(r'\b\w+\b', content.lower())
            language_patterns['common_words'].update(words)
            
            # Technical term detection
            for term in tech_terms:
                if term in content.lower():
                    language_patterns['technical_terms'][term] += 1
        
        # Convert to readable format
        language_patterns['common_words'] = dict(
            language_patterns['common_words'].most_common(20)
        )
        language_patterns['technical_terms'] = dict(
            language_patterns['technical_terms'].most_common()
        )
        
        self.analysis_results['language_patterns'] = language_patterns
    
    def _analyze_content_types(self, messages: List[Message]):
        """Analyze different types of content for processing strategies"""
        log.info("📝 Analyzing content types...")
        
        content_types = {
            'text_only': 0,
            'with_embeds': 0,
            'with_attachments': 0,
            'with_reactions': 0,
            'replies': 0,
            'thread_messages': 0,
            'polls': 0,
            'rich_content': 0,
            'embed_types': Counter(),
            'attachment_types': Counter()
        }
        
        for msg in messages:
            # Basic content type classification
            if not msg.embeds and not msg.attachments:
                content_types['text_only'] += 1
            
            if msg.embeds:
                content_types['with_embeds'] += 1
                # Analyze embed types
                for embed in msg.embeds or []:
                    embed_type = embed.get('type', 'unknown')
                    content_types['embed_types'][embed_type] += 1
            
            if msg.attachments:
                content_types['with_attachments'] += 1
                # Analyze attachment types
                for att in msg.attachments or []:
                    content_type = att.get('content_type', 'unknown')
                    content_types['attachment_types'][content_type] += 1
            
            if msg.reactions:
                content_types['with_reactions'] += 1
            
            if msg.reference:
                content_types['replies'] += 1
            
            if msg.thread:
                content_types['thread_messages'] += 1
            
            if msg.poll:
                content_types['polls'] += 1
            
            # Rich content indicator
            if (msg.embeds or msg.attachments or msg.reactions or 
                msg.poll or msg.thread):
                content_types['rich_content'] += 1
        
        # Convert counters to dicts
        content_types['embed_types'] = dict(content_types['embed_types'])
        content_types['attachment_types'] = dict(content_types['attachment_types'])
        
        self.analysis_results['content_types'] = content_types
    
    def _analyze_special_formats(self, messages: List[Message]):
        """Analyze special formatting that needs preprocessing"""
        log.info("✨ Analyzing special formats...")
        
        special_formats = {
            'discord_formatting': {
                'bold': 0,
                'italic': 0,
                'underline': 0,
                'strikethrough': 0,
                'spoiler': 0,
                'code': 0
            },
            'user_mentions': 0,
            'channel_mentions': 0,
            'role_mentions': 0,
            'custom_emojis': 0,
            'unicode_emojis': 0,
            'timestamps': 0,
            'line_breaks': 0
        }
        
        for msg in messages:
            content = msg.content or ""
            
            # Discord formatting
            special_formats['discord_formatting']['bold'] += content.count('**')
            special_formats['discord_formatting']['italic'] += content.count('*')
            special_formats['discord_formatting']['underline'] += content.count('__')
            special_formats['discord_formatting']['strikethrough'] += content.count('~~')
            special_formats['discord_formatting']['spoiler'] += content.count('||')
            special_formats['discord_formatting']['code'] += content.count('`')
            
            # Mentions
            special_formats['user_mentions'] += content.count('<@')
            special_formats['channel_mentions'] += content.count('<#')
            special_formats['role_mentions'] += content.count('<@&')
            
            # Emojis
            special_formats['custom_emojis'] += len(re.findall(r'<:[^:]+:\d+>', content))
            # Rough unicode emoji detection
            special_formats['unicode_emojis'] += len(re.findall(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]', content))
            
            # Timestamps
            special_formats['timestamps'] += content.count('<t:')
            
            # Line breaks
            special_formats['line_breaks'] += content.count('\n')
        
        self.analysis_results['special_formats'] = special_formats
    
    def _generate_preprocessing_recommendations(self):
        """Generate preprocessing recommendations based on analysis"""
        log.info("💡 Generating preprocessing recommendations...")
        
        recommendations = []
        
        # Content quality recommendations
        quality = self.analysis_results['content_quality']
        if quality['empty_content'] > quality['has_substance']:
            recommendations.append({
                'category': 'filtering',
                'recommendation': 'Filter out empty messages',
                'reason': f"{quality['empty_content']} empty messages found"
            })
        
        if quality['very_short'] > len(self.analysis_results) * 0.3:
            recommendations.append({
                'category': 'filtering',
                'recommendation': 'Consider filtering very short messages (< 10 chars)',
                'reason': f"{quality['very_short']} very short messages that may lack semantic value"
            })
        
        # Noise pattern recommendations
        noise = self.analysis_results['noise_patterns']
        if noise['bot_messages'] > 0:
            recommendations.append({
                'category': 'filtering',
                'recommendation': 'Filter bot messages for human-focused search',
                'reason': f"{noise['bot_messages']} bot messages detected"
            })
        
        if noise['emoji_only'] > len(self.analysis_results) * 0.1:
            recommendations.append({
                'category': 'filtering',
                'recommendation': 'Filter emoji-only messages',
                'reason': f"{noise['emoji_only']} emoji-only messages with limited semantic value"
            })
        
        # Language pattern recommendations
        lang = self.analysis_results['language_patterns']
        if lang['code_blocks'] > 0:
            recommendations.append({
                'category': 'processing',
                'recommendation': 'Extract and separately index code blocks',
                'reason': f"{lang['code_blocks']} messages with code blocks that need special handling"
            })
        
        if lang['urls'] > 0:
            recommendations.append({
                'category': 'processing',
                'recommendation': 'Extract and normalize URLs',
                'reason': f"{lang['urls']} URLs found that could be expanded or normalized"
            })
        
        # Content type recommendations
        content_types = self.analysis_results['content_types']
        if content_types['with_embeds'] > 0:
            recommendations.append({
                'category': 'enrichment',
                'recommendation': 'Include embed content in searchable text',
                'reason': f"{content_types['with_embeds']} messages with embeds containing valuable content"
            })
        
        if content_types['replies'] > 0:
            recommendations.append({
                'category': 'context',
                'recommendation': 'Include reply context in processing',
                'reason': f"{content_types['replies']} reply messages that benefit from parent context"
            })
        
        # Special format recommendations
        if 'special_formats' in self.analysis_results:
            special = self.analysis_results['special_formats']
            if special['user_mentions'] > 0:
                recommendations.append({
                    'category': 'processing',
                    'recommendation': 'Normalize user mentions to readable names',
                    'reason': f"{special['user_mentions']} user mentions that should be human-readable"
                })
        
        self.analysis_results['preprocessing_recommendations'] = recommendations
    
    def _has_semantic_substance(self, content: str, clean_content: str) -> bool:
        """Check if content has semantic substance worth indexing"""
        if not content:
            return False
        
        # Remove common noise
        text = clean_content or content
        text = re.sub(r'<[^>]+>', '', text)  # Remove tags
        text = re.sub(r'[^\w\s]', ' ', text)  # Remove special chars
        text = ' '.join(text.split())  # Normalize whitespace
        
        # Check for meaningful content
        words = text.split()
        if len(words) < 2:
            return False
        
        # Check for non-trivial words
        trivial_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'ok', 'yes', 'no', 'hi', 'hello', 'thanks', 'thank', 'you'}
        meaningful_words = [w for w in words if w.lower() not in trivial_words and len(w) > 2]
        
        return len(meaningful_words) >= 1
    
    def _is_mostly_symbols(self, content: str) -> bool:
        """Check if content is mostly symbols/punctuation"""
        if not content:
            return False
        
        alpha_chars = sum(1 for c in content if c.isalpha())
        total_chars = len(content)
        
        return alpha_chars / total_chars < 0.3 if total_chars > 0 else True
    
    def _is_mostly_mentions(self, content: str, mentions: List) -> bool:
        """Check if content is mostly mentions"""
        if not content:
            return False
        
        mention_chars = content.count('<@') * 20  # Rough estimate
        return mention_chars / len(content) > 0.7 if content else False
    
    def _is_emoji_only(self, content: str) -> bool:
        """Check if content is only emojis"""
        if not content:
            return False
        
        # Remove whitespace and common emoji patterns
        clean = re.sub(r'\s+', '', content)
        clean = re.sub(r'<:[^:]+:\d+>', '', clean)  # Custom emojis
        clean = re.sub(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]', '', clean)  # Unicode emojis
        
        return len(clean) == 0
    
    def _get_content_hash(self, content: str) -> Optional[str]:
        """Get a hash for content deduplication"""
        if not content or len(content) < 10:
            return None
        
        # Normalize content for comparison
        normalized = re.sub(r'\s+', ' ', content.lower().strip())
        return hash(normalized) if len(normalized) > 5 else None
    
    def _has_spam_indicators(self, msg: Message) -> bool:
        """Check for spam indicators"""
        content = msg.content or ""
        
        # Simple spam heuristics
        spam_indicators = [
            len(content) > 1000 and content.count('http') > 3,  # Many URLs
            content.count('!') > 10,  # Excessive exclamation
            content.count(content[0]) > len(content) * 0.5 if content else False,  # Repeated chars
            len(set(content.lower())) < 5 if len(content) > 20 else False  # Low char diversity
        ]
        
        return any(spam_indicators)
    
    def save_results(self, output_path: str = None):
        """Save analysis results to file"""
        if not output_path:
            output_path = f"content_preprocessing_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(output_path, 'w') as f:
            json.dump(self.analysis_results, f, indent=2, default=str)
        
        log.info(f"💾 Analysis results saved to {output_path}")
        return output_path
    
    def print_summary(self):
        """Print a summary of the analysis"""
        print("\n" + "="*60)
        print("📊 CONTENT PREPROCESSING ANALYSIS SUMMARY")
        print("="*60)
        
        total = self.analysis_results['total_messages']
        print(f"📨 Total messages analyzed: {total}")
        
        # Content quality
        quality = self.analysis_results['content_quality']
        print(f"\n📝 Content Quality:")
        print(f"  • Empty content: {quality['empty_content']} ({quality['empty_content']/total*100:.1f}%)")
        print(f"  • Has substance: {quality['has_substance']} ({quality['has_substance']/total*100:.1f}%)")
        print(f"  • Very short (<10 chars): {quality['very_short']} ({quality['very_short']/total*100:.1f}%)")
        
        # Noise patterns
        noise = self.analysis_results['noise_patterns']
        print(f"\n🔇 Noise Patterns:")
        print(f"  • Bot messages: {noise['bot_messages']}")
        print(f"  • System messages: {noise['system_messages']}")
        print(f"  • Emoji-only: {noise['emoji_only']}")
        print(f"  • Poll messages: {noise['poll_messages']}")
        
        # Content types
        content_types = self.analysis_results['content_types']
        print(f"\n📋 Content Types:")
        print(f"  • Text only: {content_types['text_only']}")
        print(f"  • With embeds: {content_types['with_embeds']}")
        print(f"  • With attachments: {content_types['with_attachments']}")
        print(f"  • Rich content: {content_types['rich_content']}")
        
        # Recommendations
        recommendations = self.analysis_results['preprocessing_recommendations']
        print(f"\n💡 Key Recommendations ({len(recommendations)}):")
        for i, rec in enumerate(recommendations[:5], 1):
            print(f"  {i}. [{rec['category'].upper()}] {rec['recommendation']}")
            print(f"     Reason: {rec['reason']}")
        
        if len(recommendations) > 5:
            print(f"  ... and {len(recommendations) - 5} more recommendations")

def main():
    """Main function to run the analysis"""
    analyzer = ContentPreprocessingAnalyzer()
    
    # Analyze sample
    results = analyzer.analyze_sample(sample_size=2000)
    
    # Print summary
    analyzer.print_summary()
    
    # Save results
    output_file = analyzer.save_results()
    print(f"\n💾 Full analysis saved to: {output_file}")
    
    analyzer.db.close()

if __name__ == "__main__":
    main()
