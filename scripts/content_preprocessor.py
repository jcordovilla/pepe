#!/usr/bin/env python3
"""
Content Preprocessor for Discord Messages

This module implements preprocessing pipeline based on content analysis recommendations
to optimize messages for embedding and search.

Key preprocessing steps:
1. Filter very short messages (<10 chars)
2. Extract and normalize URLs
3. Include embed content in searchable text
4. Include reply context
5. Clean and normalize content
"""

import os
import sys
import json
import re
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass

# Add project root to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from db import SessionLocal, Message
from utils.logger import setup_logging

setup_logging()
import logging
log = logging.getLogger(__name__)

@dataclass
class PreprocessingConfig:
    """Configuration for preprocessing pipeline"""
    min_content_length: int = 10
    include_embed_content: bool = True
    include_reply_context: bool = True
    normalize_urls: bool = True
    filter_bot_messages: bool = True
    max_embed_fields_per_message: int = 10
    max_reply_context_length: int = 200

class ContentPreprocessor:
    """Main preprocessing class for Discord messages"""
    
    def __init__(self, config: PreprocessingConfig = None):
        self.config = config or PreprocessingConfig()
        self.url_pattern = re.compile(r'https?://[^\s<>"]+')
        self.mention_pattern = re.compile(r'<@[!&]?\d+>')
        self.channel_pattern = re.compile(r'<#\d+>')
        self.emoji_pattern = re.compile(r'<a?:\w+:\d+>')
        
    def extract_embed_content(self, embeds_data: List[Dict]) -> str:
        """Extract searchable text from embed data"""
        if not embeds_data or not self.config.include_embed_content:
            return ""
            
        content_parts = []
        
        for embed in embeds_data[:self.config.max_embed_fields_per_message]:
            # Extract title and description
            if embed.get('title'):
                content_parts.append(f"EMBED_TITLE: {embed['title']}")
            if embed.get('description'):
                content_parts.append(f"EMBED_DESC: {embed['description']}")
                
            # Extract author info
            if embed.get('author', {}).get('name'):
                content_parts.append(f"EMBED_AUTHOR: {embed['author']['name']}")
                
            # Extract footer text
            if embed.get('footer', {}).get('text'):
                content_parts.append(f"EMBED_FOOTER: {embed['footer']['text']}")
                
            # Extract field content
            if embed.get('fields'):
                for field in embed['fields']:
                    if field.get('name') and field.get('value'):
                        content_parts.append(f"EMBED_FIELD: {field['name']}: {field['value']}")
        
        return " ".join(content_parts)
    
    def extract_attachment_metadata(self, attachments_data: List[Dict]) -> str:
        """Extract searchable metadata from attachments"""
        if not attachments_data:
            return ""
            
        content_parts = []
        for att in attachments_data:
            if att.get('filename'):
                # Extract file extension and type info
                filename = att['filename']
                content_parts.append(f"ATTACHMENT: {filename}")
                
                # Add content type if available
                if att.get('content_type'):
                    content_parts.append(f"FILETYPE: {att['content_type']}")
        
        return " ".join(content_parts)
    
    def normalize_urls(self, content: str) -> Tuple[str, List[str]]:
        """Extract and normalize URLs from content"""
        urls = self.url_pattern.findall(content)
        
        if self.config.normalize_urls and urls:
            # Replace URLs with normalized tokens
            normalized_content = self.url_pattern.sub('[URL]', content)
            return normalized_content, urls
        
        return content, urls
    
    def clean_discord_formatting(self, content: str) -> str:
        """Clean Discord-specific formatting elements"""
        # Replace mentions with readable text
        content = self.mention_pattern.sub('[USER_MENTION]', content)
        content = self.channel_pattern.sub('[CHANNEL_MENTION]', content)
        content = self.emoji_pattern.sub('[CUSTOM_EMOJI]', content)
        
        # Clean up extra whitespace
        content = re.sub(r'\s+', ' ', content).strip()
        
        return content
    
    def get_reply_context(self, message: Message, db_session) -> str:
        """Get context from replied-to message"""
        if not self.config.include_reply_context or not message.reference:
            return ""
            
        try:
            ref_data = message.reference
            if not ref_data or not ref_data.get('message_id'):
                return ""
                
            # Find the referenced message
            ref_message = db_session.query(Message).filter_by(
                message_id=int(ref_data['message_id'])
            ).first()
            
            if ref_message and ref_message.content:
                # Truncate reply context if too long
                reply_content = ref_message.content[:self.config.max_reply_context_length]
                if len(ref_message.content) > self.config.max_reply_context_length:
                    reply_content += "..."
                return f"REPLY_TO: {reply_content}"
                
        except Exception as e:
            log.warning(f"Error getting reply context for message {message.message_id}: {e}")
            
        return ""
    
    def should_filter_message(self, message: Message) -> Tuple[bool, str]:
        """Determine if message should be filtered out"""
        # Filter very short content
        if len(message.content or "") < self.config.min_content_length:
            if not (message.embeds or message.attachments or message.stickers):
                return True, "too_short"
        # Filter specific calendar bot "sesh" (based on database analysis)
        author = message.author or {}
        # Filter bot messages using the Discord API property if available
        if self.config.filter_bot_messages:
            if (hasattr(message, 'author') and hasattr(message.author, 'bot') and message.author.bot) or \
               (isinstance(author, dict) and author.get('bot', False)):
                return True, "bot_message"
            if author.get('username') == 'sesh':
                return True, "calendar_bot"
            if author.get('username', '').endswith('Bot') or author.get('discriminator') == '0000':
                return True, "bot_message"
        return False, ""
    
    def preprocess_message(self, message: Message, db_session) -> Optional[Dict]:
        """Preprocess a single message for embedding/search"""
        # Check if message should be filtered
        should_filter, filter_reason = self.should_filter_message(message)
        if should_filter:
            return None
            
        # Start with original content
        base_content = message.content or ""
        
        # Clean Discord formatting
        cleaned_content = self.clean_discord_formatting(base_content)
        
        # Normalize URLs
        normalized_content, extracted_urls = self.normalize_urls(cleaned_content)
        
        # Extract embed content
        embed_content = self.extract_embed_content(message.embeds or [])
        
        # Extract attachment metadata
        attachment_content = self.extract_attachment_metadata(message.attachments or [])
        
        # Get reply context
        reply_context = self.get_reply_context(message, db_session)
        
        # Combine all content for searchable text
        searchable_parts = [normalized_content]
        if embed_content:
            searchable_parts.append(embed_content)
        if attachment_content:
            searchable_parts.append(attachment_content)
        if reply_context:
            searchable_parts.append(reply_context)
            
        searchable_text = " ".join(part for part in searchable_parts if part.strip())
        
        # Create preprocessed result
        result = {
            'message_id': message.message_id,
            'original_content': base_content,
            'cleaned_content': cleaned_content,
            'searchable_text': searchable_text,
            'extracted_urls': extracted_urls,
            'has_embeds': bool(message.embeds),
            'has_attachments': bool(message.attachments),
            'has_reply_context': bool(reply_context),
            'content_length': len(searchable_text),
            'timestamp': message.timestamp,
            'channel_id': message.channel_id,
            'guild_id': message.guild_id,
            'author_id': message.author.get('id') if message.author else None,
            'message_type': message.type,
            'is_pinned': message.pinned,
            'reaction_count': len(message.reactions or []),
        }
        
        return result
    
    def preprocess_batch(self, limit: int = None, offset: int = 0) -> List[Dict]:
        """Preprocess a batch of messages"""
        db = SessionLocal()
        try:
            query = db.query(Message).order_by(Message.timestamp.desc())
            if limit:
                query = query.limit(limit).offset(offset)
                
            messages = query.all()
            log.info(f"Preprocessing {len(messages)} messages (offset: {offset})")
            
            preprocessed = []
            filtered_count = 0
            
            for message in messages:
                result = self.preprocess_message(message, db)
                if result:
                    preprocessed.append(result)
                else:
                    filtered_count += 1
                    
            log.info(f"Preprocessed {len(preprocessed)} messages, filtered {filtered_count}")
            return preprocessed
            
        finally:
            db.close()
    
    def generate_preprocessing_report(self, sample_size: int = 1000) -> Dict:
        """Generate a report on preprocessing results"""
        db = SessionLocal()
        try:
            messages = db.query(Message).limit(sample_size).all()
            
            total_messages = len(messages)
            filtered_messages = 0
            has_embeds = 0
            has_attachments = 0
            has_replies = 0
            total_urls = 0
            
            content_lengths = []
            
            for message in messages:
                result = self.preprocess_message(message, db)
                if result is None:
                    filtered_messages += 1
                    continue
                    
                content_lengths.append(result['content_length'])
                if result['has_embeds']:
                    has_embeds += 1
                if result['has_attachments']:
                    has_attachments += 1
                if result['has_reply_context']:
                    has_replies += 1
                total_urls += len(result['extracted_urls'])
            
            processed_count = total_messages - filtered_messages
            
            report = {
                'sample_size': total_messages,
                'processed_messages': processed_count,
                'filtered_messages': filtered_messages,
                'filter_rate': filtered_messages / total_messages if total_messages > 0 else 0,
                'messages_with_embeds': has_embeds,
                'messages_with_attachments': has_attachments,
                'messages_with_reply_context': has_replies,
                'total_urls_extracted': total_urls,
                'avg_content_length': sum(content_lengths) / len(content_lengths) if content_lengths else 0,
                'max_content_length': max(content_lengths) if content_lengths else 0,
                'min_content_length': min(content_lengths) if content_lengths else 0,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            return report
            
        finally:
            db.close()

def main():
    """Main function for testing preprocessing"""
    preprocessor = ContentPreprocessor()
    
    # Generate preprocessing report
    log.info("Generating preprocessing report...")
    report = preprocessor.generate_preprocessing_report(sample_size=1000)
    
    # Save report
    report_path = f"preprocessing_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, 'w') as f:
        json.dump(report, indent=2, fp=f)
    
    log.info(f"Preprocessing report saved to: {report_path}")
    log.info(f"Filter rate: {report['filter_rate']:.1%}")
    log.info(f"Avg content length: {report['avg_content_length']:.0f} chars")
    log.info(f"Messages with embeds: {report['messages_with_embeds']}")
    log.info(f"Messages with attachments: {report['messages_with_attachments']}")
    log.info(f"Total URLs extracted: {report['total_urls_extracted']}")

if __name__ == "__main__":
    main()
