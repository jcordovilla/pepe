"""
Enhanced Content Processing Service
Modernized content classification and processing with AI integration
"""

import re
import json
import logging
from typing import Dict, List, Any, Optional
from urllib.parse import urlparse
from datetime import datetime

from openai import OpenAI

logger = logging.getLogger(__name__)

class ContentProcessingService:
    """
    Modern content processing with legacy-proven classification patterns
    
    Preserves battle-tested:
    - URL analysis and filtering
    - Code detection patterns
    - Resource classification rules
    - Attachment processing logic
    """
    
    def __init__(self, openai_client: OpenAI):
        self.openai_client = openai_client
        
        # Legacy-proven patterns
        self.url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )
        
        self.code_pattern = re.compile(r'```[\s\S]*?```|`[^`]*`')
        
        # Legacy noise filtering
        self.noise_domains = {
            'tenor.com', 'giphy.com', 'discord.com', 'cdn.discordapp.com',
            'imgur.com', 'zoom.us', 'meet.google.com', 'teams.microsoft.com'
        }
        
        logger.info("🔍 Content processor initialized with legacy patterns")
    
    async def analyze_message_content(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Comprehensive message content analysis
        Enhanced from legacy detection patterns
        """
        content = message.get('content', '')
        attachments = message.get('attachments', [])
        
        analysis = {
            "message_id": message.get('message_id'),
            "content_types": [],
            "resources": [],
            "code_snippets": [],
            "urls": [],
            "attachments_processed": [],
            "classifications": []
        }
        
        if not content and not attachments:
            return analysis
        
        # Legacy URL extraction and analysis
        urls = self.url_pattern.findall(content)
        for url in urls:
            url_analysis = await self._analyze_url(url, message)
            if url_analysis:
                analysis["urls"].append(url_analysis)
        
        # Legacy code detection
        code_blocks = self.code_pattern.findall(content)
        for code in code_blocks:
            code_analysis = await self._analyze_code_snippet(code, message)
            if code_analysis:
                analysis["code_snippets"].append(code_analysis)
        
        # Legacy attachment processing
        for attachment in attachments:
            attachment_analysis = await self._analyze_attachment(attachment, message)
            if attachment_analysis:
                analysis["attachments_processed"].append(attachment_analysis)
        
        # Enhanced AI-powered classification (modern addition)
        if content:
            classifications = await self._classify_content_ai(content)
            analysis["classifications"] = classifications
        
        return analysis
    
    async def _analyze_url(self, url: str, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        URL analysis with legacy filtering patterns
        """
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            
            # Legacy noise filtering
            if domain in self.noise_domains:
                return None
            
            # Legacy-style resource classification
            resource_type = "unknown"
            if any(ext in url.lower() for ext in ['.pdf', '.doc', '.docx']):
                resource_type = "document"
            elif any(ext in url.lower() for ext in ['.py', '.js', '.html', '.css']):
                resource_type = "code"
            elif any(domain_part in domain for domain_part in ['github.com', 'stackoverflow.com']):
                resource_type = "development"
            elif any(domain_part in domain for domain_part in ['youtube.com', 'vimeo.com']):
                resource_type = "video"
            
            return {
                "url": url,
                "domain": domain,
                "type": resource_type,
                "message_id": message.get('message_id'),
                "timestamp": message.get('timestamp'),
                "author": message.get('author', {}).get('username')
            }
            
        except Exception as e:
            logger.debug(f"Error analyzing URL {url}: {e}")
            return None
    
    async def _analyze_code_snippet(self, code: str, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Code snippet analysis with legacy patterns
        """
        # Legacy language detection
        language = "unknown"
        if code.startswith("```"):
            first_line = code.split('\n')[0]
            if len(first_line) > 3:
                language = first_line[3:].strip()
        
        return {
            "code": code,
            "language": language,
            "length": len(code),
            "message_id": message.get('message_id'),
            "channel_name": message.get('channel_name'),
            "author": message.get('author', {}).get('username'),
            "timestamp": message.get('timestamp')
        }
    
    async def _analyze_attachment(self, attachment: Dict[str, Any], message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Attachment analysis with legacy patterns
        """
        return {
            "filename": attachment.get('filename'),
            "size": attachment.get('size'),
            "url": attachment.get('url'),
            "content_type": attachment.get('content_type'),
            "message_id": message.get('message_id'),
            "channel_name": message.get('channel_name'),
            "author": message.get('author', {}).get('username'),
            "timestamp": message.get('timestamp')
        }
    
    async def _classify_content_ai(self, content: str) -> List[str]:
        """
        Enhanced AI-powered content classification (modern addition)
        """
        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": "Classify the following Discord message content. Return a JSON array of classification tags like ['question', 'code_help', 'announcement', 'discussion', 'resource_sharing', 'meme', 'technical']."
                    },
                    {
                        "role": "user", 
                        "content": content[:1000]  # Limit for cost efficiency
                    }
                ],
                max_tokens=100,
                temperature=0.1
            )
            
            result = response.choices[0].message.content
            return json.loads(result)
            
        except Exception as e:
            logger.debug(f"AI classification failed: {e}")
            return ["unclassified"]
