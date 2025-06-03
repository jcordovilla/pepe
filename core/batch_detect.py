#!/usr/bin/env python3
"""
Batch Resource Detection

Analyzes stored Discord messages to detect and classify resources like links,
files, code snippets, and other valuable content for enhanced searchability.
"""

import os
import sys
import json
import logging
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse
import asyncio

from openai import OpenAI

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
PROCESSED_MARKER_DIR = Path("data/processing_markers")
RESOURCES_OUTPUT_DIR = Path("data/detected_resources")
FETCHED_MESSAGES_DIR = Path("data/fetched_messages")

# Ensure directories exist
RESOURCES_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class ResourceDetector:
    """
    Detects and classifies resources in Discord messages using AI.
    
    Uses OpenAI to intelligently analyze and filter resources:
    - Valuable content: articles, papers, tools, tutorials, documentation
    - Filtered out: GIFs, memes, internal Discord links, temporary meeting links
    """
    
    def __init__(self):
        self.stats = {
            "start_time": datetime.utcnow().isoformat(),
            "resources_detected": 0,
            "links_found": 0,
            "links_analyzed": 0,
            "links_filtered": 0,
            "code_snippets": 0,
            "attachments": 0,
            "domains": {},
            "filtered_domains": {},
            "errors": []
        }
        self.detected_resources = []
        
        # Initialize OpenAI client
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # URL regex pattern
        self.url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )
        
        # Code block pattern (markdown)
        self.code_pattern = re.compile(r'```[\s\S]*?```|`[^`]*`')
        
        # Pre-filter obvious noise domains (no need to send to LLM)
        self.noise_domains = {
            'tenor.com',          # GIFs
            'giphy.com',          # GIFs  
            'discord.com',        # Internal Discord links
            'cdn.discordapp.com', # Discord attachments (handled separately)
            'imgur.com',          # Usually memes/images
            'reddit.com',         # Can be valuable but often discussion
            'zoom.us',            # Temporary meeting links
            'meet.google.com',    # Temporary meeting links
            'teams.microsoft.com' # Temporary meeting links
        }
        
        logger.info("Intelligent resource detector initialized with OpenAI")
    
    async def detect_resources(self):
        """Main resource detection process"""
        logger.info("🔍 Starting resource detection process...")
        
        # Get all message files
        message_files = list(FETCHED_MESSAGES_DIR.glob("*_messages.json"))
        logger.info(f"📂 Found {len(message_files)} message files to process")
        
        for message_file in message_files:
            try:
                await self.process_message_file(message_file)
            except Exception as e:
                error_msg = f"Error processing {message_file}: {e}"
                logger.error(f"❌ {error_msg}")
                self.stats["errors"].append(error_msg)
        
        # Save detected resources
        if self.detected_resources:
            await self.save_resources()
        
        # Save stats
        await self.save_stats()
        
        logger.info(f"✅ Resource detection completed!")
        logger.info(f"📊 Found {self.stats['resources_detected']} valuable resources from {self.stats['links_found']} total links")
        logger.info(f"🚫 Filtered {self.stats['links_filtered']} noise links")
        logger.info(f"🔍 AI analyzed {self.stats['links_analyzed']} URLs")
    
    async def process_message_file(self, file_path: Path):
        """Process a single message file for resource detection"""
        logger.info(f"🔍 Processing {file_path.name}...")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            messages = data.get('messages', [])
            logger.info(f"   📝 Processing {len(messages)} messages...")
            
            for message in messages:
                await self.analyze_message(message)
                
        except Exception as e:
            raise Exception(f"Failed to process {file_path}: {e}")
    
    async def analyze_message(self, message: Dict[str, Any]):
        """Analyze a single message for resources"""
        content = message.get('content', '')
        attachments = message.get('attachments', [])
        
        if not content and not attachments:
            return
        
        # Extract URLs from content
        urls = self.url_pattern.findall(content)
        for url in urls:
            await self.process_url(url, message)
        
        # Extract code snippets
        code_blocks = self.code_pattern.findall(content)
        for code in code_blocks:
            await self.process_code_snippet(code, message)
        
        # Process attachments
        for attachment in attachments:
            await self.process_attachment(attachment, message)
    
    async def process_url(self, url: str, message: Dict[str, Any]):
        """Process a detected URL with intelligent filtering"""
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            
            # Track all URLs found
            self.stats["links_found"] += 1
            
            # Pre-filter obvious noise domains
            if domain in self.noise_domains:
                if domain not in self.stats["filtered_domains"]:
                    self.stats["filtered_domains"][domain] = 0
                self.stats["filtered_domains"][domain] += 1
                self.stats["links_filtered"] += 1
                logger.debug(f"🚫 Filtered noise domain: {domain}")
                return
            
            # Use AI to evaluate if this is a valuable resource
            if not await self.is_valuable_resource(url, message.get('content', ''), domain):
                if domain not in self.stats["filtered_domains"]:
                    self.stats["filtered_domains"][domain] = 0
                self.stats["filtered_domains"][domain] += 1
                self.stats["links_filtered"] += 1
                logger.debug(f"🚫 AI filtered: {url}")
                return
            
            self.stats["links_analyzed"] += 1
            
            # Track domain statistics for valuable resources
            if domain not in self.stats["domains"]:
                self.stats["domains"][domain] = 0
            self.stats["domains"][domain] += 1
            
            # Create resource entry
            resource = {
                "id": len(self.detected_resources) + 1,
                "type": "url",
                "url": url,
                "domain": domain,
                "message_id": message.get('message_id'),
                "channel_id": message.get('channel_id'),
                "channel_name": message.get('channel_name'),
                "author": message.get('author', {}).get('username'),
                "timestamp": message.get('timestamp'),
                "content_preview": message.get('content', '')[:200] + '...' if len(message.get('content', '')) > 200 else message.get('content', ''),
                "discord_url": message.get('jump_url')
            }
            
            self.detected_resources.append(resource)
            self.stats["resources_detected"] += 1
            logger.debug(f"✅ Added valuable resource: {url}")
            
        except Exception as e:
            logger.warning(f"⚠️ Error processing URL {url}: {e}")
    
    async def is_valuable_resource(self, url: str, content_context: str, domain: str) -> bool:
        """Use AI to determine if a URL represents a valuable resource"""
        try:
            # Create context for AI evaluation
            prompt = f"""Analyze this URL and context to determine if it represents a valuable educational/professional resource.

URL: {url}
Domain: {domain}
Message context: {content_context[:300]}

VALUABLE resources include:
- Educational content (articles, tutorials, documentation, courses)
- Professional tools, platforms, or software
- Research papers or academic content  
- Substantive news articles about technology/AI/business
- Software repositories or technical documentation
- Business/industry insights, reports, and analysis
- LinkedIn articles or posts with professional insights (not just status updates)
- Professional blog posts and thought leadership content

NOT VALUABLE:
- GIFs, memes, or entertainment content (tenor.com, giphy.com)
- Temporary meeting links (zoom, teams, meet.google.com)
- Internal navigation links (discord.com/channels)
- Social media posts without substantial professional content
- Image/video sharing without educational value
- Simple personal updates or casual social posts

Context clues for value:
- Words like "article", "paper", "guide", "tutorial", "insights", "analysis"
- Technical or professional terminology in the message
- References to learning, tools, or industry trends

Respond with only: VALUABLE or NOT_VALUABLE"""

            response = await asyncio.to_thread(
                self.openai_client.chat.completions.create,
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0
            )
            
            result = response.choices[0].message.content.strip().upper()
            return result == "VALUABLE"
            
        except Exception as e:
            logger.warning(f"⚠️ Error in AI evaluation for {url}: {e}")
            # Default to including the resource if AI fails
            return True
    
    async def process_code_snippet(self, code: str, message: Dict[str, Any]):
        """Process a detected code snippet"""
        try:
            # Determine language if possible
            language = "unknown"
            if code.startswith("```"):
                first_line = code.split('\n')[0]
                if len(first_line) > 3:
                    language = first_line[3:].strip()
            
            resource = {
                "id": len(self.detected_resources) + 1,
                "type": "code",
                "language": language,
                "code": code,
                "message_id": message.get('message_id'),
                "channel_id": message.get('channel_id'),
                "channel_name": message.get('channel_name'),
                "author": message.get('author', {}).get('username'),
                "timestamp": message.get('timestamp'),
                "discord_url": message.get('jump_url')
            }
            
            self.detected_resources.append(resource)
            self.stats["code_snippets"] += 1
            self.stats["resources_detected"] += 1
            
        except Exception as e:
            logger.warning(f"⚠️ Error processing code snippet: {e}")
    
    async def process_attachment(self, attachment: Dict[str, Any], message: Dict[str, Any]):
        """Process a message attachment"""
        try:
            resource = {
                "id": len(self.detected_resources) + 1,
                "type": "attachment",
                "filename": attachment.get('filename'),
                "size": attachment.get('size'),
                "url": attachment.get('url'),
                "content_type": attachment.get('content_type'),
                "message_id": message.get('message_id'),
                "channel_id": message.get('channel_id'),
                "channel_name": message.get('channel_name'),
                "author": message.get('author', {}).get('username'),
                "timestamp": message.get('timestamp'),
                "discord_url": message.get('jump_url')
            }
            
            self.detected_resources.append(resource)
            self.stats["attachments"] += 1
            self.stats["resources_detected"] += 1
            
        except Exception as e:
            logger.warning(f"⚠️ Error processing attachment: {e}")
    
    async def save_resources(self):
        """Save detected resources to JSON file"""
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        resources_file = RESOURCES_OUTPUT_DIR / f"detected_resources_{timestamp}.json"
        
        try:
            with open(resources_file, 'w', encoding='utf-8') as f:
                json.dump(self.detected_resources, f, indent=2, ensure_ascii=False)
            
            logger.info(f"💾 Detected resources saved to {resources_file}")
            
        except Exception as e:
            error_msg = f"Error saving resources: {e}"
            logger.error(f"❌ {error_msg}")
            self.stats["errors"].append(error_msg)
    
    async def save_stats(self):
        """Save detection statistics"""
        self.stats["end_time"] = datetime.utcnow().isoformat()
        
        stats_file = RESOURCES_OUTPUT_DIR / f"detection_stats_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(self.stats, f, indent=2, ensure_ascii=False)
            
            logger.info(f"📊 Detection statistics saved to {stats_file}")
            
        except Exception as e:
            logger.error(f"❌ Error saving statistics: {e}")


async def main():
    """Main function"""
    logger.info("🚀 Starting batch resource detection...")
    
    detector = ResourceDetector()
    
    try:
        await detector.detect_resources()
    except Exception as e:
        logger.error(f"❌ Fatal error in resource detection: {e}")
        sys.exit(1)
    
    logger.info("🎉 Batch resource detection completed!")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
