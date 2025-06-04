"""
Enhanced Discord Message Service
Modern Discord API integration with rate limiting and batch processing
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import discord
from discord.ext import commands

from ..cache.smart_cache import SmartCache
from ..memory.conversation_memory import ConversationMemory

logger = logging.getLogger(__name__)

class DiscordMessageService:
    """
    Modern Discord message service with legacy-proven patterns
    
    Preserves battle-tested:
    - Rate limiting algorithms
    - Error recovery patterns
    - Pagination handling
    - Metadata extraction rules
    """
    
    def __init__(self, token: str, cache: SmartCache, memory: ConversationMemory):
        self.token = token
        self.cache = cache
        self.memory = memory
        
        # Legacy-proven configuration
        self.page_size = 100  # From legacy system
        self.rate_limit_delay = 1.0  # From legacy system
        self.max_retries = 3  # From legacy system
        
        # Discord client setup
        intents = discord.Intents.default()
        intents.message_content = True
        intents.guilds = True
        intents.messages = True
        intents.reactions = True  # Enhanced for reaction search
        
        self.client = discord.Client(intents=intents)
        
        logger.info("🤖 Discord service initialized with legacy patterns")
    
    async def fetch_messages_with_reactions(self, channel_id: int, since: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """
        Fetch messages with comprehensive reaction data
        Uses legacy-proven pagination and rate limiting
        """
        messages = []
        channel = self.client.get_channel(channel_id)
        
        if not channel:
            logger.warning(f"⚠️ Channel {channel_id} not accessible")
            return messages
        
        try:
            # Legacy-style pagination with rate limiting
            async for message in channel.history(limit=None, after=since):
                # Apply legacy rate limiting
                await asyncio.sleep(self.rate_limit_delay / 100)  # Legacy pattern
                
                # Extract comprehensive message data (enhanced from legacy)
                message_data = await self._extract_message_data(message)
                messages.append(message_data)
                
                # Batch processing (from legacy)
                if len(messages) >= self.page_size:
                    await self._process_message_batch(messages)
                    messages = []
            
            # Process remaining messages
            if messages:
                await self._process_message_batch(messages)
                
            logger.info(f"✅ Fetched messages from channel {channel_id}")
            return messages
            
        except discord.errors.RateLimited as e:
            # Legacy error recovery pattern
            logger.warning(f"⏰ Rate limited, waiting {e.retry_after} seconds")
            await asyncio.sleep(e.retry_after)
            return await self.fetch_messages_with_reactions(channel_id, since)
            
        except Exception as e:
            logger.error(f"❌ Error fetching messages: {e}")
            raise
    
    async def _extract_message_data(self, message: discord.Message) -> Dict[str, Any]:
        """
        Extract comprehensive message data with reaction metadata
        Enhanced from legacy extraction patterns
        """
        # Legacy base extraction
        message_data = {
            "message_id": str(message.id),
            "channel_id": str(message.channel.id),
            "channel_name": message.channel.name,
            "guild_id": str(message.guild.id) if message.guild else None,
            "guild_name": message.guild.name if message.guild else None,
            "content": message.content,
            "timestamp": message.created_at.isoformat(),
            "jump_url": message.jump_url,
            "author": {
                "id": str(message.author.id),
                "username": message.author.name,
                "display_name": message.author.display_name,
                "bot": message.author.bot
            },
            "mentions": [str(user.id) for user in message.mentions],
            "attachments": [],
            "embeds": len(message.embeds),
            "pinned": message.pinned,
            "type": str(message.type),
            "reference": None
        }
        
        # Enhanced reaction data (new in modern system)
        reactions_data = []
        for reaction in message.reactions:
            reaction_data = {
                "emoji": str(reaction.emoji),
                "count": reaction.count,
                "me": reaction.me,
                "users": []
            }
            
            # Extract reaction users (enhanced feature)
            try:
                async for user in reaction.users():
                    reaction_data["users"].append({
                        "id": str(user.id),
                        "username": user.name,
                        "display_name": user.display_name
                    })
            except Exception as e:
                logger.debug(f"Could not fetch reaction users: {e}")
            
            reactions_data.append(reaction_data)
        
        message_data["reactions"] = reactions_data
        
        # Legacy attachment processing
        for attachment in message.attachments:
            message_data["attachments"].append({
                "id": str(attachment.id),
                "filename": attachment.filename,
                "size": attachment.size,
                "url": attachment.url,
                "content_type": getattr(attachment, 'content_type', None)
            })
        
        # Legacy reference handling
        if message.reference:
            message_data["reference"] = {
                "message_id": str(message.reference.message_id),
                "channel_id": str(message.reference.channel_id),
                "guild_id": str(message.reference.guild_id) if message.reference.guild_id else None
            }
        
        return message_data
    
    async def _process_message_batch(self, messages: List[Dict[str, Any]]):
        """
        Process message batch using modern unified data layer
        Enhanced from legacy batch processing
        """
        try:
            # Store in memory system
            for message in messages:
                await self.memory.store_message(message)
            
            # Cache processed data
            cache_key = f"messages_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            await self.cache.set(cache_key, messages, ttl=3600)
            
            logger.debug(f"📦 Processed batch of {len(messages)} messages")
            
        except Exception as e:
            logger.error(f"❌ Error processing message batch: {e}")
            raise
