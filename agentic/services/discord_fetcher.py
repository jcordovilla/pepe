"""
Discord Message Fetcher Service
Standalone service for fetching Discord messages via API for pipeline processing
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional
import discord
from discord.ext import commands

logger = logging.getLogger(__name__)

class DiscordMessageFetcher:
    """
    Standalone Discord message fetcher for pipeline operations
    
    Features:
    - Fetches messages from specified channels/guilds
    - Rate limiting and error handling
    - Saves to JSON files for pipeline processing
    - Integrates with existing data format
    """
    
    def __init__(self, token: str, config: Dict[str, Any] = None):
        self.token = token
        self.config = config or {}
        
        # Configuration
        self.page_size = self.config.get("page_size", 100)
        self.rate_limit_delay = self.config.get("rate_limit_delay", 1.0)
        self.max_retries = self.config.get("max_retries", 3)
        self.output_dir = Path(self.config.get("output_dir", "data/fetched_messages"))
        
        # Discord client setup
        intents = discord.Intents.default()
        intents.message_content = True
        intents.guilds = True
        intents.messages = True
        
        self.client = discord.Client(intents=intents)
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("🔍 Discord message fetcher initialized")
    
    async def initialize(self):
        """Initialize Discord client connection"""
        try:
            await self.client.login(self.token)
            logger.info("✅ Discord client logged in successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Discord client: {e}")
            raise
    
    async def fetch_guild_messages(
        self, 
        guild_id: str, 
        channel_ids: Optional[List[str]] = None,
        limit_per_channel: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Fetch messages from a guild (all channels or specified channels)
        
        Args:
            guild_id: Discord guild ID
            channel_ids: Optional list of specific channel IDs to fetch
            limit_per_channel: Optional limit of messages per channel
            
        Returns:
            Dictionary with fetch results
        """
        try:
            guild = await self.client.fetch_guild(guild_id)
            if not guild:
                return {"success": False, "error": "Guild not found"}
            
            logger.info(f"🏰 Fetching messages from guild: {guild.name} ({guild_id})")
            
            # Get channels
            channels = await guild.fetch_channels()
            text_channels = [ch for ch in channels if isinstance(ch, discord.TextChannel)]
            
            # Filter channels if specified
            if channel_ids:
                text_channels = [ch for ch in text_channels if str(ch.id) in channel_ids]
            
            results = {
                "guild_id": guild_id,
                "guild_name": guild.name,
                "channels_processed": 0,
                "total_messages": 0,
                "files_created": [],
                "errors": []
            }
            
            for channel in text_channels:
                try:
                    logger.info(f"📥 Fetching from #{channel.name} ({channel.id})")
                    
                    channel_result = await self.fetch_channel_messages(
                        guild_id=guild_id,
                        guild_name=guild.name,
                        channel=channel,
                        limit=limit_per_channel
                    )
                    
                    if channel_result["success"]:
                        results["channels_processed"] += 1
                        results["total_messages"] += channel_result["message_count"]
                        results["files_created"].append(channel_result["file_path"])
                    else:
                        results["errors"].append({
                            "channel": f"#{channel.name}",
                            "error": channel_result.get("error", "Unknown error")
                        })
                    
                    # Rate limiting
                    await asyncio.sleep(self.rate_limit_delay)
                    
                except Exception as e:
                    logger.error(f"❌ Error fetching from #{channel.name}: {e}")
                    results["errors"].append({
                        "channel": f"#{channel.name}",
                        "error": str(e)
                    })
            
            logger.info(f"✅ Guild fetch complete: {results['channels_processed']} channels, {results['total_messages']} messages")
            return {"success": True, **results}
            
        except Exception as e:
            logger.error(f"❌ Error fetching guild messages: {e}")
            return {"success": False, "error": str(e)}
    
    async def fetch_channel_messages(
        self,
        guild_id: str,
        guild_name: str,
        channel: discord.TextChannel,
        limit: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Fetch messages from a specific channel
        
        Args:
            guild_id: Discord guild ID
            guild_name: Discord guild name
            channel: Discord channel object
            limit: Optional limit of messages to fetch
            
        Returns:
            Dictionary with fetch results
        """
        try:
            messages = []
            message_count = 0
            
            # Fetch messages with pagination
            async for message in channel.history(limit=limit):
                try:
                    # Convert message to our standard format
                    message_data = await self._convert_message_to_dict(message)
                    messages.append(message_data)
                    message_count += 1
                    
                    # Progress logging
                    if message_count % 100 == 0:
                        logger.info(f"  📝 Fetched {message_count} messages from #{channel.name}")
                    
                except Exception as e:
                    logger.warning(f"⚠️ Error processing message {message.id}: {e}")
                    continue
            
            if not messages:
                logger.warning(f"⚠️ No messages found in #{channel.name}")
                return {"success": False, "error": "No messages found"}
            
            # Create output file
            filename = f"{guild_id}_{channel.id}_messages.json"
            file_path = self.output_dir / filename
            
            # Prepare data structure
            data = {
                "guild_id": guild_id,
                "guild_name": guild_name,
                "channel_id": str(channel.id),
                "channel_name": channel.name,
                "fetch_timestamp": datetime.now(timezone.utc).isoformat(),
                "message_count": len(messages),
                "messages": messages
            }
            
            # Save to file
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"💾 Saved {len(messages)} messages to {filename}")
            
            return {
                "success": True,
                "channel_id": str(channel.id),
                "channel_name": channel.name,
                "message_count": len(messages),
                "file_path": str(file_path)
            }
            
        except Exception as e:
            logger.error(f"❌ Error fetching channel messages: {e}")
            return {"success": False, "error": str(e)}
    
    async def _convert_message_to_dict(self, message: discord.Message) -> Dict[str, Any]:
        """Convert Discord message to dictionary format"""
        # Handle reactions
        reactions = []
        if message.reactions:
            for reaction in message.reactions:
                reactions.append({
                    "emoji": str(reaction.emoji),
                    "count": reaction.count
                })
        
        # Handle attachments
        attachments = []
        if message.attachments:
            for attachment in message.attachments:
                attachments.append({
                    "id": str(attachment.id),
                    "filename": attachment.filename,
                    "url": attachment.url,
                    "size": attachment.size,
                    "content_type": attachment.content_type
                })
        
        # Handle embeds
        embed_count = len(message.embeds) if message.embeds else 0
        
        # Handle reference (replies)
        reference = None
        if message.reference:
            reference = {
                "message_id": str(message.reference.message_id) if message.reference.message_id else None,
                "channel_id": str(message.reference.channel_id) if message.reference.channel_id else None,
                "guild_id": str(message.reference.guild_id) if message.reference.guild_id else None
            }
        
        # Handle mentions
        mentions = []
        if message.mentions:
            for user in message.mentions:
                mentions.append({
                    "id": str(user.id),
                    "username": user.name,
                    "display_name": user.display_name
                })
        
        return {
            "message_id": str(message.id),
            "channel_id": str(message.channel.id),
            "channel_name": message.channel.name,
            "guild_id": str(message.guild.id),
            "guild_name": message.guild.name,
            "content": message.content,
            "timestamp": message.created_at.isoformat(),
            "jump_url": message.jump_url,
            "author": {
                "id": str(message.author.id),
                "username": message.author.name,
                "discriminator": message.author.discriminator,
                "display_name": message.author.display_name,
                "bot": message.author.bot
            },
            "mentions": mentions,
            "reactions": reactions,
            "attachments": attachments,
            "embeds": embed_count,
            "pinned": message.pinned,
            "type": str(message.type),
            "reference": reference
        }
    
    async def close(self):
        """Close Discord client connection"""
        try:
            await self.client.close()
            logger.info("🔌 Discord client connection closed")
        except Exception as e:
            logger.error(f"❌ Error closing Discord client: {e}")
