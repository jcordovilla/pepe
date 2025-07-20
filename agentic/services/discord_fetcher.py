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
from .fetch_state_manager import FetchStateManager

logger = logging.getLogger(__name__)

class DiscordMessageFetcher:
    """
    Standalone Discord message fetcher for pipeline operations
    
    Features:
    - Fetches messages from specified channels/guilds
    - Incremental fetching with state management
    - Rate limiting and error handling
    - Saves to JSON files for pipeline processing
    - Integrates with existing data format
    """
    
    def __init__(self, token: str, config: Optional[Dict[str, Any]] = None):
        self.token = token
        self.config = config or {}
        
        # Configuration
        self.page_size = self.config.get("page_size", 100)
        self.rate_limit_delay = self.config.get("rate_limit_delay", 1.0)
        self.max_retries = self.config.get("max_retries", 3)
        self.output_dir = Path(self.config.get("output_dir", "data/fetched_messages"))
        
        # State management for incremental fetching
        self.state_manager = FetchStateManager(
            self.config.get("state_file", "data/processing_markers/fetch_state.json")
        )
        
        # Discord client setup
        intents = discord.Intents.default()
        intents.message_content = True
        intents.guilds = True
        intents.messages = True
        
        self.client = discord.Client(intents=intents)
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("🔍 Discord message fetcher initialized with incremental support")
    
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
        limit_per_channel: Optional[int] = None,
        incremental: bool = True
    ) -> Dict[str, Any]:
        """
        Fetch messages from a guild (all channels or specified channels)
        
        Args:
            guild_id: Discord guild ID
            channel_ids: Optional list of specific channel IDs to fetch
            limit_per_channel: Optional limit of messages per channel
            incremental: Whether to use incremental fetching (default: True)
            
        Returns:
            Dictionary with fetch results
        """
        try:
            guild = await self.client.fetch_guild(int(guild_id))
            if not guild:
                return {"success": False, "error": "Guild not found"}
            
            logger.info(f"🏰 Fetching messages from guild: {guild.name} ({guild_id})")
            if incremental:
                logger.info("📈 Using incremental fetching mode")
            else:
                logger.info("📥 Using full fetching mode")
            
            # Get channels
            channels = await guild.fetch_channels()
            text_channels = [ch for ch in channels if isinstance(ch, discord.TextChannel)]
            forum_channels = [ch for ch in channels if isinstance(ch, discord.ForumChannel)]
            
            # Filter out channels with "test" in the name (case-insensitive)
            text_channels = [ch for ch in text_channels if "test" not in ch.name.lower()]
            forum_channels = [ch for ch in forum_channels if "test" not in ch.name.lower()]
            
            # Filter channels if specified
            if channel_ids:
                text_channels = [ch for ch in text_channels if str(ch.id) in channel_ids]
                forum_channels = [ch for ch in forum_channels if str(ch.id) in channel_ids]
            
            results = {
                "guild_id": guild_id,
                "guild_name": guild.name,
                "channels_processed": 0,
                "total_messages": 0,
                "files_created": [],
                "files_updated": [],
                "errors": [],
                "incremental_mode": incremental,
                "forum_channels_processed": 0,
                "total_threads_processed": 0
            }
            
            for channel in text_channels:
                try:
                    logger.info(f"📥 Fetching from #{channel.name} ({channel.id})")
                    
                    channel_result = await self.fetch_channel_messages(
                        guild_id=guild_id,
                        guild_name=guild.name,
                        channel=channel,
                        limit=limit_per_channel,
                        incremental=incremental
                    )
                    
                    if channel_result["success"]:
                        results["channels_processed"] += 1
                        results["total_messages"] += channel_result["message_count"]
                        
                        if channel_result.get("file_created"):
                            results["files_created"].append(channel_result["file_path"])
                        elif channel_result.get("file_updated"):
                            results["files_updated"].append(channel_result["file_path"])
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
            
            # Process forum channels and their threads
            logger.info(f"📋 Processing {len(forum_channels)} forum channels...")
            for forum_channel in forum_channels:
                try:
                    logger.info(f"📋 Fetching from forum #{forum_channel.name} ({forum_channel.id})")
                    
                    forum_result = await self.fetch_forum_channel(
                        guild_id=guild_id,
                        guild_name=guild.name,
                        forum_channel=forum_channel,
                        limit=limit_per_channel,
                        incremental=incremental
                    )
                    
                    if forum_result["success"]:
                        results["forum_channels_processed"] += 1
                        results["total_messages"] += forum_result["message_count"]
                        results["total_threads_processed"] += forum_result["threads_processed"]
                        
                        if forum_result.get("files_created"):
                            results["files_created"].extend(forum_result["files_created"])
                        if forum_result.get("files_updated"):
                            results["files_updated"].extend(forum_result["files_updated"])
                    else:
                        results["errors"].append({
                            "channel": f"📋#{forum_channel.name}",
                            "error": forum_result.get("error", "Unknown error")
                        })
                    
                    # Rate limiting
                    await asyncio.sleep(self.rate_limit_delay)
                    
                except Exception as e:
                    logger.error(f"❌ Error fetching from forum #{forum_channel.name}: {e}")
                    results["errors"].append({
                        "channel": f"📋#{forum_channel.name}",
                        "error": str(e)
                    })
            
            logger.info(f"✅ Guild fetch complete: {results['channels_processed']} text channels, {results['forum_channels_processed']} forum channels, {results['total_threads_processed']} threads, {results['total_messages']} messages")
            return {"success": True, **results}
            
        except Exception as e:
            logger.error(f"❌ Error fetching guild messages: {e}")
            return {"success": False, "error": str(e)}
    
    async def fetch_channel_messages(
        self,
        guild_id: str,
        guild_name: str,
        channel: discord.TextChannel,
        limit: Optional[int] = None,
        incremental: bool = True
    ) -> Dict[str, Any]:
        """
        Fetch messages from a specific channel with optional incremental fetching
        
        Args:
            guild_id: Discord guild ID
            guild_name: Discord guild name
            channel: Discord channel object
            limit: Optional limit of messages to fetch
            incremental: Whether to use incremental fetching based on state
            
        Returns:
            Dictionary with fetch results
        """
        try:
            messages = []
            message_count = 0
            newest_timestamp = None
            
            # Determine starting point for incremental fetching
            after_datetime = None
            if incremental:
                last_fetch_time = self.state_manager.get_channel_last_fetch(guild_id, str(channel.id))
                if last_fetch_time:
                    after_datetime = last_fetch_time
                    logger.info(f"📈 Incremental fetch from #{channel.name} after {last_fetch_time}")
                else:
                    logger.info(f"🆕 First fetch from #{channel.name} (no previous state)")
            else:
                logger.info(f"📥 Full fetch from #{channel.name}")
            
            # Fetch messages with proper pagination handling
            if after_datetime and incremental:
                # For incremental fetching, we need to get messages AFTER our last timestamp
                # Use oldest_first=True to get them in chronological order and avoid infinite loops
                logger.info(f"  🔍 Looking for messages after {after_datetime}")
                
                # Discord.py's after parameter gets messages created after the specified time
                # We use oldest_first=True to get messages in chronological order
                async for message in channel.history(limit=limit, after=after_datetime, oldest_first=True):
                    try:
                        # Convert message to our standard format
                        message_data = await self._convert_message_to_dict(message)
                        messages.append(message_data)
                        message_count += 1
                        
                        # Track newest message timestamp for state updates
                        if newest_timestamp is None or message.created_at > newest_timestamp:
                            newest_timestamp = message.created_at
                        
                        # Progress logging
                        if message_count % 100 == 0:
                            logger.info(f"  📝 Fetched {message_count} new messages from #{channel.name}")
                        
                    except Exception as e:
                        logger.warning(f"⚠️ Error processing message {message.id}: {e}")
                        continue
                        
                if message_count == 0:
                    logger.info(f"  ✨ No new messages in #{channel.name} since {after_datetime}")
            else:
                # Full fetch mode - get all messages (or up to limit) in default order (newest first)
                async for message in channel.history(limit=limit):
                    try:
                        # Convert message to our standard format
                        message_data = await self._convert_message_to_dict(message)
                        messages.append(message_data)
                        message_count += 1
                        
                        # Track newest message timestamp for state updates
                        if newest_timestamp is None or message.created_at > newest_timestamp:
                            newest_timestamp = message.created_at
                        
                        # Progress logging
                        if message_count % 100 == 0:
                            logger.info(f"  📝 Fetched {message_count} messages from #{channel.name}")
                        
                    except Exception as e:
                        logger.warning(f"⚠️ Error processing message {message.id}: {e}")
                        continue
            
            if not messages:
                if incremental and after_datetime:
                    logger.info(f"✨ No new messages in #{channel.name} since last fetch")
                    return {"success": True, "message_count": 0, "no_new_messages": True}
                else:
                    logger.warning(f"⚠️ No messages found in #{channel.name}")
                    return {"success": False, "error": "No messages found"}
            
            # Handle file creation/updating
            filename = f"{guild_id}_{channel.id}_messages.json"
            file_path = self.output_dir / filename
            file_created = not file_path.exists()
            file_updated = False
            
            if incremental and file_path.exists() and not file_created:
                # Merge with existing data
                await self._merge_messages_to_file(file_path, messages, guild_id, guild_name, channel, incremental)
                file_updated = True
                logger.info(f"🔄 Updated {filename} with {len(messages)} new messages")
            else:
                # Create new file or overwrite (full fetch mode)
                await self._create_new_message_file(file_path, messages, guild_id, guild_name, channel)
                if file_created:
                    logger.info(f"💾 Created {filename} with {len(messages)} messages")
                else:
                    logger.info(f"💾 Overwrote {filename} with {len(messages)} messages")
            
            # Update state for incremental fetching
            if newest_timestamp:
                self.state_manager.update_channel_state(
                    guild_id=guild_id,
                    channel_id=str(channel.id),
                    channel_name=channel.name,
                    last_message_timestamp=newest_timestamp.isoformat(),
                    message_count=len(messages),
                    fetch_mode="incremental" if incremental else "full"
                )
            
            return {
                "success": True,
                "channel_id": str(channel.id),
                "channel_name": channel.name,
                "message_count": len(messages),
                "file_path": str(file_path),
                "file_created": file_created,
                "file_updated": file_updated,
                "incremental_mode": incremental
            }
            
        except Exception as e:
            logger.error(f"❌ Error fetching channel messages: {e}")
            return {"success": False, "error": str(e)}
    
    async def fetch_forum_channel(
        self,
        guild_id: str,
        guild_name: str,
        forum_channel: discord.ForumChannel,
        limit: Optional[int] = None,
        incremental: bool = True
    ) -> Dict[str, Any]:
        """
        Fetch messages from all threads in a forum channel
        
        Args:
            guild_id: Discord guild ID
            guild_name: Discord guild name
            forum_channel: Discord forum channel object
            limit: Optional limit of messages per thread
            incremental: Whether to use incremental fetching based on state
            
        Returns:
            Dictionary with fetch results
        """
        try:
            all_messages = []
            total_messages = 0
            threads_processed = 0
            files_created = []
            files_updated = []
            
            # Get all threads (active + archived)
            threads = []
            
            # Get active threads
            for thread in forum_channel.threads:
                threads.append(thread)
            
            # Get archived threads
            try:
                async for thread in forum_channel.archived_threads(limit=None):
                    threads.append(thread)
            except discord.Forbidden:
                logger.warning(f"⚠️ No permission to access archived threads in #{forum_channel.name}")
            except Exception as e:
                logger.warning(f"⚠️ Error fetching archived threads from #{forum_channel.name}: {e}")
            
            logger.info(f"  🧵 Found {len(threads)} threads in forum #{forum_channel.name}")
            
            if not threads:
                logger.info(f"  📋 No threads found in forum #{forum_channel.name}")
                return {
                    "success": True,
                    "channel_id": str(forum_channel.id),
                    "channel_name": forum_channel.name,
                    "message_count": 0,
                    "threads_processed": 0,
                    "files_created": [],
                    "files_updated": []
                }
            
            # Process each thread
            for thread in threads:
                try:
                    logger.info(f"    🧵 Processing thread: {thread.name}")
                    
                    thread_result = await self.fetch_thread_messages(
                        guild_id=guild_id,
                        guild_name=guild_name,
                        forum_channel=forum_channel,
                        thread=thread,
                        limit=limit,
                        incremental=incremental
                    )
                    
                    if thread_result["success"]:
                        threads_processed += 1
                        total_messages += thread_result["message_count"]
                        
                        if thread_result.get("file_created"):
                            files_created.append(thread_result["file_path"])
                        elif thread_result.get("file_updated"):
                            files_updated.append(thread_result["file_path"])
                        
                        # Add thread messages to overall collection
                        all_messages.extend(thread_result.get("messages", []))
                    else:
                        logger.warning(f"    ⚠️ Failed to fetch thread {thread.name}: {thread_result.get('error')}")
                    
                    # Rate limiting between threads
                    await asyncio.sleep(self.rate_limit_delay * 0.5)
                    
                except Exception as e:
                    logger.error(f"    ❌ Error processing thread {thread.name}: {e}")
                    continue
            
            # Create forum-level summary file
            if all_messages:
                forum_filename = f"{guild_id}_{forum_channel.id}_forum_messages.json"
                forum_file_path = self.output_dir / forum_filename
                
                forum_data = {
                    "guild_id": guild_id,
                    "guild_name": guild_name,
                    "channel_id": str(forum_channel.id),
                    "channel_name": forum_channel.name,
                    "channel_type": "forum",
                    "threads_processed": threads_processed,
                    "total_messages": total_messages,
                    "last_update_timestamp": datetime.now(timezone.utc).isoformat(),
                    "messages": all_messages
                }
                
                with open(forum_file_path, 'w', encoding='utf-8') as f:
                    json.dump(forum_data, f, indent=2, ensure_ascii=False)
                
                logger.info(f"  💾 Created forum summary: {forum_filename} with {total_messages} messages from {threads_processed} threads")
            
            return {
                "success": True,
                "channel_id": str(forum_channel.id),
                "channel_name": forum_channel.name,
                "message_count": total_messages,
                "threads_processed": threads_processed,
                "files_created": files_created,
                "files_updated": files_updated,
                "messages": all_messages
            }
            
        except Exception as e:
            logger.error(f"❌ Error fetching forum channel: {e}")
            return {"success": False, "error": str(e)}
    
    async def fetch_thread_messages(
        self,
        guild_id: str,
        guild_name: str,
        forum_channel: discord.ForumChannel,
        thread: discord.Thread,
        limit: Optional[int] = None,
        incremental: bool = True
    ) -> Dict[str, Any]:
        """
        Fetch messages from a specific thread
        
        Args:
            guild_id: Discord guild ID
            guild_name: Discord guild name
            forum_channel: Discord forum channel object
            thread: Discord thread object
            limit: Optional limit of messages to fetch
            incremental: Whether to use incremental fetching based on state
            
        Returns:
            Dictionary with fetch results
        """
        try:
            messages = []
            message_count = 0
            newest_timestamp = None
            
            # Determine starting point for incremental fetching
            after_datetime = None
            if incremental:
                # Use thread-specific state key
                state_key = f"{guild_id}_{forum_channel.id}_{thread.id}"
                last_fetch_time = self.state_manager.get_channel_last_fetch(guild_id, state_key)
                if last_fetch_time:
                    after_datetime = last_fetch_time
                    logger.info(f"      📈 Incremental fetch from thread {thread.name} after {last_fetch_time}")
                else:
                    logger.info(f"      🆕 First fetch from thread {thread.name} (no previous state)")
            else:
                logger.info(f"      📥 Full fetch from thread {thread.name}")
            
            # Fetch messages from thread
            if after_datetime and incremental:
                async for message in thread.history(limit=limit, after=after_datetime, oldest_first=True):
                    try:
                        message_data = await self._convert_message_to_dict(message)
                        # Add thread-specific metadata
                        message_data.update({
                            "thread_id": str(thread.id),
                            "thread_name": thread.name,
                            "forum_channel_id": str(forum_channel.id),
                            "forum_channel_name": forum_channel.name,
                            "is_forum_thread": True
                        })
                        messages.append(message_data)
                        message_count += 1
                        
                        if newest_timestamp is None or message.created_at > newest_timestamp:
                            newest_timestamp = message.created_at
                        
                    except Exception as e:
                        logger.warning(f"        ⚠️ Error processing message {message.id}: {e}")
                        continue
            else:
                async for message in thread.history(limit=limit):
                    try:
                        message_data = await self._convert_message_to_dict(message)
                        # Add thread-specific metadata
                        message_data.update({
                            "thread_id": str(thread.id),
                            "thread_name": thread.name,
                            "forum_channel_id": str(forum_channel.id),
                            "forum_channel_name": forum_channel.name,
                            "is_forum_thread": True
                        })
                        messages.append(message_data)
                        message_count += 1
                        
                        if newest_timestamp is None or message.created_at > newest_timestamp:
                            newest_timestamp = message.created_at
                        
                    except Exception as e:
                        logger.warning(f"        ⚠️ Error processing message {message.id}: {e}")
                        continue
            
            if not messages:
                if incremental and after_datetime:
                    logger.info(f"        ✨ No new messages in thread {thread.name} since last fetch")
                    return {"success": True, "message_count": 0, "no_new_messages": True}
                else:
                    logger.info(f"        📋 No messages found in thread {thread.name}")
                    return {"success": True, "message_count": 0}
            
            # Create thread-specific file
            thread_filename = f"{guild_id}_{forum_channel.id}_{thread.id}_thread_messages.json"
            thread_file_path = self.output_dir / thread_filename
            file_created = not thread_file_path.exists()
            file_updated = False
            
            thread_data = {
                "guild_id": guild_id,
                "guild_name": guild_name,
                "forum_channel_id": str(forum_channel.id),
                "forum_channel_name": forum_channel.name,
                "thread_id": str(thread.id),
                "thread_name": thread.name,
                "last_update_timestamp": datetime.now(timezone.utc).isoformat(),
                "message_count": len(messages),
                "messages": messages
            }
            
            if incremental and thread_file_path.exists():
                # Merge with existing thread data
                try:
                    with open(thread_file_path, 'r', encoding='utf-8') as f:
                        existing_data = json.load(f)
                    
                    existing_messages = existing_data.get("messages", [])
                    existing_ids = {msg.get("message_id") for msg in existing_messages}
                    
                    unique_new_messages = [
                        msg for msg in messages 
                        if msg.get("message_id") not in existing_ids
                    ]
                    
                    if unique_new_messages:
                        merged_messages = existing_messages + list(reversed(unique_new_messages))
                        thread_data["messages"] = merged_messages
                        thread_data["message_count"] = len(merged_messages)
                        file_updated = True
                        logger.info(f"        🔄 Updated thread file with {len(unique_new_messages)} new messages")
                    else:
                        logger.info(f"        📄 No new messages to add to thread file")
                        return {"success": True, "message_count": 0, "no_new_messages": True}
                        
                except Exception as e:
                    logger.warning(f"        ⚠️ Error merging thread data: {e}")
                    # Fallback to overwrite
                    file_created = True
            
            # Save thread data
            with open(thread_file_path, 'w', encoding='utf-8') as f:
                json.dump(thread_data, f, indent=2, ensure_ascii=False)
            
            if file_created:
                logger.info(f"        💾 Created thread file: {thread_filename} with {len(messages)} messages")
            elif file_updated:
                logger.info(f"        🔄 Updated thread file: {thread_filename}")
            
            # Update state for incremental fetching
            if newest_timestamp:
                state_key = f"{guild_id}_{forum_channel.id}_{thread.id}"
                self.state_manager.update_channel_state(
                    guild_id=guild_id,
                    channel_id=state_key,
                    channel_name=f"{forum_channel.name}/{thread.name}",
                    last_message_timestamp=newest_timestamp.isoformat(),
                    message_count=len(messages),
                    fetch_mode="incremental" if incremental else "full"
                )
            
            return {
                "success": True,
                "thread_id": str(thread.id),
                "thread_name": thread.name,
                "message_count": len(messages),
                "file_path": str(thread_file_path),
                "file_created": file_created,
                "file_updated": file_updated,
                "messages": messages
            }
            
        except Exception as e:
            logger.error(f"❌ Error fetching thread messages: {e}")
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
            "channel_name": getattr(message.channel, 'name', 'Unknown'),
            "guild_id": str(message.guild.id) if message.guild else None,
            "guild_name": message.guild.name if message.guild else None,
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
    
    async def _merge_messages_to_file(
        self,
        file_path: Path,
        new_messages: List[Dict[str, Any]],
        guild_id: str,
        guild_name: str,
        channel: discord.TextChannel,
        incremental: bool = True
    ) -> None:
        """Merge new messages with existing file data"""
        try:
            # Load existing data
            with open(file_path, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
            
            existing_messages = existing_data.get("messages", [])
            
            # Create a set of existing message IDs for deduplication
            existing_ids = {msg.get("message_id") for msg in existing_messages}
            
            # Filter out duplicates from new messages
            unique_new_messages = [
                msg for msg in new_messages 
                if msg.get("message_id") not in existing_ids
            ]
            
            if not unique_new_messages:
                logger.info(f"📄 No new unique messages to add to {file_path.name}")
                return
            
            # Merge messages (maintain reverse chronological order - newest first)
            if incremental:
                # New messages are in chronological order (oldest first) from incremental fetch
                # Existing messages are in reverse chronological order (newest first)
                # We need to prepend new messages and maintain newest-first order
                merged_messages = existing_messages + list(reversed(unique_new_messages))
            else:
                # For full fetch, new messages are already in newest-first order
                merged_messages = unique_new_messages + existing_messages
            
            # Update metadata
            existing_data.update({
                "guild_id": guild_id,
                "guild_name": guild_name,
                "channel_id": str(channel.id),
                "channel_name": channel.name,
                "last_update_timestamp": datetime.now(timezone.utc).isoformat(),
                "message_count": len(merged_messages),
                "messages": merged_messages
            })
            
            # Save merged data
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(existing_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"🔄 Merged {len(unique_new_messages)} new messages into {file_path.name}")
            
        except Exception as e:
            logger.error(f"❌ Error merging messages to file: {e}")
            # Fallback: create new file
            await self._create_new_message_file(file_path, new_messages, guild_id, guild_name, channel)
    
    async def _create_new_message_file(
        self,
        file_path: Path,
        messages: List[Dict[str, Any]],
        guild_id: str,
        guild_name: str,
        channel: discord.TextChannel
    ) -> None:
        """Create a new message file with provided messages"""
        try:
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
            
        except Exception as e:
            logger.error(f"❌ Error creating message file: {e}")
            raise
