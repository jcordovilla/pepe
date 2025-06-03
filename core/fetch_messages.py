#!/usr/bin/env python3
"""
Discord Message Fetcher

Fetches Discord messages with reactions data and saves them to JSON files
for processing by the agentic pipeline system.
"""

import os
import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import discord

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")

# Configuration
SKIP_CHANNEL_NAMES = {"test-channel", "🛝discord-playground"}
PAGE_SIZE = 100  # Number of messages to fetch per batch
OUTPUT_DIR = Path("data/fetched_messages")

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Discord intents
intents = discord.Intents.default()
intents.message_content = True
intents.guilds = True
intents.messages = True


class DiscordMessageFetcher(discord.Client):
    """
    Discord client for fetching messages with reactions data.
    
    Fetches messages from all accessible channels and saves them to JSON files
    for processing by the agentic pipeline.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fetch_stats = {
            "timestamp": datetime.utcnow().isoformat(),
            "guilds_processed": [],
            "channels_skipped": [],
            "errors": [],
            "total_messages_fetched": 0,
            "total_channels_processed": 0
        }
        
    async def on_ready(self):
        """Called when the bot is ready to start fetching"""
        logger.info(f"✅ Logged in as: {self.user} (ID: {self.user.id})")
        logger.info(f"📂 Output directory: {OUTPUT_DIR.absolute()}")
        
        try:
            await self.fetch_all_messages()
        except Exception as e:
            logger.error(f"❌ Error during message fetching: {e}")
            self.fetch_stats["errors"].append(str(e))
        finally:
            await self._save_stats()
            await self.close()
    
    async def fetch_all_messages(self):
        """Fetch messages from all accessible guilds and channels"""
        for guild in self.guilds:
            logger.info(f"\n📂 Processing Guild: {guild.name} (ID: {guild.id})")
            
            guild_data = {
                "guild_id": str(guild.id),
                "guild_name": guild.name,
                "channels_processed": 0,
                "messages_fetched": 0,
                "channels_skipped": 0
            }
            
            for channel in guild.text_channels:
                try:
                    # Skip test channels
                    if channel.name in SKIP_CHANNEL_NAMES:
                        logger.info(f"  🚫 Skipping test channel #{channel.name}")
                        self.fetch_stats["channels_skipped"].append({
                            "guild_name": guild.name,
                            "channel_name": channel.name,
                            "channel_id": str(channel.id),
                            "reason": "Skipped by name"
                        })
                        guild_data["channels_skipped"] += 1
                        continue
                    
                    # Check if we have permission to read this channel
                    if not channel.permissions_for(guild.me).read_message_history:
                        logger.info(f"  🚫 No permission to read #{channel.name}")
                        self.fetch_stats["channels_skipped"].append({
                            "guild_name": guild.name,
                            "channel_name": channel.name,
                            "channel_id": str(channel.id),
                            "reason": "No permission"
                        })
                        guild_data["channels_skipped"] += 1
                        continue
                    
                    # Fetch messages from this channel
                    messages = await self.fetch_channel_messages(guild, channel)
                    
                    if messages:
                        # Save messages to file
                        await self.save_messages(guild, channel, messages)
                        guild_data["messages_fetched"] += len(messages)
                        self.fetch_stats["total_messages_fetched"] += len(messages)
                        logger.info(f"  ✅ Fetched {len(messages)} messages from #{channel.name}")
                    else:
                        logger.info(f"  📫 No messages in #{channel.name}")
                    
                    guild_data["channels_processed"] += 1
                    self.fetch_stats["total_channels_processed"] += 1
                    
                except discord.Forbidden:
                    logger.info(f"  🚫 Forbidden access to #{channel.name}")
                    self.fetch_stats["channels_skipped"].append({
                        "guild_name": guild.name,
                        "channel_name": channel.name,
                        "channel_id": str(channel.id),
                        "reason": "Forbidden"
                    })
                    guild_data["channels_skipped"] += 1
                except Exception as e:
                    logger.error(f"  ❌ Error processing #{channel.name}: {e}")
                    self.fetch_stats["errors"].append(f"Channel {channel.name}: {str(e)}")
            
            self.fetch_stats["guilds_processed"].append(guild_data)
            logger.info(f"📊 Guild {guild.name} summary: "
                       f"{guild_data['messages_fetched']} messages, "
                       f"{guild_data['channels_processed']} channels processed, "
                       f"{guild_data['channels_skipped']} channels skipped")
    
    async def fetch_channel_messages(self, guild: discord.Guild, channel: discord.TextChannel) -> List[Dict[str, Any]]:
        """
        Fetch all messages from a channel with incremental loading.
        
        Args:
            guild: Discord guild
            channel: Discord text channel
            
        Returns:
            List of message dictionaries with reactions data
        """
        messages = []
        
        try:
            # Determine if we should do incremental fetch
            last_message_id = await self.get_last_fetched_message(guild, channel)
            after_object = discord.Object(id=last_message_id) if last_message_id else None
            
            logger.info(f"  📄 Fetching from #{channel.name} (after: {last_message_id})")
            
            # Fetch messages in batches
            async for message in channel.history(
                limit=None,
                after=after_object,
                oldest_first=True
            ):
                message_data = await self.format_message(message, guild, channel)
                messages.append(message_data)
                
                # Process in batches to avoid memory issues
                if len(messages) >= PAGE_SIZE * 10:  # Save every 1000 messages
                    await self.save_messages(guild, channel, messages, append=True)
                    messages.clear()
            
        except Exception as e:
            logger.error(f"Error fetching messages from #{channel.name}: {e}")
            raise
        
        return messages
    
    async def format_message(self, message: discord.Message, guild: discord.Guild, channel: discord.TextChannel) -> Dict[str, Any]:
        """
        Format a Discord message with all relevant data including reactions.
        
        Args:
            message: Discord message object
            guild: Discord guild
            channel: Discord channel
            
        Returns:
            Formatted message dictionary
        """
        # Format reactions data
        reactions = []
        for reaction in message.reactions:
            reactions.append({
                'emoji': str(reaction.emoji),
                'count': reaction.count
            })
        
        # Format author data
        author_data = {
            'id': str(message.author.id),
            'username': message.author.name,
            'discriminator': message.author.discriminator,
            'display_name': message.author.display_name,
            'bot': message.author.bot
        }
        
        # Format message data
        message_data = {
            'message_id': str(message.id),
            'channel_id': str(channel.id),
            'channel_name': channel.name,
            'guild_id': str(guild.id),
            'guild_name': guild.name,
            'content': message.content.replace('\u2028', ' ').replace('\u2029', ' ').strip(),
            'timestamp': message.created_at.isoformat(),
            'jump_url': message.jump_url,
            'author': author_data,
            'mentions': [str(user.id) for user in message.mentions],
            'reactions': reactions,
            'attachments': [
                {
                    'id': str(attachment.id),
                    'filename': attachment.filename,
                    'url': attachment.url,
                    'size': attachment.size
                } for attachment in message.attachments
            ],
            'embeds': len(message.embeds),
            'pinned': message.pinned,
            'type': str(message.type),
            'reference': {
                'message_id': str(message.reference.message_id),
                'channel_id': str(message.reference.channel_id),
                'guild_id': str(message.reference.guild_id) if message.reference.guild_id else None
            } if message.reference else None
        }
        
        return message_data
    
    async def get_last_fetched_message(self, guild: discord.Guild, channel: discord.TextChannel) -> Optional[int]:
        """
        Get the ID of the last fetched message for incremental updates.
        
        Args:
            guild: Discord guild
            channel: Discord channel
            
        Returns:
            Message ID of the last fetched message, or None
        """
        file_path = OUTPUT_DIR / f"{guild.id}_{channel.id}_messages.json"
        
        if not file_path.exists():
            return None
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if data.get('messages'):
                # Get the last message ID (messages should be sorted by timestamp)
                last_message = max(data['messages'], key=lambda x: x.get('timestamp', ''))
                return int(last_message['message_id'])
                
        except Exception as e:
            logger.warning(f"Error reading last message from {file_path}: {e}")
        
        return None
    
    async def save_messages(self, guild: discord.Guild, channel: discord.TextChannel, 
                          messages: List[Dict[str, Any]], append: bool = False):
        """
        Save messages to JSON file.
        
        Args:
            guild: Discord guild
            channel: Discord channel
            messages: List of message data
            append: Whether to append to existing file or overwrite
        """
        if not messages:
            return
        
        file_path = OUTPUT_DIR / f"{guild.id}_{channel.id}_messages.json"
        
        # Prepare data structure
        channel_data = {
            "guild_id": str(guild.id),
            "guild_name": guild.name,
            "channel_id": str(channel.id),
            "channel_name": channel.name,
            "fetch_timestamp": datetime.utcnow().isoformat(),
            "message_count": len(messages),
            "messages": messages
        }
        
        if append and file_path.exists():
            # Load existing data and append
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                
                # Merge messages (avoid duplicates by message_id)
                existing_ids = {msg['message_id'] for msg in existing_data.get('messages', [])}
                new_messages = [msg for msg in messages if msg['message_id'] not in existing_ids]
                
                if new_messages:
                    existing_data['messages'].extend(new_messages)
                    existing_data['message_count'] = len(existing_data['messages'])
                    existing_data['fetch_timestamp'] = datetime.utcnow().isoformat()
                    channel_data = existing_data
                else:
                    return  # No new messages to add
                    
            except Exception as e:
                logger.warning(f"Error reading existing file {file_path}: {e}, overwriting")
        
        # Save to file
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(channel_data, f, indent=2, ensure_ascii=False)
            
            logger.debug(f"💾 Saved {len(messages)} messages to {file_path}")
            
        except Exception as e:
            logger.error(f"Error saving messages to {file_path}: {e}")
            raise
    
    async def _save_stats(self):
        """Save fetch statistics to file"""
        stats_file = OUTPUT_DIR / f"fetch_stats_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(self.fetch_stats, f, indent=2, ensure_ascii=False)
            
            logger.info(f"📊 Statistics saved to {stats_file}")
            logger.info(f"📈 Final Stats: {self.fetch_stats['total_messages_fetched']} messages, "
                       f"{self.fetch_stats['total_channels_processed']} channels, "
                       f"{len(self.fetch_stats['errors'])} errors")
            
        except Exception as e:
            logger.error(f"Error saving statistics: {e}")


async def main():
    """Main function to run the Discord message fetcher"""
    if not DISCORD_TOKEN:
        logger.error("❌ DISCORD_TOKEN not found in environment variables")
        return
    
    logger.info("🔌 Starting Discord message fetcher...")
    logger.info(f"📁 Messages will be saved to: {OUTPUT_DIR.absolute()}")
    
    client = DiscordMessageFetcher(intents=intents)
    
    try:
        await client.start(DISCORD_TOKEN)
    except Exception as e:
        logger.error(f"❌ Error starting Discord client: {e}")
    finally:
        # Ensure we properly close the client
        if not client.is_closed():
            await client.close()
        # Give time for any pending tasks to complete
        await asyncio.sleep(1)
        logger.info("🔌 Discord message fetcher finished")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("🛑 Message fetcher stopped by user")
    except Exception as e:
        logger.error(f"❌ Error running message fetcher: {e}")
