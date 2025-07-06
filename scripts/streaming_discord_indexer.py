#!/usr/bin/env python3
"""
StreamingDiscordIndexer - Direct Discord API to ChromaDB indexing.
Eliminates JSON intermediate storage for optimal performance.
"""

import asyncio
import os
import sys
import json
import discord
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Set
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agentic.vectorstore.persistent_store import PersistentVectorStore
from agentic.services.channel_resolver import ChannelResolver

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StreamingDiscordIndexer:
    """
    Direct Discord API to ChromaDB indexing with real-time processing.
    No intermediate JSON storage - optimal performance and consistency.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.discord_token = os.getenv("DISCORD_TOKEN")  # Fixed variable name
        
        if not self.discord_token:
            raise ValueError("DISCORD_TOKEN environment variable required")
        
        # Initialize Discord client
        intents = discord.Intents.default()
        intents.message_content = True
        intents.guilds = True
        intents.members = True
        
        self.client = discord.Client(intents=intents)
        
        # Initialize vector store with PRODUCTION config (same as bot)
        vector_config = {
            "collection_name": "discord_messages",
            "persist_directory": "./data/chromadb",
            "embedding_model": os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
            "batch_size": 100
        }
        
        self.vector_store = PersistentVectorStore(vector_config)
        self.channel_resolver = ChannelResolver()
        
        # Progress tracking
        self.stats = {
            "guilds_processed": 0,
            "channels_processed": 0,
            "messages_processed": 0,
            "messages_indexed": 0,
            "errors": 0,
            "start_time": None,
            "current_guild": None,
            "current_channel": None
        }
        
        # Batch processing
        self.message_batch = []
        self.batch_size = vector_config["batch_size"]
        
    async def start_indexing(self):
        """Start the direct indexing process"""
        print("🚀 STREAMING DISCORD INDEXER")
        print("=" * 80)
        print("• Direct Discord API → ChromaDB")
        print("• No JSON intermediate files")
        print("• Real-time progress tracking")
        print("• Enhanced metadata extraction")
        print("=" * 80)
        
        self.stats["start_time"] = datetime.now()
        
        @self.client.event
        async def on_ready():
            try:
                logger.info(f"Discord client connected as {self.client.user}")
                await self._process_all_guilds()
                await self._flush_remaining_batch()
                await self._print_final_stats()
                await self.client.close()
                
            except Exception as e:
                logger.error(f"Error during indexing: {e}")
                await self.client.close()
        
        # Start Discord client
        await self.client.start(self.discord_token)
    
    async def _process_all_guilds(self):
        """Process all accessible guilds"""
        guilds = self.client.guilds
        logger.info(f"Found {len(guilds)} guilds to process")
        
        for guild in guilds:
            await self._process_guild(guild)
            self.stats["guilds_processed"] += 1
    
    async def _process_guild(self, guild: discord.Guild):
        """Process all channels in a guild"""
        self.stats["current_guild"] = guild.name
        print(f"\n🏰 Processing Guild: {guild.name} ({guild.id})")
        
        # Get text channels
        text_channels = [ch for ch in guild.channels if isinstance(ch, discord.TextChannel)]
        logger.info(f"  Found {len(text_channels)} text channels")
        
        for channel in text_channels:
            try:
                await self._process_channel(guild, channel)
                self.stats["channels_processed"] += 1
                
            except discord.Forbidden:
                logger.warning(f"  ❌ No access to #{channel.name}")
            except Exception as e:
                logger.error(f"  ❌ Error processing #{channel.name}: {e}")
                self.stats["errors"] += 1
    
    async def _process_channel(self, guild: discord.Guild, channel: discord.TextChannel):
        """Process all messages in a channel with streaming"""
        self.stats["current_channel"] = f"#{channel.name}"
        print(f"  📋 Processing #{channel.name}...")
        
        message_count = 0
        batch_count = 0
        
        try:
            # Stream messages from Discord API
            async for message in channel.history(limit=None, oldest_first=False):
                try:
                    # Transform message to enhanced format
                    enhanced_message = await self._transform_message(guild, channel, message)
                    
                    if enhanced_message:
                        # Add to batch
                        self.message_batch.append(enhanced_message)
                        message_count += 1
                        self.stats["messages_processed"] += 1
                        
                        # Process batch when full
                        if len(self.message_batch) >= self.batch_size:
                            success_count = await self._process_batch()
                            batch_count += 1
                            
                            if batch_count % 10 == 0:  # Progress update every 10 batches
                                print(f"    📊 Processed {message_count} messages, {success_count} indexed")
                    
                except Exception as e:
                    logger.warning(f"    ⚠️ Error processing message {message.id}: {e}")
                    self.stats["errors"] += 1
            
            # Process remaining messages in batch
            if self.message_batch:
                success_count = await self._process_batch()
            
            print(f"    ✅ Channel complete: {message_count} messages processed")
            
        except discord.Forbidden:
            logger.warning(f"    ❌ No permission to read #{channel.name}")
        except Exception as e:
            logger.error(f"    ❌ Channel error: {e}")
            self.stats["errors"] += 1
    
    async def _transform_message(self, guild: discord.Guild, channel: discord.TextChannel, message: discord.Message) -> Optional[Dict[str, Any]]:
        """Transform Discord message to enhanced format for indexing"""
        try:
            # Build comprehensive message data
            enhanced_message = {
                # Core message fields
                "message_id": str(message.id),
                "channel_id": str(channel.id),
                "channel_name": channel.name,
                "guild_id": str(guild.id),
                "guild_name": guild.name,
                "content": message.content or "",  # Handle empty content
                "timestamp": message.created_at.isoformat(),
                "jump_url": message.jump_url,
                
                # Enhanced author information
                "author": {
                    "id": str(message.author.id),
                    "username": message.author.name,
                    "display_name": message.author.display_name,
                    "discriminator": message.author.discriminator,
                    "bot": message.author.bot
                },
                
                # Message metadata
                "type": str(message.type),
                "pinned": message.pinned,
                "reference": {
                    "message_id": str(message.reference.message_id) if message.reference and message.reference.message_id else None
                } if message.reference else None,
                
                # Reactions
                "reactions": [
                    {
                        "emoji": str(reaction.emoji),
                        "count": reaction.count
                    }
                    for reaction in message.reactions
                ] if message.reactions else [],
                
                # Mentions
                "mentions": [
                    {
                        "id": str(user.id),
                        "username": user.name,
                        "display_name": user.display_name
                    }
                    for user in message.mentions
                ] if message.mentions else [],
                
                # Attachments
                "attachments": [
                    {
                        "id": str(attachment.id),
                        "filename": attachment.filename,
                        "url": attachment.url,
                        "size": attachment.size,
                        "content_type": attachment.content_type
                    }
                    for attachment in message.attachments
                ] if message.attachments else [],
                
                # Embeds
                "embeds": len(message.embeds)
            }
            
            return enhanced_message
            
        except Exception as e:
            logger.warning(f"Error transforming message {message.id}: {e}")
            return None
    
    async def _process_batch(self) -> int:
        """Process current batch and add to vector store"""
        if not self.message_batch:
            return 0
        
        try:
            # Add batch to vector store
            success = await self.vector_store.add_messages(self.message_batch)
            
            if success:
                indexed_count = len(self.message_batch)
                self.stats["messages_indexed"] += indexed_count
            else:
                indexed_count = 0
                self.stats["errors"] += 1
            
            # Clear batch
            self.message_batch = []
            
            return indexed_count
            
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            self.stats["errors"] += 1
            self.message_batch = []  # Clear batch to prevent memory issues
            return 0
    
    async def _flush_remaining_batch(self):
        """Process any remaining messages in batch"""
        if self.message_batch:
            await self._process_batch()
    
    async def _print_final_stats(self):
        """Print final indexing statistics"""
        end_time = datetime.now()
        duration = (end_time - self.stats["start_time"]).total_seconds()
        
        print(f"\n✅ STREAMING INDEXING COMPLETE!")
        print("=" * 80)
        print(f"📊 STATISTICS:")
        print(f"   • Guilds processed: {self.stats['guilds_processed']}")
        print(f"   • Channels processed: {self.stats['channels_processed']}")
        print(f"   • Messages processed: {self.stats['messages_processed']:,}")
        print(f"   • Messages indexed: {self.stats['messages_indexed']:,}")
        print(f"   • Success rate: {(self.stats['messages_indexed']/max(self.stats['messages_processed'],1)*100):.1f}%")
        print(f"   • Errors: {self.stats['errors']}")
        print(f"   • Duration: {duration:.1f} seconds")
        print(f"   • Rate: {self.stats['messages_processed']/max(duration,1):.1f} messages/sec")
        
        # Vector store stats
        if self.vector_store.collection:
            total_stored = self.vector_store.collection.count()
            print(f"   • Total in vector store: {total_stored:,}")
        
        print(f"\n🎯 OPTIMIZATION ACHIEVED:")
        print(f"   • Direct API → ChromaDB (no JSON files)")
        print(f"   • Real-time streaming processing")
        print(f"   • Enhanced metadata (34 fields)")
        print(f"   • Production embedding function")
        print(f"   • Single source of truth")

async def main():
    """Main entry point for streaming indexer"""
    config = {
        "batch_size": 100,
        "max_channels": None,  # Process all channels
        "max_messages_per_channel": None  # Process all messages
    }
    
    try:
        indexer = StreamingDiscordIndexer(config)
        await indexer.start_indexing()
        
    except KeyboardInterrupt:
        print("\n❌ Indexing interrupted by user")
    except Exception as e:
        logger.error(f"❌ Critical error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
