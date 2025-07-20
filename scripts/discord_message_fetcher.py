#!/usr/bin/env python3
"""
Fresh Discord Message Fetcher
Fetches messages directly from Discord and stores them in SQLite database

Note: This script automatically ignores channels that have "test" in their name
(case-insensitive) to avoid fetching test data.
"""

import asyncio
import json
import sqlite3
import os
import sys
import argparse
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import discord
import dotenv

# Load environment variables
dotenv.load_dotenv()

class DiscordMessageFetcher:
    def __init__(self):
        self.token = os.getenv('DISCORD_TOKEN')
        self.guild_id = int(os.getenv('DISCORD_GUILD_ID', '1353058864810950737'))
        self.db_path = project_root / 'data' / 'discord_messages.db'
        
        if not self.token:
            raise ValueError("DISCORD_TOKEN not found in environment variables")
        
        # Ensure data directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database with messages table"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    message_id TEXT PRIMARY KEY,
                    channel_id TEXT NOT NULL,
                    channel_name TEXT NOT NULL,
                    guild_id TEXT NOT NULL,
                    guild_name TEXT NOT NULL,
                    author_id TEXT NOT NULL,
                    author_username TEXT NOT NULL,
                    author_display_name TEXT,
                    content TEXT,
                    timestamp TEXT NOT NULL,
                    jump_url TEXT,
                    thread_id TEXT,
                    thread_name TEXT,
                    forum_channel_id TEXT,
                    forum_channel_name TEXT,
                    is_forum_thread BOOLEAN DEFAULT FALSE,
                    mentions TEXT,
                    reactions TEXT,
                    attachments TEXT,
                    embeds_count INTEGER DEFAULT 0,
                    pinned BOOLEAN DEFAULT FALSE,
                    message_type TEXT,
                    reference TEXT,
                    raw_data TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indices for better query performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_channel_id ON messages(channel_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_author_id ON messages(author_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON messages(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_thread_id ON messages(thread_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_forum_channel_id ON messages(forum_channel_id)")
            
            conn.commit()
    
    async def fetch_all_messages(self):
        """Fetch all messages from Discord server"""
        print("🚀 Starting Discord message fetch...")
        print(f"🏛️ Guild ID: {self.guild_id}")
        print(f"💾 Database: {self.db_path}")
        
        intents = discord.Intents.default()
        intents.message_content = True
        intents.guilds = True
        
        client = discord.Client(intents=intents)
        
        stats = {
            'text_channels': 0,
            'forum_channels': 0,
            'total_messages': 0,
            'total_threads': 0,
            'errors': []
        }
        
        @client.event
        async def on_ready():
            try:
                guild = client.get_guild(self.guild_id)
                if not guild:
                    print(f"❌ Guild not found: {self.guild_id}")
                    await client.close()
                    return
                
                print(f"✅ Connected to: {guild.name}")
                
                # Fetch all channels
                channels = await guild.fetch_channels()
                text_channels = [ch for ch in channels if isinstance(ch, discord.TextChannel)]
                forum_channels = [ch for ch in channels if isinstance(ch, discord.ForumChannel)]
                
                # Filter out channels with "test" in the name (case-insensitive)
                text_channels = [ch for ch in text_channels if "test" not in ch.name.lower()]
                forum_channels = [ch for ch in forum_channels if "test" not in ch.name.lower()]
                
                print(f"📊 Found {len(text_channels)} text channels and {len(forum_channels)} forum channels (excluding test channels)")
                
                # Process text channels
                for channel in text_channels:
                    try:
                        print(f"📥 Fetching from #{channel.name}")
                        await self.fetch_channel_messages(channel, stats)
                        stats['text_channels'] += 1
                    except Exception as e:
                        error_msg = f"Error fetching #{channel.name}: {e}"
                        print(f"❌ {error_msg}")
                        stats['errors'].append(error_msg)
                
                # Process forum channels
                for forum in forum_channels:
                    try:
                        print(f"📋 Fetching from forum #{forum.name}")
                        await self.fetch_forum_messages(forum, stats)
                        stats['forum_channels'] += 1
                    except Exception as e:
                        error_msg = f"Error fetching forum #{forum.name}: {e}"
                        print(f"❌ {error_msg}")
                        stats['errors'].append(error_msg)
                
                # Print final stats
                print("\n📊 Fetch Complete!")
                print(f"   💬 Text Channels: {stats['text_channels']}")
                print(f"   📋 Forum Channels: {stats['forum_channels']}")
                print(f"   📝 Total Messages: {stats['total_messages']:,}")
                print(f"   🧵 Total Threads: {stats['total_threads']}")
                print(f"   ❌ Errors: {len(stats['errors'])}")
                
                if stats['errors']:
                    print("\n⚠️ Errors encountered:")
                    for error in stats['errors'][:5]:  # Show first 5 errors
                        print(f"   • {error}")
                    if len(stats['errors']) > 5:
                        print(f"   ... and {len(stats['errors']) - 5} more")
                
            except Exception as e:
                print(f"❌ Error during fetch: {e}")
            finally:
                await client.close()
        
        await client.start(self.token)
    
    async def fetch_channel_messages(self, channel: discord.TextChannel, stats: Dict):
        """Fetch messages from a text channel"""
        try:
            messages = []
            async for message in channel.history(limit=None):
                message_data = self.convert_message_to_dict(message)
                messages.append(message_data)
                
                # Batch insert every 100 messages
                if len(messages) >= 100:
                    await self.insert_messages_batch(messages)
                    stats['total_messages'] += len(messages)
                    print(f"   ✅ {len(messages)} messages from #{channel.name}")
                    messages = []
            
            # Insert remaining messages
            if messages:
                await self.insert_messages_batch(messages)
                stats['total_messages'] += len(messages)
                print(f"   ✅ {len(messages)} messages from #{channel.name}")
                
        except discord.Forbidden:
            print(f"   ⚠️ No permission to read #{channel.name}")
        except Exception as e:
            raise e
    
    async def fetch_forum_messages(self, forum: discord.ForumChannel, stats: Dict):
        """Fetch messages from all threads in a forum channel"""
        try:
            # Get all threads (active + archived)
            threads = []
            
            # Active threads
            for thread in forum.threads:
                threads.append(thread)
            
            # Archived threads
            try:
                async for thread in forum.archived_threads(limit=None):
                    threads.append(thread)
            except discord.Forbidden:
                print(f"   ⚠️ No permission to access archived threads in #{forum.name}")
            
            print(f"   🧵 Found {len(threads)} threads in forum #{forum.name}")
            
            for thread in threads:
                try:
                    print(f"     🧵 Processing thread: {thread.name}")
                    await self.fetch_thread_messages(thread, forum, stats)
                    stats['total_threads'] += 1
                except Exception as e:
                    print(f"     ❌ Error processing thread {thread.name}: {e}")
                    
        except Exception as e:
            raise e
    
    async def fetch_thread_messages(self, thread: discord.Thread, forum: discord.ForumChannel, stats: Dict):
        """Fetch messages from a specific thread"""
        try:
            messages = []
            async for message in thread.history(limit=None):
                message_data = self.convert_message_to_dict(message)
                # Add thread-specific metadata
                message_data.update({
                    'thread_id': str(thread.id),
                    'thread_name': thread.name,
                    'forum_channel_id': str(forum.id),
                    'forum_channel_name': forum.name,
                    'is_forum_thread': True
                })
                messages.append(message_data)
                
                # Batch insert every 100 messages
                if len(messages) >= 100:
                    await self.insert_messages_batch(messages)
                    stats['total_messages'] += len(messages)
                    messages = []
            
            # Insert remaining messages
            if messages:
                await self.insert_messages_batch(messages)
                stats['total_messages'] += len(messages)
                
        except discord.Forbidden:
            print(f"       ⚠️ No permission to read thread {thread.name}")
        except Exception as e:
            raise e
    
    def convert_message_to_dict(self, message: discord.Message) -> Dict[str, Any]:
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
            "author_id": str(message.author.id),
            "author_username": message.author.name,
            "author_display_name": message.author.display_name,
            "content": message.content,
            "timestamp": message.created_at.isoformat(),
            "jump_url": message.jump_url,
            "mentions": json.dumps(mentions),
            "reactions": json.dumps(reactions),
            "attachments": json.dumps(attachments),
            "embeds_count": len(message.embeds) if message.embeds else 0,
            "pinned": message.pinned,
            "message_type": str(message.type),
            "reference": json.dumps(reference) if reference else None,
            "raw_data": json.dumps({
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
                "embeds": len(message.embeds) if message.embeds else 0,
                "pinned": message.pinned,
                "type": str(message.type),
                "reference": reference
            })
        }
    
    async def insert_messages_batch(self, messages: List[Dict[str, Any]]):
        """Insert a batch of messages into the database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            for msg in messages:
                cursor.execute("""
                    INSERT OR IGNORE INTO messages (
                        message_id, channel_id, channel_name, guild_id, guild_name,
                        author_id, author_username, author_display_name, content,
                        timestamp, jump_url, thread_id, thread_name, forum_channel_id,
                        forum_channel_name, is_forum_thread, mentions, reactions,
                        attachments, embeds_count, pinned, message_type, reference, raw_data
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    msg.get('message_id'),
                    msg.get('channel_id'),
                    msg.get('channel_name'),
                    msg.get('guild_id'),
                    msg.get('guild_name'),
                    msg.get('author_id'),
                    msg.get('author_username'),
                    msg.get('author_display_name'),
                    msg.get('content'),
                    msg.get('timestamp'),
                    msg.get('jump_url'),
                    msg.get('thread_id'),
                    msg.get('thread_name'),
                    msg.get('forum_channel_id'),
                    msg.get('forum_channel_name'),
                    msg.get('is_forum_thread', False),
                    msg.get('mentions'),
                    msg.get('reactions'),
                    msg.get('attachments'),
                    msg.get('embeds_count', 0),
                    msg.get('pinned', False),
                    msg.get('message_type'),
                    msg.get('reference'),
                    msg.get('raw_data')
                ))
            
            conn.commit()

async def main():
    """Main function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Fetch Discord messages and store them in SQLite database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python discord_message_fetcher.py --full          # Full fetch (all messages)
  python discord_message_fetcher.py --incremental   # Incremental fetch (new messages only)
  python discord_message_fetcher.py --help          # Show this help message

Note: This script automatically ignores channels that have "test" in their name
(case-insensitive) to avoid fetching test data.
        """
    )
    
    parser.add_argument(
        '--full',
        action='store_true',
        help='Perform a full fetch of all messages (default)'
    )
    
    parser.add_argument(
        '--incremental',
        action='store_true',
        help='Perform an incremental fetch of new messages only'
    )
    
    args = parser.parse_args()
    
    # If no arguments provided, default to full fetch
    if not args.full and not args.incremental:
        args.full = True
    
    try:
        fetcher = DiscordMessageFetcher()
        await fetcher.fetch_all_messages()
        print("\n🎉 Discord message fetch completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main()) 