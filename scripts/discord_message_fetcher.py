#!/usr/bin/env python3
"""
Fresh Discord Message Fetcher
Fetches messages directly from Discord and stores them in SQLite database
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
from tqdm import tqdm

# Load environment variables
dotenv.load_dotenv()

class DiscordMessageFetcher:
    def __init__(self):
        self.token = os.getenv('DISCORD_TOKEN')
        self.guild_id = int(os.getenv('DISCORD_GUILD_ID', '1353058864810950737'))
        self.db_path = project_root / 'data' / 'discord_messages.db'
        self.checkpoint_path = project_root / 'data' / 'fetching_checkpoint.json'
        
        if not self.token:
            raise ValueError("DISCORD_TOKEN not found in environment variables")
        
        # Ensure data directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self.init_database()
        
        # Load checkpoint for incremental fetching
        self.checkpoint = self.load_checkpoint()
    
    def load_checkpoint(self) -> Dict[str, Any]:
        """Load checkpoint data for incremental fetching"""
        if self.checkpoint_path.exists():
            try:
                with open(self.checkpoint_path, 'r') as f:
                    checkpoint = json.load(f)
                print(f"📋 Loaded checkpoint from {checkpoint.get('last_updated', 'unknown')}")
                return checkpoint
            except Exception as e:
                print(f"⚠️ Warning: Could not load checkpoint: {e}")
        return {
            'channel_checkpoints': {},
            'forum_checkpoints': {},
            'last_updated': None,
            'total_messages_fetched': 0
        }
    
    def save_checkpoint(self, channel_id: str, last_message_id: str, channel_type: str = 'text'):
        """Save checkpoint data for incremental fetching"""
        try:
            if channel_type == 'text':
                self.checkpoint['channel_checkpoints'][channel_id] = last_message_id
            elif channel_type == 'forum':
                self.checkpoint['forum_checkpoints'][channel_id] = last_message_id
            
            self.checkpoint['last_updated'] = datetime.now().isoformat()
            
            with open(self.checkpoint_path, 'w') as f:
                json.dump(self.checkpoint, f, indent=2)
        except Exception as e:
            print(f"⚠️ Warning: Could not save checkpoint: {e}")
    
    def get_last_message_id(self, channel_id: str, channel_type: str = 'text') -> str:
        """Get the last message ID for a channel from checkpoint"""
        if channel_type == 'text':
            return self.checkpoint['channel_checkpoints'].get(channel_id)
        elif channel_type == 'forum':
            return self.checkpoint['forum_checkpoints'].get(channel_id)
        return None
    
    def cleanup_test_messages(self):
        """Remove all messages from channels with 'test' in the name"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Delete messages from test channels
                cursor = conn.execute("""
                    DELETE FROM messages 
                    WHERE LOWER(channel_name) LIKE '%test%' 
                    OR LOWER(thread_name) LIKE '%test%'
                    OR LOWER(forum_channel_name) LIKE '%test%'
                """)
                deleted_count = cursor.rowcount
                conn.commit()
                
                if deleted_count > 0:
                    print(f"🗑️ Cleaned up {deleted_count:,} messages from test channels")
                else:
                    print("ℹ️ No test messages found to clean up")
                    
        except Exception as e:
            print(f"⚠️ Warning: Could not clean up test messages: {e}")
    
    def is_test_channel(self, channel_name: str) -> bool:
        """Check if a channel name contains 'test' (case insensitive)"""
        return 'test' in channel_name.lower()
    
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
            'errors': [],
            'new_text_channels': 0,
            'new_forum_channels': 0,
            'new_channels_messages': 0
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
                
                # Clean up any existing test messages
                self.cleanup_test_messages()
                
                # Fetch all channels
                channels = await guild.fetch_channels()
                text_channels = [ch for ch in channels if isinstance(ch, discord.TextChannel) and not self.is_test_channel(ch.name)]
                forum_channels = [ch for ch in channels if isinstance(ch, discord.ForumChannel) and not self.is_test_channel(ch.name)]
                
                # Detect new channels for incremental sync
                known_text_channel_ids = set(self.checkpoint['channel_checkpoints'].keys())
                known_forum_channel_ids = set(self.checkpoint['forum_checkpoints'].keys())
                
                new_text_channels = [ch for ch in text_channels if str(ch.id) not in known_text_channel_ids]
                new_forum_channels = [ch for ch in forum_channels if str(ch.id) not in known_forum_channel_ids]
                
                if new_text_channels or new_forum_channels:
                    print(f"🆕 Detected {len(new_text_channels)} new text channels and {len(new_forum_channels)} new forum channels!")
                    if new_text_channels:
                        print(f"   New text channels: {', '.join(['#' + ch.name for ch in new_text_channels[:5]])}")
                        if len(new_text_channels) > 5:
                            print(f"   ... and {len(new_text_channels) - 5} more")
                    if new_forum_channels:
                        print(f"   New forum channels: {', '.join(['#' + ch.name for ch in new_forum_channels[:5]])}")
                        if len(new_forum_channels) > 5:
                            print(f"   ... and {len(new_forum_channels) - 5} more")
                
                print(f"📊 Found {len(text_channels)} text channels and {len(forum_channels)} forum channels (excluding test channels)")
                print(f"   • {len(new_text_channels)} new text channels to sync from beginning")
                print(f"   • {len(text_channels) - len(new_text_channels)} existing text channels for incremental sync")
                print(f"   • {len(new_forum_channels)} new forum channels to sync from beginning")
                print(f"   • {len(forum_channels) - len(new_forum_channels)} existing forum channels for incremental sync")
                
                # Process text channels
                print(f"\n📥 Processing {len(text_channels)} text channels...")
                for channel in tqdm(text_channels, desc="📥 Text channels", unit="channel"):
                    try:
                        # Track if this is a new channel
                        is_new = str(channel.id) not in known_text_channel_ids
                        messages_before = stats['total_messages']
                        
                        await self.fetch_channel_messages(channel, stats)
                        stats['text_channels'] += 1
                        
                        if is_new:
                            stats['new_text_channels'] += 1
                            stats['new_channels_messages'] += (stats['total_messages'] - messages_before)
                    except Exception as e:
                        error_msg = f"Error fetching #{channel.name}: {e}"
                        print(f"❌ {error_msg}")
                        stats['errors'].append(error_msg)
                
                # Process forum channels
                if forum_channels:
                    print(f"\n📋 Processing {len(forum_channels)} forum channels...")
                    for forum in tqdm(forum_channels, desc="📋 Forum channels", unit="forum"):
                        try:
                            # Track if this is a new channel
                            is_new = str(forum.id) not in known_forum_channel_ids
                            messages_before = stats['total_messages']
                            
                            await self.fetch_forum_messages(forum, stats)
                            stats['forum_channels'] += 1
                            
                            if is_new:
                                stats['new_forum_channels'] += 1
                                stats['new_channels_messages'] += (stats['total_messages'] - messages_before)
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
                
                # Show new channels info if any were detected
                if stats['new_text_channels'] > 0 or stats['new_forum_channels'] > 0:
                    print(f"\n🆕 New Channels Detected:")
                    if stats['new_text_channels'] > 0:
                        print(f"   • {stats['new_text_channels']} new text channels synced")
                    if stats['new_forum_channels'] > 0:
                        print(f"   • {stats['new_forum_channels']} new forum channels synced")
                    print(f"   • {stats['new_channels_messages']:,} messages from new channels")
                
                print(f"\n   ❌ Errors: {len(stats['errors'])}")
                
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
        """Fetch messages from a text channel with incremental support"""
        try:
            messages = []
            message_count = 0
            last_message_id = None
            
            # Get last message ID from checkpoint for incremental fetching
            checkpoint_id = self.get_last_message_id(str(channel.id), 'text')
            
            if checkpoint_id:
                print(f"   🔄 Incremental fetch from message {checkpoint_id[:8]}...")
                # Fetch messages after the checkpoint
                async for message in channel.history(limit=None, after=discord.Object(id=int(checkpoint_id))):
                    message_data = self.convert_message_to_dict(message)
                    messages.append(message_data)
                    message_count += 1
                    last_message_id = str(message.id)
                    
                    # Batch insert every 100 messages
                    if len(messages) >= 100:
                        await self.insert_messages_batch(messages)
                        stats['total_messages'] += len(messages)
                        messages = []
            else:
                print(f"   📥 Full fetch (no checkpoint found)...")
                # Create progress bar for this channel
                with tqdm(desc=f"📥 #{channel.name}", unit="msgs", leave=False) as pbar:
                    async for message in channel.history(limit=None):
                        message_data = self.convert_message_to_dict(message)
                        messages.append(message_data)
                        message_count += 1
                        last_message_id = str(message.id)
                        pbar.update(1)
                        
                        # Batch insert every 100 messages
                        if len(messages) >= 100:
                            await self.insert_messages_batch(messages)
                            stats['total_messages'] += len(messages)
                            pbar.set_postfix({"batch": len(messages)})
                            messages = []
            
            # Insert remaining messages
            if messages:
                await self.insert_messages_batch(messages)
                stats['total_messages'] += len(messages)
            
            # Save checkpoint with last message ID
            # For empty channels, save the channel's last_message_id or a placeholder
            if last_message_id:
                self.save_checkpoint(str(channel.id), last_message_id, 'text')
            elif not checkpoint_id:
                # New channel with no messages - save current timestamp as checkpoint
                # This prevents re-detecting empty channels
                if channel.last_message_id:
                    self.save_checkpoint(str(channel.id), str(channel.last_message_id), 'text')
                else:
                    # Truly empty channel - save a sentinel value (current snowflake ID)
                    from datetime import datetime, timezone
                    sentinel_id = str(int(datetime.now(timezone.utc).timestamp() * 1000) << 22)
                    self.save_checkpoint(str(channel.id), sentinel_id, 'text')
            
            if checkpoint_id:
                print(f"   ✅ {message_count:,} new messages from #{channel.name}")
            else:
                print(f"   ✅ {message_count:,} messages from #{channel.name}")
                
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
            
            # Filter out test threads
            non_test_threads = [thread for thread in threads if not self.is_test_channel(thread.name)]
            skipped_test_threads = len(threads) - len(non_test_threads)
            
            if skipped_test_threads > 0:
                print(f"   🚫 Skipped {skipped_test_threads} test threads in forum #{forum.name}")
            
            print(f"   🧵 Found {len(non_test_threads)} threads in forum #{forum.name}")
            
            for thread in tqdm(non_test_threads, desc=f"Processing threads in {forum.name}"):
                try:
                    print(f"     🧵 Processing thread: {thread.name}")
                    await self.fetch_thread_messages(thread, forum, stats)
                    stats['total_threads'] += 1
                except Exception as e:
                    print(f"     ❌ Error processing thread {thread.name}: {e}")
                    
        except Exception as e:
            raise e
    
    async def fetch_thread_messages(self, thread: discord.Thread, forum: discord.ForumChannel, stats: Dict):
        """Fetch messages from a specific thread with incremental support"""
        try:
            messages = []
            message_count = 0
            last_message_id = None
            
            # Get last message ID from checkpoint for incremental fetching
            thread_checkpoint_id = self.get_last_message_id(str(thread.id), 'forum')
            
            if thread_checkpoint_id:
                print(f"       🔄 Incremental fetch from message {thread_checkpoint_id[:8]}...")
                # Fetch messages after the checkpoint
                async for message in thread.history(limit=None, after=discord.Object(id=int(thread_checkpoint_id))):
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
                    message_count += 1
                    last_message_id = str(message.id)
                    
                    # Batch insert every 100 messages
                    if len(messages) >= 100:
                        await self.insert_messages_batch(messages)
                        stats['total_messages'] += len(messages)
                        messages = []
            else:
                print(f"       📥 Full fetch (no checkpoint found)...")
                # Create progress bar for this thread
                with tqdm(desc=f"     🧵 {thread.name[:30]}", unit="msgs", leave=False) as pbar:
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
                        message_count += 1
                        last_message_id = str(message.id)
                        pbar.update(1)
                        
                        # Batch insert every 100 messages
                        if len(messages) >= 100:
                            await self.insert_messages_batch(messages)
                            stats['total_messages'] += len(messages)
                            pbar.set_postfix({"batch": len(messages)})
                            messages = []
            
            # Insert remaining messages
            if messages:
                await self.insert_messages_batch(messages)
                stats['total_messages'] += len(messages)
            
            # Save checkpoint with last message ID
            # For empty threads, save the thread's last_message_id or a placeholder
            if last_message_id:
                self.save_checkpoint(str(thread.id), last_message_id, 'forum')
            elif not thread_checkpoint_id:
                # New thread with no messages - save current timestamp as checkpoint
                if thread.last_message_id:
                    self.save_checkpoint(str(thread.id), str(thread.last_message_id), 'forum')
                else:
                    # Truly empty thread - save a sentinel value
                    from datetime import datetime, timezone
                    sentinel_id = str(int(datetime.now(timezone.utc).timestamp() * 1000) << 22)
                    self.save_checkpoint(str(thread.id), sentinel_id, 'forum')
            
            if thread_checkpoint_id:
                print(f"       ✅ {message_count:,} new messages from thread {thread.name}")
            elif message_count > 0:
                print(f"       ✅ {message_count:,} messages from thread {thread.name}")
                
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
    parser = argparse.ArgumentParser(description='Discord Message Fetcher with Incremental Support')
    parser.add_argument('--full', action='store_true', 
                       help='Force full fetch (ignore checkpoints)')
    parser.add_argument('--reset-checkpoint', action='store_true',
                       help='Reset checkpoint and start fresh')
    parser.add_argument('--cleanup-test', action='store_true',
                       help='Clean up test messages and exit (no fetching)')
    args = parser.parse_args()
    
    try:
        fetcher = DiscordMessageFetcher()
        
        # Handle checkpoint reset
        if args.reset_checkpoint:
            if fetcher.checkpoint_path.exists():
                fetcher.checkpoint_path.unlink()
                print("🗑️ Checkpoint reset - will perform full fetch")
            else:
                print("ℹ️ No checkpoint found - will perform full fetch")
        
        # Handle cleanup-only mode
        if args.cleanup_test:
            fetcher.cleanup_test_messages()
            print("✅ Test message cleanup completed")
            return
        
        # Handle full fetch mode
        if args.full:
            if fetcher.checkpoint_path.exists():
                fetcher.checkpoint_path.unlink()
                print("🔄 Full fetch mode - checkpoint cleared")
            else:
                print("🔄 Full fetch mode - no checkpoint to clear")
        
        await fetcher.fetch_all_messages()
        print("\n🎉 Discord message fetch completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main()) 