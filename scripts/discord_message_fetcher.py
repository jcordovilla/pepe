#!/usr/bin/env python3
"""
Discord Message Fetcher with Incremental Sync Support
Fetches messages from Discord and stores them in SQLite database.
Uses FetchStateManager for proper incremental fetching.
"""

import asyncio
import json
import sqlite3
import os
import sys
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import discord
import dotenv
from tqdm import tqdm

# Import only the state manager directly to avoid pulling in all dependencies
# This is a standalone script that shouldn't need langgraph, etc.
import importlib.util
state_manager_path = project_root / 'agentic' / 'services' / 'fetch_state_manager.py'
spec = importlib.util.spec_from_file_location("fetch_state_manager", state_manager_path)
fetch_state_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fetch_state_module)
FetchStateManager = fetch_state_module.FetchStateManager

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

        # Initialize state manager for incremental fetching
        self.state_manager = FetchStateManager(
            state_file=str(project_root / 'data' / 'processing_markers' / 'fetch_state.json')
        )

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

    async def fetch_all_messages(self, full_sync: bool = False):
        """
        Fetch messages from Discord server.

        Args:
            full_sync: If True, fetches all messages. If False (default), uses incremental sync.
        """
        sync_mode = "FULL" if full_sync else "INCREMENTAL"
        print(f"🚀 Starting Discord message fetch ({sync_mode} SYNC)...")
        print(f"🏛️ Guild ID: {self.guild_id}")
        print(f"💾 Database: {self.db_path}")

        # Show state summary
        state_summary = self.state_manager.get_state_summary()
        if not full_sync and state_summary['total_channels'] > 0:
            print(f"📊 Tracking {state_summary['total_channels']} channels from previous syncs")

        intents = discord.Intents.default()
        intents.message_content = True
        intents.guilds = True

        client = discord.Client(intents=intents)

        stats = {
            'text_channels': 0,
            'forum_channels': 0,
            'total_messages': 0,
            'new_messages': 0,
            'total_threads': 0,
            'errors': [],
            'new_channels': 0,
            'channels_with_updates': 0
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

                print(f"📊 Found {len(text_channels)} text channels and {len(forum_channels)} forum channels")

                # Process text channels
                print(f"\n💬 Processing {len(text_channels)} text channels...")
                for channel in tqdm(text_channels, desc="💬 Text channels", unit="ch"):
                    try:
                        new_count = await self.fetch_channel_messages(
                            channel,
                            stats,
                            str(guild.id),
                            guild.name,
                            incremental=not full_sync
                        )
                        stats['text_channels'] += 1
                        if new_count > 0:
                            stats['channels_with_updates'] += 1
                    except Exception as e:
                        error_msg = f"Error fetching #{channel.name}: {e}"
                        print(f"\n❌ {error_msg}")
                        stats['errors'].append(error_msg)

                # Process forum channels
                if forum_channels:
                    print(f"\n📋 Processing {len(forum_channels)} forum channels...")
                    for forum in tqdm(forum_channels, desc="📋 Forum channels", unit="forum"):
                        try:
                            await self.fetch_forum_messages(
                                forum,
                                stats,
                                str(guild.id),
                                guild.name,
                                incremental=not full_sync
                            )
                            stats['forum_channels'] += 1
                        except Exception as e:
                            error_msg = f"Error fetching forum #{forum.name}: {e}"
                            print(f"\n❌ {error_msg}")
                            stats['errors'].append(error_msg)

                # Print final stats
                print("\n" + "=" * 60)
                print("📊 Fetch Complete!")
                print("=" * 60)
                print(f"   💬 Text Channels processed: {stats['text_channels']}")
                print(f"   📋 Forum Channels processed: {stats['forum_channels']}")
                print(f"   🧵 Total Threads: {stats['total_threads']}")
                print(f"   📝 New messages fetched: {stats['new_messages']:,}")
                print(f"   📂 Channels with updates: {stats['channels_with_updates']}")

                if stats['new_channels'] > 0:
                    print(f"   🆕 New channels discovered: {stats['new_channels']}")

                if stats['errors']:
                    print(f"   ❌ Errors: {len(stats['errors'])}")
                    for error in stats['errors'][:5]:
                        print(f"      • {error}")
                    if len(stats['errors']) > 5:
                        print(f"      ... and {len(stats['errors']) - 5} more")

                # Query database for total count
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute("SELECT COUNT(*) FROM messages")
                    total_in_db = cursor.fetchone()[0]
                    print(f"\n   💾 Total messages in database: {total_in_db:,}")

            except Exception as e:
                print(f"❌ Error during fetch: {e}")
                import traceback
                traceback.print_exc()
            finally:
                await client.close()

        await client.start(self.token)
        return stats

    async def fetch_channel_messages(
        self,
        channel: discord.TextChannel,
        stats: Dict,
        guild_id: str,
        guild_name: str,
        incremental: bool = True
    ) -> int:
        """
        Fetch messages from a text channel with incremental support.

        Returns the number of new messages fetched.
        """
        try:
            # Get last fetch timestamp for incremental sync
            after_datetime: Optional[datetime] = None
            is_new_channel = False

            if incremental:
                after_datetime = self.state_manager.get_channel_last_fetch(guild_id, str(channel.id))
                if after_datetime:
                    # Add a small buffer to avoid missing messages at the boundary
                    pass  # Discord API handles this correctly
                else:
                    is_new_channel = True
                    stats['new_channels'] += 1

            messages = []
            message_count = 0
            last_message_timestamp: Optional[str] = None

            # Fetch messages - use 'after' parameter for incremental sync
            if after_datetime and incremental:
                # Incremental: fetch only new messages after last sync
                async for message in channel.history(limit=None, after=after_datetime, oldest_first=True):
                    message_data = self.convert_message_to_dict(message, guild_id, guild_name)
                    messages.append(message_data)
                    message_count += 1
                    last_message_timestamp = message.created_at.isoformat()

                    # Batch insert every 100 messages
                    if len(messages) >= 100:
                        await self.insert_messages_batch(messages)
                        messages = []
            else:
                # Full sync: fetch all messages
                async for message in channel.history(limit=None):
                    message_data = self.convert_message_to_dict(message, guild_id, guild_name)
                    messages.append(message_data)
                    message_count += 1

                    # Track the newest message timestamp
                    if last_message_timestamp is None:
                        last_message_timestamp = message.created_at.isoformat()

                    # Batch insert every 100 messages
                    if len(messages) >= 100:
                        await self.insert_messages_batch(messages)
                        messages = []

            # Insert remaining messages
            if messages:
                await self.insert_messages_batch(messages)

            # Update state with the newest message timestamp
            if last_message_timestamp:
                self.state_manager.update_channel_state(
                    guild_id=guild_id,
                    channel_id=str(channel.id),
                    channel_name=channel.name,
                    last_message_timestamp=last_message_timestamp,
                    message_count=message_count,
                    fetch_mode="full" if not incremental or is_new_channel else "incremental"
                )
            elif not after_datetime:
                # Empty channel on first sync - still record it to avoid re-scanning
                self.state_manager.update_channel_state(
                    guild_id=guild_id,
                    channel_id=str(channel.id),
                    channel_name=channel.name,
                    last_message_timestamp=datetime.now(timezone.utc).isoformat(),
                    message_count=0,
                    fetch_mode="full"
                )

            stats['new_messages'] += message_count

            if message_count > 0:
                mode_str = "🆕 new" if is_new_channel else ("📥 new" if incremental else "📝")
                tqdm.write(f"   {mode_str} #{channel.name}: {message_count:,} messages")

            return message_count

        except discord.Forbidden:
            tqdm.write(f"   ⚠️ No permission to read #{channel.name}")
            return 0
        except Exception as e:
            raise e

    async def fetch_forum_messages(
        self,
        forum: discord.ForumChannel,
        stats: Dict,
        guild_id: str,
        guild_name: str,
        incremental: bool = True
    ):
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
                tqdm.write(f"   ⚠️ No permission to access archived threads in #{forum.name}")

            if threads:
                tqdm.write(f"   🧵 Forum #{forum.name}: {len(threads)} threads")

            for thread in threads:
                try:
                    new_count = await self.fetch_thread_messages(
                        thread,
                        forum,
                        stats,
                        guild_id,
                        guild_name,
                        incremental=incremental
                    )
                    stats['total_threads'] += 1
                    if new_count > 0:
                        stats['channels_with_updates'] += 1
                except Exception as e:
                    tqdm.write(f"     ❌ Error processing thread {thread.name}: {e}")

        except Exception as e:
            raise e

    async def fetch_thread_messages(
        self,
        thread: discord.Thread,
        forum: discord.ForumChannel,
        stats: Dict,
        guild_id: str,
        guild_name: str,
        incremental: bool = True
    ) -> int:
        """Fetch messages from a specific thread with incremental support"""
        try:
            # Use composite key for thread state
            thread_key = f"{forum.id}_{thread.id}"

            # Get last fetch timestamp for incremental sync
            after_datetime: Optional[datetime] = None
            is_new_thread = False

            if incremental:
                after_datetime = self.state_manager.get_channel_last_fetch(guild_id, thread_key)
                if not after_datetime:
                    is_new_thread = True

            messages = []
            message_count = 0
            last_message_timestamp: Optional[str] = None

            # Fetch messages
            if after_datetime and incremental:
                async for message in thread.history(limit=None, after=after_datetime, oldest_first=True):
                    message_data = self.convert_message_to_dict(message, guild_id, guild_name)
                    message_data.update({
                        'thread_id': str(thread.id),
                        'thread_name': thread.name,
                        'forum_channel_id': str(forum.id),
                        'forum_channel_name': forum.name,
                        'is_forum_thread': True
                    })
                    messages.append(message_data)
                    message_count += 1
                    last_message_timestamp = message.created_at.isoformat()

                    if len(messages) >= 100:
                        await self.insert_messages_batch(messages)
                        messages = []
            else:
                async for message in thread.history(limit=None):
                    message_data = self.convert_message_to_dict(message, guild_id, guild_name)
                    message_data.update({
                        'thread_id': str(thread.id),
                        'thread_name': thread.name,
                        'forum_channel_id': str(forum.id),
                        'forum_channel_name': forum.name,
                        'is_forum_thread': True
                    })
                    messages.append(message_data)
                    message_count += 1

                    if last_message_timestamp is None:
                        last_message_timestamp = message.created_at.isoformat()

                    if len(messages) >= 100:
                        await self.insert_messages_batch(messages)
                        messages = []

            # Insert remaining messages
            if messages:
                await self.insert_messages_batch(messages)

            # Update state
            if last_message_timestamp:
                self.state_manager.update_channel_state(
                    guild_id=guild_id,
                    channel_id=thread_key,
                    channel_name=f"{forum.name}/{thread.name}",
                    last_message_timestamp=last_message_timestamp,
                    message_count=message_count,
                    fetch_mode="full" if not incremental or is_new_thread else "incremental"
                )

            stats['new_messages'] += message_count

            if message_count > 0:
                tqdm.write(f"       📥 Thread {thread.name}: {message_count:,} messages")

            return message_count

        except discord.Forbidden:
            tqdm.write(f"       ⚠️ No permission to read thread {thread.name}")
            return 0
        except Exception as e:
            raise e

    def convert_message_to_dict(
        self,
        message: discord.Message,
        guild_id: str,
        guild_name: str
    ) -> Dict[str, Any]:
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
            "guild_id": guild_id,
            "guild_name": guild_name,
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
                "guild_id": guild_id,
                "guild_name": guild_name,
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
    import argparse

    parser = argparse.ArgumentParser(description='Discord Message Fetcher with Incremental Sync')
    parser.add_argument('--full', action='store_true', help='Perform full sync instead of incremental')
    parser.add_argument('--reset-state', action='store_true', help='Reset sync state before fetching')
    args = parser.parse_args()

    try:
        fetcher = DiscordMessageFetcher()

        if args.reset_state:
            print("🔄 Resetting sync state...")
            fetcher.state_manager.reset_state()

        await fetcher.fetch_all_messages(full_sync=args.full)
        print("\n🎉 Discord message fetch completed successfully!")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
