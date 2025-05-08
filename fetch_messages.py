# fetch_messages.py
import os
import asyncio
from datetime import datetime
from db import SessionLocal, Message
from dotenv import load_dotenv
import discord
from utils.logger import setup_logging

setup_logging()

import logging
log = logging.getLogger(__name__)

load_dotenv()
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")

# Skip these channels by name or ID
SKIP_CHANNEL_NAMES = {"test-channel", "🛝discord-playground"}

intents = discord.Intents.default()
intents.message_content = True
intents.guilds = True
intents.messages = True

PAGE_SIZE = 100  # Number of messages to fetch per batch

class DiscordFetcher(discord.Client):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sync_log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "guilds_synced": [],
            "channels_skipped": [],
            "errors": [],
            "total_messages_synced": 0
        }

    async def on_ready(self):
        log.info(f"✅ Logged in as: {self.user} (ID: {self.user.id})")

        for guild in self.guilds:
            log.info(f"\n📂 Guild: {guild.name} (ID: {guild.id})")
            self.sync_log_entry["guilds_synced"].append({
                "guild_name": guild.name,
                "guild_id": str(guild.id)
            })

            for channel in guild.text_channels:
                if channel.name in SKIP_CHANNEL_NAMES:
                    log.info(f"  🚫 Skipping test channel #{channel.name}")
                    self.sync_log_entry["channels_skipped"].append({
                        "guild_name": guild.name,
                        "channel_name": channel.name,
                        "channel_id": str(channel.id),
                        "reason": "Skipped by name"
                    })
                    continue

                # Determine last synced message for this channel
                db = SessionLocal()
                last_msg = (
                    db.query(Message)
                      .filter_by(guild_id=guild.id, channel_id=channel.id)
                      .order_by(Message.message_id.desc())
                      .first()
                )
                db.close()
                after_id = last_msg.message_id if last_msg else None

                log.info(f"  📄 Channel: #{channel.name} | after={after_id}")
                fetched = []
                try:
                    # Paginate history in fixed-size batches
                    while True:
                        batch = []
                        async for message in channel.history(
                            limit=PAGE_SIZE,
                            after=discord.Object(id=after_id) if after_id else None,
                            oldest_first=True
                        ):
                            batch.append(message)
                        if not batch:
                            break
                        # Collect fetched messages
                        for message in batch:
                            fetched.append({
                                'message_id': str(message.id),
                                'channel_id': str(channel.id),
                                'guild_id': str(guild.id),
                                'content': message.content.replace('\u2028', ' ').replace('\u2029', ' ').strip(),
                                'timestamp': message.created_at.isoformat(),
                                'jump_url': message.jump_url,
                                'author': {
                                    'id': str(message.author.id),
                                    'username': message.author.name,
                                    'discriminator': message.author.discriminator,
                                },
                                'mentions': [str(user.id) for user in message.mentions],
                                'reactions': [
                                    {'emoji': str(r.emoji), 'count': r.count} for r in message.reactions
                                ],
                            })
                        # Update cursor to last message of this batch
                        after_id = batch[-1].id
                except discord.Forbidden:
                    log.info(f"    🚫 Skipped channel #{channel.name}: insufficient permissions")
                    self.sync_log_entry["channels_skipped"].append({
                        "guild_name": guild.name,
                        "channel_name": channel.name,
                        "channel_id": str(channel.id),
                        "reason": "Forbidden"
                    })
                    continue
                except Exception as e:
                    log.error(f"    ❌ Error in channel {channel.name}: {e}")
                    self.sync_log_entry["errors"].append(str(e))
                    continue

                if fetched:
                    log.info(f"    ✅ Fetched {len(fetched)} new messages")
                    self.sync_log_entry["total_messages_synced"] += len(fetched)

                    db = SessionLocal()
                    for m in fetched:
                        ts = datetime.fromisoformat(m['timestamp'])
                        db_msg = Message(
                            guild_id=int(m['guild_id']),
                            channel_id=int(m['channel_id']),
                            message_id=int(m['message_id']),
                            content=m['content'],
                            timestamp=ts,
                            author=m['author'],
                            mention_ids=[int(mid) for mid in m['mentions']],
                            reactions=[{'emoji': r['emoji'], 'count': r['count']} for r in m['reactions']],
                            jump_url=m.get('jump_url')
                        )
                        db.merge(db_msg)
                    db.commit()
                    db.close()
                else:
                    log.info(f"    📫 No new messages")

        log.info("🔌 Sync complete, closing connection...")
        await self.close()

async def main():
    log.info("🔌 Connecting to Discord...")
    client = DiscordFetcher(intents=intents)
    try:
        await client.start(DISCORD_TOKEN)
    finally:
        await client.close()
        log.info("🔌 Disconnected from Discord.")

if __name__ == "__main__":
    asyncio.run(main())