# fetch_messages.py
import os
import asyncio
import re
from datetime import datetime
from db import SessionLocal, Message
from dotenv import load_dotenv
import discord

load_dotenv()
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")

# Skip these channels by name or ID
SKIP_CHANNEL_NAMES = {"test-channel", "🛝discord-playground"}

intents = discord.Intents.default()
intents.message_content = True
intents.guilds = True
intents.messages = True

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
        print(f"✅ Logged in as: {self.user} (ID: {self.user.id})")

        for guild in self.guilds:
            print(f"\n📂 Guild: {guild.name} (ID: {guild.id})")
            self.sync_log_entry["guilds_synced"].append({
                "guild_name": guild.name,
                "guild_id": str(guild.id)
            })

            for channel in guild.text_channels:
                if channel.name in SKIP_CHANNEL_NAMES:
                    print(f"  🚫 Skipping test channel #{channel.name}")
                    self.sync_log_entry["channels_skipped"].append({
                        "guild_name": guild.name,
                        "channel_name": channel.name,
                        "channel_id": str(channel.id),
                        "reason": "Skipped by name"
                    })
                    continue

                # Determine last synced message for this channel to avoid refetching
                db = SessionLocal()
                last_msg = (
                    db.query(Message)
                      .filter_by(guild_id=guild.id, channel_id=channel.id)
                      .order_by(Message.message_id.desc())
                      .first()
                )
                db.close()
                last_message_id = last_msg.message_id if last_msg else None

                print(f"  📄 Channel: #{channel.name} (ID: {channel.id}) | after={last_message_id}")
                new_messages = []
                try:
                    async for message in channel.history(
                        limit=None,
                        after=discord.Object(id=last_message_id) if last_message_id else None,
                        oldest_first=True
                    ):
                        new_messages.append({
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
                except discord.Forbidden:
                    print(f"    🚫 Skipped channel #{channel.name}: insufficient permissions")
                    self.sync_log_entry["channels_skipped"].append({
                        "guild_name": guild.name,
                        "channel_name": channel.name,
                        "channel_id": str(channel.id),
                        "reason": "Forbidden - missing read permissions"
                    })
                    continue
                except Exception as e:
                    print(f"    ❌ Error fetching channel #{channel.name}: {e}")
                    self.sync_log_entry["errors"].append(str(e))
                    continue

                if new_messages:
                    print(f"    ✅ {len(new_messages)} new message(s)")
                    self.sync_log_entry["total_messages_synced"] += len(new_messages)

                    db = SessionLocal()
                    for m in new_messages:
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
                    print(f"    📫 No new messages")

# Top-level runner
def create_and_run():
    print("🔌 Connecting to Discord...")
    client = DiscordFetcher(intents=intents)
    try:
        asyncio.run(client.start(DISCORD_TOKEN))
    finally:
        print("🔌 Disconnected from Discord.")

if __name__ == "__main__":
    create_and_run()