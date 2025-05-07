import os
import json
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

def extract_emojis(text):
    unicode_emojis = re.findall(r'[\U00010000-\U0010ffff]', text)
    custom_emojis = re.findall(r'<a?:\w+:\d+>', text)
    return {
        'unicode_emojis': unicode_emojis,
        'custom_emojis': custom_emojis
    }

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
                    continue
                print(f"  📄 Channel: #{channel.name} (ID: {channel.id})")

                last_message_id = None  # No JSON store; sync all or implement your own checkpoint
                new_messages = []
                try:
                    async for message in channel.history(
                        limit=None,
                        after=discord.Object(id=last_message_id) if last_message_id else None,
                        oldest_first=True
                    ):
                        new_messages.append({
                            'id': str(message.id),
                            'message_id': str(message.id),
                            'channel_id': str(channel.id),
                            'guild_id': str(guild.id),
                            'content': message.content.replace('\u2028', ' ').replace('\u2029', ' ').strip(),
                            'timestamp': message.created_at.isoformat(),
                            'edited_timestamp': message.edited_at.isoformat() if message.edited_at else None,
                            'jump_url': message.jump_url,
                            'author': {
                                'id': str(message.author.id),
                                'name': message.author.name,
                                'discriminator': message.author.discriminator,
                                'display_name': getattr(message.author, "display_name", message.author.name),
                                'is_bot': message.author.bot,
                                'avatar_url': str(message.author.avatar.url) if message.author.avatar else None,
                            },
                            'mentions': [str(user.id) for user in message.mentions],
                            'reactions': [
                                {
                                    'emoji': str(reaction.emoji),
                                    'count': reaction.count
                                } for reaction in message.reactions
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
                    print(f"    ❌ Error fetching channel #{channel.name}: {str(e)}")
                    self.sync_log_entry["errors"].append(str(e))
                    continue

                if new_messages:
                    print(f"    ✅ {len(new_messages)} new message(s)")
                    self.sync_log_entry["total_messages_synced"] += len(new_messages)

                    # Upsert each message into the SQLite DB
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

        await self.close()

async def update_discord_messages():
    print("🔌 Connecting to Discord...")
    client = DiscordFetcher(intents=intents)
    await client.start(DISCORD_TOKEN)

if __name__ == "__main__":
    asyncio.run(update_discord_messages())
    print("🔌 Disconnecting from Discord...")
