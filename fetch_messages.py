import os
import json
import asyncio
import re
from datetime import datetime
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
    def __init__(self, data_store, **kwargs):
        super().__init__(**kwargs)
        self.data_store = data_store
        self.new_data = {}
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
            self.new_data.setdefault(guild.name, {})
            self.sync_log_entry["guilds_synced"].append({
                "guild_name": guild.name,
                "guild_id": str(guild.id)
            })

            for channel in guild.text_channels:
                if channel.name in SKIP_CHANNEL_NAMES:
                    print(f"  🚫 Skipping test channel #{channel.name}")
                    continue
                print(f"  📄 Channel: #{channel.name} (ID: {channel.id})")

                existing_msgs = self.data_store.get(guild.name, {}).get(channel.name, [])
                last_message_id = existing_msgs[-1]['id'] if existing_msgs else None

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
                            'user_presence': {
                                'status': str(message.author.status),
                                'activity': str(message.author.activity.name) if message.author.activity else None,
                            } if isinstance(message.author, discord.Member) else None,
                            'guild': {
                                'id': str(guild.id),
                                'name': guild.name,
                                'member_count': guild.member_count,
                            } if guild else None,
                            'channel': {
                                'id': str(channel.id),
                                'name': channel.name,
                                'type': str(channel.type),
                                'topic': getattr(channel, 'topic', None),
                            },
                            'thread': {
                                'id': str(message.thread.id),
                                'name': message.thread.name
                            } if message.thread else None,
                            'mentions': [str(user.id) for user in message.mentions],
                            'mention_everyone': message.mention_everyone,
                            'mention_roles': [str(role.id) for role in message.role_mentions],
                            'referenced_message_id': str(message.reference.message_id) if message.reference else None,
                            'attachments': [
                                {
                                    'url': att.url,
                                    'filename': att.filename,
                                    'size': att.size,
                                    'content_type': att.content_type
                                } for att in message.attachments
                            ],
                            'embeds': [embed.to_dict() for embed in message.embeds],
                            'reactions': [
                                {
                                    'emoji': str(reaction.emoji),
                                    'count': reaction.count,
                                    'me': reaction.me
                                } for reaction in message.reactions
                            ],
                            'emoji_stats': extract_emojis(message.content),
                            'pinned': message.pinned,
                            'flags': message.flags.value if message.flags else None,
                            'nonce': message.nonce,
                            'type': str(message.type),
                            'is_system': message.is_system(),
                            'mentions_everyone': message.mention_everyone,
                            'message_type': str(message.type),
                            'has_reactions': bool(message.reactions),
                            'is_pinned': message.pinned
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
                    self.new_data[guild.name].setdefault(channel.name, []).extend(new_messages)
                else:
                    print(f"    📫 No new messages")

        await self.close()

def load_existing_data(path='discord_messages.json'):
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_combined_data(existing, new_data, path='discord_messages.json'):
    for guild, channels in new_data.items():
        existing.setdefault(guild, {})
        for channel, messages in channels.items():
            existing[guild].setdefault(channel, [])
            existing[guild][channel].extend(messages)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)

def save_sync_log(sync_log_entry):
    os.makedirs("logs", exist_ok=True)
    date_str = datetime.utcnow().strftime("%Y-%m-%d")
    log_path = os.path.join("logs", f"sync_log_{date_str}.jsonl")
    with open(log_path, 'a', encoding='utf-8') as log_f:
        log_f.write(json.dumps(sync_log_entry) + '\n')

def update_discord_messages():
    print("🔌 Connecting to Discord...")
    data_store = load_existing_data()
    client = DiscordFetcher(data_store=data_store, intents=intents)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(client.start(DISCORD_TOKEN))

    if client.new_data:
        save_combined_data(data_store, client.new_data)
        save_sync_log(client.sync_log_entry)
        print("\n📂 Data updated.")
        print(f"📝 Total messages synced: {client.sync_log_entry['total_messages_synced']}")
    else:
        print("\n✅ No new messages found.")

if __name__ == "__main__":
    update_discord_messages()
    print("🔌 Disconnecting from Discord...")
