# fetch_messages.py
import os
import sys
import asyncio
from datetime import datetime

# Add project root to Python path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from db import SessionLocal, Message
from dotenv import load_dotenv
import discord
from utils.logger import setup_logging

setup_logging()

import logging
log = logging.getLogger(__name__)

# Load .env file from project root
env_path = os.path.join(project_root, '.env')
load_dotenv(env_path)
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")

# Debug: Print token info (masked for security)
if DISCORD_TOKEN:
    log.info(f"✅ Discord token loaded: {DISCORD_TOKEN[:10]}...{DISCORD_TOKEN[-5:]}")
else:
    log.error("❌ Discord token not found in environment variables")
    log.info(f"Looking for .env file at: {env_path}")
    log.info(f"File exists: {os.path.exists(env_path)}")

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
            
            # 🔍 PRE-SCAN: Check all channels before processing
            total_text_channels = len(guild.text_channels)
            accessible_channels = []
            inaccessible_channels = []
            skipped_channels = []
            
            log.info(f"  🔍 Pre-scanning {total_text_channels} text channels...")
            
            for channel in guild.text_channels:
                if channel.name in SKIP_CHANNEL_NAMES:
                    skipped_channels.append(channel)
                    log.info(f"    🚫 Will skip: #{channel.name} (ID: {channel.id}) - Test channel")
                    continue
                    
                # Test channel access by checking permissions
                try:
                    permissions = channel.permissions_for(guild.me)
                    if permissions.read_messages and permissions.read_message_history:
                        accessible_channels.append(channel)
                        log.info(f"    ✅ Accessible: #{channel.name} (ID: {channel.id})")
                    else:
                        inaccessible_channels.append(channel)
                        log.info(f"    🚫 No permissions: #{channel.name} (ID: {channel.id})")
                except Exception as e:
                    inaccessible_channels.append(channel)
                    log.info(f"    ❌ Error checking: #{channel.name} (ID: {channel.id}) - {e}")
            
            # Summary of pre-scan
            log.info(f"\n  📊 Channel Access Summary:")
            log.info(f"    Total channels: {total_text_channels}")
            log.info(f"    Accessible: {len(accessible_channels)}")
            log.info(f"    Inaccessible: {len(inaccessible_channels)}")
            log.info(f"    Skipped by config: {len(skipped_channels)}")
            log.info(f"  📦 Processing {len(accessible_channels)} accessible channels...\n")

            # Process only accessible channels  
            for channel in accessible_channels:
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

                log.info(f"  📄 Channel: #{channel.name} (ID: {channel.id}) | after={after_id}")
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
                            # Helper function to safely extract attachment data
                            attachments_data = []
                            for att in message.attachments:
                                attachments_data.append({
                                    'id': str(att.id),
                                    'filename': att.filename,
                                    'size': att.size,
                                    'url': att.url,
                                    'proxy_url': att.proxy_url,
                                    'content_type': getattr(att, 'content_type', None),
                                    'width': getattr(att, 'width', None),
                                    'height': getattr(att, 'height', None)
                                })

                            # Helper function to safely extract embed data
                            embeds_data = []
                            for embed in message.embeds:
                                embed_dict = {
                                    'title': embed.title,
                                    'description': embed.description,
                                    'url': embed.url,
                                    'color': embed.color.value if embed.color else None,
                                    'timestamp': embed.timestamp.isoformat() if embed.timestamp else None,
                                    'type': embed.type
                                }
                                if embed.author:
                                    embed_dict['author'] = {
                                        'name': embed.author.name,
                                        'url': embed.author.url,
                                        'icon_url': embed.author.icon_url
                                    }
                                if embed.footer:
                                    embed_dict['footer'] = {
                                        'text': embed.footer.text,
                                        'icon_url': embed.footer.icon_url
                                    }
                                if embed.thumbnail:
                                    embed_dict['thumbnail'] = {
                                        'url': embed.thumbnail.url,
                                        'width': embed.thumbnail.width,
                                        'height': embed.thumbnail.height
                                    }
                                if embed.image:
                                    embed_dict['image'] = {
                                        'url': embed.image.url,
                                        'width': embed.image.width,
                                        'height': embed.image.height
                                    }
                                if embed.fields:
                                    embed_dict['fields'] = [
                                        {
                                            'name': field.name,
                                            'value': field.value,
                                            'inline': field.inline
                                        } for field in embed.fields
                                    ]
                                embeds_data.append(embed_dict)

                            # Helper function to extract sticker data
                            stickers_data = []
                            for sticker in getattr(message, 'stickers', []):
                                stickers_data.append({
                                    'id': str(sticker.id),
                                    'name': sticker.name,
                                    'format': str(sticker.format),
                                    'url': sticker.url if hasattr(sticker, 'url') else None
                                })

                            # Helper function to extract component data
                            components_data = []
                            for component in getattr(message, 'components', []):
                                component_dict = {
                                    'type': component.type.value if hasattr(component.type, 'value') else str(component.type)
                                }
                                if hasattr(component, 'children'):
                                    component_dict['children'] = []
                                    for child in component.children:
                                        child_dict = {
                                            'type': child.type.value if hasattr(child.type, 'value') else str(child.type)
                                        }
                                        if hasattr(child, 'label'):
                                            child_dict['label'] = child.label
                                        if hasattr(child, 'custom_id'):
                                            child_dict['custom_id'] = child.custom_id
                                        if hasattr(child, 'url'):
                                            child_dict['url'] = child.url
                                        component_dict['children'].append(child_dict)
                                components_data.append(component_dict)

                            # Extract message reference data for replies
                            reference_data = None
                            if message.reference:
                                reference_data = {
                                    'message_id': str(message.reference.message_id) if message.reference.message_id else None,
                                    'channel_id': str(message.reference.channel_id) if message.reference.channel_id else None,
                                    'guild_id': str(message.reference.guild_id) if message.reference.guild_id else None
                                }

                            # Extract thread data
                            thread_data = None
                            if hasattr(message, 'thread') and message.thread:
                                thread_data = {
                                    'id': str(message.thread.id),
                                    'name': message.thread.name,
                                    'archived': getattr(message.thread, 'archived', None),
                                    'auto_archive_duration': getattr(message.thread, 'auto_archive_duration', None),
                                    'locked': getattr(message.thread, 'locked', None)
                                }

                            # Extract poll data
                            poll_data = None
                            if hasattr(message, 'poll') and message.poll:
                                try:
                                    # Handle poll question - it might be a string or an object with .text
                                    question_text = None
                                    if hasattr(message.poll, 'question') and message.poll.question:
                                        if hasattr(message.poll.question, 'text'):
                                            question_text = message.poll.question.text
                                        else:
                                            question_text = str(message.poll.question)
                                    
                                    poll_data = {
                                        'question': question_text,
                                        'answers': [],
                                        'expiry': message.poll.expiry.isoformat() if hasattr(message.poll, 'expiry') and message.poll.expiry else None,
                                        'allow_multiselect': getattr(message.poll, 'allow_multiselect', False),
                                        'layout_type': str(message.poll.layout_type) if hasattr(message.poll, 'layout_type') else None
                                    }
                                    
                                    # Handle poll answers - similar defensive approach
                                    if hasattr(message.poll, 'answers') and message.poll.answers:
                                        for answer in message.poll.answers:
                                            answer_data = {
                                                'id': getattr(answer, 'id', None),
                                                'emoji': str(answer.emoji) if getattr(answer, 'emoji', None) else None
                                            }
                                            
                                            # Handle answer text - might be string or object with .text
                                            if hasattr(answer, 'text') and answer.text:
                                                if hasattr(answer.text, 'text'):
                                                    answer_data['text'] = answer.text.text
                                                else:
                                                    answer_data['text'] = str(answer.text)
                                            else:
                                                answer_data['text'] = None
                                                
                                            poll_data['answers'].append(answer_data)
                                except Exception as e:
                                    log.warning(f"Error parsing poll data in message {message.id}: {e}")
                                    poll_data = {'error': f'Failed to parse poll: {str(e)}'}

                            # Extract activity data
                            activity_data = None
                            if message.activity:
                                activity_data = {
                                    'type': message.activity.type.value if hasattr(message.activity.type, 'value') else str(message.activity.type),
                                    'party_id': getattr(message.activity, 'party_id', None)
                                }

                            # Extract application data
                            application_data = None
                            if message.application:
                                application_data = {
                                    'id': str(message.application.id),
                                    'name': message.application.name,
                                    'description': message.application.description,
                                    'icon': str(message.application.icon) if message.application.icon else None,
                                    'cover_image': str(getattr(message.application, 'cover_image', None)) if getattr(message.application, 'cover_image', None) else None
                                }

                            fetched.append({
                                # Existing fields
                                'message_id': str(message.id),
                                'channel_id': str(channel.id),
                                'guild_id': str(guild.id),
                                'channel_name': channel.name, 
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
                                
                                # New essential metadata fields
                                'edited_at': message.edited_at.isoformat() if message.edited_at else None,
                                'type': str(message.type),
                                'flags': message.flags.value if message.flags else 0,
                                'tts': message.tts,
                                'pinned': message.pinned,
                                
                                # Rich content fields
                                'embeds': embeds_data,
                                'attachments': attachments_data,
                                'stickers': stickers_data,
                                'components': components_data,
                                
                                # Reply/thread context
                                'reference': reference_data,
                                'thread': thread_data,
                                
                                # Advanced metadata
                                'webhook_id': str(message.webhook_id) if message.webhook_id else None,
                                'application_id': str(message.application.id) if message.application else None,
                                'application': application_data,
                                'activity': activity_data,
                                'poll': poll_data,
                                
                                # Raw mention arrays
                                'raw_mentions': [str(m) for m in getattr(message, 'raw_mentions', [])],
                                'raw_channel_mentions': [str(m) for m in getattr(message, 'raw_channel_mentions', [])],
                                'raw_role_mentions': [str(m) for m in getattr(message, 'raw_role_mentions', [])],
                                
                                # Derived content
                                'clean_content': message.clean_content,
                                'system_content': message.system_content if hasattr(message, 'system_content') else None,
                                
                                # Additional mention data
                                'channel_mentions': [str(ch.id) for ch in message.channel_mentions],
                                'role_mentions': [str(role.id) for role in message.role_mentions],
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
                        edited_ts = datetime.fromisoformat(m['edited_at']) if m['edited_at'] else None
                        
                        db_msg = Message(
                            guild_id=int(m['guild_id']),
                            channel_id=int(m['channel_id']),
                            channel_name=channel.name,
                            message_id=int(m['message_id']),
                            content=m['content'],
                            timestamp=ts,
                            author=m['author'],
                            mention_ids=[int(mid) for mid in m['mentions']],
                            reactions=[{'emoji': r['emoji'], 'count': r['count']} for r in m['reactions']],
                            jump_url=m.get('jump_url'),
                            
                            # New essential metadata fields
                            edited_at=edited_ts,
                            type=m.get('type'),
                            flags=m.get('flags', 0),
                            tts=m.get('tts', False),
                            pinned=m.get('pinned', False),
                            
                            # Rich content fields
                            embeds=m.get('embeds'),
                            attachments=m.get('attachments'),
                            stickers=m.get('stickers'),
                            components=m.get('components'),
                            
                            # Reply/thread context
                            reference=m.get('reference'),
                            thread=m.get('thread'),
                            
                            # Advanced metadata
                            webhook_id=m.get('webhook_id'),
                            application_id=m.get('application_id'),
                            application=m.get('application'),
                            activity=m.get('activity'),
                            poll=m.get('poll'),
                            
                            # Raw mention arrays
                            raw_mentions=m.get('raw_mentions'),
                            raw_channel_mentions=m.get('raw_channel_mentions'),
                            raw_role_mentions=m.get('raw_role_mentions'),
                            
                            # Derived content
                            clean_content=m.get('clean_content'),
                            system_content=m.get('system_content'),
                            
                            # Additional mention data
                            channel_mentions=m.get('channel_mentions'),
                            role_mentions=m.get('role_mentions')
                        )
                        db.merge(db_msg)
                    db.commit()
                    db.close()
                else:
                    log.info(f"    📫 No new messages")

            # Log pre-identified inaccessible channels
            for channel in inaccessible_channels:
                self.sync_log_entry["channels_skipped"].append({
                    "guild_name": guild.name,
                    "channel_name": channel.name,
                    "channel_id": str(channel.id),
                    "reason": "No permissions (pre-scan)"
                })
                
            # Log skipped channels  
            for channel in skipped_channels:
                self.sync_log_entry["channels_skipped"].append({
                    "guild_name": guild.name,
                    "channel_name": channel.name,
                    "channel_id": str(channel.id),
                    "reason": "Skipped by name"
                })

        # After fetching and saving messages, print summary stats
        total_channels = len(guild.text_channels)
        total_accessible = len(accessible_channels)
        total_inaccessible = len(inaccessible_channels)
        total_skipped = len(skipped_channels)
        total_new_messages = self.sync_log_entry["total_messages_synced"]
        
        # total_messages_including_past is not directly available, calculate it
        db = SessionLocal()
        total_messages_including_past = db.query(Message).filter_by(guild_id=guild.id).count()
        db.close()

        print(f"\n📊 Final Summary for guild {guild.name}:")
        print(f"  Total channels: {total_channels}")
        print(f"  ✅ Accessible & processed: {total_accessible}")
        print(f"  🚫 Inaccessible (no permissions): {total_inaccessible}")
        print(f"  ⏭️  Skipped by configuration: {total_skipped}")
        print(f"  📨 New messages fetched: {total_new_messages}")
        print(f"  📚 Total messages in database: {total_messages_including_past}")

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