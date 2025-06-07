# Discord Message Fields Enhancement

This document describes the comprehensive Discord message fields that are now captured and stored by the enhanced fetch system.

## Overview

The Discord message fetching system has been upgraded to capture **all available Discord API message fields**, providing rich metadata and content information for enhanced search, analysis, and interaction capabilities.

## Field Categories

### 🔑 Core Message Fields (Original)
These fields were already captured:

- **`message_id`** - Unique Discord message ID
- **`channel_id`** - Channel where message was posted
- **`guild_id`** - Server/guild ID
- **`channel_name`** - Human-readable channel name
- **`content`** - Text content of the message
- **`timestamp`** - When the message was created
- **`author`** - Author information (id, username, discriminator)
- **`mentions`** - Array of mentioned user IDs
- **`reactions`** - Reactions with emoji and count
- **`jump_url`** - Direct link to view message in Discord

### ⭐ Essential Metadata Fields (New)
Critical message metadata for better analysis:

- **`edited_at`** - When message was last edited (null if never edited)
- **`type`** - Message type (`default`, `reply`, `system`, `application_command`, etc.)
- **`flags`** - Discord message flags bitmask (crossposted, urgent, ephemeral, etc.)
- **`tts`** - Whether message uses text-to-speech
- **`pinned`** - Whether message is pinned in channel

### 🎨 Rich Content Fields (New)
Advanced content and media:

- **`embeds`** - Rich embed objects with titles, descriptions, images, etc.
- **`attachments`** - File attachments with metadata (filename, size, dimensions, URLs)
- **`stickers`** - Discord stickers used in the message
- **`components`** - Interactive components (buttons, select menus, etc.)

### 🔗 Reply/Thread Context (New)
Message relationships and threading:

- **`reference`** - Reference to original message for replies
- **`thread`** - Associated thread information if message starts/is in a thread

### 🤖 Advanced Metadata (New)
Bot, application, and system integration:

- **`webhook_id`** - ID if message was sent by a webhook
- **`application_id`** - ID if message was sent by an application/bot
- **`application`** - Full application data (name, description, icon, etc.)
- **`activity`** - Rich presence activity data
- **`poll`** - Poll data if message contains a poll

### 📝 Mention Arrays (New)
Enhanced mention tracking:

- **`raw_mentions`** - Raw user mention data from Discord
- **`raw_channel_mentions`** - Raw channel mention data
- **`raw_role_mentions`** - Raw role mention data
- **`channel_mentions`** - Processed channel mention IDs
- **`role_mentions`** - Processed role mention IDs

### 🔤 Derived Content (New)
Processed and alternative content representations:

- **`clean_content`** - Content with mentions resolved to readable names
- **`system_content`** - System message content (for system message types)

## Rich Content Examples

### Embeds
Embeds capture rich link previews, bot messages, and formatted content:
```json
{
  "title": "Documentation Link",
  "description": "Complete guide to Discord API",
  "url": "https://discord.com/developers/docs",
  "color": 5814783,
  "thumbnail": {"url": "https://...", "width": 400, "height": 300},
  "fields": [
    {"name": "Version", "value": "v10", "inline": true}
  ]
}
```

### Attachments
File uploads with comprehensive metadata:
```json
{
  "id": "123456789",
  "filename": "diagram.png",
  "size": 245760,
  "url": "https://cdn.discordapp.com/attachments/...",
  "content_type": "image/png",
  "width": 1920,
  "height": 1080
}
```

### Message References (Replies)
Track reply chains and conversations:
```json
{
  "message_id": "987654321",
  "channel_id": "123456789",
  "guild_id": "987654321"
}
```

### Polls
Discord poll data with answers and metadata:
```json
{
  "question": "What should we discuss next?",
  "answers": [
    {"id": 1, "text": "AI Ethics", "emoji": "🤖"},
    {"id": 2, "text": "Machine Learning", "emoji": "📊"}
  ],
  "expiry": "2025-06-08T12:00:00Z",
  "allow_multiselect": false
}
```

## Message Types

The `type` field indicates the nature of the message:

- **`default`** - Regular user message
- **`recipient_add`** - User added to group DM
- **`recipient_remove`** - User removed from group DM
- **`call`** - Call started
- **`channel_name_change`** - Channel name changed
- **`channel_icon_change`** - Channel icon changed
- **`pins_add`** - Message pinned
- **`guild_member_join`** - Member joined server
- **`user_premium_guild_subscription`** - User boosted server
- **`user_premium_guild_subscription_tier_1`** - Server reached Boost level 1
- **`user_premium_guild_subscription_tier_2`** - Server reached Boost level 2
- **`user_premium_guild_subscription_tier_3`** - Server reached Boost level 3
- **`channel_follow_add`** - Channel followed
- **`guild_discovery_disqualified`** - Server disqualified from Discovery
- **`guild_discovery_requalified`** - Server requalified for Discovery
- **`reply`** - Reply to another message
- **`application_command`** - Slash command response
- **`thread_starter_message`** - Message that started a thread
- **`guild_invite_reminder`** - Server invite reminder
- **`context_menu_command`** - Context menu command response

## Message Flags

The `flags` field is a bitmask with these possible values:

- **1** (`CROSSPOSTED`) - Message has been crossposted
- **2** (`IS_CROSSPOST`) - Message is a crosspost from another channel
- **4** (`SUPPRESS_EMBEDS`) - Do not include embeds when serializing
- **8** (`SOURCE_MESSAGE_DELETED`) - Source message for this crosspost has been deleted
- **16** (`URGENT`) - Message is urgent
- **64** (`HAS_THREAD`) - Message has an associated thread
- **128** (`EPHEMERAL`) - Message is ephemeral (only visible to user)
- **256** (`LOADING`) - Message is loading
- **4096** (`SUPPRESS_NOTIFICATIONS`) - Message does not trigger push/desktop notifications

## Database Schema Updates

The database has been migrated to include all new fields with proper indexing and data types:

- **JSON fields** for complex data (embeds, attachments, components, etc.)
- **Boolean fields** for flags (tts, pinned)
- **DateTime fields** for timestamps (edited_at)
- **String fields** for IDs and types
- **Text fields** for content variations (clean_content, system_content)

## Search and Analysis Benefits

These enhanced fields enable:

1. **Rich Content Discovery** - Find messages with specific media types, embeds, or attachments
2. **Conversation Threading** - Track reply chains and thread relationships
3. **Bot/System Message Analysis** - Identify and analyze automated content
4. **Edit History Tracking** - Find recently edited messages
5. **Reaction Analysis** - Enhanced emoji and reaction patterns
6. **Poll Data Mining** - Analyze community polls and voting patterns
7. **Link Preview Analysis** - Rich embed content for better search relevance
8. **File Sharing Patterns** - Track document and media sharing behaviors

## Migration Notes

- All existing messages retain their original fields
- New fields are nullable and backward-compatible
- The migration adds columns without data loss
- Future message fetches will populate all new fields
- Search tools automatically handle both old and new field formats

## Performance Considerations

- JSON fields are efficiently stored and indexed
- Complex queries can filter by message type, flags, or content types
- Attachment metadata enables media-specific searches
- Thread relationships support conversation analysis

## API Compatibility

The enhanced system maintains full backward compatibility while providing access to all Discord API message capabilities. Tools and searches will automatically benefit from the additional metadata without requiring code changes.
