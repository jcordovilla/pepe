# models.py

from datetime import datetime
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field

class Author(BaseModel):
    id: int = Field(..., description="The Discord user ID of the author")
    username: str = Field(..., description="The display name of the author")
    discriminator: Optional[str] = Field(None, description="The 4-digit Discord discriminator")

class Reaction(BaseModel):
    emoji: str = Field(..., description="The emoji used in the reaction")
    count: int = Field(..., description="How many times this emoji was used")

class Attachment(BaseModel):
    id: str = Field(..., description="Attachment ID")
    filename: str = Field(..., description="Name of the attached file")
    size: int = Field(..., description="Size of the file in bytes")
    url: str = Field(..., description="Source URL of the file")
    proxy_url: str = Field(..., description="Proxied URL of the file")
    content_type: Optional[str] = Field(None, description="The attachment's media type")
    width: Optional[int] = Field(None, description="Width of the attachment if image/video")
    height: Optional[int] = Field(None, description="Height of the attachment if image/video")

class EmbedField(BaseModel):
    name: str = Field(..., description="Name of the field")
    value: str = Field(..., description="Value of the field")
    inline: bool = Field(False, description="Whether the field is displayed inline")

class EmbedAuthor(BaseModel):
    name: Optional[str] = Field(None, description="Name of author")
    url: Optional[str] = Field(None, description="URL of author")
    icon_url: Optional[str] = Field(None, description="URL of author icon")

class EmbedFooter(BaseModel):
    text: str = Field(..., description="Footer text")
    icon_url: Optional[str] = Field(None, description="URL of footer icon")

class EmbedMedia(BaseModel):
    url: str = Field(..., description="Source URL of the media")
    width: Optional[int] = Field(None, description="Width of the media")
    height: Optional[int] = Field(None, description="Height of the media")

class Embed(BaseModel):
    title: Optional[str] = Field(None, description="Title of embed")
    description: Optional[str] = Field(None, description="Description of embed")
    url: Optional[str] = Field(None, description="URL of embed")
    color: Optional[int] = Field(None, description="Color code of the embed")
    timestamp: Optional[datetime] = Field(None, description="Timestamp of embed content")
    type: Optional[str] = Field(None, description="Type of embed")
    author: Optional[EmbedAuthor] = Field(None, description="Author information")
    footer: Optional[EmbedFooter] = Field(None, description="Footer information")
    thumbnail: Optional[EmbedMedia] = Field(None, description="Thumbnail information")
    image: Optional[EmbedMedia] = Field(None, description="Image information")
    fields: Optional[List[EmbedField]] = Field(None, description="Fields information")

class Sticker(BaseModel):
    id: str = Field(..., description="ID of the sticker")
    name: str = Field(..., description="Name of the sticker")
    format: str = Field(..., description="Format type of the sticker")
    url: Optional[str] = Field(None, description="URL of the sticker")

class MessageReference(BaseModel):
    message_id: Optional[str] = Field(None, description="ID of the originating message")
    channel_id: Optional[str] = Field(None, description="ID of the originating message's channel")
    guild_id: Optional[str] = Field(None, description="ID of the originating message's guild")

class Thread(BaseModel):
    id: str = Field(..., description="ID of the thread")
    name: str = Field(..., description="Name of the thread")
    archived: Optional[bool] = Field(None, description="Whether the thread is archived")
    auto_archive_duration: Optional[int] = Field(None, description="Duration before auto-archive")
    locked: Optional[bool] = Field(None, description="Whether the thread is locked")

class Application(BaseModel):
    id: str = Field(..., description="ID of the application")
    name: str = Field(..., description="Name of the application")
    description: str = Field(..., description="Description of the application")
    icon: Optional[str] = Field(None, description="Icon hash of the application")
    cover_image: Optional[str] = Field(None, description="Cover image hash of the application")

class Activity(BaseModel):
    type: str = Field(..., description="Type of message activity")
    party_id: Optional[str] = Field(None, description="Party ID from a Rich Presence event")

class PollAnswer(BaseModel):
    id: int = Field(..., description="ID of the answer")
    text: Optional[str] = Field(None, description="Text of the answer")
    emoji: Optional[str] = Field(None, description="Emoji of the answer")

class Poll(BaseModel):
    question: Optional[str] = Field(None, description="The question of the poll")
    answers: List[PollAnswer] = Field(default_factory=list, description="Answers available in the poll")
    expiry: Optional[datetime] = Field(None, description="When the poll expires")
    allow_multiselect: bool = Field(False, description="Whether multiple answers are allowed")
    layout_type: Optional[str] = Field(None, description="Layout type of the poll")

class DiscordMessage(BaseModel):
    # Existing fields
    guild_id: int = Field(..., description="The Discord guild (server) ID")
    channel_id: int = Field(..., description="The Discord channel ID")
    message_id: int = Field(..., description="The unique message ID")
    content: str = Field(..., description="The text content of the message")
    timestamp: datetime = Field(..., description="When the message was sent (ISO format)")
    author: Author = Field(..., description="Author information")
    mention_ids: List[int] = Field(default_factory=list, description="List of user IDs mentioned")
    reactions: List[Reaction] = Field(default_factory=list, description="Reactions on the message")
    jump_url: Optional[str] = Field(None, description="Link to view the message in Discord")
    
    # New essential metadata fields
    edited_at: Optional[datetime] = Field(None, description="When the message was last edited")
    type: Optional[str] = Field(None, description="Type of message (default, reply, system, etc.)")
    flags: int = Field(0, description="Message flags bitmask")
    tts: bool = Field(False, description="Whether this was a text-to-speech message")
    pinned: bool = Field(False, description="Whether this message is pinned")
    
    # Rich content fields
    embeds: Optional[List[Dict[str, Any]]] = Field(None, description="Rich embed objects")
    attachments: Optional[List[Dict[str, Any]]] = Field(None, description="File attachments")
    stickers: Optional[List[Dict[str, Any]]] = Field(None, description="Sticker objects")
    components: Optional[List[Dict[str, Any]]] = Field(None, description="Interactive components")
    
    # Reply/thread context
    reference: Optional[MessageReference] = Field(None, description="Message reference for replies")
    thread: Optional[Thread] = Field(None, description="Associated thread information")
    
    # Advanced metadata
    webhook_id: Optional[str] = Field(None, description="ID of the webhook that sent the message")
    application_id: Optional[str] = Field(None, description="ID of the application that sent the message")
    application: Optional[Application] = Field(None, description="Application that sent the message")
    activity: Optional[Activity] = Field(None, description="Rich presence activity")
    poll: Optional[Poll] = Field(None, description="Poll data if message contains poll")
    
    # Raw mention arrays
    raw_mentions: Optional[List[str]] = Field(None, description="Raw user mention data")
    raw_channel_mentions: Optional[List[str]] = Field(None, description="Raw channel mention data")
    raw_role_mentions: Optional[List[str]] = Field(None, description="Raw role mention data")
    
    # Derived content
    clean_content: Optional[str] = Field(None, description="Content with mentions resolved to names")
    system_content: Optional[str] = Field(None, description="System message content")
    
    # Additional mention data
    channel_mentions: Optional[List[str]] = Field(None, description="Mentioned channel IDs")
    role_mentions: Optional[List[str]] = Field(None, description="Mentioned role IDs")
