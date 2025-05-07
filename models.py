# models.py

from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field


class Author(BaseModel):
    id: int = Field(..., description="The Discord user ID of the author")
    username: str = Field(..., description="The display name of the author")
    discriminator: Optional[str] = Field(None, description="The 4-digit Discord discriminator")


class Reaction(BaseModel):
    emoji: str = Field(..., description="The emoji used in the reaction")
    count: int = Field(..., description="How many times this emoji was used")


class DiscordMessage(BaseModel):
    guild_id: int = Field(..., description="The Discord guild (server) ID")
    channel_id: int = Field(..., description="The Discord channel ID")
    message_id: int = Field(..., description="The unique message ID")
    content: str = Field(..., description="The text content of the message")
    timestamp: datetime = Field(..., description="When the message was sent (ISO format)")
    author: Author = Field(..., description="Author information")
    mention_ids: List[int] = Field(default_factory=list, description="List of user IDs mentioned")
    reactions: List[Reaction] = Field(default_factory=list, description="Reactions on the message")
    jump_url: Optional[str] = Field(None, description="Link to view the message in Discord")
