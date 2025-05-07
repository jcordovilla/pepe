import sys
import os
import json
import pytest

# Add project root to Python path so we can import models.py
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models import DiscordMessage

def test_discord_messages_v2_structure_and_schema():
    # 1. Load the migrated JSON
    with open("discord_messages_v2.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    # 2. Top‐level key check
    assert "guilds" in data, "Missing top-level 'guilds' key"
    guilds = data["guilds"]
    assert isinstance(guilds, dict) and guilds, "No guilds present in data"

    # 3. Pick the first guild
    first_gid, guild = next(iter(guilds.items()))
    assert guild.get("channels"), f"No channels under guild {first_gid}"

    # 4. Pick the first channel
    first_cid, channel = next(iter(guild["channels"].items()))
    messages = channel.get("messages")
    assert isinstance(messages, list), "'messages' should be a list"
    assert messages, "No messages found in first channel"

    # 5. Validate the first message with our Pydantic model
    first_msg = messages[0]
    # This will raise if the structure or types are wrong
    validated = DiscordMessage.model_validate(first_msg)
    assert validated.content, "Message content is empty"
