# Script to migrate Discord messages from an old JSON format to a new structured format
# migrate_messages.py

import json
from collections import defaultdict
from models import DiscordMessage

# 1. Load your old JSON extract
with open("discord_messages.json", "r", encoding="utf-8") as f:
    old_messages = json.load(f)

# 2. Prepare the new structure
new_data = {"guilds": {}}

for raw in old_messages:
    try:
        # 3. Validate & parse with Pydantic
        msg = DiscordMessage.parse_obj(raw)
    except Exception as e:
        print(f"Skipping invalid record: {e}")
        continue

    gid = str(msg.guild_id)
    cid = str(msg.channel_id)

    # 4. Ensure guild entry exists
    if gid not in new_data["guilds"]:
        # Attempt to capture guild name if it was in the raw data
        guild_name = raw.get("guild_name") or raw.get("guild") or ""
        new_data["guilds"][gid] = {"name": guild_name, "channels": {}}

    # 5. Ensure channel entry exists
    if cid not in new_data["guilds"][gid]["channels"]:
        channel_name = raw.get("channel_name") or raw.get("channel") or ""
        new_data["guilds"][gid]["channels"][cid] = {
            "name": channel_name,
            "messages": []
        }

    # 6. Append the full, validated dict
    new_data["guilds"][gid]["channels"][cid]["messages"].append(
        msg.dict()
    )

# 7. Write out the migrated JSON
with open("discord_messages_v2.json", "w", encoding="utf-8") as f:
    json.dump(new_data, f, ensure_ascii=False, indent=2)

print("Migration complete: discord_messages_v2.json created.")
