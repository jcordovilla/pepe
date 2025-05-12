# migrate_messages.py

import json
from db.models import DiscordMessage
from collections import defaultdict

def transform_raw(raw: dict) -> dict:
    """
    Extract and convert fields from the old Discord export to our new schema.
    """
    return {
        "guild_id": int(raw["guild_id"]),
        "channel_id": int(raw["channel_id"]),
        "message_id": int(raw.get("message_id") or raw.get("id")),
        "content": raw.get("content", ""),
        "timestamp": raw["timestamp"],  # Pydantic will parse the ISO string
        "author": {
            "id": int(raw["author"]["id"]),
            "username": raw["author"]["name"],
            "discriminator": raw["author"].get("discriminator")
        },
        "mention_ids": [int(mid) for mid in raw.get("mentions", [])],
        "reactions": [
            {"emoji": r["emoji"], "count": r["count"]}
            for r in raw.get("reactions", [])
        ],
        "jump_url": raw.get("jump_url")
    }

def main():
    # 1. Load your old JSON, which is a dict of guild_name → channel_name → [messages]
    with open("discord_messages.json", "r", encoding="utf-8") as f:
        old_data = json.load(f)

    # 2. Prepare the new ID-first structure
    new_data = {"guilds": {}}

    for guild_name, channels in old_data.items():
        for channel_name, messages in channels.items():
            for raw in messages:
                try:
                    # 3. Transform fields and validate with Pydantic
                    clean = transform_raw(raw)
                    msg = DiscordMessage.model_validate(clean)
                except Exception as e:
                    print(f"Skipping invalid record (guild={guild_name}, channel={channel_name}): {e}")
                    continue

                gid = str(msg.guild_id)
                cid = str(msg.channel_id)

                # 4. Ensure guild entry
                guild_entry = new_data["guilds"].setdefault(gid, {
                    "name": guild_name,
                    "channels": {}
                })

                # 5. Ensure channel entry
                channel_entry = guild_entry["channels"].setdefault(cid, {
                    "name": channel_name,
                    "messages": []
                })

                # 6. Append validated message dict
                channel_entry["messages"].append(msg.model_dump())

    # 7. Write out the migrated JSON
    with open("discord_messages_v2.json", "w", encoding="utf-8") as f:
        json.dump(new_data, f, ensure_ascii=False, indent=2, default=str)

    # 8. Print summary

    print("Migration complete: discord_messages_v2.json created.")

if __name__ == "__main__":
    main()
