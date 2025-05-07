import json
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from typing import Literal

def build_jump_url(msg):
    """Safely construct a valid Discord jump URL."""
    gid = msg.get("guild_id")
    cid = msg.get("channel_id")
    mid = msg.get("message_id") or msg.get("id")

    if all(id and id.isdigit() for id in [gid, cid, mid]):
        return f"https://discord.com/channels/{gid}/{cid}/{mid}"
    return None

DATA_FILE = "discord_messages.json"

def load_messages():
    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

# 📌 Summarize Weekly Activity
def summarize_weekly_activity(output_format: Literal["text", "json"] = "text", week_ending_iso: str = None):
    data = load_messages()
    week_end = datetime.fromisoformat(week_ending_iso) if week_ending_iso else datetime.now(timezone.utc)
    week_start = week_end - timedelta(days=7)
    summary = defaultdict(list)

    for guild, channels in data.items():
        for channel, messages in channels.items():
            for msg in messages:
                ts = datetime.fromisoformat(msg["timestamp"])
                if week_start <= ts <= week_end:
                    summary[f"{guild}/#{channel}"].append(msg["content"])

    if output_format == "json":
        return json.dumps(summary, indent=2, ensure_ascii=False)

    return "\n".join(
        [f"🗓️ Weekly Summary {week_start.date()} → {week_end.date()}"] +
        [f"\n#{ch} ({len(msgs)} messages):\n" + "\n".join(f"- {m}" for m in msgs[:3])
         for ch, msgs in summary.items()]
    )

# 📌 Get Server Stats
def get_server_stats(output_format: Literal["text", "json"] = "text", top_n: int = 10):
    data = load_messages()
    author_counter = Counter()
    channel_counter = Counter()

    for guild, channels in data.items():
        for channel, messages in channels.items():
            for msg in messages:
                author_counter[msg.get("author", "Unknown")] += 1
                channel_counter[f"{guild}/#{channel}"] += 1

    if output_format == "json":
        return json.dumps({
            "top_authors": author_counter.most_common(top_n),
            "top_channels": channel_counter.most_common(top_n),
            "total_messages": sum(author_counter.values())
        }, indent=2, ensure_ascii=False)

    text = [
        f"📊 Total messages: {sum(author_counter.values())}",
        "\n👤 Top Authors:",
        *[f"- {author}: {count}" for author, count in author_counter.most_common(top_n)],
        "\n📺 Top Channels:",
        *[f"- {channel}: {count}" for channel, count in channel_counter.most_common(top_n)]
    ]
    return "\n".join(text)

# 📌 Extract Feedback and Event Ideas
def extract_feedback_and_ideas(output_format: Literal["text", "json"] = "text"):
    data = load_messages()
    keywords = ["idea", "event", "feedback", "suggest", "should", "could", "recommend", "wish"]
    found = []

    for guild, channels in data.items():
        for channel, messages in channels.items():
            for msg in messages:
                if any(k in msg["content"].lower() for k in keywords):
                    found.append({
                        "guild": guild,
                        "channel": channel,
                        "author": msg["author"],
                        "timestamp": msg["timestamp"],
                        "content": msg["content"]
                    })

    if output_format == "json":
        return json.dumps(found, indent=2, ensure_ascii=False)

    return "\n".join([
        f"[{f['timestamp']}] {f['author']} in {f['guild']}/#{f['channel']}\n→ {f['content']}"
        for f in found[:20]
    ])

# 📌 Most Reacted Messages
def get_most_reacted_messages(output_format: Literal["text", "json"] = "text", top_n: int = 5):
    data = load_messages()
    reactions = []

    for guild, channels in data.items():
        for channel, messages in channels.items():
            for msg in messages:
                total_reactions = len(msg.get("reactions", []))
                if total_reactions > 0:
                    reactions.append({
                        "guild": guild,
                        "channel": channel,
                        "author": msg["author"],
                        "content": msg["content"],
                        "total_reactions": total_reactions
                    })

    reactions.sort(key=lambda x: x["total_reactions"], reverse=True)
    top_reacted = reactions[:top_n]

    if output_format == "json":
        return json.dumps(top_reacted, indent=2, ensure_ascii=False)

    return "\n".join([
        f"{m['guild']}/#{m['channel']} - {m['author']}: {m['content']} ({m['total_reactions']} reactions)"
        for m in top_reacted
    ])

# 📌 Messages Mentioning a User
def find_messages_mentioning_user(user_id: str, output_format: Literal["text", "json"] = "text"):
    data = load_messages()
    mentions = []

    for guild, channels in data.items():
        for channel, messages in channels.items():
            for msg in messages:
                if user_id in msg.get("mention_ids", []):
                    mentions.append({
                        "guild": guild,
                        "channel": channel,
                        "author": msg["author"],
                        "timestamp": msg["timestamp"],
                        "content": msg["content"]
                    })

    if output_format == "json":
        return json.dumps(mentions, indent=2, ensure_ascii=False)

    return "\n".join([
        f"[{m['timestamp']}] {m['author']} mentioned user in {m['guild']}/#{m['channel']}: {m['content']}"
        for m in mentions
    ])

# 📌 Pinned Messages
def get_pinned_messages(output_format: Literal["text", "json"] = "text"):
    data = load_messages()
    pinned = []

    for guild, channels in data.items():
        for channel, messages in channels.items():
            for msg in messages:
                if msg.get("pinned", False):
                    pinned.append({
                        "guild": guild,
                        "channel": channel,
                        "author": msg["author"],
                        "timestamp": msg["timestamp"],
                        "content": msg["content"]
                    })

    if output_format == "json":
        return json.dumps(pinned, indent=2, ensure_ascii=False)

    return "\n".join([
        f"📌 [{p['timestamp']}] {p['author']} pinned in {p['guild']}/#{p['channel']}: {p['content']}"
        for p in pinned
    ])

# 📌 Analyze Message Types
def analyze_message_types(output_format: Literal["text", "json"] = "text"):
    data = load_messages()
    type_counter = Counter()

    for guild, channels in data.items():
        for channel, messages in channels.items():
            for msg in messages:
                msg_type = msg.get("type", "default")
                type_counter[msg_type] += 1

    if output_format == "json":
        return json.dumps(type_counter.most_common(), indent=2, ensure_ascii=False)

    return "\n".join([f"{t}: {c} messages" for t, c in type_counter.most_common()])

# 📌 Find Messages with Keywords
def find_messages_with_keywords(keywords: list, output_format: Literal["text", "json"] = "text", top_n: int = 10):
    data = load_messages()
    results = []

    for guild, channels in data.items():
        for channel, messages in channels.items():
            for msg in messages:
                if any(keyword.lower() in msg["content"].lower() for keyword in keywords):
                    results.append({
                        "guild": guild,
                        "channel": channel,
                        "author": msg["author"],
                        "timestamp": msg["timestamp"],
                        "content": msg["content"]
                    })

    results = results[:top_n]

    if output_format == "json":
        return json.dumps(results, indent=2, ensure_ascii=False)

    return "\n".join([
        f"[{m['timestamp']}] {m['author']} in {m['guild']}/#{m['channel']}: {m['content']}"
        for m in results
    ])

# 📌 Find users with specific profiles or skills
def find_users_by_skill(skill_query: str, output_format: Literal["text", "json"] = "text", top_n: int = 10):
    """
    Search for users who mention specific skills, expertise, or interests in their messages.
    
    Args:
        skill_query (str): The skill or keyword to search for.
        output_format (str): "text" or "json" format for output.
        top_n (int): Maximum number of user matches to return.

    Returns:
        str or JSON: List of matching users and sample messages.
    """
    data = load_messages()
    matches = []

    skill_lower = skill_query.lower()

    for guild, channels in data.items():
        for channel, messages in channels.items():
            for msg in messages:
                content_lower = msg["content"].lower()
                if skill_lower in content_lower:
                    matches.append({
                        "guild": guild,
                        "channel": channel,
                        "author": msg["author"],
                        "author_id": msg.get("author_id"),
                        "timestamp": msg["timestamp"],
                        "content": msg["content"]
                    })

    # Deduplicate users but allow multiple relevant messages per user if needed
    matches.sort(key=lambda x: x["timestamp"])  # Earliest first

    # If too many results, truncate to top N unique users
    unique_users = {}
    for m in matches:
        uid = m["author_id"]
        if uid not in unique_users:
            unique_users[uid] = m
        if len(unique_users) >= top_n:
            break

    result_list = list(unique_users.values())

    if output_format == "json":
        return json.dumps(result_list, indent=2, ensure_ascii=False)

    lines = []
    for m in result_list:
        jump_url = m.get("jump_url") or build_jump_url(m)
        lines.append(
            f"👤 {m['author']} (ID: {m.get('author_id', '?')})\n"
            f"🕓 {m['timestamp']} in {m['guild']}/#{m['channel']}\n"
            f"📝 {m['content']}"
            + (f"\n🔗 [Jump to message]({jump_url})" if jump_url else "")
        )
    return "\n\n".join(lines)

