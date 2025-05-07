import json
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from typing import Literal, Optional
from utils import build_jump_url  # centralized URL builder
from db import SessionLocal, Message
from time_parser import parse_timeframe  # parse natural-language timeframes

DATA_FILE = "discord_messages.json"

def load_messages():
    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

# 📌 Summarize Weekly Activity (legacy)
def summarize_weekly_activity(
    output_format: Literal["text", "json"] = "text",
    week_ending_iso: Optional[str] = None
):
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

    header = f"📅 Weekly Summary {week_start.date()} → {week_end.date()}"
    lines = [header]
    for ch, msgs in summary.items():
        lines.append(f"\n**{ch}** ({len(msgs)} messages):")
        for m in msgs[:3]:
            lines.append(f"- {m}")
    return "\n".join(lines)

# 📌 Get Server Stats
def get_server_stats(
    output_format: Literal["text", "json"] = "text",
    top_n: int = 10
):
    data = load_messages()
    author_counter = Counter()
    channel_counter = Counter()

    for guild, channels in data.items():
        for channel, messages in channels.items():
            for msg in messages:
                author = msg.get("author") or "Unknown"
                author_counter[json.dumps(author)] += 1
                channel_counter[f"{guild}/#{channel}"] += 1

    if output_format == "json":
        return json.dumps({
            "top_authors": author_counter.most_common(top_n),
            "top_channels": channel_counter.most_common(top_n),
            "total_messages": sum(author_counter.values())
        }, indent=2, ensure_ascii=False)

    lines = [f"📊 Total messages: {sum(author_counter.values())}", "\n👤 Top Authors:"]
    for author, count in author_counter.most_common(top_n):
        lines.append(f"- {author}: {count}")
    lines.append("\n📺 Top Channels:")
    for channel, count in channel_counter.most_common(top_n):
        lines.append(f"- {channel}: {count}")
    return "\n".join(lines)

# 📌 Extract Feedback and Event Ideas
def extract_feedback_and_ideas(
    output_format: Literal["text", "json"] = "text"
):
    data = load_messages()
    keywords = ["idea", "event", "feedback", "suggest", "should", "could", "recommend", "wish"]
    found = []

    for guild, channels in data.items():
        for channel, messages in channels.items():
            for msg in messages:
                content = msg.get("content", "")
                if any(k in content.lower() for k in keywords):
                    found.append({
                        "guild": guild,
                        "channel": channel,
                        "author": msg.get("author"),
                        "timestamp": msg.get("timestamp"),
                        "content": content
                    })

    if output_format == "json":
        return json.dumps(found, indent=2, ensure_ascii=False)

    lines = []
    for f in found[:20]:
        lines.append(f"[{f['timestamp']}] {f['author']} in {f['guild']}/#{f['channel']}: {f['content']}")
    return "\n".join(lines)

# 📌 Most Reacted Messages
def get_most_reacted_messages(
    output_format: Literal["text", "json"] = "text",
    top_n: int = 5
):
    data = load_messages()
    reactions = []

    for guild, channels in data.items():
        for channel, messages in channels.items():
            for msg in messages:
                total_reactions = sum(r.get("count", 0) for r in msg.get("reactions", []))
                if total_reactions > 0:
                    reactions.append({
                        "guild": guild,
                        "channel": channel,
                        "author": msg.get("author"),
                        "content": msg.get("content", ""),
                        "total_reactions": total_reactions
                    })

    reactions.sort(key=lambda x: x["total_reactions"], reverse=True)
    top_reacted = reactions[:top_n]

    if output_format == "json":
        return json.dumps(top_reacted, indent=2, ensure_ascii=False)

    lines = []
    for m in top_reacted:
        lines.append(f"{m['guild']}/#{m['channel']} - {m['author']}: {m['content']} ({m['total_reactions']} reactions)")
    return "\n".join(lines)

# 📌 Messages Mentioning a User
def find_messages_mentioning_user(
    user_id: str,
    output_format: Literal["text", "json"] = "text"
):
    data = load_messages()
    mentions = []

    for guild, channels in data.items():
        for channel, messages in channels.items():
            for msg in messages:
                if user_id in msg.get("mention_ids", []):
                    mentions.append({
                        "guild": guild,
                        "channel": channel,
                        "author": msg.get("author"),
                        "timestamp": msg.get("timestamp"),
                        "content": msg.get("content", "")
                    })

    if output_format == "json":
        return json.dumps(mentions, indent=2, ensure_ascii=False)

    lines = []
    for m in mentions:
        lines.append(f"[{m['timestamp']}] {m['author']} mentioned user in {m['guild']}/#{m['channel']}: {m['content']}")
    return "\n".join(lines)

# 📌 Pinned Messages
def get_pinned_messages(
    output_format: Literal["text", "json"] = "text"
):
    data = load_messages()
    pinned = []

    for guild, channels in data.items():
        for channel, messages in channels.items():
            for msg in messages:
                if msg.get("pinned", False):
                    pinned.append({
                        "guild": guild,
                        "channel": channel,
                        "author": msg.get("author"),
                        "timestamp": msg.get("timestamp"),
                        "content": msg.get("content", "")
                    })

    if output_format == "json":
        return json.dumps(pinned, indent=2, ensure_ascii=False)

    lines = []
    for p in pinned:
        lines.append(f"📌 [{p['timestamp']}] {p['author']} pinned in {p['guild']}/#{p['channel']}: {p['content']}")
    return "\n".join(lines)

# 📌 Analyze Message Types
def analyze_message_types(
    output_format: Literal["text", "json"] = "text"
):
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
def find_messages_with_keywords(
    keywords: list,
    output_format: Literal["text", "json"] = "text",
    top_n: int = 10
):
    data = load_messages()
    results = []

    for guild, channels in data.items():
        for channel, messages in channels.items():
            for msg in messages:
                if any(keyword.lower() in msg.get("content", "").lower() for keyword in keywords):
                    results.append({
                        "guild": guild,
                        "channel": channel,
                        "author": msg.get("author"),
                        "timestamp": msg.get("timestamp"),
                        "content": msg.get("content", "")
                    })

    results = results[:top_n]

    if output_format == "json":
        return json.dumps(results, indent=2, ensure_ascii=False)

    lines = []
    for m in results:
        lines.append(f"[{m['timestamp']}] {m['author']} in {m['guild']}/#{m['channel']}: {m['content']}")
    return "\n".join(lines)

# 📌 Find users with specific profiles or skills
def find_users_by_skill(
    skill_query: str,
    output_format: Literal["text", "json"] = "text",
    top_n: int = 10
):
    data = load_messages()
    matches = []

    skill_lower = skill_query.lower()

    for guild, channels in data.items():
        for channel, messages in channels.items():
            for msg in messages:
                content_lower = msg.get("content", "").lower()
                if skill_lower in content_lower:
                    matches.append({
                        "guild": guild,
                        "channel": channel,
                        "author": msg.get("author"),
                        "author_id": msg.get("author_id"),
                        "message_id": msg.get("message_id"),
                        "guild_id": msg.get("guild_id"),
                        "channel_id": msg.get("channel_id"),
                        "timestamp": msg.get("timestamp"),
                        "content": msg.get("content", ""),
                        "jump_url": msg.get("jump_url")
                    })

    # Deduplicate users but allow multiple relevant messages per user
    matches.sort(key=lambda x: x["timestamp"])
    unique_users = {}
    for m in matches:
        uid = m.get("author_id")
        if uid not in unique_users:
            unique_users[uid] = m
        if len(unique_users) >= top_n:
            break
    result_list = list(unique_users.values())

    if output_format == "json":
        return json.dumps(result_list, indent=2, ensure_ascii=False)

    lines = []
    for m in result_list:
        url = m.get("jump_url") or build_jump_url(
            int(m.get("guild_id")), int(m.get("channel_id")), int(m.get("message_id"))
        )
        lines.append(
            f"👤 {m.get('author')} (ID: {m.get('author_id')})\n"
            f"🕓 {m.get('timestamp')} in {m.get('guild')}/#{m.get('channel')}\n"
            f"📝 {m.get('content')}" + (f"\n🔗 Jump to message: {url}" if url else "")
        )
    return "\n\n".join(lines)

# 📌 --- New Tool: Summarize Messages in Range ---
def summarize_messages_in_range(
    start_iso: str,
    end_iso: str,
    guild_id: Optional[int] = None,
    channel_id: Optional[int] = None,
    output_format: Literal["text", "json"] = "text"
):
    """
    Summarize messages between start_iso and end_iso.
    Optionally filter by guild_id and/or channel_id.
    """
    # Parse ISO timestamps
    start_dt = datetime.fromisoformat(start_iso)
    end_dt = datetime.fromisoformat(end_iso)

    # Query SQLite DB
    session = SessionLocal()
    query = session.query(Message).filter(
        Message.timestamp >= start_dt,
        Message.timestamp <= end_dt
    )
    if guild_id:
        query = query.filter(Message.guild_id == guild_id)
    if channel_id:
        query = query.filter(Message.channel_id == channel_id)

    msgs = query.order_by(Message.timestamp).all()
    session.close()

    # Group by channel for summary
    by_channel = {}
    for m in msgs:
        key = f"{m.guild_id}/#{m.channel_id}"
        by_channel.setdefault(key, []).append(m)

    # JSON output
    if output_format == "json":
        out = {ch: [msg.content for msg in lst] for ch, lst in by_channel.items()}
        return json.dumps(out, ensure_ascii=False, indent=2)

    # Text summary
    lines = [f"📅 Messages from {start_dt.date()} to {end_dt.date()}"]
    for ch, lst in by_channel.items():
        lines.append(f"\n**{ch}**: {len(lst)} messages")
        for m in lst[:3]:
            ts = m.timestamp.isoformat()
            snippet = m.content[:100].replace("\n", " ")
            lines.append(f"- [{ts}] {snippet}…")
    return "\n".join(lines)