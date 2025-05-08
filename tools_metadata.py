from tools import (
    search_messages,
    get_most_reacted_messages,
    find_users_by_skill,
    summarize_messages
)

TOOLS_METADATA = [
    {
        "name": "search_messages",
        "description": "Hybrid keyword + semantic search over messages, with optional guild/channel/author filters.",
        "parameters": {
            "type": "object",
            "properties": {
            "query":       { "type": "string" },
            "keyword":     { "type": "string" },
            "guild_id":    { "type": "integer" },
            "channel_id":  { "type": "integer" },
            "author_name": { "type": "string" },
            "k":           { "type": "integer", "default": 5 }
            },
            "required": ["query"]
        }
    },
    {
        "name": "get_most_reacted_messages",
        "description": "Return the top N messages by reaction count, optionally scoped by guild and/or channel.",
        "parameters": {
            "type": "object",
            "properties": {
            "guild_id":   { "type": "integer" },
            "channel_id": { "type": "integer" },
            "top_n":      { "type": "integer", "default": 5 }
            },
            "required": []
        }
    },
    {
        "name": "find_users_by_skill",
        "description": "Identify authors whose messages mention a given skill keyword, with an example message and jump URL.",
        "parameters": {
            "type": "object",
            "properties": {
            "skill":      { "type": "string" },
            "guild_id":   { "type": "integer" },
            "channel_id": { "type": "integer" }
            },
            "required": ["skill"]
        }
    },
    {
        "name": "summarize_messages",
        "description": "Summarize messages between two ISO datetimes, optionally by guild/channel.",
        "parameters": {
            "type": "object",
            "properties": {
            "start_iso":  { "type": "string", "format": "date-time" },
            "end_iso":    { "type": "string", "format": "date-time" },
            "guild_id":   { "type": "integer" },
            "channel_id": { "type": "integer" },
            "as_json":    { "type": "boolean", "default": False }
            },
            "required": ["start_iso", "end_iso"]
        }
    }
]
