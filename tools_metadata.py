from tools import (
    search_messages,
    get_most_reacted_messages,
    find_users_by_skill,
    summarize_messages,
    summarize_messages_in_range
)

TOOLS_METADATA = [
    {
        "name": "search_messages",
        "description": "Hybrid keyword + semantic search over Discord messages, with optional guild, channel (by ID or name), or author filters.",
        "parameters": {
            "type": "object",
            "properties": {
                "query":         {"type": "string"},
                "keyword":       {"type": "string"},
                "guild_id":      {"type": "integer"},
                "channel_id":    {"type": "integer"},
                "channel_name":  {"type": "string"},
                "author_name":   {"type": "string"},
                "k":             {"type": "integer", "default": 5}
            },
            "required": ["query"]
        }
    },
    {
        "name": "get_most_reacted_messages",
        "description": "Return the top N messages by total reaction count, optionally scoped by guild and/or channel (by ID or name).",
        "parameters": {
            "type": "object",
            "properties": {
                "guild_id":     {"type": "integer"},
                "channel_id":   {"type": "integer"},
                "channel_name": {"type": "string"},
                "top_n":        {"type": "integer", "default": 5}
            },
            "required": []
        }
    },
    {
        "name": "find_users_by_skill",
        "description": "Identify users whose messages mention a specific skill keyword, optionally filtered by guild and/or channel (by ID or name).",
        "parameters": {
            "type": "object",
            "properties": {
                "skill":        {"type": "string"},
                "guild_id":     {"type": "integer"},
                "channel_id":   {"type": "integer"},
                "channel_name": {"type": "string"}
            },
            "required": ["skill"]
        }
    },
    {
        "name": "summarize_messages",
        "description": "Summarize messages sent between two ISO datetimes, optionally filtered by guild and/or channel (by ID or name), returning text or JSON.",
        "parameters": {
            "type": "object",
            "properties": {
                "start_iso":    {"type": "string", "format": "date-time"},
                "end_iso":      {"type": "string", "format": "date-time"},
                "guild_id":     {"type": "integer"},
                "channel_id":   {"type": "integer"},
                "channel_name": {"type": "string"},
                "as_json":      {"type": "boolean", "default": False}
            },
            "required": ["start_iso", "end_iso"]
        }
    },
    {
        "name": "summarize_messages_in_range",
        "description": "Legacy wrapper: summarize messages between two ISO datetimes, scoped by guild and/or channel (by ID or name). Returns text or JSON based on output_format.",
        "parameters": {
            "type": "object",
            "properties": {
                "start_iso":    {"type": "string", "format": "date-time"},
                "end_iso":      {"type": "string", "format": "date-time"},
                "guild_id":     {"type": "integer"},
                "channel_id":   {"type": "integer"},
                "channel_name": {"type": "string"},
                "output_format":{"type": "string", "enum": ["text","json"], "default": "text"}
            },
            "required": ["start_iso", "end_iso"]
        }
    }
]
