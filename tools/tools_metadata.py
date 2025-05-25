# tools_metadata.py

# Only include the two core tools: search_messages and summarize_messages

TOOLS = [
    {
        "name": "search_messages",
        "description": "Hybrid semantic and keyword search over Discord messages. Supports filtering by guild, channel, author, and skill/keyword.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query (natural language)"},
                "k": {"type": "integer", "description": "Number of results to return", "default": 5},
                "keyword": {"type": "string", "description": "Optional keyword or skill to filter by", "nullable": True},
                "guild_id": {"type": "integer", "description": "Guild/server ID", "nullable": True},
                "channel_id": {"type": "integer", "description": "Channel ID", "nullable": True},
                "channel_name": {"type": "string", "description": "Channel name (if ID not known)", "nullable": True},
                "author_name": {"type": "string", "description": "Author username to filter by", "nullable": True}
            },
            "required": ["query"]
        }
    },
    {
        "name": "summarize_messages",
        "description": "Summarize all messages in a given time range, optionally filtered by guild or channel.",
        "parameters": {
            "type": "object",
            "properties": {
                "start_iso": {"type": "string", "description": "Start time (ISO 8601)"},
                "end_iso": {"type": "string", "description": "End time (ISO 8601)"},
                "guild_id": {"type": "integer", "description": "Guild/server ID", "nullable": True},
                "channel_id": {"type": "integer", "description": "Channel ID", "nullable": True},
                "channel_name": {"type": "string", "description": "Channel name (if ID not known)", "nullable": True},
                "as_json": {"type": "boolean", "description": "Return summary as JSON", "default": False}
            },
            "required": ["start_iso", "end_iso"]
        }
    },
    {
        "name": "validate_data_availability",
        "description": "Check if the database has messages and return their count, available channels, and date range.",
        "parameters": {
            "type": "object",
            "properties": {}
        }
    },
    {
        "name": "extract_skill_terms",
        "description": "Extract skill-related terms from a query to enhance search capabilities.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The query to analyze for skill terms"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "get_channels",
        "description": "Get a list of all available channels with their IDs and message counts.",
        "parameters": {
            "type": "object",
            "properties": {
                "guild_id": {"type": "integer", "description": "Optional guild/server ID to filter channels", "nullable": True}
            }
        }
    },
    {
        "name": "resolve_channel_name",
        "description": "Convert a channel name to its ID, optionally scoped to a specific guild.",
        "parameters": {
            "type": "object",
            "properties": {
                "channel_name": {"type": "string", "description": "The channel name to resolve"},
                "guild_id": {"type": "integer", "description": "Optional guild/server ID to scope the search", "nullable": True}
            },
            "required": ["channel_name"]
        }
    }
]

# Deprecated or auxiliary tool definitions (commented out for tidying):
# ...existing code for other tools, now commented out...
