# tools_metadata.py

TOOLS_METADATA = [
  {
    "name": "search_messages",
    "description": "Search Discord messages by keyword and/or semantically, with optional guild or channel filters.",
    "parameters": {
      "type": "object",
      "properties": {
        "query":         { "type": "string", "description": "Natural-language search query" },
        "k":             { "type": "integer", "minimum": 1, "maximum": 20, "default": 5 },
        "keyword":       { "type": "string",  "description": "Exact keyword pre-filter (optional)" },
        "guild_id":      { "type": "integer", "description": "Discord guild ID (optional)" },
        "channel_id":    { "type": "integer", "description": "Discord channel ID (optional)" },
        "channel_name":  { "type": "string",  "description": "Discord channel name (optional)" },
        "author_name":   { "type": "string",  "description": "Author username filter (optional)" }
      },
      "required": ["query"]
    }
  },
  {
    "name": "summarize_messages",
    "description": "Summarize Discord messages within a given ISO time range, scoped to a guild or channel.",
    "parameters": {
      "type": "object",
      "properties": {
        "start_iso":    { "type": "string",  "format": "date-time", "description": "Start of time window in ISO format" },
        "end_iso":      { "type": "string",  "format": "date-time", "description": "End of time window in ISO format" },
        "guild_id":     { "type": "integer", "description": "Discord guild ID (optional)" },
        "channel_id":   { "type": "integer", "description": "Discord channel ID (optional)" },
        "channel_name": { "type": "string",  "description": "Discord channel name (optional)" },
        "as_json":      { "type": "boolean", "default": false, "description": "Return structured JSON if true" }
      },
      "required": ["start_iso","end_iso"]
    }
  }
]
