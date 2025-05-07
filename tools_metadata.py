from tools import (
    search_messages,
    summarize_weekly_activity,
    get_server_stats,
    extract_feedback_and_ideas,
    get_most_reacted_messages,
    find_messages_mentioning_user,
    get_pinned_messages,
    analyze_message_types,
    find_messages_with_keywords,
    find_users_by_skill,
    summarize_messages_in_range
)

TOOLS_METADATA = [
    {
        "name": "SummarizeWeeklyActivity",
        "function": summarize_weekly_activity,
        "description": "Summarizes weekly messages posted across all Discord channels, optionally specifying the week end date.",
    },
    {
        "name": "GetServerStats",
        "function": get_server_stats,
        "description": "Returns statistics like top users and active channels from the server's message history.",
    },
    {
        "name": "ExtractFeedbackAndIdeas",
        "function": extract_feedback_and_ideas,
        "description": "Extracts suggestions, feedback, and event ideas from past Discord discussions based on keyword scanning.",
    },
    {
        "name": "GetMostReactedMessages",
        "function": get_most_reacted_messages,
        "description": "Finds the most reacted messages across the server based on total reaction counts.",
    },
    {
        "name": "FindMessagesMentioningUser",
        "function": find_messages_mentioning_user,
        "description": "Finds all messages that mention a specific user by their ID.",
    },
    {
        "name": "GetPinnedMessages",
        "function": get_pinned_messages,
        "description": "Retrieves all messages that have been pinned in various Discord channels.",
    },
    {
        "name": "AnalyzeMessageTypes",
        "function": analyze_message_types,
        "description": "Counts different types of messages posted, grouped by message type (e.g., default, reply, etc.).",
    },
    {
        "name": "FindMessagesWithKeywords",
        "function": find_messages_with_keywords,
        "description": "Searches all messages for specified keywords and returns matches.",
    },
    {
        "name": "FindUsersBySkill",
        "function": find_users_by_skill,
        "description": "Searches Discord messages to find users mentioning specific skills, expertise, or introductions. Use for queries like 'Who knows Python?' or 'Find experts in ML.'"
    },
        {
        "name": "SummarizeMessagesInRange",
        "function": summarize_messages_in_range,
        "description": (
            "Summarizes messages sent between start_iso and end_iso. "
            "Optionally filter by guild_id and/or channel_id to scope the summary."
        ),
    },
        {
        "name": "search_messages",
        "function": search_messages,
        "description": (
            "Search messages by keywords, then rerank semantically. "
            "Supports optional guild_id, channel_id, and author_id filters."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query":      {"type": "string", "description": "Text to search for"},
                "guild_id":   {"type": "integer", "description": "Filter by guild ID"},
                "channel_id": {"type": "integer", "description": "Filter by channel ID"},
                "author_id":  {"type": "integer", "description": "Filter by author ID"},
                "top_n":      {"type": "integer", "description": "Number of results to return"}
            },
            "required": ["query"]
        }
    }
]
