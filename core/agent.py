import os
from dotenv import load_dotenv
from typing import Any

from langchain.chat_models import ChatOpenAI
from langchain.tools import StructuredTool
from langchain.agents import initialize_agent, AgentType

from tools.tools import search_messages, summarize_messages, validate_data_availability

# ─── Env & LLM Setup ────────────────────────────────────────────────────────────

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

MODEL_NAME = os.getenv("GPT_MODEL", "gpt-4-turbo")

llm = ChatOpenAI(
    model_name=MODEL_NAME,
    openai_api_key=OPENAI_API_KEY,
    temperature=0.0
)

# ─── Tool Registry ──────────────────────────────────────────────────────────────

tools = [
    StructuredTool.from_function(
        search_messages,
        name="search_messages",
        description="Search Discord messages by keyword or semantically, scoped by guild_id, channel_id, or channel_name. Always returns message content, author, timestamp, jump_url, and metadata."
    ),
    StructuredTool.from_function(
        summarize_messages,
        name="summarize_messages",
        description="Summarize Discord messages between two ISO datetimes, scoped by guild_id, channel_id, or channel_name."
    ),
    StructuredTool.from_function(
        validate_data_availability,
        name="validate_data_availability",
        description="Check if the database has messages and return their count, available channels, and date range."
    ),
]

# ─── System Prompt ──────────────────────────────────────────────────────────────

SYSTEM_MESSAGE = """
You are a specialized Discord data assistant. To handle any user request:

1. ALWAYS check for time expressions first (e.g. "past 2 days", "last week", "yesterday"). If found:
   - Call parse_timeframe(text) to get start_iso and end_iso
   - Use these timestamps with summarize_messages() to get a time-windowed summary

2. For non-time queries or after getting time context:
   - Use search_messages() to find specific messages
   - Include any parsed timestamps in your search parameters

3. Always use function-calling; never answer directly without invoking these tools.
   - For time-based queries, you MUST call parse_timeframe() first
   - Then use the returned timestamps with summarize_messages() or search_messages()

4. For search results, ALWAYS return a list of messages with the following fields for each message:
   - author (username and display_name if available)
   - timestamp (ISO format)
   - content (the actual message text)
   - jump_url (a direct link to the message)
   - channel_name, guild_id, channel_id, message_id
   Do NOT return only summaries or links—always include the actual message content and metadata.

5. For channel name queries, resolve human-friendly names (e.g., "#general", "#dev") to channel IDs. If the channel is unknown, reply with a clear note (e.g., "Channel '#foo' not found.").

6. For data availability queries (e.g., "How many messages are in the database?"), call the data availability tool and return the count, date range, and available channels.

7. For output formatting, always include jump URLs and message content in the results. If no results, reply with a clear message.

8. For invalid or missing fields, reply with a clear error message.
"""

# ─── Agent Initialization ────────────────────────────────────────────────────────

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
    system_message=SYSTEM_MESSAGE,
    handle_parsing_errors=True  # Better error handling for tool parsing
)

# ─── Public Entry Point ─────────────────────────────────────────────────────────

def get_agent_answer(query: str) -> Any:
    """
    Send the user's raw query to the function-calling agent.
    
    Args:
        query: The user's question or request
        
    Returns:
        The agent's response
        
    Raises:
        ValueError: If the query is empty
        Exception: For other errors during agent execution
    """
    if not query.strip():
        raise ValueError("Query cannot be empty")

    # Data availability queries
    lower_query = query.lower().strip()
    if any(
        kw in lower_query for kw in [
            "what data is currently cached", "what data is available", "how many messages", "data availability", "database status", "message count", "what's in the database", "how much data", "how many channels", "date range"
        ]
    ):
        from tools.tools import validate_data_availability
        data = validate_data_availability()
        if data["status"] == "ok":
            channels = ", ".join([f"{name} ({count})" for name, count in data["channels"].items()])
            return (
                f"Data available: {data['count']} messages across {len(data['channels'])} channels. "
                f"Date range: {data['date_range']['oldest']} to {data['date_range']['newest']}.\n"
                f"Channels: {channels}"
            )
        else:
            return data["message"]

    # Fallback/clarification for vague queries
    lower_query = query.lower()
    if not any(
        kw in lower_query for kw in ["channel", "#", "time", "day", "week", "month", "year", "keyword", "messages", "summarize", "find", "search", "between", "from", "to", "in "]
    ):
        return "Which channel, timeframe, or keyword would you like to search or summarize? Please specify so I can help you."
    
    try:
        return agent.run(query)
    except Exception as e:
        raise Exception(f"Agent failed to process query: {str(e)}")
