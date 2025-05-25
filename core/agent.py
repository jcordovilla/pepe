import os
from dotenv import load_dotenv
from typing import Any

from langchain_openai import ChatOpenAI
from langchain.tools import StructuredTool
from langchain.agents import initialize_agent, AgentType

from tools.tools import (
    search_messages,
    summarize_messages,
    validate_data_availability,
    extract_skill_terms,
    get_channels,
    resolve_channel_name
)

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
    StructuredTool.from_function(
        extract_skill_terms,
        name="extract_skill_terms",
        description="Extract skill-related terms from a query to enhance search capabilities. Useful for identifying technical skills, programming languages, or frameworks mentioned in queries."
    ),
    StructuredTool.from_function(
        get_channels,
        name="get_channels",
        description="Get a list of all available channels with their IDs and message counts. Optionally filter by guild_id."
    ),
    StructuredTool.from_function(
        resolve_channel_name,
        name="resolve_channel_name",
        description="Convert a channel name to its ID, optionally scoped to a specific guild. Useful for resolving human-readable channel names to their IDs."
    ),
]

# ─── System Prompt ──────────────────────────────────────────────────────────────

SYSTEM_MESSAGE = """
You are a specialized Discord data assistant powered by FAISS vector search and LangChain. To handle any user request:

1. For semantic search queries (natural language questions about content):
   - Use search_messages() with the raw query for semantic similarity search
   - FAISS will automatically find semantically similar messages
   - No need to break down the query - let the vector search handle it
   - Example: "What did people say about Python async programming?" -> direct search

2. For time-bounded queries:
   - If time expression present (e.g. "past 2 days", "last week"):
     * Call parse_timeframe(text) to get start_iso and end_iso
     * Use these timestamps with summarize_messages()
   - If no time expression:
     * Use the full available date range from validate_data_availability()
     * This ensures comprehensive search across all history

3. For hybrid search (combining semantic and keyword):
   - Use search_messages() with both query and keyword parameters
   - Let FAISS handle semantic similarity
   - Use keyword for exact matches
   - Example: "Find discussions about Python async programming" + keyword="asyncio"

4. For channel-specific queries:
   - First use get_channels() to verify channel existence
   - Use resolve_channel_name() to convert names to IDs
   - Then use search_messages() with channel_id
   - Example: "What was discussed in #general-chat about async?"

5. For skill/technical queries:
   - Use extract_skill_terms() to identify technical terms
   - Combine with semantic search for better results
   - Example: "Find discussions about Python async programming" -> extract "Python", "async" as terms

6. Always use function-calling with these best practices:
   - For semantic search: Use raw natural language queries
   - For time-based: Always include timestamps
   - For hybrid: Combine semantic + keywords
   - For channel-specific: Always verify channel first

7. For search results, return messages with these fields:
   - author (username and display_name)
   - timestamp (ISO format)
   - content (full message text)
   - jump_url (direct link)
   - channel_name, guild_id, channel_id, message_id
   Always include full message content and metadata.

8. For channel resolution:
   - Use resolve_channel_name() for human-friendly names
   - Handle unknown channels gracefully
   - Example: "#general" -> resolve to channel_id

9. For data availability:
   - Use validate_data_availability() to check database state
   - Return count, date range, and channels
   - Use this to inform search scope

10. For output formatting:
    - Include jump URLs and full message content
    - Group related messages by topic
    - Include search context (timeframe, channel)
    - For no results, explain search parameters used

11. Error handling:
    - For invalid fields: Clear error messages
    - For no results: Explain search parameters
    - For API errors: Retry with fallback parameters
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

    # Always use search path for common human question patterns
    search_triggers = [
        "what was discussed about",
        "what did people say about",
        "what did people talk about",
        "what was said about",
        "what were the discussions about",
        "what was talked about",
        "what was mentioned about",
        "what did users say about",
        "what did members say about",
        "what was the conversation about"
    ]
    q_lower = query.strip().lower()
    if any(q_lower.startswith(trigger) for trigger in search_triggers):
        try:
            result = agent.run(query)
            # Propagate empty list or dict directly
            if result == [] or result == {}:
                return result
            return result
        except Exception as e:
            raise Exception(f"Agent failed to process query: {str(e)}")

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

    # Remove the fallback/clarification for vague queries since we now handle all queries
    try:
        result = agent.run(query)
        # Propagate empty list or dict directly
        if result == [] or result == {}:
            return result
        return result
    except Exception as e:
        raise Exception(f"Agent failed to process query: {str(e)}")
