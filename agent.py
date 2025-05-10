import os
from dotenv import load_dotenv
from typing import Any

from langchain.chat_models import ChatOpenAI
from langchain.tools import StructuredTool
from langchain.agents import initialize_agent, AgentType

from tools import search_messages, summarize_messages
from time_parser import parse_timeframe

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
        parse_timeframe,
        name="parse_timeframe",
        description="Parse a natural-language timeframe (e.g. 'last week') into (start_iso, end_iso)."
    ),
    StructuredTool.from_function(
        search_messages,
        name="search_messages",
        description="Search Discord messages by keyword or semantically, scoped by guild_id, channel_id, or channel_name."
    ),
    StructuredTool.from_function(
        summarize_messages,
        name="summarize_messages",
        description="Summarize Discord messages between two ISO datetimes, scoped by guild_id, channel_id, or channel_name."
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
        
    try:
        return agent.run(query)
    except Exception as e:
        raise Exception(f"Agent failed to process query: {str(e)}")
