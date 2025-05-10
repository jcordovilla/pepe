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
MODEL_NAME     = os.getenv("GPT_MODEL", "gpt-4-turbo")

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
1. If it involves time-based queries, call `parse_timeframe(text)` first.
2. To retrieve messages, call `search_messages(query, k, keyword, guild_id, channel_id, channel_name, author_name)`.
3. To summarize a window, call `summarize_messages(start_iso, end_iso, guild_id, channel_id, channel_name, as_json)`.
Always use function-calling; do not answer directly without invoking these tools.
"""

# ─── Agent Initialization ────────────────────────────────────────────────────────

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
    system_message=SYSTEM_MESSAGE
)

# ─── Public Entry Point ─────────────────────────────────────────────────────────

def get_agent_answer(query: str) -> Any:
    """
    Send the user's raw query to the function-calling agent.
    """
    return agent.run(query)
