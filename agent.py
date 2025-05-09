import os
from dotenv import load_dotenv
from typing import Any, Optional

from langchain.chat_models import ChatOpenAI
from langchain.tools import StructuredTool
from langchain.agents import initialize_agent, AgentType

from tools import (
    summarize_messages,
    search_messages,
    get_most_reacted_messages,
    find_users_by_skill
)
from rag_engine import get_answer as discord_rag_search
from time_parser import parse_timeframe

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("GPT_MODEL", "gpt-4-turbo-2024-04-09")

# Initialize LLM
llm = ChatOpenAI(
    model_name=MODEL_NAME,
    openai_api_key=OPENAI_API_KEY,
    temperature=0.0
)

# Register tools
tools = [
    StructuredTool.from_function(
        parse_timeframe,
        name="parse_timeframe",
        description="Parse a natural-language timeframe into start and end ISO datetimes.",
    ),
    StructuredTool.from_function(
        summarize_messages,
        name="summarize_messages",
        description="Summarize messages between two ISO datetimes, optionally scoped by guild/channel.",
    ),
    StructuredTool.from_function(
        search_messages,
        name="search_messages",
        description="Perform hybrid keyword and semantic search over Discord messages with filters.",
    ),
    StructuredTool.from_function(
        get_most_reacted_messages,
        name="get_most_reacted_messages",
        description="Get top N messages by reaction count, with optional filters.",
    ),
    StructuredTool.from_function(
        find_users_by_skill,
        name="find_users_by_skill",
        description="Find authors whose messages mention a skill keyword, with example and URL.",
    ),
    StructuredTool.from_function(
        discord_rag_search,
        name="discord_rag_search",
        description="Run a GPT-powered RAG query over Discord messages.",
    )
]

# --- System prompt to enforce tool usage ---
SYSTEM_MESSAGE = '''
You are a Discord assistant that must always use the provided tools to answer user questions about Discord data.
Do NOT rely on your internal knowledge or mention any knowledge cutoff. Instead:
- For time-based summaries, call `parse_timeframe` then `summarize_messages`.
- For searches, call `search_messages` with query, guild_id or channel_name.
- For reaction stats, call `get_most_reacted_messages`.
- For skill lookups, call `find_users_by_skill`.
- For RAG queries, call `discord_rag_search`.
Always invoke the tools via function calling; do not produce your own summaries.
'''

# Initialize agent with function-calling and custom system message
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
    system_message=SYSTEM_MESSAGE
)

# Expose main entrypoint
def get_agent_answer(query: str) -> Any:
    """
    Main entry point for your app or REPL. Queries are sent directly to the agent.
    """
    return agent.run(query)
