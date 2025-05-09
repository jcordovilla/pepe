import os
from dotenv import load_dotenv
from typing import Any

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
API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("GPT_MODEL", "gpt-4-turbo-2024-04-09")

# Initialize LLM
temp_llm = ChatOpenAI(
    model_name=MODEL_NAME,
    openai_api_key=API_KEY,
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

# Initialize agent with function-calling
agent = initialize_agent(
    tools,
    temp_llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True
)

# Prefix to guide summary chaining
def _with_prefix(query: str) -> str:
    return (
        "Summarize messages command:\n"
        "When asking to summarize by natural language timeframe, first use parse_timeframe, then summarize_messages.\n"
        f"User: {query}"
    )

# Main entrypoint
def get_agent_answer(query: str) -> Any:
    """
    Entrypoint: if the user asks to summarize messages in natural language,
    manually parse the timeframe and call summarize_messages.
    Otherwise, delegate to the agent.
    """
    q = query.strip().lower()
    # Detect natural-language summary requests
    if q.startswith("summarize messages"):
        # e.g. "Summarize messages from last week in #non-coders-learning"
        # extract the phrase between "from" and "in"
        import re

        m = re.search(r"from (.?) in", q)
        if not m:
            return "⚠️ Please specify a timeframe, e.g. 'from last week in #channel'."
        timeframe = m.group(1)
        # parse range
        start_dt, end_dt = parse_timeframe(timeframe)
        # now call the summarizer directly
        return summarize_messages(
            start_dt.isoformat(),
            end_dt.isoformat(),
            # you can hard-code guild/channel if you prefer, or let the user supply IDs
        )
    # fallback to general agent for everything else
    return agent.run(_with_prefix(query))
