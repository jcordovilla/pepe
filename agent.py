# agent.py

import os
from dotenv import load_dotenv
from typing import Optional, Any
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.tools import StructuredTool
from pydantic import BaseModel, Field

from time_parser import parse_timeframe
from tools import (
    summarize_messages,
    search_messages,
    get_most_reacted_messages,
    find_users_by_skill,
    get_answer
)

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME      = os.getenv("GPT_MODEL", "gpt-4-turbo-2024-04-09")

# --- System prompt ---
SYSTEM_MESSAGE = """
You are GenAI Pathfinder Assistant, a versatile AI designed to help users explore, summarize, and analyze Discord community conversations.

You can:
- Parse natural-language timeframes (e.g., "last week") into concrete datetimes.
- Summarize messages in a given date range and (optionally) channel.
- Perform hybrid searches over your Discord history.
- Highlight most reacted messages and find users by skill keyword.

Behavior guidelines:
- Always use the structured tools when possible.
- Return JSON only when explicitly requested.
- Keep answers concise, include author/timestamp/channel snippets and jump URLs.
"""

# --- Initialize the LLM ---
llm = ChatOpenAI(
    model_name=MODEL_NAME,
    temperature=0.0,
    openai_api_key=OPENAI_API_KEY,
)

# --- Define schemas for each tool ---
class ParseTimeframeSchema(BaseModel):
    text: str = Field(..., description="Natural-language timeframe (e.g. 'last week', 'April 1-7').")

class SummarizeSchema(BaseModel):
    start_iso: str                      = Field(..., description="ISO start datetime for summary.")
    end_iso:   str                      = Field(..., description="ISO end datetime for summary.")
    guild_id:  Optional[int] = Field(None, description="Discord guild ID filter.")
    channel_id: Optional[int] = Field(None, description="Discord channel ID filter.")
    as_json:   Optional[bool] = Field(False, description="Return JSON object if True.")

class SearchSchema(BaseModel):
    query:       str              = Field(..., description="Search query string.")
    keyword:     Optional[str]    = Field(None, description="Exact keyword pre-filter.")
    guild_id:    Optional[int]    = Field(None, description="Discord guild ID filter.")
    channel_id:  Optional[int]    = Field(None, description="Discord channel ID filter.")
    author_name: Optional[str]    = Field(None, description="Fuzzy author filter.")
    k:           Optional[int]    = Field(5, description="Number of top results to return.")

class MostReactedSchema(BaseModel):
    guild_id:   Optional[int] = Field(None, description="Discord guild ID filter.")
    channel_id: Optional[int] = Field(None, description="Discord channel ID filter.")
    top_n:      Optional[int] = Field(5, description="Number of top messages to return.")

class FindUsersSchema(BaseModel):
    skill:      str              = Field(..., description="Skill keyword to search for.")
    guild_id:   Optional[int]    = Field(None, description="Discord guild ID filter.")
    channel_id: Optional[int]    = Field(None, description="Discord channel ID filter.")

class RAGSearchInput(BaseModel):
    query: str = Field(..., description="Search query string.")
    k: Optional[int] = Field(5, description="Number of top results to return.")
    guild_id: Optional[int] = Field(None, description="Discord guild ID filter.")
    channel_id: Optional[int] = Field(None, description="Discord channel ID filter.")
    channel_name: Optional[str] = Field(None, description="Discord channel name filter (e.g., 'general').")

# --- Wrap each function as a StructuredTool ---
tools = [
    StructuredTool.from_function(
        parse_timeframe,
        name="parse_timeframe",
        description="Convert a natural-language timeframe into (start_iso, end_iso).",
        args_schema=ParseTimeframeSchema
    ),
    StructuredTool.from_function(
        summarize_messages,
        name="summarize_messages",
        description="Summarize Discord messages between two ISO datetimes.",
        args_schema=SummarizeSchema
    ),
    StructuredTool.from_function(
        search_messages,
        name="search_messages",
        description="Hybrid keyword+semantic search over Discord messages.",
        args_schema=SearchSchema
    ),
    StructuredTool.from_function(
        get_most_reacted_messages,
        name="get_most_reacted_messages",
        description="Retrieve the top-N most reacted messages.",
        args_schema=MostReactedSchema
    ),
    StructuredTool.from_function(
        find_users_by_skill,
        name="find_users_by_skill",
        description="Find users mentioning a specific skill in messages.",
        args_schema=FindUsersSchema
    ),
    StructuredTool.from_function(
        get_answer,
        name="discord_rag_search",
        description="Run a GPT-powered RAG query over Discord messages.",
        args_schema=RAGSearchInput
    )
]

# --- Create the agent using OPENAI_FUNCTIONS mode ---
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
    system_message=SYSTEM_MESSAGE
)

def get_agent_answer(query: str) -> Any:
    """
    Main entry point for your app or REPL.
    Normalizes timeframes then runs the agent.
    """
    # Re-use your existing normalization if desired:
    # normalized = normalize_timeframe_phrases(query)
    # return agent.run(normalized)

    return agent.run(query)
