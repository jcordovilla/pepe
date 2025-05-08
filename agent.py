import os
from dotenv import load_dotenv
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from langchain.chat_models import ChatOpenAI
from langchain.tools import StructuredTool
from langchain.agents import initialize_agent, AgentType

from tools import (
    summarize_messages,
    search_messages,
    get_most_reacted_messages,
    find_users_by_skill
)
from rag_engine import get_answer
from time_parser import parse_timeframe

# Load environment variables
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("GPT_MODEL", "gpt-4-turbo-2024-04-09")

# Initialize LLM with function-calling support
llm = ChatOpenAI(
    model_name=MODEL_NAME,
    openai_api_key=API_KEY,
    temperature=0
)

# --- Define argument schemas for tools ---
class ParseTimeframeSchema(BaseModel):
    text: str = Field(..., description="Natural-language timeframe, e.g. 'last week', 'April 1-7'")

class RAGSearchInput(BaseModel):
    query: str = Field(..., description="Natural-language question for RAG search")
    k: int = Field(5, description="Number of top context matches to retrieve")
    as_json: bool = Field(False, description="Return raw JSON answer if True")
    return_matches: bool = Field(False, description="Include context matches list if True")

class SummarizeSchema(BaseModel):
    start_iso: str = Field(..., description="ISO 8601 start datetime for summary range")
    end_iso: str = Field(..., description="ISO 8601 end datetime for summary range")
    guild_id: Optional[int] = Field(None, description="Discord guild ID filter (optional)")
    channel_id: Optional[int] = Field(None, description="Discord channel ID filter (optional)")
    as_json: bool = Field(False, description="Return JSON object if True")

class SearchSchema(BaseModel):
    query: str = Field(..., description="Search query string")
    keyword: Optional[str] = Field(None, description="Exact keyword pre-filter, optional")
    guild_id: Optional[int] = Field(None, description="Discord guild ID filter, optional")
    channel_id: Optional[int] = Field(None, description="Discord channel ID filter, optional")
    author_name: Optional[str] = Field(None, description="Fuzzy-match author username, optional")
    k: int = Field(5, description="Number of top results to return")

class MostReactedSchema(BaseModel):
    guild_id: Optional[int] = Field(None, description="Discord guild ID filter, optional")
    channel_id: Optional[int] = Field(None, description="Discord channel ID filter, optional")
    top_n: int = Field(5, description="Number of top messages by reaction count to return")

class FindUsersSchema(BaseModel):
    skill: str = Field(..., description="Skill keyword to search in message content")
    guild_id: Optional[int] = Field(None, description="Discord guild ID filter, optional")
    channel_id: Optional[int] = Field(None, description="Discord channel ID filter, optional")

# --- Register tools ---
tools = [
    StructuredTool.from_function(
        parse_timeframe,
        name="parse_timeframe",
        description="Convert a natural-language timeframe into a start and end ISO datetime.",
        args_schema=ParseTimeframeSchema
    ),
    StructuredTool.from_function(
        summarize_messages,
        name="summarize_messages",
        description="Summarize Discord messages between two ISO datetimes, optionally scoped by guild/channel.",
        args_schema=SummarizeSchema
    ),
    StructuredTool.from_function(
        search_messages,
        name="search_messages",
        description="Hybrid keyword + semantic search over Discord messages with optional filters.",
        args_schema=SearchSchema
    ),
    StructuredTool.from_function(
        get_most_reacted_messages,
        name="get_most_reacted_messages",
        description="Return the top N messages by reaction count, optionally scoped by guild/channel.",
        args_schema=MostReactedSchema
    ),
    StructuredTool.from_function(
        find_users_by_skill,
        name="find_users_by_skill",
        description="Identify authors whose messages mention a given skill keyword, with example message and URL.",
        args_schema=FindUsersSchema
    ),
    StructuredTool.from_function(
        get_answer,
        name="discord_rag_search",
        description="Retrieve and answer questions using GPT-powered RAG over Discord messages.",
        args_schema=RAGSearchInput
    )
]

# --- Initialize agent ---
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True
)

# --- Expose main entrypoint ---
def get_agent_answer(query: str) -> str:
    """
    Send a user query to the agent and return the response.
    """
    return agent.run(query)
