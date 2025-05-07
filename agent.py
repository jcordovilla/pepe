import os
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain.agents import AgentExecutor
from langchain.agents.openai_functions_agent.agent_token_buffer_memory import AgentTokenBufferMemory
from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from tools_metadata import TOOLS_METADATA
from rag_engine import get_answer
from pydantic import BaseModel
from langchain.tools import StructuredTool
import dateparser
from datetime import datetime, timezone
import re

load_dotenv()

def normalize_timeframe_phrases(prompt: str) -> str:
    now = datetime.now(timezone.utc)

    def safe_parse(expression, default_text=None, date_only=False):
        dt = dateparser.parse(expression, settings={"RELATIVE_BASE": now})
        if dt:
            return dt.date().isoformat() if date_only else dt.isoformat()
        else:
            print(f"⚠️ Could not parse time expression: '{expression}'")
            return default_text or expression  # fallback: keep the original string

    patterns = [
        (r"\blast\s+(\d+)\s*(hours?|days?)\b", lambda m:
            f"since {safe_parse(m.group(0), default_text='an unspecified recent time')}"),
        (r"\bthis\s+week\b", lambda m:
            f"since {safe_parse('monday this week', default_text='monday this week')}"),
        (r"\byesterday\b", lambda m:
            f"on {safe_parse('yesterday', default_text='yesterday', date_only=True)}"),
        (r"\btoday\b", lambda m:
            f"on {safe_parse('today', default_text='today', date_only=True)}"),
        (r"\bthis\s+month\b", lambda m:
            f"since {safe_parse('first day of this month', default_text='the start of this month')}"),
        (r"\bfrom\s+(.+?)\s+to\s+(.+?)\b", lambda m:
            f"from {safe_parse(m.group(1), m.group(1))} to {safe_parse(m.group(2), m.group(2))}")
    ]

    updated = prompt
    for pattern, repl in patterns:
        updated = re.sub(pattern, repl, updated, flags=re.IGNORECASE)
    print("🔍 Normalized prompt:", updated)
    return updated

# 🔧 System prompt for the OpenAI agent
SYSTEM_MESSAGE = """
You are GenAI Pathfinder Assistant, a versatile AI designed to help users explore, summarize, and analyze Discord community conversations.

You can:
- Retrieve and summarize relevant Discord messages (RAG).
- Call specialized tools to calculate server statistics, extract feedback, find users with skills, highlight pinned messages, and more.

Behavior guidelines:
- Prefer using tools when a matching capability is available.
- If needed, perform direct semantic search using the message database.
- Always answer concisely and clearly unless JSON output is explicitly requested.
- When quoting messages, include: author, timestamp, channel, short snippet, and link (jump_url).
- Group, summarize, and explain findings when possible.
- If uncertain or insufficient context, suggest clarifying questions.

Goal:
Maximize clarity, utility, and actionable insights for the user — whether the request is simple or complex.
"""

# 🔧 Set up LLM
gpt_model = os.getenv("GPT_MODEL", "gpt-4-turbo-2024-04-09")
llm = ChatOpenAI(model=gpt_model, temperature=0)

# 🔧 Structured input for DiscordRAGSearch
class RAGSearchInput(BaseModel):
    query: str
    k: int = 5
    as_json: bool = False
    return_matches: bool = False

# 🔧 Register tools from tools_metadata
from langchain.agents import Tool
tools = [
    Tool(
        name=tool["name"],
        func=tool["function"],
        description=tool["description"]
    )
    for tool in TOOLS_METADATA
]

# 🔧 Add structured RAG search tool
tools.append(
    StructuredTool.from_function(
        name="DiscordRAGSearch",
        func=get_answer,
        description="Searches Discord messages using GPT-powered semantic search. Use for questions like 'Who mentioned onboarding last week?'",
        args_schema=RAGSearchInput
    )
)

# 🔧 (Optional) tool startup test
def test_registered_tools():
    failures = []
    for tool in tools:
        try:
            tool.func("test")  # Works for single-arg tools
        except Exception as e:
            failures.append((tool.name, str(e)))

    if failures:
        print("❌ Tool errors detected during startup:")
        for name, error in failures:
            print(f"- {name}: {error}")
    else:
        print("✅ All tools passed basic startup test.")

# 🔧 Create agent with OpenAI Functions support
agent = OpenAIFunctionsAgent.from_llm_and_tools(
    llm=llm,
    tools=tools,
    system_message=SYSTEM_MESSAGE,
)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True
)

# ✅ Main function called by your bot or app
def get_agent_answer(query: str) -> str:
    normalized = normalize_timeframe_phrases(query)
    return agent_executor.run(normalized)
