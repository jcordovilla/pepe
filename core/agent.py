from typing import Any, Dict, List
import logging
from langchain.tools import StructuredTool
from langchain.agents import initialize_agent, AgentType
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from core.ai_client import get_ai_client
from core.config import get_config

logger = logging.getLogger(__name__)

from tools.tools import (
    search_messages,
    summarize_messages,
    validate_data_availability,
    extract_skill_terms,
    get_channels,
    resolve_channel_name
)

# ─── Local LLM Wrapper for LangChain ────────────────────────────────────────────

class OllamaLLM(LLM):
    """LangChain-compatible wrapper for our Ollama AI client."""
    
    @property
    def _llm_type(self) -> str:
        return "ollama"
    
    def _call(
        self,
        prompt: str,
        stop: List[str] = None,
        run_manager: CallbackManagerForLLMRun = None,
        **kwargs: Any,
    ) -> str:
        """Call the Ollama model via our AI client."""
        ai_client = get_ai_client()
        messages = [{"role": "user", "content": prompt}]
        return ai_client.chat_completion(messages)

# ─── LLM Setup ──────────────────────────────────────────────────────────────────

config = get_config()
llm = OllamaLLM()

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

# Create a simple chain-based agent that works better with local models
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Create agent prompt
agent_prompt = PromptTemplate(
    input_variables=["query", "tools"],
    template="""You are a helpful Discord bot assistant with access to these tools:

{tools}

User Query: {query}

Think through this step by step:
1. What type of information is the user asking for?
2. Which tool(s) would be most helpful?
3. What parameters should I use?

Based on the user's query, determine the best approach and provide a helpful response. If you need to search for messages, channel information, or summaries, explain what you're looking for and provide relevant results.

Response:"""
)

# Create the chain
agent_chain = LLMChain(llm=llm, prompt=agent_prompt)

# Helper function to format tools for the prompt
def format_tools_for_prompt():
    tool_descriptions = []
    for tool in tools:
        tool_descriptions.append(f"- {tool.name}: {tool.description}")
    return "\n".join(tool_descriptions)

def execute_agent_query(query: str) -> str:
    """Execute agent query using local model approach."""
    try:
        # First, let the LLM understand the query
        tools_text = format_tools_for_prompt()
        response = agent_chain.run(query=query, tools=tools_text)
        
        # Simple intent detection and tool calling
        query_lower = query.lower()
        
        # Channel listing
        if any(word in query_lower for word in ['channel', 'channels', 'list']):
            try:
                channels = get_channels()
                if channels:
                    channel_list = "\n".join([f"- **{ch['name']}** ({ch['id']})" for ch in channels[:20]])
                    return f"Here are the available channels:\n{channel_list}\n\n{response}"
                else:
                    return "No channels found."
            except Exception as e:
                logger.error(f"Error getting channels: {e}")
        
        # Message search
        if any(word in query_lower for word in ['search', 'find', 'message', 'about']):
            try:
                # Extract search parameters from query (simplified)
                search_results = search_messages(query=query, k=5)
                if search_results:
                    formatted_results = []
                    for msg in search_results[:3]:
                        author = msg.get('author', {}).get('username', 'Unknown')
                        content = msg.get('content', '')[:100] + '...' if len(msg.get('content', '')) > 100 else msg.get('content', '')
                        channel = msg.get('channel_name', 'Unknown')
                        formatted_results.append(f"**{author}** in #{channel}: {content}")
                    
                    results_text = "\n".join(formatted_results)
                    return f"Here are some relevant messages:\n{results_text}\n\n{response}"
                else:
                    return f"No messages found for your query. {response}"
            except Exception as e:
                logger.error(f"Error searching messages: {e}")
        
        return response
        
    except Exception as e:
        logger.error(f"Agent execution error: {e}")
        return f"I apologize, but I encountered an error processing your request: {str(e)}"

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
            "what data is currently cached", "what data is available", "how many messages", 
            "data availability", "database status", "message count", "what's in the database", 
            "how much data", "how many channels", "date range"
        ]
    ):
        from tools.tools import validate_data_availability
        try:
            data = validate_data_availability()
            if data.get("status") == "ok":
                channels = ", ".join([f"{name} ({count})" for name, count in data["channels"].items()])
                return (
                    f"Data available: {data['count']} messages across {len(data['channels'])} channels. "
                    f"Date range: {data['date_range']['oldest']} to {data['date_range']['newest']}.\n"
                    f"Channels: {channels}"
                )
            else:
                return data.get("message", "Unable to check data availability")
        except Exception as e:
            logger.error(f"Data availability check failed: {e}")
            return "Unable to check data availability at this time."

    # Use the new simplified agent approach
    try:
        result = execute_agent_query(query)
        return result
    except Exception as e:
        logger.error(f"Agent failed to process query: {e}")
        return f"I apologize, but I encountered an error processing your request: {str(e)}"
