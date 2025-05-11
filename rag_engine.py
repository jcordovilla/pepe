# rag_engine.py

import os
from typing import Any, Dict, List, Optional, Union
from dotenv import load_dotenv
from openai import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from utils.helpers import build_jump_url
from tools import resolve_channel_name, summarize_messages, validate_data_availability
from time_parser import parse_timeframe, extract_time_reference, extract_channel_reference, extract_content_reference
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import json
from functools import lru_cache
from sqlalchemy.orm import Session
from db import Message, get_db_session
import logging
import time
from prometheus_client import Counter, Histogram, start_http_server
import threading

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Prometheus metrics
QUERY_COUNT = Counter('rag_query_total', 'Total number of RAG queries')
QUERY_ERRORS = Counter('rag_query_errors', 'Number of failed RAG queries')
QUERY_DURATION = Histogram('rag_query_duration_seconds', 'Time spent processing RAG queries')
VECTOR_SEARCH_DURATION = Histogram('vector_search_duration_seconds', 'Time spent in vector search')
OPENAI_API_DURATION = Histogram('openai_api_duration_seconds', 'Time spent in OpenAI API calls')

# Start Prometheus metrics server in a separate thread
def start_metrics_server(port: int = 8000):
    """Start Prometheus metrics server."""
    try:
        start_http_server(port)
        logger.info(f"Metrics server started on port {port}")
    except Exception as e:
        logger.error(f"Failed to start metrics server: {e}")

# Start metrics server in background
metrics_thread = threading.Thread(target=start_metrics_server, daemon=True)
metrics_thread.start()

# ——— Load config & initialize clients ———
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GPT_MODEL       = os.getenv("GPT_MODEL", "gpt-4-turbo-2024-04-09")
INDEX_DIR       = "index_faiss"

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
openai_client   = OpenAI(api_key=OPENAI_API_KEY)

def load_vectorstore() -> FAISS:
    """
    Load the locally saved FAISS index from disk.
    """
    try:
        logger.info("Loading vector store from disk")
        return FAISS.load_local(INDEX_DIR, embedding_model, allow_dangerous_deserialization=True)
    except Exception as e:
        logger.error(f"Failed to load vector store: {e}")
        raise

def get_top_k_matches(
    query: str,
    k: int = 5,
    guild_id: Optional[int]      = None,
    channel_id: Optional[int]    = None,
    channel_name: Optional[str]  = None
) -> List[Dict[str, Any]]:
    """
    Retrieve the top-k FAISS matches for 'query',
    optionally scoped to a guild, channel_id, or channel_name
    via manual post-filtering.
    """
    with VECTOR_SEARCH_DURATION.time():
        try:
            logger.info(f"Searching for matches: query='{query}', k={k}")
            store = load_vectorstore()
            # 1) Fetch extra candidates for reliable filtering
            fetch_k = k * 10
            retriever = store.as_retriever(search_kwargs={"k": fetch_k})
            docs = retriever.get_relevant_documents(query)
            metas = [doc.metadata for doc in docs]

            # 2) Resolve a human channel_name to ID if needed
            if channel_name and not channel_id:
                resolved = resolve_channel_name(channel_name, guild_id)
                if resolved is None:
                    logger.warning(f"Unknown channel name: {channel_name}")
                    raise ValueError(f"Unknown channel name: {channel_name}")
                channel_id = resolved

            # 3) Manual metadata filters
            if guild_id is not None:
                metas = [m for m in metas if m.get("guild_id") == str(guild_id)]
            if channel_id is not None:
                metas = [m for m in metas if m.get("channel_id") == str(channel_id)]

            # 4) Return the top-k of whatever remains
            results = metas[:k]
            logger.info(f"Found {len(results)} matches")
            return results
        except Exception as e:
            logger.error(f"Error in vector search: {e}")
            raise

def safe_jump_url(metadata: Dict[str, Any]) -> str:
    """
    Ensure the metadata contains a valid jump_url, constructing one if missing.
    """
    url = metadata.get("jump_url")
    if url:
        return url
    try:
        gid = int(metadata["guild_id"])
        cid = int(metadata["channel_id"])
        mid = int(metadata.get("message_id") or metadata.get("id"))
        return build_jump_url(gid, cid, mid)
    except Exception:
        return ""

def build_prompt(
    matches: List[Dict[str, Any]],
    question: str,
    as_json: bool
) -> List[Dict[str, str]]:
    """
    Construct a ChatML prompt with context from matching messages.
    """
    context_lines: List[str] = []
    for m in matches:
        author      = m.get("author", {})
        author_name = author.get("display_name") or author.get("username") or "Unknown"
        ts          = m.get("timestamp", "")
        ch_name     = m.get("channel_name") or m.get("channel_id")
        content     = m.get("content", "").replace("\n", " ")
        url         = safe_jump_url(m)
        line = f"**{author_name}** (_{ts}_ in **#{ch_name}**):\n{content}"
        if url:
            line += f"\n[🔗 View Message]({url})"
        context_lines.append(line)

    instructions = (
        "You are a knowledgeable assistant specialized in Discord server data.\n\n"
        "Use RAG to answer the user's question based on the provided context. "
        "Include author, timestamp, channel, snippets, and URLs.\n"
    )
    if as_json:
        instructions += "Return results as a JSON array with fields: guild_id, channel_id, message_id, content, timestamp, author, jump_url.\n"

    return [
        {"role": "system", "content": "You are a Discord message analyst."},
        {"role": "user",   "content": instructions + "\nContext:\n" + "\n\n".join(context_lines)
                                  + f"\n\nUser's question: {question}\n"}
    ]

def get_answer(
    query: str,
    k: int                = 5,
    as_json: bool          = False,
    return_matches: bool   = False,
    guild_id: Optional[int]     = None,
    channel_id: Optional[int]   = None,
    channel_name: Optional[str] = None
) -> Any:
    """
    Perform a RAG-based query: retrieve matches, build a prompt, and ask OpenAI.
    Returns either a string (answer) or (answer, matches) if return_matches=True.
    """
    try:
        matches = get_top_k_matches(
            query, k,
            guild_id=guild_id,
            channel_id=channel_id,
            channel_name=channel_name
        )
        if not matches and not return_matches:
            return "⚠️ I couldn't find relevant messages. Try rephrasing your question or being more specific."

        chat_messages = build_prompt(matches, query, as_json)
        response = openai_client.chat.completions.create(
            model=GPT_MODEL,
            messages=chat_messages,
            temperature=0.7,
            max_tokens=1000,
            **({"response_format": {"type": "json_object"}} if as_json else {})
        )
        answer = response.choices[0].message.content
        return (answer, matches) if return_matches else answer

    except Exception as e:
        err = f"❌ Error during RAG retrieval: {e}"
        return (err, []) if return_matches else err

def get_agent_answer(query: str, channel_id: Optional[int] = None) -> str:
    """
    Process a natural language query and return a response.
    Handles time-based, channel-specific, and content-based queries.
    """
    QUERY_COUNT.inc()
    start_time = time.time()
    
    try:
        logger.info(f"Processing query: {query}")
        
        # Validate data availability first
        data_status = validate_data_availability()
        logger.info(f"Data availability status: {data_status}")
        
        # Extract time reference if present
        time_ref = extract_time_reference(query)
        if time_ref:
            logger.info(f"Extracted time reference: {time_ref}")
            start, end = parse_timeframe(time_ref)
            start_iso = start.isoformat()
            end_iso = end.isoformat()
        else:
            # Default to last 7 days
            end = datetime.now(ZoneInfo("UTC"))
            start = end - timedelta(days=7)
            start_iso = start.isoformat()
            end_iso = end.isoformat()
            logger.info(f"Using default timeframe: {start_iso} to {end_iso}")

        # Get messages for the timeframe
        with OPENAI_API_DURATION.time():
            messages = summarize_messages(
                start_iso=start_iso,
                end_iso=end_iso,
                channel_id=channel_id,
                as_json=True
            )

        if not messages or not messages.get("summary"):
            logger.warning("No messages found in specified timeframe")
            return "⚠️ No messages found in the specified timeframe."

        # Build a more structured response
        response = {
            "timeframe": f"From {start.strftime('%Y-%m-%d %H:%M')} to {end.strftime('%Y-%m-%d %H:%M')} UTC",
            "channel": f"Channel ID: {channel_id}" if channel_id else "All channels",
            "summary": messages["summary"],
            "note": messages.get("note", "")
        }
        
        # Ensure the response is properly formatted JSON
        try:
            # First try to parse the summary as JSON if it's a string
            if isinstance(response["summary"], str):
                try:
                    summary_json = json.loads(response["summary"])
                    response["summary"] = summary_json
                except json.JSONDecodeError:
                    # If it's not valid JSON, keep it as a string
                    pass
            
            # Convert the entire response to a formatted JSON string
            formatted_response = json.dumps(response, indent=2, ensure_ascii=False)
            logger.info("Successfully processed query")
            return formatted_response
            
        except Exception as e:
            logger.error(f"Error formatting response: {e}")
            # Fallback to a simpler format if JSON formatting fails
            return f"""Timeframe: {response['timeframe']}
Channel: {response['channel']}
Summary: {response['summary']}
Note: {response['note']}"""
        
    except Exception as e:
        QUERY_ERRORS.inc()
        logger.error(f"Error processing query: {e}")
        return f"❌ Error processing query: {str(e)}"
    finally:
        duration = time.time() - start_time
        QUERY_DURATION.observe(duration)
        logger.info(f"Query completed in {duration:.2f} seconds")

# ——— Convenience aliases for the app ———
search_messages    = get_top_k_matches
discord_rag_search = get_answer

def preprocess_query(query: str) -> Dict[str, Any]:
    """Extract and validate query components."""
    components = {
        "time_reference": extract_time_reference(query),
        "channel_reference": extract_channel_reference(query),
        "content_reference": extract_content_reference(query)
    }
    
    # Validate components
    if not any(components.values()):
        raise ValueError("Query must contain at least one valid reference")
    
    return components

def log_query_performance(query: str, duration: float, success: bool):
    """Log query performance metrics."""
    with open("query_performance.log", "a") as f:
        f.write(json.dumps({
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "duration": duration,
            "success": success
        }) + "\n")
