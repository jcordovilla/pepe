# rag_engine.py

import os
from typing import Any, Dict, List, Optional, Union
import numpy as np
from core.ai_client import get_ai_client
from core.config import get_config
from utils.helpers import build_jump_url
from tools.tools import resolve_channel_name, summarize_messages, validate_data_availability
from tools.time_parser import parse_timeframe, extract_time_reference, extract_channel_reference, extract_content_reference
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
AI_API_DURATION = Histogram('ai_api_duration_seconds', 'Time spent in AI API calls')

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
config = get_config()
INDEX_DIR = config.faiss_index_path
ai_client = get_ai_client()

class LocalVectorStore:
    """Simple vector store using FAISS with local embeddings."""
    
    def __init__(self, index_path: str):
        self.index_path = index_path
        self._index = None
        self._metadata = None
    
    def load(self):
        """Load FAISS index and metadata from disk."""
        import faiss
        import pickle
        
        try:
            logger.info("Loading vector store from disk")
            # Load FAISS index
            self._index = faiss.read_index(f"{self.index_path}/index.faiss")
            # Load metadata
            with open(f"{self.index_path}/index.pkl", "rb") as f:
                self._metadata = pickle.load(f)
            logger.info(f"Loaded vector store with {self._index.ntotal} vectors")
        except Exception as e:
            logger.error(f"Failed to load vector store: {e}")
            raise
    
    def search(self, query_text: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar messages."""
        if self._index is None:
            self.load()
        
        # Create query embedding
        query_embedding = ai_client.create_embeddings(query_text)
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Search FAISS index
        scores, indices = self._index.search(query_embedding.astype('float32'), k * 10)  # Get more for filtering
        
        # Return metadata for found documents
        results = []
        for i, idx in enumerate(indices[0]):
            if idx >= 0 and idx < len(self._metadata):
                meta = self._metadata[idx].copy()
                meta['score'] = float(scores[0][i])
                results.append(meta)
        
        return results

# Global vector store
_vector_store = None

def load_vectorstore() -> LocalVectorStore:
    """Load the locally saved FAISS index from disk."""
    global _vector_store
    if _vector_store is None:
        _vector_store = LocalVectorStore(INDEX_DIR)
        _vector_store.load()
    return _vector_store

def vector_search(metas: List[dict], k: int) -> List[dict]:
    """
    Always return the top-k regardless of score (no threshold filtering).
    """
    return metas[:k]

def get_top_k_matches(
    query: str,
    k: int = 5,
    guild_id: Optional[int]      = None,
    channel_id: Optional[int]    = None,
    channel_name: Optional[str]  = None
) -> List[Dict[str, Any]]:
    """
    Retrieve the top-k matches for 'query',
    optionally scoped to a guild, channel_id, or channel_name
    via manual post-filtering.
    """
    with VECTOR_SEARCH_DURATION.time():
        try:
            logger.info(f"Searching for matches: query='{query}', k={k}")
            store = load_vectorstore()
            
            # Get search results from local vector store
            results = store.search(query, k * 10)  # Get extra for filtering

            # 2) Resolve a human channel_name to ID if needed
            if channel_name and not channel_id:
                resolved = resolve_channel_name(channel_name, guild_id)
                if resolved is None:
                    logger.warning(f"Unknown channel name: {channel_name}")
                    raise ValueError(f"Unknown channel name: {channel_name}")
                channel_id = resolved

            # 3) Manual metadata filters
            if guild_id is not None:
                results = [r for r in results if r.get("guild_id") == str(guild_id)]
            if channel_id is not None:
                results = [r for r in results if r.get("channel_id") == str(channel_id)]

            # 4) Return top-k results
            final_results = vector_search(results, k)
            logger.info(f"Found {len(final_results)} matches")
            return final_results
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
    Perform a RAG-based query: retrieve matches, build a prompt, and ask local AI.
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
        
        with AI_API_DURATION.time():
            if as_json:
                # For JSON responses, add specific instruction
                chat_messages[-1]["content"] += "\n\nIMPORTANT: Return only valid JSON, no other text."
            
            answer = ai_client.chat_completion(
                chat_messages,
                temperature=0.7,
                max_tokens=1000
            )
        
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
        with AI_API_DURATION.time():
            messages = summarize_messages(
                start_iso=start_iso,
                end_iso=end_iso,
                channel_id=channel_id,
                as_json=True
            )

        if isinstance(messages, str):
            # Handle error messages
            return messages

        if not messages or not messages.get("messages"):
            return f"⚠️ No messages found in the specified timeframe ({start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')})"

        # Build a more structured response
        response = {
            "timeframe": f"From {start.strftime('%Y-%m-%d %H:%M')} to {end.strftime('%Y-%m-%d %H:%M')} UTC",
            "channel": f"Channel: {messages.get('channel_name', 'All channels')}",
            "summary": messages.get("summary", ""),
            "note": messages.get("note", ""),
            "messages": []
        }

        # Process messages and ensure jump URLs are included
        for msg in messages.get("messages", []):
            processed_msg = {
                "author": msg.get("author", {}),
                "timestamp": msg.get("timestamp", ""),
                "content": msg.get("content", ""),
                "channel_name": msg.get("channel_name", ""),
                "jump_url": msg.get("jump_url", "")
            }
            
            # If jump_url is missing, try to build it
            if not processed_msg["jump_url"]:
                try:
                    guild_id = msg.get("guild_id")
                    channel_id = msg.get("channel_id")
                    message_id = msg.get("message_id")
                    if all([guild_id, channel_id, message_id]):
                        processed_msg["jump_url"] = build_jump_url(guild_id, channel_id, message_id)
                except Exception as e:
                    logger.warning(f"Failed to build jump URL: {e}")
            
            response["messages"].append(processed_msg)

        return response

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
