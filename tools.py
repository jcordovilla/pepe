# tools.py

import os
import json
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from db import SessionLocal, Message
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from openai import OpenAI
from utils import (
    validate_channel_id,
    validate_guild_id,
    validate_channel_name
)
from functools import lru_cache
import time
import logging
from typing import Callable, TypeVar, cast

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Type variable for generic function typing
T = TypeVar('T')

def cache_with_timeout(timeout_seconds: int = 300) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Cache decorator with timeout.
    Args:
        timeout_seconds: Cache timeout in seconds (default: 5 minutes)
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        cache = {}
        last_updated = {}
        
        def wrapper(*args: Any, **kwargs: Any) -> T:
            # Create cache key from function name and arguments
            key = f"{func.__name__}:{str(args)}:{str(kwargs)}"
            current_time = time.time()
            
            # Check if cache is valid
            if key in cache and key in last_updated:
                if current_time - last_updated[key] < timeout_seconds:
                    logger.debug(f"Cache hit for {func.__name__}")
                    return cache[key]
            
            # Cache miss or expired, call function
            result = func(*args, **kwargs)
            cache[key] = result
            last_updated[key] = current_time
            logger.debug(f"Cache miss for {func.__name__}")
            return result
            
        return cast(Callable[..., T], wrapper)
    return decorator

# setup
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

EMBED_MODEL = "text-embedding-3-small"
GPT_MODEL   = os.getenv("GPT_MODEL", "gpt-4-turbo")
INDEX_DIR   = "index_faiss"

# init clients
_embedding = OpenAIEmbeddings(model=EMBED_MODEL, openai_api_key=OPENAI_API_KEY)
_openai    = OpenAI(api_key=OPENAI_API_KEY)

@cache_with_timeout(timeout_seconds=3600)  # Cache for 1 hour
def _load_store() -> FAISS:
    """Load the FAISS index from disk."""
    try:
        logger.info("Loading FAISS index from disk")
        return FAISS.load_local(INDEX_DIR, _embedding, allow_dangerous_deserialization=True)
    except Exception as e:
        logger.error(f"Failed to load FAISS index from {INDEX_DIR}: {e}")
        raise RuntimeError(f"Failed to load FAISS index from {INDEX_DIR}: {e}")

@cache_with_timeout(timeout_seconds=300)  # Cache for 5 minutes
def resolve_channel_name(channel_name: str, guild_id: Optional[int] = None) -> Optional[int]:
    """
    Given a human-friendly channel_name (e.g. "non-coders-learning")
    and an optional guild_id, return the corresponding channel_id
    from the database, or None if not found.
    """
    if not validate_channel_name(channel_name):
        logger.error(f"Invalid channel name format: {channel_name}")
        raise ValueError(f"Invalid channel name format: {channel_name}")
        
    if guild_id is not None and not validate_guild_id(guild_id):
        logger.error(f"Invalid guild ID format: {guild_id}")
        raise ValueError(f"Invalid guild ID format: {guild_id}")
        
    session = SessionLocal()
    try:
        logger.info(f"Resolving channel name: {channel_name}")
        query = session.query(Message.channel_id).filter(Message.channel_name == channel_name)
        if guild_id is not None:
            query = query.filter(Message.guild_id == guild_id)
        result = query.first()
        return result[0] if result else None
    except Exception as e:
        logger.error(f"Error resolving channel name: {e}")
        raise
    finally:
        session.close()

def search_messages(
    query: str,
    k: int = 5,
    keyword: Optional[str] = None,
    guild_id: Optional[int] = None,
    channel_id: Optional[int] = None,
    channel_name: Optional[str] = None,
    author_name: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Enhanced hybrid keyword + semantic search with improved relevance.
    """
    if not query.strip():
        raise ValueError("Query cannot be empty")
        
    # Validate IDs
    if guild_id is not None and not validate_guild_id(guild_id):
        raise ValueError(f"Invalid guild ID format: {guild_id}")
        
    if channel_id is not None and not validate_channel_id(channel_id):
        raise ValueError(f"Invalid channel ID format: {channel_id}")
        
    if channel_name is not None and not validate_channel_name(channel_name):
        raise ValueError(f"Invalid channel name format: {channel_name}")

    # Resolve channel_name → channel_id
    if channel_name and not channel_id:
        channel_id = resolve_channel_name(channel_name, guild_id)
        if not channel_id:
            raise ValueError(f"Unknown channel: {channel_name}")

    session = SessionLocal()
    try:
        # 1) Enhanced keyword pre-filter
        db_q = session.query(Message)
        if guild_id:
            db_q = db_q.filter(Message.guild_id == guild_id)
        if channel_id:
            db_q = db_q.filter(Message.channel_id == channel_id)
        if author_name:
            db_q = db_q.filter(Message.author["username"].as_string() == author_name)
        
        # Improved keyword matching
        if keyword:
            # Split keyword into terms for better matching
            terms = keyword.split()
            for term in terms:
                db_q = db_q.filter(Message.content.ilike(f"%{term}%"))
        
        # Get more candidates for better semantic ranking
        candidates = db_q.limit(k * 10).all()

        if not candidates:
            return []

        # 2) Enhanced semantic reranking
        texts = [m.content for m in candidates]
        metas = [{
            **m.__dict__,
            'relevance_score': 0.0  # Will be updated during ranking
        } for m in candidates]
        
        # Create temporary FAISS index
        temp_store = FAISS.from_texts(
            texts=texts,
            embedding=_embedding,
            metadatas=metas
        )
        
        # Get more documents for better coverage
        retriever = temp_store.as_retriever(search_kwargs={"k": k * 2})
        docs = retriever.get_relevant_documents(query)
        
        # Extract and sort by relevance
        results = [d.metadata for d in docs]
        results.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        
        # Return top k results
        return results[:k]
    finally:
        session.close()

def summarize_messages(
    start_iso: str,
    end_iso: str,
    guild_id: Optional[int] = None,
    channel_id: Optional[int] = None,
    channel_name: Optional[str] = None,
    as_json: bool = False
) -> Union[str, Dict[str, Any]]:
    """
    Summarize all messages in [start_iso, end_iso].
    Returns either a text summary or JSON, per `as_json`.
    """
    # Debug logging
    print(f"\nDEBUG: Query Parameters:")
    print(f"Start ISO: {start_iso}")
    print(f"End ISO: {end_iso}")
    print(f"Guild ID: {guild_id}")
    print(f"Channel ID: {channel_id}")
    print(f"Channel Name: {channel_name}")

    # Validate IDs
    if guild_id is not None and not validate_guild_id(guild_id):
        raise ValueError(f"Invalid guild ID format: {guild_id}")
        
    if channel_id is not None and not validate_channel_id(channel_id):
        raise ValueError(f"Invalid channel ID format: {channel_id}")
        
    if channel_name is not None and not validate_channel_name(channel_name):
        raise ValueError(f"Invalid channel name format: {channel_name}")

    # validate time inputs
    try:
        start = datetime.fromisoformat(start_iso)
        end = datetime.fromisoformat(end_iso)
        if end < start:
            raise ValueError("End time must be after start time")
        
        # Debug logging
        print(f"\nDEBUG: Parsed Timestamps:")
        print(f"Start: {start}")
        print(f"End: {end}")
    except Exception as e:
        raise ValueError(f"Invalid ISO datetime: {e}")

    # resolve channel_name → channel_id
    if channel_name and not channel_id:
        channel_id = resolve_channel_name(channel_name, guild_id)
        if not channel_id:
            raise ValueError(f"Unknown channel: {channel_name}")

    # load messages from DB
    session = SessionLocal()
    try:
        q = session.query(Message)
        q = q.filter(Message.timestamp.between(start, end))
        if guild_id:
            q = q.filter(Message.guild_id == guild_id)
        if channel_id:
            q = q.filter(Message.channel_id == channel_id)
        
        # Debug logging
        print("\nDEBUG: SQL Query:")
        print(str(q.statement.compile(compile_kwargs={"literal_binds": True})))
        
        msgs = q.order_by(Message.timestamp).all()
        
        # Debug logging
        print(f"\nDEBUG: Query Results:")
        print(f"Number of messages found: {len(msgs)}")
        if msgs:
            print(f"First message timestamp: {msgs[0].timestamp}")
            print(f"Last message timestamp: {msgs[-1].timestamp}")

        if not msgs:
            return {"summary": "", "note": "No messages in that timeframe"} if as_json else "⚠️ No messages found in the specified timeframe."

        # build prompt with clear time context
        context = "\n\n".join(
            f"**{m.author['username']}** (_{m.timestamp}_): {m.content}"
            for m in msgs
        )
        instructions = (
            f"Summarize the following Discord messages from {start.date()} to {end.date()}:\n\n"
            "Include author, timestamp, and key points. Return JSON if requested."
        )
        prompt = f"{instructions}\n\n{context}\n\nSummary:"

        # chat completion
        resp = _openai.chat.completions.create(
            model=GPT_MODEL,
            messages=[
                {"role": "system", "content": instructions},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=800,
            **({"response_format": {"type": "json_object"}} if as_json else {})
        )
        result = resp.choices[0].message.content
        return json.loads(result) if as_json else result

    finally:
        session.close()

def get_channels() -> List[Dict[str, Any]]:
    """
    Get a list of all available channels from the database.
    Returns a list of dictionaries containing channel information.
    """
    session = SessionLocal()
    try:
        # Query distinct channels
        channels = session.query(
            Message.channel_id,
            Message.channel_name
        ).distinct().all()
        
        # Format results
        return [
            {
                "id": channel_id,
                "name": channel_name
            }
            for channel_id, channel_name in channels
            if channel_id and channel_name  # Filter out None values
        ]
    finally:
        session.close()
