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
from sqlalchemy import func
import re

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Type variable for generic function typing
T = TypeVar('T')

# Helper: cache_with_timeout

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
            key = f"{func.__name__}:{str(args)}:{str(kwargs)}"
            current_time = time.time()
            if key in cache and key in last_updated:
                if current_time - last_updated[key] < timeout_seconds:
                    logger.debug(f"Cache hit for {func.__name__}")
                    return cache[key]
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

# @cache_with_timeout(timeout_seconds=3600)  # Cache for 1 hour
# def _load_store() -> FAISS:
#     """Load the FAISS index from disk."""
#     try:
#         logger.info("Loading FAISS index from disk")
#         return FAISS.load_local(INDEX_DIR, _embedding, allow_dangerous_deserialization=True)
#     except Exception as e:
#         logger.error(f"Failed to load FAISS index from {INDEX_DIR}: {e}")
#         raise RuntimeError(f"Failed to load FAISS index from {INDEX_DIR}: {e}")

def search_messages(
    query: Optional[str] = None,
    k: int = 5,
    keyword: Optional[str] = None,
    guild_id: Optional[int] = None,
    channel_id: Optional[int] = None,
    channel_name: Optional[str] = None,
    author_name: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Enhanced hybrid keyword + semantic search with improved relevance.
    Always returns a list of dicts with required metadata: author, timestamp, content, jump_url, channel_name, guild_id, channel_id, message_id.
    Handles unknown channels gracefully.
    """
    # Default query to empty string if not provided
    if query is None:
        query = ""
    if not isinstance(query, str):
        raise ValueError("Query must be a string.")
    # Allow empty query for author/channel-only searches

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
            # Instead of raising, return a clear result
            return [{
                "note": f"Channel '{channel_name}' not found.",
                "messages": []
            }]

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
        
        # Improved keyword matching for skill-based queries
        if keyword:
            terms = keyword.split()
            for term in terms:
                db_q = db_q.filter(Message.content.ilike(f"%{term}%"))
        else:
            skill_terms = extract_skill_terms(query)
            if skill_terms:
                for term in skill_terms:
                    db_q = db_q.filter(Message.content.ilike(f"%{term}%"))
        
        candidates = db_q.limit(k * 10).all()
        if not candidates:
            return []
        
        # 2) Enhanced semantic reranking
        texts = [m.content for m in candidates]
        metas = [{
            **m.__dict__,
            'relevance_score': 0.0
        } for m in candidates]
        temp_store = FAISS.from_texts(
            texts=texts,
            embedding=_embedding,
            metadatas=metas
        )
        retriever = temp_store.as_retriever(search_kwargs={"k": k * 2})
        docs = retriever.get_relevant_documents(query)
        results = [d.metadata for d in docs]
        results.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        # Always return required metadata for each result
        formatted = []
        for m in results[:k]:
            formatted.append({
                "author": m.get("author", {}),
                "timestamp": m.get("timestamp"),
                "content": m.get("content"),
                "jump_url": build_jump_url(m.get("guild_id"), m.get("channel_id"), m.get("message_id")),
                "channel_name": m.get("channel_name"),
                "guild_id": m.get("guild_id"),
                "channel_id": m.get("channel_id"),
                "message_id": m.get("message_id")
            })
        return formatted
    finally:
        session.close()

# Helper: resolve_channel_name
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

# def extract_skill_terms(query: str) -> List[str]:
#     """
#     Extract skill-related terms from a query string.
#     """
#     # Common skill-related patterns
#     patterns = [
#         r'experience in (\w+)',
#         r'expertise in (\w+)',
#         r'knowledge of (\w+)',
#         r'background in (\w+)',
#         r'proficient in (\w+)',
#         r'skilled in (\w+)',
#         r'with (\w+) experience',
#         r'with (\w+) expertise',
#         r'with (\w+) knowledge',
#         r'with (\w+) background',
#         r'with (\w+) proficiency',
#         r'with (\w+) skills'
#     ]
    
#     terms = []
#     for pattern in patterns:
#         matches = re.finditer(pattern, query.lower())
#         for match in matches:
#             term = match.group(1)
#             if term not in terms:
#                 terms.append(term)
    
#     return terms

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
        # Handle 'Z' suffix for UTC
        if isinstance(start_iso, str) and start_iso.endswith('Z'):
            start_iso = start_iso.replace('Z', '+00:00')
        if isinstance(end_iso, str) and end_iso.endswith('Z'):
            end_iso = end_iso.replace('Z', '+00:00')
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

        # Convert messages to dictionaries with all necessary fields
        message_dicts = []
        for m in msgs:
            msg_dict = {
                "author": {
                    "username": m.author.get("username", "Unknown"),
                    "display_name": m.author.get("display_name", ""),
                    "id": m.author.get("id")
                },
                "timestamp": m.timestamp.isoformat(),
                "content": m.content,
                "jump_url": build_jump_url(m.guild_id, m.channel_id, m.message_id),
                "channel_name": m.channel_name,
                "guild_id": m.guild_id,
                "channel_id": m.channel_id,
                "message_id": m.message_id
            }
            message_dicts.append(msg_dict)

        if as_json:
            return {
                "timeframe": f"From {start.date()} to {end.date()} UTC",
                "channel": channel_name or "All channels",
                "messages": message_dicts
            }
        else:
            # Return the list of message dictionaries for formatting
            return message_dicts

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

def validate_data_availability() -> Dict[str, Any]:
    """
    Check if the database has messages and return their distribution.
    Returns a dictionary with status and message counts.
    """
    session = SessionLocal()
    try:
        # Get total message count
        total_messages = session.query(Message).count()
        
        if total_messages == 0:
            return {
                "status": "error",
                "message": "No messages found in the database",
                "count": 0
            }
        
        # Get message count by channel
        channel_counts = session.query(
            Message.channel_name,
            func.count(Message.id)
        ).group_by(Message.channel_name).all()
        
        # Get date range
        oldest = session.query(Message.timestamp).order_by(Message.timestamp.asc()).first()
        newest = session.query(Message.timestamp).order_by(Message.timestamp.desc()).first()
        
        return {
            "status": "ok",
            "message": "Data available",
            "count": total_messages,
            "channels": dict(channel_counts),
            "date_range": {
                "oldest": oldest[0].isoformat() if oldest else None,
                "newest": newest[0].isoformat() if newest else None
            }
        }
    except Exception as e:
        logger.error(f"Error validating data availability: {e}")
        return {
            "status": "error",
            "message": f"Error checking data: {str(e)}",
            "count": 0
        }
    finally:
        session.close()

def extract_skill_terms(query: str) -> List[str]:
    """
    Extract skill-related terms from a query string.
    """
    patterns = [
        r'experience in (\w+)',
        r'expertise in (\w+)',
        r'knowledge of (\w+)',
        r'background in (\w+)',
        r'proficient in (\w+)',
        r'skilled in (\w+)',
        r'with (\w+) experience',
        r'with (\w+) expertise',
        r'with (\w+) knowledge',
        r'with (\w+) background',
        r'with (\w+) proficiency',
        r'with (\w+) skills'
    ]
    terms = []
    for pattern in patterns:
        matches = re.finditer(pattern, query.lower())
        for match in matches:
            term = match.group(1)
            if term not in terms:
                terms.append(term)
    return terms

# Helper: build_jump_url

def build_jump_url(guild_id: int, channel_id: int, message_id: int) -> str:
    """
    Build a Discord message jump URL from guild, channel, and message IDs.
    Args:
        guild_id: The Discord guild (server) ID
        channel_id: The Discord channel ID
        message_id: The Discord message ID
    Returns:
        A Discord message jump URL in the format: https://discord.com/channels/{guild_id}/{channel_id}/{message_id}
    """
    return f"https://discord.com/channels/{guild_id}/{channel_id}/{message_id}"
