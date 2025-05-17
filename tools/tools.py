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
    Hybrid keyword + semantic search. Always returns a list of dicts with required metadata: author, timestamp, content, jump_url, channel_name, guild_id, channel_id, message_id.
    Handles unknown channels gracefully. No summaries or paraphrasing.
    """
    # Default query to empty string if not provided
    if query is None:
        query = ""
    if not isinstance(query, str):
        raise ValueError("Query must be a string.")

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
            return []  # Unknown channel, return empty result

    session = SessionLocal()
    try:
        db_q = session.query(Message)
        if guild_id:
            db_q = db_q.filter(Message.guild_id == guild_id)
        if channel_id:
            db_q = db_q.filter(Message.channel_id == channel_id)
        if author_name:
            db_q = db_q.filter(Message.author_name == author_name)

        # Fetch more candidates for post-filtering
        candidates = db_q.order_by(Message.timestamp.desc()).limit(k * 20).all()
        if not candidates:
            return []

        # Author filter: handle author_name as username in author or author_name fields
        if author_name:
            filtered_candidates = []
            for m in candidates:
                # Try author_name (string or dict)
                author_name_val = getattr(m, "author_name", None)
                match = False
                if isinstance(author_name_val, dict):
                    if author_name_val.get("username") == author_name:
                        match = True
                elif isinstance(author_name_val, str):
                    if author_name_val == author_name:
                        match = True
                # Try author (dict)
                if not match:
                    author_val = getattr(m, "author", None)
                    if isinstance(author_val, dict):
                        if author_val.get("username") == author_name:
                            match = True
                if match:
                    filtered_candidates.append(m)
            candidates = filtered_candidates

        def is_real_message(msg_content: str, keyword: Optional[str] = None) -> bool:
            # Ignore section headers, lines with only dashes, or lines that look like bot headers
            if not msg_content:
                return False
            lines = [l.strip() for l in msg_content.split('\n') if l.strip()]
            # If all lines are headers or separators, skip
            if all(l.startswith('---') or l.endswith('---') or re.match(r'^[\W_]+$', l) for l in lines):
                return False
            # If keyword is present, ensure it's in a non-header line (not just in a heading)
            if keyword:
                for l in lines:
                    # Only count as a match if the keyword is in a line that is not a heading or separator
                    if keyword.lower() in l.lower() and not (l.startswith('---') or l.endswith('---') or re.match(r'^[\W_]+$', l) or re.match(r'^[\W_ ]*welcome[\W_ ]*$', l, re.IGNORECASE)):
                        return True
                return False
            return True

        filtered = []
        # Hybrid search: both query and keyword must be present if both are given
        if keyword and query:
            for m in candidates:
                if m.content and keyword.lower() in m.content.lower() and query.lower() in m.content.lower():
                    if is_real_message(m.content, keyword):
                        filtered.append(m)
            # If no matches for both, try to return matches for at least one term (with a note)
            if not filtered:
                for m in candidates:
                    if m.content and (keyword.lower() in m.content.lower() or query.lower() in m.content.lower()):
                        if is_real_message(m.content, keyword):
                            filtered.append(m)
        elif keyword:
            for m in candidates:
                if m.content and keyword.lower() in m.content.lower():
                    if is_real_message(m.content, keyword):
                        filtered.append(m)
        elif query:
            for m in candidates:
                if m.content and query.lower() in m.content.lower():
                    if is_real_message(m.content, query):
                        filtered.append(m)
        else:
            for m in candidates:
                if is_real_message(m.content):
                    filtered = candidates
                    break

        if not filtered:
            return []

        # Sort by timestamp descending and take top k
        filtered = sorted(filtered, key=lambda m: m.timestamp, reverse=True)[:k]

        formatted = []
        for m in filtered:
            # Robust author extraction
            username = None
            display_name = None
            # Try author_name (string or dict)
            author_name_val = getattr(m, "author_name", None)
            if isinstance(author_name_val, dict):
                username = author_name_val.get("username")
                display_name = author_name_val.get("display_name")
            elif isinstance(author_name_val, str):
                username = author_name_val
            # Try author (dict)
            if not username or not display_name:
                author_val = getattr(m, "author", None)
                if isinstance(author_val, dict):
                    if not username:
                        username = author_val.get("username")
                    if not display_name:
                        display_name = author_val.get("display_name")
            # Fallbacks
            if not username:
                username = "Unknown"
            if not display_name:
                display_name = username
            author = {
                "username": username,
                "display_name": display_name
            }
            jump_url = getattr(m, "jump_url", None)
            if not jump_url:
                try:
                    jump_url = build_jump_url(m.guild_id, m.channel_id, m.message_id)
                except Exception:
                    jump_url = ""
            formatted.append({
                "author": author,
                "timestamp": m.timestamp.isoformat() if hasattr(m.timestamp, 'isoformat') else str(m.timestamp),
                "content": m.content,
                "jump_url": jump_url,
                "channel_name": getattr(m, "channel_name", None),
                "guild_id": getattr(m, "guild_id", None),
                "channel_id": getattr(m, "channel_id", None),
                "message_id": getattr(m, "message_id", None) or getattr(m, "id", None)
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
        if isinstance(start_iso, str) and start_iso.endswith('Z'):
            start_iso = start_iso.replace('Z', '+00:00')
        if isinstance(end_iso, str) and end_iso.endswith('Z'):
            end_iso = end_iso.replace('Z', '+00:00')
        start = datetime.fromisoformat(start_iso)
        end = datetime.fromisoformat(end_iso)
        if end < start:
            if as_json:
                return {"error": "End time must be after start time"}
            else:
                return "End time must be after start time"
    except Exception as e:
        if as_json:
            return {"error": f"Invalid ISO datetime: {e}"}
        else:
            return f"Invalid ISO datetime: {e}"
    if channel_name and not channel_id:
        channel_id = resolve_channel_name(channel_name, guild_id)
        if not channel_id:
            if as_json:
                return {"error": "Unknown channel"}
            else:
                return "Unknown channel"
    session = SessionLocal()
    try:
        q = session.query(Message)
        q = q.filter(Message.timestamp.between(start, end))
        if guild_id:
            q = q.filter(Message.guild_id == guild_id)
        if channel_id:
            q = q.filter(Message.channel_id == channel_id)
        msgs = q.order_by(Message.timestamp).all()
        if not msgs:
            if as_json:
                return {"summary": "", "note": "No messages in that timeframe"}
            else:
                return "⚠️ No messages found in the specified timeframe."
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
                "summary": "",  # Placeholder, actual summary logic can be added
                "note": "",     # Placeholder, actual note logic can be added
                "messages": message_dicts
            }
        else:
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
