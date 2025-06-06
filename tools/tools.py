# tools.py

import os
import json
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from db import SessionLocal, Message
from core.ai_client import get_ai_client
from core.config import get_config
from utils import (
    validate_channel_id,
    validate_guild_id,
    validate_channel_name
)
from utils.helpers import build_jump_url
from functools import lru_cache
import time
import logging
from typing import Callable, TypeVar, cast
from sqlalchemy import func
import re
import numpy as np
import faiss

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

# Load configuration and AI client
config = get_config()
ai_client = get_ai_client()

INDEX_DIR = config.faiss_index_path

# Initialize FAISS store lazily
_faiss_store = None

def _get_faiss_store():
    """Get or initialize FAISS store."""
    global _faiss_store
    if _faiss_store is None:
        try:
            logger.info(f"Loading FAISS index from {INDEX_DIR}")
            # Load FAISS index
            index = faiss.read_index(f"{INDEX_DIR}/index.faiss")
            # Load metadata
            with open(f"{INDEX_DIR}/index.pkl", "rb") as f:
                import pickle
                metadata = pickle.load(f)
            _faiss_store = {"index": index, "metadata": metadata}
            logger.info(f"Loaded FAISS index with {index.ntotal} vectors")
        except Exception as e:
            logger.error(f"Failed to load FAISS index: {e}")
            # Create empty index if not found
            dimension = config.models.embedding_dimension
            index = faiss.IndexFlatL2(dimension)
            _faiss_store = {"index": index, "metadata": []}
            logger.warning("Created empty FAISS index")
    return _faiss_store

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
    Hybrid search combining FAISS vector similarity with keyword filtering.
    Uses local embeddings for semantic search.
    """
    logger.info(f"Searching messages with params: query={query}, k={k}, channel_id={channel_id}, channel_name={channel_name}")
    
    # Get messages from database with filters
    session = SessionLocal()
    try:
        db_q = session.query(Message)
        if guild_id:
            db_q = db_q.filter(Message.guild_id == guild_id)
        if channel_id:
            db_q = db_q.filter(Message.channel_id == channel_id)
        elif channel_name:
            # Try to find channel by name
            channel_matches = session.query(Message.channel_id)\
                .filter(Message.channel_name.ilike(f"%{channel_name}%"))\
                .distinct()\
                .all()
            if channel_matches:
                channel_ids = [match[0] for match in channel_matches]
                logger.info(f"Found channel IDs for name '{channel_name}': {channel_ids}")
                db_q = db_q.filter(Message.channel_id.in_(channel_ids))
            else:
                logger.warning(f"No channels found matching name: {channel_name}")

        # Fetch candidates
        candidates = db_q.order_by(Message.timestamp.desc()).limit(k * 20).all()
        logger.info(f"Found {len(candidates)} candidate messages")
        
        if not candidates:
            return []

        # If we have a query, use semantic search
        if query:
            try:
                # Use FAISS store for semantic search
                store = _get_faiss_store()
                query_embedding = ai_client.create_embeddings(query)
                
                if store["index"].ntotal > 0:
                    # Search in FAISS index
                    distances, indices = store["index"].search(
                        query_embedding.reshape(1, -1), min(k * 2, store["index"].ntotal)
                    )
                    
                    # Map back to messages
                    semantic_results = []
                    for idx, distance in zip(indices[0], distances[0]):
                        if idx < len(store["metadata"]):
                            meta = store["metadata"][idx]
                            # Find corresponding message in candidates
                            for msg in candidates:
                                if (hasattr(meta, 'get') and meta.get('message_id') == msg.message_id) or \
                                   (isinstance(meta, dict) and meta.get('message_id') == msg.message_id):
                                    semantic_results.append(msg)
                                    break
                    
                    # Use semantic results if found, otherwise fall back to candidates
                    messages = semantic_results[:k] if semantic_results else candidates[:k]
                else:
                    # No index available, use database results
                    messages = candidates[:k]
            except Exception as e:
                logger.error(f"Semantic search failed: {e}")
                messages = candidates[:k]
        else:
            # No query, just return recent messages
            messages = candidates[:k]

        # Post-process results
        results = []
        for msg in messages:
            # Apply keyword filter if specified
            if keyword and keyword.lower() not in msg.content.lower():
                continue
                
            # Apply author filter if specified
            if author_name:
                author = msg.author
                if not (author_name.lower() in author.get("username", "").lower() or 
                       author_name.lower() in author.get("display_name", "").lower()):
                    continue
                    continue
                    
            # Format result
            results.append({
                "author": msg.author,
                "timestamp": msg.timestamp.isoformat() if msg.timestamp else None,
                "content": msg.content,
                "jump_url": build_jump_url(
                    msg.guild_id,
                    msg.channel_id,
                    getattr(msg, 'message_id', getattr(msg, 'id', None))
                ),
                "channel_name": msg.channel_name,
                "guild_id": msg.guild_id,
                "channel_id": msg.channel_id,
                "message_id": getattr(msg, 'message_id', getattr(msg, 'id', None))
            })
            
            if len(results) >= k:
                break
                
        logger.info(f"Returning {len(results)} results")
        return results
    except Exception as e:
        logger.error(f"Error in search_messages: {str(e)}", exc_info=True)
        return []
    finally:
        session.close()

# Helper: resolve_channel_name
@cache_with_timeout(timeout_seconds=300)  # Cache for 5 minutes

def resolve_channel_name(*args, **kwargs) -> Optional[int]:
    """
    Given a human-friendly channel_name (e.g. "non-coders-learning")
    and an optional guild_id, return the corresponding channel_id
    from the database, or None if not found.
    Accepts *args and **kwargs for compatibility with agent tool-calling.
    Also handles Discord's channel mention format (<#channel_id>).
    """
    # Extract channel_name and guild_id from args or kwargs
    channel_name = None
    guild_id = None
    # Try kwargs first
    if 'channel_name' in kwargs:
        channel_name = kwargs['channel_name']
    if 'guild_id' in kwargs:
        guild_id = kwargs['guild_id']
    # Fallback to positional args
    if channel_name is None and args:
        channel_name = args[0]
    if guild_id is None and len(args) > 1:
        guild_id = args[1]

    # Handle Discord channel mention format (<#channel_id>)
    if channel_name and channel_name.startswith('<#') and channel_name.endswith('>'):
        try:
            return int(channel_name[2:-1])
        except ValueError:
            return None

    if not channel_name or not validate_channel_name(channel_name):
        return None
    if guild_id is not None and not validate_guild_id(guild_id):
        return None
    session = SessionLocal()
    try:
        q = session.query(Message.channel_id).filter(Message.channel_name == channel_name)
        if guild_id:
            q = q.filter(Message.guild_id == guild_id)
        result = q.first()
        if result:
            return result[0]
        else:
            return None
    except Exception as e:
        return None
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
    Generate a summary of messages using local AI models.
    
    Args:
        start_iso (str): Start timestamp in ISO format
        end_iso (str): End timestamp in ISO format
        guild_id (int): Optional guild ID filter
        channel_id (int): Optional channel ID filter
        channel_name (str): Optional channel name filter
        as_json (bool): Return JSON format if True
        
    Returns:
        Summary text or JSON with messages and metadata
    """
    # Get messages from database
    session = SessionLocal()
    try:
        q = session.query(Message)
        q = q.filter(Message.timestamp.between(start_iso, end_iso))
        if guild_id:
            q = q.filter(Message.guild_id == guild_id)
        if channel_id:
            q = q.filter(Message.channel_id == channel_id)
            
        messages = q.order_by(Message.timestamp).all()
        
        if not messages:
            return {
                "summary": "",
                "note": "No messages found in timeframe",
                "messages": []
            } if as_json else "No messages found in timeframe"
        
        # Prepare messages for summarization
        message_texts = []
        for msg in messages:
            author_name = msg.author.get('username', 'Unknown')
            timestamp = msg.timestamp.strftime('%Y-%m-%d %H:%M')
            message_texts.append(f"[{timestamp}] {author_name}: {msg.content}")
        
        # Create combined text for summarization
        combined_text = "\n".join(message_texts)
        
        # Generate summary using local AI
        prompt = f"""Analyze and summarize the following Discord messages. 
Focus on key topics, discussions, and insights. Be concise but informative:

{combined_text}

Summary:"""
        
        summary = ai_client.chat_completion([
            {"role": "system", "content": "You are a helpful assistant that summarizes Discord conversations concisely."},
            {"role": "user", "content": prompt}
        ])
        
        # Format output
        if as_json:
            return {
                "summary": summary,
                "timeframe": f"{start_iso} to {end_iso}",
                "message_count": len(messages),
                "messages": [
                    {
                        "author": msg.author.get('username', 'Unknown'),
                        "content": msg.content,
                        "timestamp": msg.timestamp.isoformat(),
                        "channel_name": msg.channel_name,
                        "jump_url": build_jump_url(msg.guild_id, msg.channel_id, msg.message_id)
                    }
                    for msg in messages[:10]  # Limit to first 10 messages
                ]
            }
        else:
            return f"**Summary ({len(messages)} messages from {start_iso} to {end_iso}):**\n{summary}"
        
        # Format output
        if as_json:
            return {
                "summary": result["result"],
                "note": f"Summary of {len(messages)} messages",
                "messages": [
                    {
                        "author": msg.author,
                        "timestamp": msg.timestamp.isoformat(),
                        "content": msg.content,
                        "jump_url": build_jump_url(
                            msg.guild_id,
                            msg.channel_id,
                            getattr(msg, 'message_id', getattr(msg, 'id', None))
                        ),
                        "channel_name": msg.channel_name,
                        "guild_id": msg.guild_id,
                        "channel_id": msg.channel_id,
                        "message_id": getattr(msg, 'message_id', getattr(msg, 'id', None))
                    }
                    for msg in messages
                ]
            }
        else:
            return result["result"]
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
