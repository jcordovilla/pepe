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
import time
import logging
from sqlalchemy import func
import re
import numpy as np
import faiss

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load configuration and AI client
config = get_config()
ai_client = get_ai_client()

INDEX_DIR = config.faiss_index_path

# Initialize FAISS store lazily
_faiss_store = None
_enhanced_faiss_store = None

def _get_enhanced_faiss_store():
    """Get or initialize enhanced FAISS store."""
    global _enhanced_faiss_store
    if _enhanced_faiss_store is None:
        try:
            # Find the most recent enhanced index
            import glob
            enhanced_indices = glob.glob("/Users/jose/Documents/apps/discord-bot/enhanced_faiss_*.index")
            if enhanced_indices:
                # Get the most recent one
                latest_index = max(enhanced_indices, key=lambda x: os.path.getctime(x))
                base_name = latest_index.replace('.index', '')
                metadata_file = f"{base_name}_metadata.json"
                
                logger.info(f"Loading enhanced FAISS index from {latest_index}")
                # Load FAISS index
                index = faiss.read_index(latest_index)
                # Load metadata
                with open(metadata_file, "r") as f:
                    import json
                    data = json.load(f)
                    metadata = data["metadata"]
                
                _enhanced_faiss_store = {"index": index, "metadata": metadata}
                logger.info(f"Loaded enhanced FAISS index with {index.ntotal} vectors")
            else:
                logger.warning("No enhanced FAISS index found")
                _enhanced_faiss_store = None
        except Exception as e:
            logger.error(f"Failed to load enhanced FAISS index: {e}")
            _enhanced_faiss_store = None
    return _enhanced_faiss_store

def _create_embedding_with_correct_model(text: str):
    """Create embedding using the correct model that matches the enhanced index."""
    try:
        from sentence_transformers import SentenceTransformer
        # Use the same model that was used to build the enhanced index
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embedding = model.encode([text])
        return embedding
    except Exception as e:
        logger.error(f"Failed to create embedding with correct model: {e}")
        # Fall back to AI client
        return ai_client.create_embeddings(text)

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
                # Try enhanced FAISS store first
                enhanced_store = _get_enhanced_faiss_store()
                
                semantic_results = []
                
                if enhanced_store and enhanced_store["index"].ntotal > 0:
                    logger.info("Using enhanced FAISS index for semantic search")
                    # Use correct embedding model for enhanced index
                    try:
                        if SentenceTransformer is not None:
                            embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
                            query_embedding = embedding_model.encode([query])
                        else:
                            query_embedding = ai_client.create_embeddings(query)
                            query_embedding = query_embedding.reshape(1, -1)
                    except Exception as e:
                        logger.error(f"Failed to load correct embedding model: {e}")
                        query_embedding = ai_client.create_embeddings(query)
                        query_embedding = query_embedding.reshape(1, -1)
                    
                    # Search in enhanced FAISS index
                    distances, indices = enhanced_store["index"].search(
                        query_embedding, min(k * 2, enhanced_store["index"].ntotal)
                    )
                    
                    logger.info(f"Enhanced search returned indices: {indices[0][:5]}")
                    
                    # Map back to messages using enhanced metadata
                    for idx, distance in zip(indices[0], distances[0]):
                        if 0 <= idx < len(enhanced_store["metadata"]):
                            meta = enhanced_store["metadata"][idx]
                            # Find corresponding message in candidates
                            for msg in candidates:
                                if meta.get('message_id') == msg.message_id:
                                    semantic_results.append(msg)
                                    logger.info(f"Found match for message_id: {msg.message_id}")
                                    break
                        else:
                            logger.warning(f"Invalid index {idx} (metadata size: {len(enhanced_store['metadata'])})")
                    
                    logger.info(f"Enhanced search found {len(semantic_results)} semantic results")
                
                # If no enhanced results, fall back to regular FAISS
                if not semantic_results:
                    logger.info("Falling back to regular FAISS index")
                    store = _get_faiss_store()
                    
                    if store["index"].ntotal > 0:
                        # Search in regular FAISS index
                        distances, indices = store["index"].search(
                            query_embedding.reshape(1, -1), min(k * 2, store["index"].ntotal)
                        )
                        
                        # Map back to messages
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

def test_enhanced_faiss_search(
    query: str,
    k: int = 5
) -> Dict[str, Any]:
    """
    Test function specifically for enhanced FAISS index.
    Returns detailed information about the search process and results.
    """
    logger.info(f"Testing enhanced FAISS search with query: {query}")
    
    try:
        # Load enhanced FAISS store
        enhanced_store = _get_enhanced_faiss_store()
        if not enhanced_store:
            return {
                "status": "error",
                "message": "Enhanced FAISS index not available",
                "results": []
            }
        
        # Create query embedding
        if SentenceTransformer is not None:
            embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            query_embedding = embedding_model.encode([query])
        else:
            query_embedding = ai_client.create_embeddings(query)
            query_embedding = query_embedding.reshape(1, -1)
        
        logger.info(f"Query embedding shape: {query_embedding.shape}")
        
        # Search in enhanced index
        distances, indices = enhanced_store["index"].search(
            query_embedding.reshape(1, -1), min(k, enhanced_store["index"].ntotal)
        )
        
        logger.info(f"Search returned {len(indices[0])} results")
        logger.info(f"Distances: {distances[0]}")
        logger.info(f"Indices: {indices[0]}")
        
        # Process results
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(enhanced_store["metadata"]):
                meta = enhanced_store["metadata"][idx]
                result = {
                    "message_id": meta.get('message_id'),
                    "author_name": meta.get('author_name'),
                    "channel_name": meta.get('channel_name'),
                    "content": meta.get('processed_content', meta.get('original_content', '')),
                    "timestamp": meta.get('timestamp'),
                    "distance": float(distance),
                    "has_embeds": meta.get('has_embeds', False),
                    "has_attachments": meta.get('has_attachments', False),
                    "attachment_types": meta.get('attachment_types', []),
                    "content_length": meta.get('content_length', 0)
                }
                results.append(result)
            else:
                logger.warning(f"Index {idx} out of range for metadata (size: {len(enhanced_store['metadata'])})")
        
        return {
            "status": "success",
            "index_info": {
                "total_vectors": enhanced_store["index"].ntotal,
                "metadata_count": len(enhanced_store["metadata"])
            },
            "query": query,
            "results_count": len(results),
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Enhanced FAISS test failed: {e}", exc_info=True)
        return {
            "status": "error",
            "message": str(e),
            "results": []
        }
