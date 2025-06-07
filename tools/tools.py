"""
Enhanced tools system for Discord bot queries with 768D embedding architecture.

Features:
- Hybrid semantic search using msmarco-distilbert-base-v4 (768D)
- Multi-tier FAISS index support (community, enhanced, standard)
- Intelligent channel resolution and validation
- Comprehensive message summarization with local AI
- Data availability validation and reporting
- Robust error handling and fallback mechanisms

Architecture:
- Integrates with upgraded 768D embedding model for superior semantic understanding
- Supports multiple FAISS indices for different use cases
- Uses local AI client for summarization and processing
- Provides comprehensive metadata for message results
- Compatible with enhanced RAG engine capabilities
"""

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
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SentenceTransformer = None  
    SENTENCE_TRANSFORMERS_AVAILABLE = False

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
_community_faiss_store = None
_embedding_model = None

def get_768d_embedding_model():
    """
    Get the 768D embedding model (msmarco-distilbert-base-v4) with proper error handling.
    Returns None if unavailable and logs appropriate warnings.
    """
    global _embedding_model
    
    if _embedding_model is not None:
        return _embedding_model
    
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        logger.warning(
            "SentenceTransformers library not available. "
            "Install with: pip install sentence-transformers"
        )
        return None
        
    try:
        logger.info("Loading msmarco-distilbert-base-v4 embedding model (768D)...")
        _embedding_model = SentenceTransformer("msmarco-distilbert-base-v4")
        logger.info("Successfully loaded 768D embedding model")
        return _embedding_model
    except Exception as e:
        logger.error(f"Failed to load 768D embedding model: {e}")
        return None

def create_query_embedding(query: str) -> Optional[np.ndarray]:
    """
    Create embedding for query using the 768D model with fallback support.
    
    Args:
        query: Query string to embed
        
    Returns:
        768-dimensional embedding array or None if failed
    """
    if not query or not query.strip():
        logger.warning("Empty query provided for embedding")
        return None
        
    # Primary: Use 768D SentenceTransformer model
    model = get_768d_embedding_model()
    if model is not None:
        try:
            embedding = model.encode([query.strip()], convert_to_numpy=True)
            logger.debug(f"Created 768D embedding for query (shape: {embedding.shape})")
            return embedding
        except Exception as e:
            logger.error(f"Failed to create embedding with SentenceTransformer: {e}")
    
    # Fallback: Use AI client
    try:
        logger.info("Falling back to AI client for embeddings")
        embedding = ai_client.create_embeddings(query.strip())
        if isinstance(embedding, np.ndarray):
            return embedding.reshape(1, -1)
        else:
            logger.warning("AI client returned non-array embedding")
            return None
    except Exception as e:
        logger.error(f"Failed to create embedding with AI client: {e}")
        return None

def validate_embedding_compatibility(embedding: np.ndarray, expected_dim: int = 768) -> bool:
    """
    Validate embedding dimensions for 768D architecture compatibility.
    
    Args:
        embedding: Embedding array to validate
        expected_dim: Expected embedding dimension (default 768)
        
    Returns:
        True if compatible, False otherwise
    """
    if embedding is None:
        return False
    
    if not isinstance(embedding, np.ndarray):
        logger.warning(f"Embedding is not numpy array: {type(embedding)}")
        return False
        
    if len(embedding.shape) != 2:
        logger.warning(f"Embedding has wrong shape: {embedding.shape}")
        return False
        
    if embedding.shape[1] != expected_dim:
        logger.warning(f"Embedding dimension mismatch: {embedding.shape[1]} != {expected_dim}")
        return False
        
    return True

def _get_community_faiss_store():
    """Get or initialize community-focused FAISS store with enhanced error handling."""
    global _community_faiss_store
    if _community_faiss_store is None:
        try:
            # Find the most recent community index
            import glob
            community_indices = glob.glob("/Users/jose/Documents/apps/discord-bot/data/indices/community_faiss_*.index")
            if community_indices:
                # Get the most recent one
                latest_index = max(community_indices, key=lambda x: os.path.getctime(x))
                base_name = latest_index.replace('.index', '')
                metadata_file = f"{base_name}_metadata.json"
                
                if not os.path.exists(metadata_file):
                    logger.error(f"Metadata file not found: {metadata_file}")
                    _community_faiss_store = None
                    return None
                
                logger.info(f"Loading community FAISS index from {latest_index}")
                # Load FAISS index
                index = faiss.read_index(latest_index)
                # Load metadata
                with open(metadata_file, "r") as f:
                    data = json.load(f)
                    metadata = data.get("metadata", [])
                
                _community_faiss_store = {"index": index, "metadata": metadata}
                logger.info(f"Loaded community FAISS index with {index.ntotal} vectors")
            else:
                logger.warning("No community FAISS index found")
                _community_faiss_store = None
        except Exception as e:
            logger.error(f"Failed to load community FAISS index: {e}")
            _community_faiss_store = None
    return _community_faiss_store

def _get_enhanced_faiss_store():
    """Get or initialize enhanced FAISS store with enhanced error handling."""
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
                
                if not os.path.exists(metadata_file):
                    logger.error(f"Metadata file not found: {metadata_file}")
                    _enhanced_faiss_store = None
                    return None
                
                logger.info(f"Loading enhanced FAISS index from {latest_index}")
                # Load FAISS index
                index = faiss.read_index(latest_index)
                # Load metadata
                with open(metadata_file, "r") as f:
                    data = json.load(f)
                    metadata = data.get("metadata", [])
                
                _enhanced_faiss_store = {"index": index, "metadata": metadata}
                logger.info(f"Loaded enhanced FAISS index with {index.ntotal} vectors")
            else:
                logger.warning("No enhanced FAISS index found")
                _enhanced_faiss_store = None
        except Exception as e:
            logger.error(f"Failed to load enhanced FAISS index: {e}")
            _enhanced_faiss_store = None
    return _enhanced_faiss_store

def _get_faiss_store():
    """Get or initialize standard FAISS store with enhanced error handling."""
    global _faiss_store
    if _faiss_store is None:
        try:
            logger.info(f"Loading standard FAISS index from {INDEX_DIR}")
            index_path = f"{INDEX_DIR}/index.faiss"
            metadata_path = f"{INDEX_DIR}/index.pkl"
            
            if not os.path.exists(index_path):
                logger.warning(f"Standard FAISS index not found at {index_path}")
                # Create empty index if not found
                dimension = config.models.embedding_dimension
                index = faiss.IndexFlatL2(dimension)
                _faiss_store = {"index": index, "metadata": []}
                logger.warning("Created empty standard FAISS index")
                return _faiss_store
            
            # Load FAISS index
            index = faiss.read_index(index_path)
            # Load metadata
            if os.path.exists(metadata_path):
                with open(metadata_path, "rb") as f:
                    import pickle
                    metadata = pickle.load(f)
            else:
                logger.warning(f"Metadata file not found: {metadata_path}")
                metadata = []
                
            _faiss_store = {"index": index, "metadata": metadata}
            logger.info(f"Loaded standard FAISS index with {index.ntotal} vectors")
        except Exception as e:
            logger.error(f"Failed to load standard FAISS index: {e}")
            # Create empty index if loading failed
            dimension = config.models.embedding_dimension
            index = faiss.IndexFlatL2(dimension)
            _faiss_store = {"index": index, "metadata": []}
            logger.warning("Created empty standard FAISS index due to loading error")
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
                # Create query embedding using enhanced 768D model
                query_embedding = create_query_embedding(query)
                if query_embedding is None:
                    logger.error("Failed to create query embedding, falling back to keyword search")
                    messages = candidates[:k]
                else:
                    # Validate embedding compatibility
                    if not validate_embedding_compatibility(query_embedding):
                        logger.error("Query embedding not compatible with 768D architecture")
                        messages = candidates[:k]
                    else:
                        # Try community FAISS store first (best features)
                        community_store = _get_community_faiss_store()
                        enhanced_store = _get_enhanced_faiss_store()
                        
                        semantic_results = []
                        
                        if community_store and community_store["index"].ntotal > 0:
                            logger.info("Using community FAISS index for semantic search")
                            
                            # Search in community FAISS index
                            distances, indices = community_store["index"].search(
                                query_embedding, min(k * 2, community_store["index"].ntotal)
                            )
                            
                            logger.info(f"Community search returned indices: {indices[0][:5]}")
                            
                            # Map back to messages using community metadata
                            semantic_messages = []
                            for idx, distance in zip(indices[0], distances[0]):
                                if 0 <= idx < len(community_store["metadata"]):
                                    meta = community_store["metadata"][idx]
                                    message_id = meta.get('message_id')
                                    
                                    # First try to find in candidates for efficiency
                                    found_in_candidates = False
                                    for msg in candidates:
                                        if msg.message_id == message_id:
                                            semantic_messages.append(msg)
                                            found_in_candidates = True
                                            logger.info(f"Found candidate match for message_id: {message_id}")
                                            break
                                    
                                    # If not in candidates, query database directly
                                    if not found_in_candidates:
                                        try:
                                            msg = session.query(Message).filter(Message.message_id == message_id).first()
                                            if msg:
                                                # Apply filters
                                                if guild_id and msg.guild_id != guild_id:
                                                    continue
                                                if channel_id and msg.channel_id != channel_id:
                                                    continue
                                                elif channel_name and channel_name.lower() not in msg.channel_name.lower():
                                                    continue
                                                
                                                semantic_messages.append(msg)
                                                logger.info(f"Found database match for message_id: {message_id}")
                                        except Exception as e:
                                            logger.warning(f"Error querying message {message_id}: {e}")
                                            continue
                                else:
                                    logger.warning(f"Invalid index {idx} (metadata size: {len(community_store['metadata'])})")
                            
                            semantic_results = semantic_messages
                            logger.info(f"Community search found {len(semantic_results)} semantic results")
                        
                        # Fall back to enhanced FAISS if community search didn't work
                        elif enhanced_store and enhanced_store["index"].ntotal > 0:
                            logger.info("Using enhanced FAISS index for semantic search")
                            
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
            
    except Exception as e:
        logger.error(f"Error summarizing messages: {e}")
        error_msg = f"Error generating summary: {str(e)}"
        return {"error": error_msg} if as_json else error_msg
    finally:
        session.close()

def get_channels() -> List[Dict[str, Any]]:
    """
    Enhanced channel listing with additional metadata.
    
    Returns:
        List of channel dictionaries with enhanced information
    """
    session = SessionLocal()
    try:
        # Get channel information with message counts and date ranges
        channel_info = session.query(
            Message.channel_id,
            Message.channel_name,
            func.count(Message.message_id).label('message_count'),
            func.min(Message.timestamp).label('oldest_message'),
            func.max(Message.timestamp).label('newest_message')
        ).group_by(
            Message.channel_id, 
            Message.channel_name
        ).order_by(
            func.count(Message.message_id).desc()
        ).all()
        
        channels = []
        for info in channel_info:
            channels.append({
                "id": info.channel_id,
                "name": info.channel_name,
                "message_count": info.message_count,
                "oldest_message": info.oldest_message.isoformat() if info.oldest_message else None,
                "newest_message": info.newest_message.isoformat() if info.newest_message else None,
                "activity_level": "high" if info.message_count > 100 else "medium" if info.message_count > 20 else "low"
            })
        
        logger.info(f"Retrieved {len(channels)} channels")
        return channels
        
    except Exception as e:
        logger.error(f"Error retrieving channels: {e}")
        return []
    finally:
        session.close()

def validate_data_availability() -> Dict[str, Any]:
    """
    Enhanced data availability validation with 768D architecture info.
    
    Returns:
        Dictionary with data status, counts, and system health
    """
    session = SessionLocal()
    try:
        # Get message count
        total_messages = session.query(func.count(Message.message_id)).scalar()
        
        if total_messages == 0:
            return {
                "status": "error",
                "message": "No messages found in database",
                "count": 0,
                "channels": {},
                "date_range": {},
                "system_health": {
                    "embedding_model": "not_available",
                    "faiss_indices": []
                }
            }
        
        # Get channel breakdown
        channel_data = session.query(
            Message.channel_name, 
            func.count(Message.message_id).label('count')
        ).group_by(Message.channel_name).all()
        
        channels = {name: count for name, count in channel_data}
        
        # Get date range
        oldest = session.query(func.min(Message.timestamp)).scalar()
        newest = session.query(func.max(Message.timestamp)).scalar()
        
        date_range = {
            "oldest": oldest.isoformat() if oldest else "Unknown",
            "newest": newest.isoformat() if newest else "Unknown"
        }
        
        # Check system health
        system_health = {
            "embedding_model": "available" if get_768d_embedding_model() else "unavailable",
            "sentence_transformers": SENTENCE_TRANSFORMERS_AVAILABLE,
            "faiss_indices": []
        }
        
        # Check FAISS indices availability
        if _get_community_faiss_store():
            system_health["faiss_indices"].append("community")
        if _get_enhanced_faiss_store():
            system_health["faiss_indices"].append("enhanced")
        if _get_faiss_store():
            system_health["faiss_indices"].append("standard")
        
        return {
            "status": "ok",
            "count": total_messages,
            "channels": channels,
            "date_range": date_range,
            "system_health": system_health
        }
        
    except Exception as e:
        logger.error(f"Error validating data availability: {e}")
        return {
            "status": "error",
            "message": f"Database error: {str(e)}",
            "count": 0,
            "channels": {},
            "date_range": {},
            "system_health": {
                "embedding_model": "error",
                "faiss_indices": []
            }
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
            embedding_model = SentenceTransformer("msmarco-distilbert-base-v4")
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

def find_community_experts(
    skill_or_topic: str,
    k: int = 5,
    min_expertise_score: float = 0.3,
    guild_id: Optional[int] = None,
    channel_id: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Find community experts based on skills or topics using community metadata.
    
    Args:
        skill_or_topic: The skill or topic to search for experts in
        k: Number of expert results to return
        min_expertise_score: Minimum expertise confidence score (0.0-1.0)
        guild_id: Optional guild filter
        channel_id: Optional channel filter
        
    Returns:
        List of expert information with expertise scores and examples
    """
    logger.info(f"Finding experts for: {skill_or_topic}")
    
    try:
        community_store = _get_community_faiss_store()
        if not community_store:
            logger.warning("Community FAISS index not available")
            return []
        
        # Create query embedding
        if SentenceTransformer is not None:
            embedding_model = SentenceTransformer("msmarco-distilbert-base-v4")
            query_embedding = embedding_model.encode([skill_or_topic])
        else:
            query_embedding = ai_client.create_embeddings(skill_or_topic)
            query_embedding = query_embedding.reshape(1, -1)
        
        # Search in community index
        distances, indices = community_store["index"].search(
            query_embedding.reshape(1, -1), min(k * 10, community_store["index"].ntotal)
        )
        
        # Aggregate experts by author
        expert_data = {}
        skill_lower = skill_or_topic.lower()
        
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(community_store["metadata"]):
                meta = community_store["metadata"][idx]
                
                # Filter by guild/channel if specified
                if guild_id and meta.get('guild_id') != guild_id:
                    continue
                if channel_id and meta.get('channel_id') != channel_id:
                    continue
                
                # Check if message has relevant skills or expertise
                skill_keywords = meta.get('skill_keywords', [])
                expertise_indicators = meta.get('expertise_indicators', {})
                expertise_confidence = meta.get('expertise_confidence', 0.0)
                
                # Skip if expertise score too low
                if expertise_confidence < min_expertise_score:
                    continue
                
                # Check if skill matches
                skill_match = any(skill_lower in keyword.lower() for keyword in skill_keywords)
                expertise_match = any(skill_lower in key.lower() for key in expertise_indicators.keys())
                
                if skill_match or expertise_match or distance < 0.7:  # Semantic similarity threshold
                    author_name = meta.get('author_name')
                    if author_name:
                        if author_name not in expert_data:
                            expert_data[author_name] = {
                                'author_name': author_name,
                                'author_id': meta.get('author_id'),
                                'total_expertise_score': 0.0,
                                'skill_mentions': 0,
                                'example_messages': [],
                                'skills': set(),
                                'channels_active': set()
                            }
                        
                        expert = expert_data[author_name]
                        expert['total_expertise_score'] += expertise_confidence
                        expert['skill_mentions'] += 1
                        expert['skills'].update(skill_keywords)
                        expert['channels_active'].add(meta.get('channel_name', ''))
                        
                        # Add example message if it's good quality
                        if len(expert['example_messages']) < 3 and meta.get('content_length', 0) > 50:
                            expert['example_messages'].append({
                                'content': meta.get('processed_content', '')[:200] + '...',
                                'timestamp': meta.get('timestamp'),
                                'channel_name': meta.get('channel_name'),
                                'expertise_score': expertise_confidence,
                                'jump_url': build_jump_url(
                                    meta.get('guild_id'),
                                    meta.get('channel_id'),
                                    meta.get('message_id')
                                )
                            })
        
        # Sort experts by combined score
        experts = []
        for author_name, data in expert_data.items():
            avg_expertise = data['total_expertise_score'] / max(data['skill_mentions'], 1)
            combined_score = avg_expertise * (1 + 0.1 * data['skill_mentions'])  # Boost for frequency
            
            experts.append({
                'author_name': author_name,
                'author_id': data['author_id'],
                'expertise_score': round(avg_expertise, 3),
                'combined_score': round(combined_score, 3),
                'skill_mentions': data['skill_mentions'],
                'skills': list(data['skills'])[:10],  # Top skills
                'channels_active': list(data['channels_active']),
                'example_messages': data['example_messages']
            })
        
        # Sort by combined score and return top k
        experts.sort(key=lambda x: x['combined_score'], reverse=True)
        return experts[:k]
        
    except Exception as e:
        logger.error(f"Error finding community experts: {e}", exc_info=True)
        return []

def search_conversation_threads(
    query: str,
    k: int = 5,
    include_resolved: bool = True,
    min_participants: int = 2,
    guild_id: Optional[int] = None,
    channel_id: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Search for conversation threads based on content and thread metadata.
    
    Args:
        query: Search query for thread content
        k: Number of threads to return
        include_resolved: Whether to include resolved conversations
        min_participants: Minimum number of participants in thread
        guild_id: Optional guild filter
        channel_id: Optional channel filter
        
    Returns:
        List of conversation thread information
    """
    logger.info(f"Searching conversation threads for: {query}")
    
    try:
        community_store = _get_community_faiss_store()
        if not community_store:
            logger.warning("Community FAISS index not available")
            return []
        
        # Create query embedding
        if SentenceTransformer is not None:
            embedding_model = SentenceTransformer("msmarco-distilbert-base-v4")
            query_embedding = embedding_model.encode([query])
        else:
            query_embedding = ai_client.create_embeddings(query)
            query_embedding = query_embedding.reshape(1, -1)
        
        # Search in community index
        distances, indices = community_store["index"].search(
            query_embedding.reshape(1, -1), min(k * 20, community_store["index"].ntotal)
        )
        
        # Group messages by conversation thread
        threads = {}
        
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(community_store["metadata"]):
                meta = community_store["metadata"][idx]
                
                # Filter by guild/channel if specified
                if guild_id and meta.get('guild_id') != guild_id:
                    continue
                if channel_id and meta.get('channel_id') != channel_id:
                    continue
                
                thread_id = meta.get('conversation_thread_id')
                if thread_id:
                    if thread_id not in threads:
                        threads[thread_id] = {
                            'thread_id': thread_id,
                            'channel_name': meta.get('channel_name'),
                            'guild_id': meta.get('guild_id'),
                            'channel_id': meta.get('channel_id'),
                            'participants': set(),
                            'messages': [],
                            'question_resolved': None,
                            'has_questions': False,
                            'has_solutions': False,
                            'total_engagement': 0,
                            'start_time': None,
                            'end_time': None
                        }
                    
                    thread = threads[thread_id]
                    thread['participants'].update(meta.get('thread_participants', []))
                    thread['messages'].append({
                        'content': meta.get('processed_content', ''),
                        'author_name': meta.get('author_name'),
                        'timestamp': meta.get('timestamp'),
                        'distance': float(distance),
                        'question_indicators': meta.get('question_indicators', False),
                        'solution_indicators': meta.get('solution_indicators', False),
                        'help_seeking': meta.get('help_seeking', False),
                        'help_providing': meta.get('help_providing', False),
                        'message_id': meta.get('message_id')
                    })
                    
                    # Update thread metadata
                    if meta.get('question_indicators'):
                        thread['has_questions'] = True
                    if meta.get('solution_indicators'):
                        thread['has_solutions'] = True
                    if meta.get('question_resolved') is not None:
                        thread['question_resolved'] = meta.get('question_resolved')
                    
                    # Update time range
                    timestamp = meta.get('timestamp')
                    if timestamp:
                        if not thread['start_time'] or timestamp < thread['start_time']:
                            thread['start_time'] = timestamp
                        if not thread['end_time'] or timestamp > thread['end_time']:
                            thread['end_time'] = timestamp
        
        # Filter and rank threads
        filtered_threads = []
        for thread_id, thread in threads.items():
            # Filter by participant count
            if len(thread['participants']) < min_participants:
                continue
            
            # Filter by resolution status if needed
            if not include_resolved and thread['question_resolved']:
                continue
            
            # Calculate thread score
            avg_distance = sum(msg['distance'] for msg in thread['messages']) / len(thread['messages'])
            engagement_score = len(thread['participants']) * len(thread['messages'])
            thread_score = (1 / (avg_distance + 0.1)) * (1 + 0.01 * engagement_score)
            
            # Sort messages by timestamp
            thread['messages'].sort(key=lambda x: x['timestamp'])
            
            # Prepare result
            filtered_threads.append({
                'thread_id': thread_id,
                'channel_name': thread['channel_name'],
                'participants': list(thread['participants']),
                'participant_count': len(thread['participants']),
                'message_count': len(thread['messages']),
                'has_questions': thread['has_questions'],
                'has_solutions': thread['has_solutions'],
                'question_resolved': thread['question_resolved'],
                'start_time': thread['start_time'],
                'end_time': thread['end_time'],
                'thread_score': round(thread_score, 3),
                'summary': thread['messages'][0]['content'][:200] + '...' if thread['messages'] else '',
                'key_messages': thread['messages'][:5],  # Top 5 messages
                'jump_url': build_jump_url(
                    thread['guild_id'],
                    thread['channel_id'],
                    thread['messages'][0]['message_id']
                ) if thread['messages'] else None
            })
        
        # Sort by thread score and return top k
        filtered_threads.sort(key=lambda x: x['thread_score'], reverse=True)
        return filtered_threads[:k]
        
    except Exception as e:
        logger.error(f"Error searching conversation threads: {e}", exc_info=True)
        return []

def analyze_community_engagement(
    author_name: Optional[str] = None,
    channel_name: Optional[str] = None,
    time_period_days: int = 30,
    guild_id: Optional[int] = None
) -> Dict[str, Any]:
    """
    Analyze community engagement patterns using community metadata.
    
    Args:
        author_name: Optional specific author to analyze
        channel_name: Optional specific channel to analyze
        time_period_days: Number of days to look back
        guild_id: Optional guild filter
        
    Returns:
        Engagement analysis with metrics and insights
    """
    logger.info(f"Analyzing community engagement for author={author_name}, channel={channel_name}")
    
    try:
        community_store = _get_community_faiss_store()
        if not community_store:
            logger.warning("Community FAISS index not available")
            return {}
        
        from datetime import datetime, timedelta
        cutoff_date = datetime.now() - timedelta(days=time_period_days)
        
        # Analyze metadata
        engagement_data = {
            'total_messages': 0,
            'unique_authors': set(),
            'help_seeking_messages': 0,
            'help_providing_messages': 0,
            'questions_asked': 0,
            'solutions_provided': 0,
            'technical_discussions': 0,
            'skill_mentions': {},
            'author_activity': {},
            'channel_activity': {},
            'conversation_threads': set(),
            'resolved_questions': 0,
            'unresolved_questions': 0
        }
        
        for meta in community_store["metadata"]:
            # Filter by time
            timestamp_str = meta.get('timestamp', '')
            try:
                msg_time = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                if msg_time < cutoff_date:
                    continue
            except:
                continue
            
            # Filter by guild
            if guild_id and meta.get('guild_id') != guild_id:
                continue
            
            # Filter by author
            if author_name and meta.get('author_name') != author_name:
                continue
            
            # Filter by channel
            if channel_name and meta.get('channel_name') != channel_name:
                continue
            
            # Collect engagement metrics
            engagement_data['total_messages'] += 1
            engagement_data['unique_authors'].add(meta.get('author_name'))
            
            if meta.get('help_seeking'):
                engagement_data['help_seeking_messages'] += 1
            if meta.get('help_providing'):
                engagement_data['help_providing_messages'] += 1
            if meta.get('question_indicators'):
                engagement_data['questions_asked'] += 1
            if meta.get('solution_indicators'):
                engagement_data['solutions_provided'] += 1
            if meta.get('has_technical_skills'):
                engagement_data['technical_discussions'] += 1
            
            # Track skills
            for skill in meta.get('skill_keywords', []):
                if skill not in engagement_data['skill_mentions']:
                    engagement_data['skill_mentions'][skill] = 0
                engagement_data['skill_mentions'][skill] += 1
            
            # Track author activity
            author = meta.get('author_name')
            if author:
                if author not in engagement_data['author_activity']:
                    engagement_data['author_activity'][author] = {
                        'messages': 0,
                        'help_seeking': 0,
                        'help_providing': 0,
                        'questions': 0,
                        'solutions': 0,
                        'expertise_score': 0.0
                    }
                activity = engagement_data['author_activity'][author]
                activity['messages'] += 1
                activity['help_seeking'] += int(meta.get('help_seeking', False))
                activity['help_providing'] += int(meta.get('help_providing', False))
                activity['questions'] += int(meta.get('question_indicators', False))
                activity['solutions'] += int(meta.get('solution_indicators', False))
                activity['expertise_score'] += meta.get('expertise_confidence', 0.0)
            
            # Track channel activity
            channel = meta.get('channel_name')
            if channel:
                if channel not in engagement_data['channel_activity']:
                    engagement_data['channel_activity'][channel] = 0
                engagement_data['channel_activity'][channel] += 1
            
            # Track conversation threads
            thread_id = meta.get('conversation_thread_id')
            if thread_id:
                engagement_data['conversation_threads'].add(thread_id)
            
            # Track question resolution
            if meta.get('question_resolved') is True:
                engagement_data['resolved_questions'] += 1
            elif meta.get('question_resolved') is False:
                engagement_data['unresolved_questions'] += 1
        
        # Calculate final metrics
        unique_author_count = len(engagement_data['unique_authors'])
        thread_count = len(engagement_data['conversation_threads'])
        
        # Top skills
        top_skills = sorted(
            engagement_data['skill_mentions'].items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        # Top contributors
        top_contributors = []
        for author, activity in engagement_data['author_activity'].items():
            avg_expertise = activity['expertise_score'] / max(activity['messages'], 1)
            contribution_score = (
                activity['help_providing'] * 2 +
                activity['solutions'] * 3 +
                activity['messages'] * 0.5 +
                avg_expertise * 10
            )
            top_contributors.append({
                'author_name': author,
                'contribution_score': round(contribution_score, 2),
                'messages': activity['messages'],
                'help_providing': activity['help_providing'],
                'solutions': activity['solutions'],
                'avg_expertise': round(avg_expertise, 3)
            })
        
        top_contributors.sort(key=lambda x: x['contribution_score'], reverse=True)
        
        # Most active channels
        top_channels = sorted(
            engagement_data['channel_activity'].items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        return {
            'analysis_period_days': time_period_days,
            'total_messages': engagement_data['total_messages'],
            'unique_authors': unique_author_count,
            'conversation_threads': thread_count,
            'engagement_metrics': {
                'help_seeking_messages': engagement_data['help_seeking_messages'],
                'help_providing_messages': engagement_data['help_providing_messages'],
                'questions_asked': engagement_data['questions_asked'],
                'solutions_provided': engagement_data['solutions_provided'],
                'technical_discussions': engagement_data['technical_discussions'],
                'resolved_questions': engagement_data['resolved_questions'],
                'unresolved_questions': engagement_data['unresolved_questions']
            },
            'top_skills': top_skills,
            'top_contributors': top_contributors[:10],
            'most_active_channels': top_channels,
            'community_health': {
                'help_ratio': round(
                    engagement_data['help_providing_messages'] / max(engagement_data['help_seeking_messages'], 1),
                    2
                ),
                'resolution_ratio': round(
                    engagement_data['resolved_questions'] / max(
                        engagement_data['resolved_questions'] + engagement_data['unresolved_questions'], 1
                    ),
                    2
                ),
                'engagement_per_author': round(
                    engagement_data['total_messages'] / max(unique_author_count, 1),
                    1
                )
            }
        }
        
    except Exception as e:
        logger.error(f"Error analyzing community engagement: {e}", exc_info=True)
        return {}

def test_community_search(
    query: str,
    k: int = 5,
    search_type: str = "general"
) -> Dict[str, Any]:
    """
    Test function specifically for community FAISS index with detailed analysis.
    
    Args:
        query: Search query
        k: Number of results
        search_type: Type of search - "general", "experts", "threads", "engagement"
        
    Returns:
        Detailed search results with community metadata
    """
    logger.info(f"Testing community search with query: {query}, type: {search_type}")
    
    try:
        community_store = _get_community_faiss_store()
        if not community_store:
            return {
                "status": "error",
                "message": "Community FAISS index not available",
                "results": []
            }
        
        if search_type == "experts":
            results = find_community_experts(query, k=k)
            return {
                "status": "success",
                "search_type": "experts",
                "query": query,
                "results_count": len(results),
                "results": results
            }
        
        elif search_type == "threads":
            results = search_conversation_threads(query, k=k)
            return {
                "status": "success",
                "search_type": "conversation_threads",
                "query": query,
                "results_count": len(results),
                "results": results
            }
        
        elif search_type == "engagement":
            results = analyze_community_engagement(time_period_days=30)
            return {
                "status": "success",
                "search_type": "engagement_analysis",
                "query": query,
                "results": results
            }
        
        else:  # general search
            # Create query embedding
            if SentenceTransformer is not None:
                embedding_model = SentenceTransformer("msmarco-distilbert-base-v4")
                query_embedding = embedding_model.encode([query])
            else:
                query_embedding = ai_client.create_embeddings(query)
                query_embedding = query_embedding.reshape(1, -1)
            
            logger.info(f"Query embedding shape: {query_embedding.shape}")
            
            # Search in community index
            distances, indices = community_store["index"].search(
                query_embedding.reshape(1, -1), min(k, community_store["index"].ntotal)
            )
            
            logger.info(f"Search returned {len(indices[0])} results")
            logger.info(f"Distances: {distances[0]}")
            logger.info(f"Indices: {indices[0]}")
            
            # Process results with community metadata
            results = []
            for idx, distance in zip(indices[0], distances[0]):
                if idx < len(community_store["metadata"]):
                    meta = community_store["metadata"][idx]
                    result = {
                        "message_id": meta.get('message_id'),
                        "author_name": meta.get('author_name'),
                        "channel_name": meta.get('channel_name'),
                        "content": meta.get('processed_content', meta.get('original_content', '')),
                        "timestamp": meta.get('timestamp'),
                        "distance": float(distance),
                        "community_features": {
                            "skill_keywords": meta.get('skill_keywords', []),
                            "expertise_confidence": meta.get('expertise_confidence', 0.0),
                            "has_technical_skills": meta.get('has_technical_skills', False),
                            "question_indicators": meta.get('question_indicators', False),
                            "solution_indicators": meta.get('solution_indicators', False),
                            "help_seeking": meta.get('help_seeking', False),
                            "help_providing": meta.get('help_providing', False),
                            "conversation_thread_id": meta.get('conversation_thread_id'),
                            "thread_participants": meta.get('thread_participants', []),
                            "question_resolved": meta.get('question_resolved')
                        },
                        "content_metadata": {
                            "content_length": meta.get('content_length', 0),
                            "has_embeds": meta.get('has_embeds', False),
                            "has_attachments": meta.get('has_attachments', False),
                            "attachment_types": meta.get('attachment_types', []),
                            "extracted_urls": meta.get('extracted_urls', [])
                        }
                    }
                    results.append(result)
                else:
                    logger.warning(f"Index {idx} out of range for metadata (size: {len(community_store['metadata'])})")
            
            return {
                "status": "success",
                "search_type": "general",
                "index_info": {
                    "total_vectors": community_store["index"].ntotal,
                    "metadata_count": len(community_store["metadata"])
                },
                "query": query,
                "results_count": len(results),
                "results": results
            }
        
    except Exception as e:
        logger.error(f"Community search test failed: {e}", exc_info=True)
        return {
            "status": "error",
            "message": str(e),
            "results": []
        }
