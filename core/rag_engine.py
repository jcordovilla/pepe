# rag_engine.py

import os
from typing import Any, Dict, List, Optional, Union
import numpy as np
from core.ai_client import get_ai_client
from core.config import get_config
from core.enhanced_fallback_system import EnhancedFallbackSystem
from utils.helpers import build_jump_url
from tools.tools import resolve_channel_name, summarize_messages, validate_data_availability
from tools.time_parser import parse_timeframe, extract_time_reference, extract_channel_reference, extract_content_reference
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import json
from sqlalchemy.orm import Session
from db import Message, get_db_session
import logging
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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
        self._message_order = None
    
    def load(self):
        """Load FAISS index and metadata from disk."""
        import faiss
        import json
        import os
        
    def load(self):
        """Load FAISS index and metadata from disk."""
        import faiss
        import json
        import os
        
        try:
            logger.info("Loading vector store from disk")
            
            # Check if enhanced index format (single files)
            if os.path.exists(f"{self.index_path}.index"):
                # Enhanced index format
                self._index = faiss.read_index(f"{self.index_path}.index")
                with open(f"{self.index_path}_metadata.json", "r") as f:
                    data = json.load(f)
                    self._metadata = data.get('metadata', {})
                    self._message_order = data.get('message_order', data.get('id_mapping', []))
                logger.info(f"Loaded enhanced vector store with {self._index.ntotal} vectors")
            else:
                # Standard index format (directory structure)
                self._index = faiss.read_index(f"{self.index_path}/faiss_index.index")
                with open(f"{self.index_path}/metadata.json", "r") as f:
                    data = json.load(f)
                    self._metadata = data.get('metadata', {})
                    self._message_order = data.get('message_order', data.get('id_mapping', []))
                logger.info(f"Loaded standard vector store with {self._index.ntotal} vectors")
                
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
            if idx >= 0 and idx < len(self._message_order):
                message_id = str(self._message_order[idx])
                if message_id in self._metadata:
                    meta = self._metadata[message_id].copy()
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
        "Answer the user's question based on the provided Discord message context. "
        "For search queries, list the relevant messages with their metadata. "
        "Include author names, timestamps, channel names, and URLs when available.\n"
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
            # Use enhanced fallback system to provide intelligent response
            try:
                from core.enhanced_fallback_system import EnhancedFallbackSystem
                fallback_system = EnhancedFallbackSystem()
                
                # Determine capability based on query patterns
                capability = "general"
                if any(kw in query.lower() for kw in ["analyze", "analysis", "patterns", "data", "statistics"]):
                    capability = "server_data_analysis"
                elif any(kw in query.lower() for kw in ["feedback", "summary", "summarize"]):
                    capability = "feedback_summarization"
                elif any(kw in query.lower() for kw in ["trending", "popular", "topics"]):
                    capability = "trending_topics"
                elif any(kw in query.lower() for kw in ["question", "answer", "qa", "q&a"]):
                    capability = "qa_concepts"
                elif any(kw in query.lower() for kw in ["engagement", "statistics", "stats", "metrics"]):
                    capability = "statistics_generation"
                elif any(kw in query.lower() for kw in ["channel", "structure", "organization"]):
                    capability = "server_structure_analysis"
                
                fallback_response = fallback_system.generate_intelligent_fallback(
                    query=query,
                    capability=capability,
                    available_channels=None,  # Could enhance with channel list
                    timeframe=None  # Could enhance with timeframe detection
                )
                
                return fallback_response["response"]
                
            except Exception as fallback_error:
                logger.error(f"Fallback system failed: {fallback_error}")
                return "⚠️ I couldn't find relevant messages. Try rephrasing your question or being more specific."

        chat_messages = build_prompt(matches, query, as_json)
        
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

        # Check if key_topics are requested
        include_key_topics = "key_topics" in query.lower() or "key topics" in query.lower()
        
        # Get messages for the timeframe
        messages = summarize_messages(
            start_iso=start_iso,
            end_iso=end_iso,
            channel_id=channel_id,
            as_json=True,
            include_key_topics=include_key_topics
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
        
        # Add key_topics if present
        if "key_topics" in messages:
            response["key_topics"] = messages["key_topics"]

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
        logger.error(f"Error processing query: {e}")
        return f"❌ Error processing query: {str(e)}"
    finally:
        duration = time.time() - start_time
        logger.info(f"Query completed in {duration:.2f} seconds")

# ——— Resource Search Integration ———

class ResourceVectorStore:
    """Resource-specific vector store using FAISS with resource embeddings."""
    
    def __init__(self, 
                 index_path: str = "data/indices/resource_faiss_20250607_220423.index",
                 metadata_path: str = "data/indices/resource_faiss_20250607_220423_metadata.json",
                 model_name: str = "msmarco-distilbert-base-v4"):
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.model_name = model_name
        self._index = None
        self._metadata = None
        self._model = None
    
    def load(self):
        """Load FAISS index, metadata, and embedding model."""
        if self._index is None:
            logger.info(f"Loading resource FAISS index from: {self.index_path}")
            import faiss
            self._index = faiss.read_index(self.index_path)
            
            logger.info(f"Loading resource metadata from: {self.metadata_path}")
            with open(self.metadata_path, 'r') as f:
                data = json.load(f)
                self._metadata = data['metadata']
            
            logger.info(f"Loading sentence transformer model: {self.model_name}")
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
            except ImportError:
                logger.error("sentence_transformers not installed. Resource search unavailable.")
                raise ImportError("sentence_transformers required for resource search")
            
            logger.info(f"Resource store loaded: {self._index.ntotal} vectors, {len(self._metadata)} metadata entries")
    
    def search(self, query_text: str, k: int = 5, filters: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Search for similar resources."""
        if self._index is None:
            self.load()
        
        # Create query embedding
        query_embedding = self._model.encode([query_text])
        
        # Search FAISS index
        scores, indices = self._index.search(query_embedding.astype('float32'), k * 2)
        
        # Return metadata for found resources
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self._metadata):
                meta = self._metadata[idx].copy()
                meta['similarity_score'] = float(score)
                
                # Apply filters if provided
                if filters and not self._apply_filters(meta, filters):
                    continue
                
                results.append(meta)
                
                if len(results) >= k:
                    break
        
        return results
    
    def _apply_filters(self, resource: Dict, filters: Dict) -> bool:
        """Apply optional filters to search results."""
        for key, value in filters.items():
            if key == 'tag' and resource.get('tag') != value:
                return False
            elif key == 'domain' and resource.get('domain') != value:
                return False
            elif key == 'is_tool' and resource.get('is_tool') != value:
                return False
            elif key == 'is_article' and resource.get('is_article') != value:
                return False
            elif key == 'is_youtube' and resource.get('is_youtube') != value:
                return False
            elif key == 'author' and resource.get('author') != value:
                return False
        return True

# Global resource store instance
_resource_store = None

def load_resource_vectorstore() -> ResourceVectorStore:
    """Load the resource vector store (singleton pattern)."""
    global _resource_store
    if _resource_store is None:
        _resource_store = ResourceVectorStore()
        try:
            _resource_store.load()
        except Exception as e:
            logger.warning(f"Resource store unavailable: {e}")
            return None
    return _resource_store

def get_top_k_resource_matches(
    query: str,
    k: int = 5,
    filters: Optional[Dict] = None
) -> List[Dict[str, Any]]:
    """
    Retrieve the top-k resource matches for 'query'.
    """
    try:
        logger.info(f"Searching for resource matches: query='{query}', k={k}")
        store = load_resource_vectorstore()
        
        if store is None:
            logger.warning("Resource store not available")
            return []
        
        # Get search results from resource vector store
        results = store.search(query, k, filters)
        
        logger.info(f"Found {len(results)} resource matches")
        return results
    except Exception as e:
        logger.error(f"Error in resource search: {e}")
        return []

def build_resource_prompt(
    resource_matches: List[Dict[str, Any]],
    question: str,
    as_json: bool = False
) -> List[Dict[str, str]]:
    """
    Construct a ChatML prompt with context from matching resources.
    """
    context_lines: List[str] = []
    for r in resource_matches:
        title = r.get("title", "Untitled Resource")
        description = r.get("description", "No description")
        url = r.get("resource_url", "")
        domain = r.get("domain", "")
        tag = r.get("tag", "Other")
        author = r.get("author", "Unknown")
        score = r.get("similarity_score", 0)
        
        line = f"**{title}** ({tag})\n"
        line += f"📝 {description}\n"
        line += f"🌐 Domain: {domain} | 👤 Shared by: {author} | 📊 Relevance: {score:.3f}"
        if url:
            line += f"\n🔗 [View Resource]({url})"
        context_lines.append(line)

    instructions = (
        "You are a knowledgeable assistant with access to a curated collection of AI, technology, and educational resources.\n\n"
        "Use the provided resource context to answer the user's question. "
        "Include resource titles, descriptions, URLs, and relevance scores when helpful.\n"
        "Focus on the most relevant resources and explain how they relate to the user's query.\n"
    )
    if as_json:
        instructions += "Return results as a JSON array with fields: resource_id, title, description, resource_url, tag, domain, author, similarity_score.\n"

    return [
        {"role": "system", "content": "You are a resource discovery assistant for AI and technology resources."},
        {"role": "user", "content": instructions + "\nResource Context:\n" + "\n\n".join(context_lines)
                                  + f"\n\nUser's question: {question}\n"}
    ]

def get_resource_answer(
    query: str,
    k: int = 5,
    as_json: bool = False,
    return_matches: bool = False,
    filters: Optional[Dict] = None
) -> Any:
    """
    Perform a resource-based RAG query: retrieve resource matches, build a prompt, and ask AI.
    Returns either a string (answer) or (answer, matches) if return_matches=True.
    """
    try:
        matches = get_top_k_resource_matches(query, k, filters)
        
        if not matches and not return_matches:
            return "⚠️ I couldn't find relevant resources. Try rephrasing your question or using different keywords."

        chat_messages = build_resource_prompt(matches, query, as_json)
        
        if as_json:
            # For JSON responses, add specific instruction
            chat_messages[-1]["content"] += "\n\nIMPORTANT: Return only valid JSON, no other text."
        
        answer = ai_client.chat_completion(
            chat_messages,
            temperature=0.7,
            max_tokens=1500
        )
        
        return (answer, matches) if return_matches else answer

    except Exception as e:
        err = f"❌ Error during resource retrieval: {e}"
        return (err, []) if return_matches else err

def get_hybrid_answer(
    query: str,
    k_messages: int = 3,
    k_resources: int = 3,
    as_json: bool = False,
    return_matches: bool = False,
    guild_id: Optional[int] = None,
    channel_id: Optional[int] = None,
    channel_name: Optional[str] = None,
    resource_filters: Optional[Dict] = None
) -> Any:
    """
    Perform a hybrid RAG query combining both Discord messages and resources.
    """
    try:
        # Get both message and resource matches
        message_matches = get_top_k_matches(
            query, k_messages,
            guild_id=guild_id,
            channel_id=channel_id,
            channel_name=channel_name
        )
        
        resource_matches = get_top_k_resource_matches(query, k_resources, resource_filters)
        
        if not message_matches and not resource_matches and not return_matches:
            return "⚠️ I couldn't find relevant messages or resources. Try rephrasing your question or being more specific."

        # Build hybrid context
        context_lines: List[str] = []
        
        # Add message context
        if message_matches:
            context_lines.append("## Discord Messages Context")
            for m in message_matches:
                author = m.get("author", {})
                author_name = author.get("display_name") or author.get("username") or "Unknown"
                ts = m.get("timestamp", "")
                ch_name = m.get("channel_name") or m.get("channel_id")
                content = m.get("content", "").replace("\n", " ")
                url = safe_jump_url(m)
                line = f"**{author_name}** (_{ts}_ in **#{ch_name}**):\n{content}"
                if url:
                    line += f"\n[🔗 View Message]({url})"
                context_lines.append(line)
        
        # Add resource context
        if resource_matches:
            context_lines.append("\n## Resource Library Context")
            for r in resource_matches:
                title = r.get("title", "Untitled Resource")
                description = r.get("description", "No description")
                url = r.get("resource_url", "")
                domain = r.get("domain", "")
                tag = r.get("tag", "Other")
                author = r.get("author", "Unknown")
                score = r.get("similarity_score", 0)
                
                line = f"**{title}** ({tag})\n"
                line += f"📝 {description}\n"
                line += f"🌐 Domain: {domain} | 👤 Shared by: {author} | 📊 Relevance: {score:.3f}"
                if url:
                    line += f"\n🔗 [View Resource]({url})"
                context_lines.append(line)

        instructions = (
            "You are a knowledgeable assistant with access to both Discord community conversations and a curated resource library.\n\n"
            "Use both the Discord messages and the resource context to provide a comprehensive answer. "
            "Combine insights from community discussions with relevant resources when helpful.\n"
            "Include author names, timestamps, URLs, and resource details when relevant.\n"
        )
        
        if as_json:
            instructions += "Return results as a JSON object with 'messages' and 'resources' arrays.\n"

        chat_messages = [
            {"role": "system", "content": "You are a comprehensive Discord community and resource assistant."},
            {"role": "user", "content": instructions + "\nContext:\n" + "\n\n".join(context_lines)
                                      + f"\n\nUser's question: {query}\n"}
        ]
        
        if as_json:
            chat_messages[-1]["content"] += "\n\nIMPORTANT: Return only valid JSON, no other text."
        
        answer = ai_client.chat_completion(
            chat_messages,
            temperature=0.7,
            max_tokens=2000
        )
        
        all_matches = {
            'messages': message_matches,
            'resources': resource_matches
        }
        
        return (answer, all_matches) if return_matches else answer

    except Exception as e:
        err = f"❌ Error during hybrid retrieval: {e}"
        return (err, {'messages': [], 'resources': []}) if return_matches else err

# Convenience aliases for the app
search_messages = get_top_k_matches
search_resources = get_top_k_resource_matches
discord_rag_search = get_answer
resource_rag_search = get_resource_answer
hybrid_rag_search = get_hybrid_answer

# ——— Resource Search Integration ———

class ResourceVectorStore:
    """Resource-specific vector store using FAISS with resource embeddings."""
    
    def __init__(self, 
                 index_path: str = "data/indices/resource_faiss_20250607_220423.index",
                 metadata_path: str = "data/indices/resource_faiss_20250607_220423_metadata.json",
                 model_name: str = "msmarco-distilbert-base-v4"):
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.model_name = model_name
        self._index = None
        self._metadata = None
        self._model = None
    
    def load(self):
        """Load FAISS index, metadata, and embedding model."""
        if self._index is None:
            logger.info(f"Loading resource FAISS index from: {self.index_path}")
            import faiss
            self._index = faiss.read_index(self.index_path)
            
            logger.info(f"Loading resource metadata from: {self.metadata_path}")
            with open(self.metadata_path, 'r') as f:
                data = json.load(f)
                self._metadata = data['metadata']
            
            logger.info(f"Loading sentence transformer model: {self.model_name}")
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            
            logger.info(f"Resource store loaded: {self._index.ntotal} vectors, {len(self._metadata)} metadata entries")
    
    def search(self, query_text: str, k: int = 5, filters: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Search for similar resources."""
        if self._index is None:
            self.load()
        
        # Create query embedding
        query_embedding = self._model.encode([query_text])
        
        # Search FAISS index
        scores, indices = self._index.search(query_embedding.astype('float32'), k * 2)
        
        # Return metadata for found resources
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self._metadata):
                meta = self._metadata[idx].copy()
                meta['similarity_score'] = float(score)
                
                # Apply filters if provided
                if filters and not self._apply_filters(meta, filters):
                    continue
                
                results.append(meta)
                
                if len(results) >= k:
                    break
        
        return results
    
    def _apply_filters(self, resource: Dict, filters: Dict) -> bool:
        """Apply optional filters to search results."""
        for key, value in filters.items():
            if key == 'tag' and resource.get('tag') != value:
                return False
            elif key == 'domain' and resource.get('domain') != value:
                return False
            elif key == 'is_tool' and resource.get('is_tool') != value:
                return False
            elif key == 'is_article' and resource.get('is_article') != value:
                return False
            elif key == 'is_youtube' and resource.get('is_youtube') != value:
                return False
            elif key == 'author' and resource.get('author') != value:
                return False
        return True

# Global resource store instance
_resource_store = None

def load_resource_vectorstore() -> ResourceVectorStore:
    """Load the resource vector store (singleton pattern)."""
    global _resource_store
    if _resource_store is None:
        _resource_store = ResourceVectorStore()
        _resource_store.load()
    return _resource_store

def get_top_k_resource_matches(
    query: str,
    k: int = 5,
    filters: Optional[Dict] = None
) -> List[Dict[str, Any]]:
    """
    Retrieve the top-k resource matches for 'query'.
    """
    try:
        logger.info(f"Searching for resource matches: query='{query}', k={k}")
        store = load_resource_vectorstore()
        
        # Get search results from resource vector store
        results = store.search(query, k, filters)
        
        logger.info(f"Found {len(results)} resource matches")
        return results
    except Exception as e:
        logger.error(f"Error in resource search: {e}")
        return []

def build_resource_prompt(
    resource_matches: List[Dict[str, Any]],
    question: str,
    as_json: bool = False
) -> List[Dict[str, str]]:
    """
    Construct a ChatML prompt with context from matching resources.
    """
    context_lines: List[str] = []
    for r in resource_matches:
        title = r.get("title", "Untitled Resource")
        description = r.get("description", "No description")
        url = r.get("resource_url", "")
        domain = r.get("domain", "")
        tag = r.get("tag", "Other")
        author = r.get("author", "Unknown")
        score = r.get("similarity_score", 0)
        
        line = f"**{title}** ({tag})\n"
        line += f"📝 {description}\n"
        line += f"🌐 Domain: {domain} | 👤 Shared by: {author} | 📊 Relevance: {score:.3f}"
        if url:
            line += f"\n🔗 [View Resource]({url})"
        context_lines.append(line)

    instructions = (
        "You are a knowledgeable assistant with access to a curated collection of AI, technology, and educational resources.\n\n"
        "Use the provided resource context to answer the user's question. "
        "Include resource titles, descriptions, URLs, and relevance scores when helpful.\n"
        "Focus on the most relevant resources and explain how they relate to the user's query.\n"
    )
    if as_json:
        instructions += "Return results as a JSON array with fields: resource_id, title, description, resource_url, tag, domain, author, similarity_score.\n"

    return [
        {"role": "system", "content": "You are a resource discovery assistant for AI and technology resources."},
        {"role": "user", "content": instructions + "\nResource Context:\n" + "\n\n".join(context_lines)
                                  + f"\n\nUser's question: {question}\n"}
    ]

def get_resource_answer(
    query: str,
    k: int = 5,
    as_json: bool = False,
    return_matches: bool = False,
    filters: Optional[Dict] = None
) -> Any:
    """
    Perform a resource-based RAG query: retrieve resource matches, build a prompt, and ask AI.
    Returns either a string (answer) or (answer, matches) if return_matches=True.
    """
    try:
        matches = get_top_k_resource_matches(query, k, filters)
        
        if not matches and not return_matches:
            return "⚠️ I couldn't find relevant resources. Try rephrasing your question or using different keywords."

        chat_messages = build_resource_prompt(matches, query, as_json)
        
        if as_json:
            # For JSON responses, add specific instruction
            chat_messages[-1]["content"] += "\n\nIMPORTANT: Return only valid JSON, no other text."
        
        answer = ai_client.chat_completion(
            chat_messages,
            temperature=0.7,
            max_tokens=1500
        )
        
        return (answer, matches) if return_matches else answer

    except Exception as e:
        err = f"❌ Error during resource retrieval: {e}"
        return (err, []) if return_matches else err

def get_hybrid_answer(
    query: str,
    k_messages: int = 3,
    k_resources: int = 3,
    as_json: bool = False,
    return_matches: bool = False,
    guild_id: Optional[int] = None,
    channel_id: Optional[int] = None,
    channel_name: Optional[str] = None,
    resource_filters: Optional[Dict] = None
) -> Any:
    """
    Perform a hybrid RAG query combining both Discord messages and resources.
    """
    try:
        # Get both message and resource matches
        message_matches = get_top_k_matches(
            query, k_messages,
            guild_id=guild_id,
            channel_id=channel_id,
            channel_name=channel_name
        )
        
        resource_matches = get_top_k_resource_matches(query, k_resources, resource_filters)
        
        if not message_matches and not resource_matches and not return_matches:
            return "⚠️ I couldn't find relevant messages or resources. Try rephrasing your question or being more specific."

        # Build hybrid context
        context_lines: List[str] = []
        
        # Add message context
        if message_matches:
            context_lines.append("## Discord Messages Context")
            for m in message_matches:
                author = m.get("author", {})
                author_name = author.get("display_name") or author.get("username") or "Unknown"
                ts = m.get("timestamp", "")
                ch_name = m.get("channel_name") or m.get("channel_id")
                content = m.get("content", "").replace("\n", " ")
                url = safe_jump_url(m)
                line = f"**{author_name}** (_{ts}_ in **#{ch_name}**):\n{content}"
                if url:
                    line += f"\n[🔗 View Message]({url})"
                context_lines.append(line)
        
        # Add resource context
        if resource_matches:
            context_lines.append("\n## Resource Library Context")
            for r in resource_matches:
                title = r.get("title", "Untitled Resource")
                description = r.get("description", "No description")
                url = r.get("resource_url", "")
                domain = r.get("domain", "")
                tag = r.get("tag", "Other")
                author = r.get("author", "Unknown")
                score = r.get("similarity_score", 0)
                
                line = f"**{title}** ({tag})\n"
                line += f"📝 {description}\n"
                line += f"🌐 Domain: {domain} | 👤 Shared by: {author} | 📊 Relevance: {score:.3f}"
                if url:
                    line += f"\n🔗 [View Resource]({url})"
                context_lines.append(line)

        instructions = (
            "You are a knowledgeable assistant with access to both Discord community conversations and a curated resource library.\n\n"
            "Use both the Discord messages and the resource context to provide a comprehensive answer. "
            "Combine insights from community discussions with relevant resources when helpful.\n"
            "Include author names, timestamps, URLs, and resource details when relevant.\n"
        )
        
        if as_json:
            instructions += "Return results as a JSON object with 'messages' and 'resources' arrays.\n"

        chat_messages = [
            {"role": "system", "content": "You are a comprehensive Discord community and resource assistant."},
            {"role": "user", "content": instructions + "\nContext:\n" + "\n\n".join(context_lines)
                                      + f"\n\nUser's question: {query}\n"}
        ]
        
        if as_json:
            chat_messages[-1]["content"] += "\n\nIMPORTANT: Return only valid JSON, no other text."
        
        answer = ai_client.chat_completion(
            chat_messages,
            temperature=0.7,
            max_tokens=2000
        )
        
        all_matches = {
            'messages': message_matches,
            'resources': resource_matches
        }
        
        return (answer, all_matches) if return_matches else answer

    except Exception as e:
        err = f"❌ Error during hybrid retrieval: {e}"
        return (err, {'messages': [], 'resources': []}) if return_matches else err

# ——— Enhanced Convenience Functions ———
