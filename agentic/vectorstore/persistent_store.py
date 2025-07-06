"""
Persistent Vector Store using ChromaDB

Replaces the real-time FAISS indexing with a persistent, scalable vector store.
"""

import os
import logging
import shutil
from typing import Dict, List, Any, Optional, Union
import asyncio
from datetime import datetime
import json

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import numpy as np

from ..cache.smart_cache import SmartCache

logger = logging.getLogger(__name__)


class PersistentVectorStore:
    """
    ChromaDB-based persistent vector store for Discord messages.
    
    Provides semantic search, keyword search, and filtering capabilities
    with persistent storage and optimized performance.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Configuration
        self.collection_name = config.get("collection_name", "discord_messages")
        self.persist_directory = config.get("persist_directory", "./data/chromadb")
        self.embedding_model = config.get("embedding_model", os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"))
        self.chunk_size = config.get("chunk_size", 1000)
        self.batch_size = config.get("batch_size", 100)
        
        # Initialize collection as None, will be set in _init_chromadb
        self.collection: Optional[Any] = None
        self.client: Optional[Any] = None
        self.embedding_function: Optional[Any] = None
        
        # Performance tracking - Initialize before ChromaDB
        self.stats: Dict[str, Any] = {
            "total_documents": 0,
            "total_searches": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "index_updates": 0
        }
        
        # Initialize ChromaDB
        self._init_chromadb()
        
        # Initialize cache
        self.cache = SmartCache(config.get("cache", {}))
        
        logger.info(f"PersistentVectorStore initialized with collection: {self.collection_name}")
    
    def _init_chromadb(self):
        """Initialize ChromaDB client and collection."""
        try:
            # Ensure persist directory exists
            os.makedirs(self.persist_directory, exist_ok=True)
            
            # Initialize ChromaDB client with persistence
            # Set environment variables to disable telemetry completely
            os.environ["ANONYMIZED_TELEMETRY"] = "False"
            os.environ["CHROMA_TELEMETRY_CAPTURE"] = "False"
            os.environ["CHROMA_TELEMETRY_ENABLED"] = "False"
            
            try:
                self.client = chromadb.PersistentClient(
                    path=self.persist_directory,
                    settings=Settings(
                        anonymized_telemetry=False,
                        allow_reset=False  # FIXED: Prevent accidental resets that cause data loss
                    )
                )
            except Exception as client_error:
                # Fallback without settings if there are compatibility issues
                logger.warning(f"Error creating client with settings, trying fallback: {client_error}")
                self.client = chromadb.PersistentClient(path=self.persist_directory)
            
            # Initialize embedding function - use single OpenAI API key
            api_key = os.getenv("OPENAI_API_KEY")
            
            if not api_key or api_key == "test-key-for-testing":
                # For testing scenarios, create a simple mock embedding function
                logger.warning("Using test/mock configuration - creating collection with default embedding function")
                self.embedding_function = embedding_functions.DefaultEmbeddingFunction()
            else:
                self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
                    api_key=api_key,
                    model_name=self.embedding_model
                )
            
            # Get or create collection
            try:
                self.collection = self.client.get_collection(
                    name=self.collection_name,
                    embedding_function=self.embedding_function
                )
                logger.info(f"Loaded existing collection: {self.collection_name}")
            except ValueError as embedding_error:
                # Handle embedding function mismatch - recreate collection
                if "Embedding function name mismatch" in str(embedding_error):
                    logger.warning(f"Embedding function mismatch detected, recreating collection")
                    try:
                        # Delete the existing collection
                        self.client.delete_collection(name=self.collection_name)
                        logger.info(f"Deleted collection with mismatched embedding function")
                        # Create new collection with correct embedding function
                        self.collection = self.client.create_collection(
                            name=self.collection_name,
                            embedding_function=self.embedding_function,
                            metadata={"description": "Discord messages vector store"}
                        )
                        logger.info(f"Created new collection with correct embedding function: {self.collection_name}")
                    except Exception as recreate_error:
                        logger.error(f"Failed to recreate collection: {recreate_error}")
                        raise recreate_error
                elif "does not exist" in str(embedding_error):
                    logger.warning(f"Collection does not exist, creating new collection: {self.collection_name}")
                    try:
                        self.collection = self.client.create_collection(
                            name=self.collection_name,
                            embedding_function=self.embedding_function,
                            metadata={"description": "Discord messages vector store"}
                        )
                        logger.info(f"Created new collection: {self.collection_name}")
                    except Exception as create_error:
                        logger.error(f"Failed to create collection: {create_error}")
                        raise create_error
                else:
                    raise embedding_error
            except Exception as get_error:
                try:
                    self.collection = self.client.create_collection(
                        name=self.collection_name,
                        embedding_function=self.embedding_function,
                        metadata={"description": "Discord messages vector store"}
                    )
                    logger.info(f"Created new collection: {self.collection_name}")
                except Exception as create_error:
                    # If creation fails because collection exists, try to get it again
                    if "already exists" in str(create_error).lower():
                        try:
                            self.collection = self.client.get_collection(
                                name=self.collection_name,
                                embedding_function=self.embedding_function
                            )
                            logger.info(f"Retrieved existing collection after creation conflict: {self.collection_name}")
                        except Exception:
                            logger.error(f"Failed to get collection after creation conflict: {create_error}")
                            raise create_error
                    else:
                        logger.error(f"Failed to create collection: {create_error}")
                        raise create_error
            
            # Update stats
            if self.collection:
                self.stats["total_documents"] = self.collection.count()
            else:
                logger.warning("Collection not initialized, stats will be incomplete")
            
        except Exception as e:
            logger.error(f"Error initializing ChromaDB: {e}")
            raise
    
    def _ensure_collection(self) -> bool:
        """Ensure collection is available for operations"""
        if self.collection is None:
            logger.error("ChromaDB collection not initialized")
            return False
        return True
    
    async def add_messages(self, messages: List[Dict[str, Any]]) -> bool:
        """
        Add or update messages in the vector store.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            True if successful, False otherwise
        """
        if not self._ensure_collection():
            return False
            
        try:
            if not messages:
                return True
            
            # Prepare documents for insertion
            documents = []
            metadatas = []
            ids = []
            
            for msg in messages:
                try:
                    # Ensure msg is a dictionary
                    if not isinstance(msg, dict):
                        logger.warning(f"Skipping non-dict message: {type(msg)}")
                        continue
                    
                    # Create document text - handle empty content gracefully
                    content = msg.get("content", "").strip()
                    
                    # If no text content, create a description from other elements
                    if not content:
                        content_parts = []
                        
                        # Add attachment info
                        attachments = msg.get("attachments", [])
                        if self._safe_len(attachments) > 0:
                            attachment_names = []
                            for att in attachments[:3]:  # Max 3 attachment names
                                if isinstance(att, dict) and att.get("filename"):
                                    attachment_names.append(att["filename"])
                            if attachment_names:
                                content_parts.append(f"Attachments: {', '.join(attachment_names)}")
                        
                        # Add embed info  
                        embeds = msg.get("embeds", [])
                        if self._safe_len(embeds) > 0:
                            content_parts.append(f"Rich embed content ({self._safe_len(embeds)} embeds)")
                        
                        # Add system message info
                        message_type = msg.get("type", "")
                        if message_type and str(message_type) != "MessageType.default":
                            content_parts.append(f"System message: {message_type}")
                        
                        # Create synthetic content for indexing
                        if content_parts:
                            content = " | ".join(content_parts)
                        else:
                            # Skip truly empty messages
                            continue
                    
                    # Create unique ID
                    message_id = str(msg.get("message_id", ""))
                    if not message_id:
                        continue
                
                    # Prepare comprehensive metadata with all available Discord fields
                    # Handle reactions - can be list or int
                    reactions = msg.get("reactions", [])
                    if isinstance(reactions, list):
                        total_reactions = sum(r.get("count", 0) for r in reactions) if reactions else 0
                        reaction_emojis = [r.get("emoji", "") for r in reactions] if reactions else []
                    elif isinstance(reactions, int):
                        total_reactions = reactions
                        reaction_emojis = []
                    else:
                        total_reactions = 0
                        reaction_emojis = []
                    
                    # Author information (comprehensive) - handle string author fields
                    author = msg.get("author", {})
                    if isinstance(author, str):
                        # Handle case where author is a string (username only)
                        author = {"username": author, "display_name": author}
                    elif not isinstance(author, dict):
                        author = {}
                    
                    # Handle mentions - can be list or int
                    mentions = msg.get("mentions", [])
                    if isinstance(mentions, list):
                        mentioned_user_ids = [str(m.get("id", "")) for m in mentions] if mentions else []
                        mentioned_user_names = [m.get("display_name", m.get("username", "")) for m in mentions] if mentions else []
                        mention_count = len(mentions)
                    elif isinstance(mentions, int):
                        mentioned_user_ids = []
                        mentioned_user_names = []
                        mention_count = mentions
                    else:
                        mentioned_user_ids = []
                        mentioned_user_names = []
                        mention_count = 0
                    
                    # Handle attachments - can be list or int
                    attachments = msg.get("attachments", [])
                    if isinstance(attachments, list):
                        attachment_urls = [a.get("url", "") for a in attachments] if attachments else []
                        attachment_filenames = [a.get("filename", "") for a in attachments] if attachments else []
                        attachment_types = [a.get("content_type", "") for a in attachments] if attachments else []
                        attachment_sizes = [a.get("size", 0) for a in attachments] if attachments else []
                        attachment_count = len(attachments)
                    elif isinstance(attachments, int):
                        attachment_urls = []
                        attachment_filenames = []
                        attachment_types = []
                        attachment_sizes = []
                        attachment_count = attachments
                    else:
                        attachment_urls = []
                        attachment_filenames = []
                        attachment_types = []
                        attachment_sizes = []
                        attachment_count = 0
                    
                    # Embed information - handle both int and list formats
                    embeds = msg.get("embeds", [])
                    if isinstance(embeds, int):
                        embed_count = embeds  # Discord sometimes stores embed count as int
                    elif isinstance(embeds, list):
                        embed_count = len(embeds)
                    else:
                        embed_count = 0
                    
                    # Content analysis
                    import re
                    has_code_block = bool(re.search(r'```|`[^`]+`', content))
                    has_urls = bool(re.search(r'https?://', content))
                    word_count = len(content.split()) if content else 0
                    
                    # Reference information (for replies)
                    reference = msg.get("reference")
                    reply_to_message_id = str(reference.get("message_id", "")) if reference else ""
                    
                    metadata = {
                        # Core message fields
                        "message_id": message_id,
                        "channel_id": str(msg.get("channel_id", "")),
                        "channel_name": msg.get("channel_name", ""),
                        "guild_id": str(msg.get("guild_id", "")),
                        "guild_name": msg.get("guild_name", ""),
                        "timestamp": msg.get("timestamp", ""),
                        "jump_url": msg.get("jump_url", ""),
                        
                        # Author fields (comprehensive)
                        "author_id": str(author.get("id", "")),
                        "author_username": author.get("username", ""),
                        "author_display_name": author.get("display_name", ""),
                        "author_discriminator": author.get("discriminator", ""),
                        "author_bot": author.get("bot", False),
                        
                        # Message metadata
                        "message_type": str(msg.get("type", "default")),
                        "pinned": msg.get("pinned", False),
                        "reply_to_message_id": reply_to_message_id,
                        
                        # Content analysis
                        "content_length": len(content),
                        "word_count": word_count,
                        "has_code_block": has_code_block,
                        "has_urls": has_urls,
                        
                        # Reactions (enhanced)
                        "total_reactions": total_reactions,
                        "reaction_emojis": ",".join(reaction_emojis),
                        "reaction_details": json.dumps(reactions) if isinstance(reactions, list) else str(reactions),
                        
                        # Mentions (comprehensive)
                        "mentioned_user_ids": ",".join(mentioned_user_ids),
                        "mentioned_user_names": ",".join(mentioned_user_names),
                        "mention_count": mention_count,
                        
                        # Attachments (comprehensive)
                        "attachment_urls": ",".join(attachment_urls),
                        "attachment_filenames": ",".join(attachment_filenames), 
                        "attachment_types": ",".join(attachment_types),
                        "attachment_sizes": ",".join(str(s) for s in attachment_sizes),
                        "attachment_count": attachment_count,
                        
                        # Embeds
                        "embed_count": embed_count,
                        "has_embeds": embed_count > 0,
                        
                        # System metadata
                        "indexed_at": datetime.utcnow().isoformat(),
                        "processing_version": "2.0",  # Track data structure version
                    }
                    
                    documents.append(content)
                    metadatas.append(metadata)
                    ids.append(message_id)
                    
                except Exception as msg_error:
                    # Handle both dict and non-dict messages in error reporting
                    if isinstance(msg, dict):
                        message_id = msg.get('message_id', 'unknown')
                    else:
                        message_id = f"string-{type(msg).__name__}"
                    logger.warning(f"Error processing message {message_id}: {msg_error}")
                    continue
            
            if not documents:
                logger.warning("No valid documents to add")
                return True
            
            # Add to collection in batches
            for i in range(0, len(documents), self.batch_size):
                batch_docs = documents[i:i + self.batch_size]
                batch_metas = metadatas[i:i + self.batch_size]
                batch_ids = ids[i:i + self.batch_size]
                
                try:
                    assert self.collection is not None, "Collection not initialized"
                    await asyncio.to_thread(
                        self.collection.upsert,
                        documents=batch_docs,
                        metadatas=batch_metas,
                        ids=batch_ids
                    )
                except Exception as e:
                    logger.error(f"Error adding batch {i//self.batch_size + 1}: {e}")
                    continue
            
            # Update stats
            assert self.collection is not None, "Collection not initialized"
            self.stats["total_documents"] = await asyncio.to_thread(self.collection.count)
            self.stats["index_updates"] += 1
            
            logger.info(f"Added {len(documents)} messages to vector store")
            return True
            
        except Exception as e:
            logger.error(f"Error adding messages to vector store: {e}")
            return False
    
    async def similarity_search(
        self,
        query: str,
        k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        min_score: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Perform semantic similarity search.
        
        Args:
            query: Search query
            k: Number of results to return
            filters: Optional metadata filters
            min_score: Minimum similarity score
            
        Returns:
            List of search results with metadata
        """
        try:
            # Check cache first
            cache_key = f"similarity:{hash(query)}:{k}:{hash(str(filters))}"
            cached_results = await self.cache.get(cache_key)
            
            if cached_results:
                self.stats["cache_hits"] += 1
                return cached_results
            
            # Build where clause from filters
            where_clause = self._build_where_clause(filters) if filters else None
            
            # Perform similarity search
            assert self.collection is not None, "Collection not initialized"
            results = await asyncio.to_thread(
                self.collection.query,
                query_texts=[query],
                n_results=k,
                where=where_clause,
                include=["documents", "metadatas", "distances"]
            )
            
            # Process results
            processed_results = self._process_search_results(results, min_score)
            
            # Cache results
            await self.cache.set(cache_key, processed_results, ttl=1800)  # 30 min TTL
            
            self.stats["total_searches"] += 1
            logger.info(f"Similarity search found {len(processed_results)} results")
            
            return processed_results
            
        except Exception as e:
            logger.error(f"Error in similarity search: {e}")
            return []
    
    async def keyword_search(
        self,
        keywords: List[str],
        k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform keyword-based search.
        
        Args:
            keywords: List of keywords to search
            k: Number of results to return
            filters: Optional metadata filters
            
        Returns:
            List of search results
        """
        try:
            # Check cache
            cache_key = f"keyword:{hash(str(keywords))}:{k}:{hash(str(filters))}"
            cached_results = await self.cache.get(cache_key)
            
            if cached_results:
                self.stats["cache_hits"] += 1
                return cached_results
            
            # Build where clause
            where_clause = self._build_where_clause(filters) if filters else None
            
            # For keyword search, we'll use a simple text matching approach
            # This is a limitation of ChromaDB - it doesn't have built-in keyword search
            # We'll use similarity search with the keywords joined
            keyword_query = " ".join(keywords)
            
            assert self.collection is not None, "Collection not initialized"
            results = await asyncio.to_thread(
                self.collection.query,
                query_texts=[keyword_query],
                n_results=k * 2,  # Get more results to filter
                where=where_clause,
                include=["documents", "metadatas", "distances"]
            )
            
            # Filter results that actually contain the keywords
            filtered_results = self._filter_by_keywords(results, keywords)
            
            # Limit to requested number
            filtered_results = filtered_results[:k]
            
            # Cache results
            await self.cache.set(cache_key, filtered_results, ttl=1800)
            
            self.stats["total_searches"] += 1
            logger.info(f"Keyword search found {len(filtered_results)} results")
            
            return filtered_results
            
        except Exception as e:
            logger.error(f"Error in keyword search: {e}")
            return []
    
    async def filter_search(
        self,
        filters: Dict[str, Any],
        k: int = 10,
        sort_by: str = "timestamp"
    ) -> List[Dict[str, Any]]:
        """
        Search messages using only filters (no text query).
        
        Args:
            filters: Metadata filters
            k: Number of results to return
            sort_by: Field to sort by
            
        Returns:
            List of filtered results
        """
        try:
            # Check cache
            cache_key = f"filter:{hash(str(filters))}:{k}:{sort_by}"
            cached_results = await self.cache.get(cache_key)
            
            if cached_results:
                self.stats["cache_hits"] += 1
                return cached_results
            
            # Build where clause
            where_clause = self._build_where_clause(filters) if filters else None
            
            # Get all matching documents (ChromaDB doesn't support sorting directly)
            # We'll retrieve more than needed and sort in Python
            assert self.collection is not None, "Collection not initialized"
            results = await asyncio.to_thread(
                self.collection.get,
                where=where_clause,
                limit=k * 5,  # Get more to allow for sorting
                include=["documents", "metadatas"]
            )
            
            # Process and sort results
            processed_results = []
            
            if results and results.get("documents"):
                documents = results["documents"]
                metadatas = results["metadatas"]
                
                for i, doc in enumerate(documents):
                    metadata = metadatas[i] if i < len(metadatas) else {}
                    
                    result = {
                        "content": doc,
                        "score": 1.0,  # No similarity score for filter-only search
                        **metadata
                    }
                    
                    processed_results.append(result)
                
                # Sort results
                if sort_by in ["timestamp", "indexed_at"]:
                    processed_results.sort(
                        key=lambda x: x.get(sort_by, ""),
                        reverse=True
                    )
                elif sort_by == "content_length":
                    processed_results.sort(
                        key=lambda x: x.get(sort_by, 0),
                        reverse=True
                    )
                
                # Limit to requested number
                processed_results = processed_results[:k]
            
            # Cache results
            await self.cache.set(cache_key, processed_results, ttl=1800)
            
            self.stats["total_searches"] += 1
            logger.info(f"Filter search found {len(processed_results)} results")
            
            return processed_results
            
        except Exception as e:
            logger.error(f"Error in filter search: {e}")
            return []
    
    async def delete_messages(self, message_ids: List[str]) -> bool:
        """
        Delete messages from the vector store.
        
        Args:
            message_ids: List of message IDs to delete
            
        Returns:
            True if successful
        """
        if not self._ensure_collection():
            return False
            
        try:
            if not message_ids:
                return True
            
            # Delete in batches
            for i in range(0, len(message_ids), self.batch_size):
                batch_ids = message_ids[i:i + self.batch_size]
                
                try:
                    assert self.collection is not None, "Collection not initialized"
                    await asyncio.to_thread(self.collection.delete, ids=batch_ids)
                except Exception as e:
                    logger.error(f"Error deleting batch {i//self.batch_size + 1}: {e}")
                    continue
            
            # Update stats
            assert self.collection is not None, "Collection not initialized"
            self.stats["total_documents"] = await asyncio.to_thread(self.collection.count)
            
            logger.info(f"Deleted {len(message_ids)} messages from vector store")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting messages: {e}")
            return False
    
    async def update_message(self, message_id: str, updated_data: Dict[str, Any]) -> bool:
        """
        Update an existing message in the vector store.
        
        Args:
            message_id: ID of message to update
            updated_data: New message data
            
        Returns:
            True if successful
        """
        if not self._ensure_collection():
            return False
            
        try:
            content = updated_data.get("content", "").strip()
            if not content:
                return False
            
            # Prepare metadata
            reactions = updated_data.get("reactions", [])
            total_reactions = sum(r.get("count", 0) for r in reactions) if reactions else 0
            reaction_emojis = [r.get("emoji", "") for r in reactions] if reactions else []
            
            metadata = {
                "message_id": message_id,
                "channel_id": str(updated_data.get("channel_id", "")),
                "channel_name": updated_data.get("channel_name", ""),
                "guild_id": str(updated_data.get("guild_id", "")),
                "author_id": str(updated_data.get("author", {}).get("id", "")),
                "author_username": updated_data.get("author", {}).get("username", ""),
                "author_display_name": updated_data.get("author", {}).get("display_name", ""),
                "timestamp": updated_data.get("timestamp", ""),
                "jump_url": updated_data.get("jump_url", ""),
                "content_length": len(content),
                "total_reactions": total_reactions,
                "reaction_emojis": ",".join(reaction_emojis),
                "indexed_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat()
            }
            
            # Update using upsert
            assert self.collection is not None, "Collection not initialized"
            await asyncio.to_thread(
                self.collection.upsert,
                documents=[content],
                metadatas=[metadata],
                ids=[message_id]
            )
            
            logger.info(f"Updated message {message_id} in vector store")
            return True
            
        except Exception as e:
            logger.error(f"Error updating message {message_id}: {e}")
            return False
    
    def _build_where_clause(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """Build ChromaDB where clause from filters."""
        where_clause = {}
        
        for key, value in filters.items():
            if isinstance(value, dict):
                # Handle operators like $gt, $lt, $in, etc.
                where_clause[key] = value
            elif isinstance(value, list):
                # Convert list to $in operator
                where_clause[key] = {"$in": value}
            else:
                # Simple equality
                where_clause[key] = value
        
        return where_clause
    
    def _process_search_results(
        self,
        results: Dict[str, Any],
        min_score: float = 0.0
    ) -> List[Dict[str, Any]]:
        """Process raw ChromaDB search results."""
        processed = []
        
        if not results or not results.get("documents"):
            return processed
        
        documents = results["documents"][0] if results["documents"] else []
        metadatas = results["metadatas"][0] if results["metadatas"] else []
        distances = results["distances"][0] if results["distances"] else []
        
        for i, doc in enumerate(documents):
            # Convert distance to similarity score (ChromaDB returns distances)
            distance = distances[i] if i < len(distances) else 1.0
            score = max(0, 1 - distance)  # Convert distance to similarity
            
            if score < min_score:
                continue
            
            metadata = metadatas[i] if i < len(metadatas) else {}
            
            # Restructure author information to match expected format - with comprehensive extraction
            author_info = {}
            
            # Method 1: Direct author fields in metadata root
            if "author_username" in metadata:
                author_info["username"] = metadata["author_username"]
            if "author_id" in metadata:
                author_info["id"] = metadata["author_id"]
            if "author_display_name" in metadata:
                author_info["display_name"] = metadata["author_display_name"]
                
            # Method 2: Nested author object with direct fields
            if isinstance(metadata.get("author"), dict):
                if not author_info.get("username") and "username" in metadata["author"]:
                    author_info["username"] = metadata["author"]["username"]
                if not author_info.get("id") and "id" in metadata["author"]:
                    author_info["id"] = metadata["author"]["id"]
                if not author_info.get("display_name") and "display_name" in metadata["author"]:
                    author_info["display_name"] = metadata["author"]["display_name"]
                # Some systems store it as name instead of username
                if not author_info.get("username") and "name" in metadata["author"]:
                    author_info["username"] = metadata["author"]["name"]
            
            # Method 3: Serialized author object (from JSON string)
            if not author_info.get("username") and isinstance(metadata.get("author"), str):
                try:
                    author_data = json.loads(metadata["author"])
                    if isinstance(author_data, dict):
                        if "username" in author_data:
                            author_info["username"] = author_data["username"]
                        if "id" in author_data:
                            author_info["id"] = author_data["id"]
                except (json.JSONDecodeError, TypeError):
                    # Not a valid JSON string, will be handled by fallbacks
                    pass
                
            # Fallback 1: String author field as username
            if not author_info.get("username") and isinstance(metadata.get("author"), str):
                author_info["username"] = metadata["author"]
                
            # Fallback 2: Ensure at least some username is present (avoid "Unknown")
            if not author_info.get("username"):
                # Try to use any other identifying information from metadata
                possible_username_fields = ["user", "user_name", "username", "name", "sender"]
                for field in possible_username_fields:
                    if field in metadata and metadata[field]:
                        author_info["username"] = metadata[field]
                        break
                        
            # Final fallback to prevent "Unknown" display
            if not author_info.get("username"):
                # Use message ID or a timestamp as last resort
                if "message_id" in metadata:
                    author_info["username"] = f"User-{metadata['message_id'][-6:]}"
                else:
                    # Absolute last resort
                    author_info["username"] = f"User-{i}"

            # Ensure we're not passing back an empty author object
            if not author_info:
                author_info = {"username": f"User-{i}"}
            
            # Create result with proper author structure - ensuring we have a valid author field
            result = {
                "content": doc,
                "score": score,
                "author": author_info,
                # Include other metadata but exclude the individual author fields
                **{k: v for k, v in metadata.items() 
                   if not k.startswith("author_") and k != "author"}
            }
            
            processed.append(result)
        
        return processed
    
    def _filter_by_keywords(
        self,
        results: Dict[str, Any],
        keywords: List[str]
    ) -> List[Dict[str, Any]]:
        """Filter search results to only include those containing keywords."""
        processed = self._process_search_results(results)
        filtered = []
        
        for result in processed:
            content = result.get("content", "").lower()
            
            # Check if any keyword is present
            if any(keyword.lower() in content for keyword in keywords):
                # Calculate keyword score
                keyword_matches = sum(1 for kw in keywords if kw.lower() in content)
                keyword_score = keyword_matches / len(keywords)
                
                # Combine with similarity score
                result["keyword_score"] = keyword_score
                result["combined_score"] = (result["score"] * 0.6) + (keyword_score * 0.4)
                
                filtered.append(result)
        
        # Sort by combined score
        filtered.sort(key=lambda x: x.get("combined_score", 0), reverse=True)
        
        return filtered
    
    async def reaction_search(
        self,
        reaction: str = "",
        k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        sort_by: str = "total_reactions"
    ) -> List[Dict[str, Any]]:
        """
        Search messages based on reaction data.
        
        Args:
            reaction: Specific emoji/reaction to search for (optional)
            k: Number of results to return
            filters: Optional metadata filters
            sort_by: Field to sort by ("total_reactions" or "timestamp")
            
        Returns:
            List of search results sorted by reaction count
        """
        if not self._ensure_collection():
            return []
            
        try:
            # Check cache
            cache_key = f"reaction:{hash(reaction)}:{k}:{hash(str(filters))}:{sort_by}"
            cached_results = await self.cache.get(cache_key)
            
            if cached_results:
                self.stats["cache_hits"] += 1
                return cached_results
            
            # Build where clause from filters
            where_clause = self._build_where_clause(filters) if filters else {}
            
            # Add reaction-specific filters
            if reaction:
                # Since ChromaDB doesn't support $contains or $like for string pattern matching,
                # we'll retrieve all messages with reactions and filter in Python
                if where_clause and "$and" in where_clause:
                    where_clause["$and"].append({
                        "total_reactions": {"$gt": 0}
                    })
                elif where_clause:
                    # Convert existing filters to $and format
                    existing_filters = where_clause.copy()
                    where_clause = {
                        "$and": [
                            existing_filters,
                            {"total_reactions": {"$gt": 0}}
                        ]
                    }
                else:
                    where_clause = {
                        "total_reactions": {"$gt": 0}
                    }
            else:
                # Search for messages with any reactions (total_reactions > 0)
                if where_clause and "$and" in where_clause:
                    where_clause["$and"].append({
                        "total_reactions": {"$gt": 0}
                    })
                elif where_clause:
                    # Convert existing filters to $and format
                    existing_filters = where_clause.copy()
                    where_clause = {
                        "$and": [
                            existing_filters,
                            {"total_reactions": {"$gt": 0}}
                        ]
                    }
                else:
                    where_clause = {
                        "total_reactions": {"$gt": 0}
                    }
            
            # Get messages with reactions - fetch more to allow for sorting
            assert self.collection is not None, "Collection not initialized"
            results = await asyncio.to_thread(
                self.collection.get,
                where=where_clause,
                limit=k * 3,  # Get more results to allow for proper sorting
                include=["documents", "metadatas"]
            )
            
            # Process and sort results
            processed_results = []
            
            if results and results.get("documents"):
                documents = results["documents"]
                metadatas = results["metadatas"]
                
                for i, doc in enumerate(documents):
                    metadata = metadatas[i] if i < len(metadatas) else {}
                    
                    # Only include messages with reactions
                    total_reactions = metadata.get("total_reactions", 0)
                    if isinstance(total_reactions, str):
                        try:
                            total_reactions = int(total_reactions)
                        except (ValueError, TypeError):
                            total_reactions = 0
                    
                    if total_reactions > 0:
                        # If searching for specific reaction, filter by emoji
                        if reaction:
                            reaction_emojis = metadata.get("reaction_emojis", "")
                            if reaction not in reaction_emojis:
                                continue  # Skip this message if it doesn't have the specific emoji
                        
                        result = {
                            "content": doc,
                            "score": 1.0,  # No similarity score for reaction search
                            "reaction_count": total_reactions,
                            **metadata
                        }
                        processed_results.append(result)
                
                # Sort results by reaction count (descending) or timestamp
                if sort_by == "total_reactions":
                    processed_results.sort(
                        key=lambda x: x.get("reaction_count", 0),
                        reverse=True
                    )
                elif sort_by == "timestamp":
                    processed_results.sort(
                        key=lambda x: x.get("timestamp", ""),
                        reverse=True
                    )
                
                # Limit to requested number
                processed_results = processed_results[:k]
            
            # Cache results
            await self.cache.set(cache_key, processed_results, ttl=1800)
            
            self.stats["total_searches"] += 1
            self.stats["cache_misses"] += 1
            
            logger.info(f"Reaction search found {len(processed_results)} results")
            return processed_results
            
        except Exception as e:
            logger.error(f"Error in reaction search: {e}")
            return []
    
    async def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store collection.
        
        Returns:
            Dictionary with collection statistics
        """
        try:
            # Get basic count
            assert self.collection is not None, "Collection not initialized"
            total_count = await asyncio.to_thread(self.collection.count)
            
            # Get sample of documents to analyze
            sample_results = await asyncio.to_thread(
                self.collection.get,
                limit=min(1000, total_count),
                include=["metadatas"]
            )
            
            stats = {
                "total_documents": total_count,
                "collection_name": self.collection_name,
                "embedding_model": self.embedding_model,
                "last_updated": datetime.utcnow().isoformat()
            }
            
            # Analyze metadata if we have documents
            if sample_results and sample_results.get("metadatas"):
                metadatas = sample_results["metadatas"]
                
                # Count by channel
                channels = {}
                guilds = {}
                authors = {}
                content_lengths = []
                timestamps = []
                
                for metadata in metadatas:
                    # Channel analysis
                    channel_id = metadata.get("channel_id", "unknown")
                    channel_name = metadata.get("channel_name", "Unknown")
                    channels[channel_id] = {
                        "name": channel_name,
                        "count": channels.get(channel_id, {}).get("count", 0) + 1
                    }
                    
                    # Guild analysis
                    guild_id = metadata.get("guild_id", "unknown")
                    guilds[guild_id] = guilds.get(guild_id, 0) + 1
                    
                    # Author analysis
                    author_id = metadata.get("author_id", "unknown")
                    author_username = metadata.get("author_username", "Unknown")
                    authors[author_id] = {
                        "username": author_username,
                        "count": authors.get(author_id, {}).get("count", 0) + 1
                    }
                    
                    # Content length analysis
                    content_length = metadata.get("content_length", 0)
                    if isinstance(content_length, (int, float)):
                        content_lengths.append(content_length)
                    
                    # Timestamp analysis
                    timestamp = metadata.get("timestamp")
                    if timestamp:
                        timestamps.append(timestamp)
                
                # Calculate statistics
                if content_lengths:
                    stats["content_stats"] = {
                        "avg_length": sum(content_lengths) / len(content_lengths),
                        "min_length": min(content_lengths),
                        "max_length": max(content_lengths),
                        "total_tokens": sum(content_lengths)
                    }
                
                # Top channels
                stats["top_channels"] = sorted(
                    [{"id": k, **v} for k, v in channels.items()],
                    key=lambda x: x["count"],
                    reverse=True
                )[:10]
                
                # Top authors
                stats["top_authors"] = sorted(
                    [{"id": k, **v} for k, v in authors.items()],
                    key=lambda x: x["count"],
                    reverse=True
                )[:10]
                
                # Guild distribution
                stats["guild_distribution"] = dict(sorted(
                    guilds.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:10])
                
                # Date range
                if timestamps:
                    sorted_timestamps = sorted(timestamps)
                    stats["date_range"] = {
                        "earliest": sorted_timestamps[0],
                        "latest": sorted_timestamps[-1]
                    }
            
            # Add performance stats
            stats["performance_stats"] = self.stats.copy()
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {
                "total_documents": 0,
                "collection_name": self.collection_name,
                "error": str(e)
            }
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on the vector store.
        
        Returns:
            Health status information
        """
        health = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "checks": {}
        }
        
        try:
            # Check ChromaDB connection
            try:
                assert self.collection is not None, "Collection not initialized"
                count = await asyncio.to_thread(self.collection.count)
                health["checks"]["chromadb"] = {
                    "status": "healthy",
                    "document_count": count
                }
            except Exception as e:
                health["checks"]["chromadb"] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
                health["status"] = "unhealthy"
            
            # Check cache system
            try:
                test_key = "health_check_test"
                await self.cache.set(test_key, "test_value", ttl=60)
                cached_value = await self.cache.get(test_key)
                await self.cache.delete(test_key)
                
                if cached_value == "test_value":
                    health["checks"]["cache"] = {"status": "healthy"}
                else:
                    health["checks"]["cache"] = {
                        "status": "degraded",
                        "message": "Cache not working properly"
                    }
            except Exception as e:
                health["checks"]["cache"] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
            
            # Check disk space
            try:
                disk_usage = shutil.disk_usage(self.persist_directory)
                free_space_gb = disk_usage.free / (1024**3)
                
                if free_space_gb > 1.0:  # More than 1GB free
                    health["checks"]["disk_space"] = {
                        "status": "healthy",
                        "free_space_gb": round(free_space_gb, 2)
                    }
                else:
                    health["checks"]["disk_space"] = {
                        "status": "warning",
                        "free_space_gb": round(free_space_gb, 2),
                        "message": "Low disk space"
                    }
                    if health["status"] == "healthy":
                        health["status"] = "degraded"
                        
            except Exception as e:
                health["checks"]["disk_space"] = {
                    "status": "unknown",
                    "error": str(e)
                }
            
            return health
            
        except Exception as e:
            logger.error(f"Error during health check: {e}")
            return {
                "status": "unhealthy",
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e)
            }
    
    async def optimize(self) -> bool:
        """
        Optimize the vector store performance.
        
        Returns:
            True if optimization successful
        """
        try:
            logger.info("Starting vector store optimization...")
            
            # Clean up cache
            cleaned_entries = await self.cache.cleanup_expired()
            logger.info(f"Cleaned up {cleaned_entries} expired cache entries")
            
            # ChromaDB doesn't have explicit optimization commands
            # But we can perform some maintenance tasks
            
            # Get current collection stats
            assert self.collection is not None, "Collection not initialized"
            current_count = await asyncio.to_thread(self.collection.count)
            logger.info(f"Current collection has {current_count} documents")
            
            # Update internal stats
            self.stats["total_documents"] = current_count
            
            logger.info("Vector store optimization completed")
            return True
            
        except Exception as e:
            logger.error(f"Error during optimization: {e}")
            return False
    
    async def close(self):
        """Close the vector store and cleanup resources."""
        try:
            # Close cache
            await self.cache.close()
            
            # ChromaDB client doesn't need explicit closing
            # but we can clear references
            self.collection = None
            self.client = None
            
            logger.info("PersistentVectorStore closed successfully")
            
        except Exception as e:
            logger.error(f"Error closing vector store: {e}")
    
    def _safe_len(self, obj):
        """Safely get length of an object, handling cases where it might be an int or other non-iterable"""
        try:
            if obj is None:
                return 0
            if isinstance(obj, (list, tuple, str, dict)):
                return len(obj)
            if isinstance(obj, int):
                return obj if obj > 0 else 0
            return 0
        except:
            return 0
