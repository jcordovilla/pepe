"""
Channel Resolution Service

Provides channel name-to-ID resolution and mapping functionality
to make the system resilient to channel name changes.
"""

import logging
import sqlite3
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import re
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ChannelInfo:
    """Channel information container."""
    id: str
    name: str
    guild_id: str
    message_count: int
    aliases: List[str]


class ChannelResolver:
    """
    Service for resolving channel names to IDs and managing channel mappings.
    
    This service provides:
    1. Channel name-to-ID resolution
    2. Fuzzy matching for partial channel names
    3. Alias support for common variations
    4. Caching for performance
    """
    
    def __init__(self, chromadb_path: str = "./data/chromadb/chroma.sqlite3"):
        self.chromadb_path = chromadb_path
        self._channel_cache: Dict[str, ChannelInfo] = {}
        self._name_to_id_cache: Dict[str, str] = {}
        self._alias_map: Dict[str, str] = {}
        self._initialize_aliases()
        
    def _initialize_aliases(self):
        """Initialize common channel name aliases."""
        # Common aliases for channel names
        self._alias_map = {
            # Agent-related channels
            "agent-ops": ["🦾agent-ops", "agentops", "agent ops"],
            "agent-dev": ["🤖agent-dev", "agentdev", "agent dev"],
            "netarch-agents": ["🏛🤖netarch-agents", "netarch agents"],
            "agent-ops-resources": ["🗂agent-ops-resources", "agent ops resources"],
            
            # General channels
            "general-chat": ["🏘general-chat", "general chat"],
            "admin-general-chat": ["🏘admin-general-chat", "admin general chat"],
            "introductions": ["👋introductions"],
            "admin-introductions": ["👋admin-introductions", "admin introductions"],
            
            # Learning channels
            "non-coders-learning": ["❌💻non-coders-learning", "non coders learning"],
            "ai-philosophy-ethics": ["📚ai-philosophy-ethics", "ai philosophy ethics"],
            "ai-practical-applications": ["🛠ai-practical-applications", "ai practical applications"],
            
            # Support channels
            "q-and-a-questions": ["❓q-and-a-questions", "q and a questions", "questions"],
            "buddy-support": ["❓buddy-support", "buddy support"],
            
            # Community channels
            "community": ["🌎-community"],
            "showcase": ["showcase"],
            "feedback-submissions": ["📥feedback-submissions", "feedback submissions"],
        }
        
    def refresh_cache(self) -> bool:
        """Refresh the channel cache from ChromaDB."""
        try:
            if not Path(self.chromadb_path).exists():
                logger.error(f"ChromaDB not found at {self.chromadb_path}")
                return False
                
            conn = sqlite3.connect(self.chromadb_path)
            
            # Get all channels with their metadata
            query = '''
            SELECT DISTINCT 
                m1.string_value as channel_name,
                m2.string_value as channel_id,
                m3.string_value as guild_id,
                COUNT(*) as message_count
            FROM embedding_metadata m1
            JOIN embedding_metadata m2 ON m1.id = m2.id
            LEFT JOIN embedding_metadata m3 ON m1.id = m3.id AND m3.key = 'guild_id'
            WHERE m1.key = 'channel_name' 
              AND m2.key = 'channel_id'
            GROUP BY m1.string_value, m2.string_value, m3.string_value
            ORDER BY message_count DESC
            '''
            
            results = conn.execute(query).fetchall()
            conn.close()
            
            # Clear caches
            self._channel_cache.clear()
            self._name_to_id_cache.clear()
            
            # Build cache
            for row in results:
                channel_name, channel_id, guild_id, message_count = row
                
                # Create channel info
                aliases = self._get_aliases_for_channel(channel_name)
                channel_info = ChannelInfo(
                    id=channel_id,
                    name=channel_name,
                    guild_id=guild_id or "",
                    message_count=message_count,
                    aliases=aliases
                )
                
                # Cache by ID and name
                self._channel_cache[channel_id] = channel_info
                self._name_to_id_cache[channel_name.lower()] = channel_id
                
                # Cache aliases
                for alias in aliases:
                    self._name_to_id_cache[alias.lower()] = channel_id
                    
            logger.info(f"Refreshed channel cache with {len(self._channel_cache)} channels")
            return True
            
        except Exception as e:
            logger.error(f"Error refreshing channel cache: {e}")
            return False
    
    def _get_aliases_for_channel(self, channel_name: str) -> List[str]:
        """Get aliases for a given channel name."""
        aliases = []
        
        # Check direct aliases
        for base_name, alias_list in self._alias_map.items():
            if channel_name in alias_list:
                aliases.extend([a for a in alias_list if a != channel_name])
                aliases.append(base_name)
                break
        
        # Add common variations
        clean_name = self._clean_channel_name(channel_name)
        if clean_name != channel_name:
            aliases.append(clean_name)
            
        # Add hyphenated and spaced versions
        if "-" in clean_name:
            aliases.append(clean_name.replace("-", " "))
        if " " in clean_name:
            aliases.append(clean_name.replace(" ", "-"))
            
        return list(set(aliases))  # Remove duplicates
    
    def _clean_channel_name(self, name: str) -> str:
        """Clean channel name by removing emojis and special characters."""
        # Remove leading emojis and special characters
        clean = re.sub(r'^[^\w\s-]+', '', name)
        # Remove extra whitespace
        clean = re.sub(r'\s+', ' ', clean.strip())
        return clean
    
    def resolve_channel_name(self, name: str, guild_id: Optional[str] = None) -> Optional[str]:
        """
        Resolve a channel name to its ID.
        
        Args:
            name: Channel name (with or without #)
            guild_id: Optional guild ID to scope the search
            
        Returns:
            Channel ID if found, None otherwise
        """
        if not name:
            return None
            
        # Ensure cache is populated
        if not self._channel_cache:
            self.refresh_cache()
        
        # Clean the input name
        clean_name = name.lstrip('#').strip().lower()
        
        # Direct lookup
        if clean_name in self._name_to_id_cache:
            channel_id = self._name_to_id_cache[clean_name]
            
            # If guild_id specified, verify it matches
            if guild_id:
                channel_info = self._channel_cache.get(channel_id)
                if channel_info and channel_info.guild_id != guild_id:
                    return None
                    
            return channel_id
        
        # Fuzzy matching - find channels that contain the search term
        candidates = []
        for cached_name, channel_id in self._name_to_id_cache.items():
            channel_info = self._channel_cache.get(channel_id)
            
            # Skip if guild doesn't match
            if guild_id and channel_info and channel_info.guild_id != guild_id:
                continue
                
            # Check if search term is contained in channel name
            if clean_name in cached_name:
                candidates.append((channel_id, channel_info))
        
        # If we have candidates, prefer the one with the most messages
        if candidates:
            best_candidate = max(candidates, key=lambda x: x[1].message_count if x[1] else 0)
            return best_candidate[0]
        
        return None
    
    def get_channel_info(self, channel_id: str) -> Optional[ChannelInfo]:
        """Get channel information by ID."""
        if not self._channel_cache:
            self.refresh_cache()
        return self._channel_cache.get(channel_id)
    
    def list_channels(self, guild_id: Optional[str] = None, pattern: Optional[str] = None) -> List[ChannelInfo]:
        """
        List channels, optionally filtered by guild and/or pattern.
        
        Args:
            guild_id: Optional guild ID filter
            pattern: Optional name pattern filter
            
        Returns:
            List of channel information
        """
        if not self._channel_cache:
            self.refresh_cache()
        
        channels = list(self._channel_cache.values())
        
        # Filter by guild
        if guild_id:
            channels = [c for c in channels if c.guild_id == guild_id]
        
        # Filter by pattern
        if pattern:
            pattern_lower = pattern.lower()
            channels = [c for c in channels 
                       if pattern_lower in c.name.lower() 
                       or any(pattern_lower in alias.lower() for alias in c.aliases)]
        
        # Sort by message count (descending)
        channels.sort(key=lambda x: x.message_count, reverse=True)
        return channels
    
    def get_similar_channels(self, name: str, limit: int = 5) -> List[Tuple[ChannelInfo, float]]:
        """
        Get channels similar to the given name with similarity scores.
        
        Args:
            name: Channel name to find similar channels for
            limit: Maximum number of results
            
        Returns:
            List of (ChannelInfo, similarity_score) tuples
        """
        if not self._channel_cache:
            self.refresh_cache()
        
        clean_name = name.lstrip('#').strip().lower()
        candidates = []
        
        for channel in self._channel_cache.values():
            # Calculate similarity score
            score = self._calculate_similarity(clean_name, channel.name.lower())
            
            # Also check aliases
            for alias in channel.aliases:
                alias_score = self._calculate_similarity(clean_name, alias.lower())
                score = max(score, alias_score)
            
            if score > 0:
                candidates.append((channel, score))
        
        # Sort by score (descending) and limit results
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:limit]
    
    def _calculate_similarity(self, name1: str, name2: str) -> float:
        """Calculate similarity score between two channel names."""
        # Exact match
        if name1 == name2:
            return 1.0
        
        # Contains match
        if name1 in name2 or name2 in name1:
            return 0.8
        
        # Word overlap
        words1 = set(name1.split())
        words2 = set(name2.split())
        
        if words1 and words2:
            overlap = len(words1.intersection(words2))
            total = len(words1.union(words2))
            return overlap / total if total > 0 else 0.0
        
        return 0.0
    
    def resolve_channel_id_to_name(self, channel_id: str) -> Optional[str]:
        """
        Resolve a channel ID to its full name (with emojis) from the vector store.
        
        This is critical for Discord mentions like <#1371647370911154228> which need
        to be mapped to actual channel names in the vector store.
        """
        if not channel_id:
            return None
        
        # Check cache first
        if channel_id in self._channel_cache:
            return self._channel_cache[channel_id].name
        
        # Query the vector store directly for this channel ID
        try:
            from agentic.vectorstore.persistent_store import PersistentVectorStore
            config = {
                "collection_name": "discord_messages",
                "persist_directory": "./data/chromadb",
                "embedding_model": os.getenv("EMBEDDING_MODEL", "msmarco-distilbert-base-v4"),
            "embedding_type": os.getenv("EMBEDDING_TYPE", "sentence_transformers")
            }
            vector_store = PersistentVectorStore(config)
            
            if vector_store.collection:
                # Get metadata for this channel ID
                results = vector_store.collection.get(
                    where={"channel_id": channel_id},
                    limit=1,
                    include=["metadatas"]
                )
                
                if results and results.get("metadatas"):
                    metadata = results["metadatas"][0]
                    channel_name = metadata.get("channel_name", "")
                    if channel_name:
                        # Cache the result
                        channel_info = ChannelInfo(
                            id=channel_id,
                            name=channel_name,
                            guild_id=metadata.get("guild_id", ""),
                            message_count=1,
                            aliases=[]
                        )
                        self._channel_cache[channel_id] = channel_info
                        self._name_to_id_cache[channel_name] = channel_id
                        logger.info(f"Resolved channel ID {channel_id} -> '{channel_name}'")
                        return channel_name
                        
        except Exception as e:
            logger.error(f"Error resolving channel ID {channel_id}: {e}")
        
        return None
