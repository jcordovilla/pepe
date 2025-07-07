#!/usr/bin/env python3
"""
Test and Fix Channel Resolution

The bot receives channel IDs from Discord mentions but needs to map them
to the full emoji channel names stored in the vector store.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agentic.services.channel_resolver import ChannelResolver
from agentic.vectorstore.persistent_store import PersistentVectorStore

async def test_channel_resolution():
    """Test channel resolution functionality"""
    print("🔍 Testing Channel Resolution")
    print("=" * 60)
    
    try:
        # Initialize services
        resolver = ChannelResolver("./data/chromadb/chroma.sqlite3")
        
        # First, let's see what channels are available
        print("📋 Refreshing channel cache...")
        success = resolver.refresh_cache()
        if success:
            print("✅ Channel cache refreshed")
        else:
            print("❌ Failed to refresh channel cache")
        
        # Get vector store to see actual channel names
        config = {
            "collection_name": "discord_messages",
            "persist_directory": "./data/chromadb",
            "embedding_model": "text-embedding-3-small"
        }
        
        vector_store = PersistentVectorStore(config)
        
        # Get actual channel names from vector store
        print("\n📊 Getting channel names from vector store...")
        if vector_store.collection:
            sample_data = vector_store.collection.get(limit=1000, include=["metadatas"])
            if sample_data and sample_data.get("metadatas"):
                channel_mapping = {}
                for metadata in sample_data["metadatas"]:
                    channel_id = metadata.get("channel_id", "")
                    channel_name = metadata.get("channel_name", "")
                    if channel_id and channel_name:
                        channel_mapping[channel_id] = channel_name
                
                print(f"   Found {len(channel_mapping)} unique channels:")
                for channel_id, channel_name in sorted(channel_mapping.items(), key=lambda x: x[1]):
                    print(f"   {channel_id} -> '{channel_name}'")
                
                # Test specific channel IDs from Discord
                test_channel_ids = [
                    "1371647370911154228",  # From the Discord query
                    "1365732945859444767",  # From the Discord query
                    "1353448986408779877",  # ai-philosophy-ethics
                ]
                
                print("\n🔍 Testing specific channel ID resolution:")
                for channel_id in test_channel_ids:
                    if channel_id in channel_mapping:
                        channel_name = channel_mapping[channel_id]
                        print(f"   ✅ {channel_id} -> '{channel_name}'")
                        
                        # Test search with this channel
                        print(f"      Testing search in this channel...")
                        results = await vector_store.similarity_search(
                            "discussion", 
                            k=3, 
                            filters={"channel_name": channel_name}
                        )
                        print(f"      Found {len(results)} results")
                    else:
                        print(f"   ❌ {channel_id} -> Not found in vector store")
                
                # Test the resolver's functionality
                print("\n🔧 Testing channel resolver methods:")
                resolver._channel_cache = {}
                resolver._name_to_id_cache = {}
                
                # Manually populate cache with real data
                for channel_id, channel_name in channel_mapping.items():
                    from agentic.services.channel_resolver import ChannelInfo
                    channel_info = ChannelInfo(
                        id=channel_id,
                        name=channel_name,
                        guild_id="1353058864810950737",
                        message_count=1,
                        aliases=[]
                    )
                    resolver._channel_cache[channel_id] = channel_info
                    resolver._name_to_id_cache[channel_name] = channel_id
                
                # Test resolution
                for channel_id in test_channel_ids:
                    resolved_name = resolver.resolve_channel_id_to_name(channel_id)
                    print(f"   Channel ID {channel_id} -> '{resolved_name}'")
        
        print("\n✅ Channel resolution testing completed")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

def create_channel_mapping_fix():
    """Create a fix for the channel mapping issue"""
    print("\n🔧 Creating Channel Mapping Fix")
    print("=" * 60)
    
    # This would be a method to add to the ChannelResolver class
    fix_code = '''
def resolve_channel_id_to_name(self, channel_id: str) -> Optional[str]:
    """Resolve a channel ID to its full name (with emojis) from the vector store."""
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
            "embedding_model": "text-embedding-3-small"
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
                    from agentic.services.channel_resolver import ChannelInfo
                    channel_info = ChannelInfo(
                        id=channel_id,
                        name=channel_name,
                        guild_id=metadata.get("guild_id", ""),
                        message_count=1,
                        aliases=[]
                    )
                    self._channel_cache[channel_id] = channel_info
                    self._name_to_id_cache[channel_name] = channel_id
                    return channel_name
                    
    except Exception as e:
        logger.error(f"Error resolving channel ID {channel_id}: {e}")
    
    return None
'''
    
    print("📝 Suggested fix method to add to ChannelResolver:")
    print(fix_code)
    
    return fix_code

if __name__ == "__main__":
    asyncio.run(test_channel_resolution())
    create_channel_mapping_fix()
