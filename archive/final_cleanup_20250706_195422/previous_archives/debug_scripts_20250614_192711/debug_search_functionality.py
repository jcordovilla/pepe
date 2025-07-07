#!/usr/bin/env python3
"""
Debug script to test actual search functionality and channel filtering.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agentic.vectorstore.persistent_store import PersistentVectorStore

async def test_vector_search():
    """Test vector store search functionality"""
    print("🔍 Testing Vector Store Search Functionality")
    print("=" * 60)
    
    try:
        # Initialize vector store
        config = {
            "collection_name": "discord_messages",
            "persist_directory": "./data/chromadb",
            "embedding_model": "text-embedding-3-small"
        }
        
        vector_store = PersistentVectorStore(config)
        
        # Get basic stats
        stats = await vector_store.get_collection_stats()
        print(f"📊 Total documents: {stats.get('total_documents', 0)}")
        
        # Test 1: Basic search without filters
        print("\n🔍 Test 1: Basic semantic search (no filters)")
        results = await vector_store.similarity_search("machine learning", k=5)
        print(f"   Found {len(results)} results")
        for i, result in enumerate(results[:3], 1):
            content = result.get('content', '')[:100] + "..." if len(result.get('content', '')) > 100 else result.get('content', '')
            channel = result.get('channel_name', 'Unknown')
            print(f"   {i}. Channel: {channel}")
            print(f"      Content: {content}")
        
        # Test 2: Channel filtering with exact channel name
        print("\n🔍 Test 2: Search with channel filter (exact name)")
        channel_name = "📚ai-philosophy-ethics"  # From the stats above
        results = await vector_store.similarity_search(
            "AI", 
            k=5, 
            filters={"channel_name": channel_name}
        )
        print(f"   Searching in channel: {channel_name}")
        print(f"   Found {len(results)} results")
        for i, result in enumerate(results[:2], 1):
            content = result.get('content', '')[:100] + "..." if len(result.get('content', '')) > 100 else result.get('content', '')
            print(f"   {i}. {content}")
        
        # Test 3: Filter search
        print("\n🔍 Test 3: Filter search (recent messages)")
        results = await vector_store.filter_search(
            filters={"channel_name": channel_name},
            k=5,
            sort_by="timestamp"
        )
        print(f"   Found {len(results)} results with filter search")
        
        # Test 4: Channel name variations
        print("\n🔍 Test 4: Testing channel name variations")
        test_channels = [
            "ai-philosophy-ethics",  # Without emoji
            "📚ai-philosophy-ethics",  # With emoji
            "1353448986408779877",  # Channel ID
        ]
        
        for channel in test_channels:
            results = await vector_store.similarity_search(
                "discussion", 
                k=3, 
                filters={"channel_name": channel}
            )
            print(f"   '{channel}': {len(results)} results")
        
        # Test 5: List actual channel names in the vector store
        print("\n📋 Test 5: Actual channel names in vector store")
        if vector_store.collection:
            # Get a sample of metadata to see channel names
            sample_data = vector_store.collection.get(limit=50, include=["metadatas"])
            if sample_data and sample_data.get("metadatas"):
                unique_channels = set()
                for metadata in sample_data["metadatas"]:
                    channel_name = metadata.get("channel_name", "Unknown")
                    if channel_name != "Unknown":
                        unique_channels.add(channel_name)
                
                print(f"   Found {len(unique_channels)} unique channels in sample:")
                for channel in sorted(unique_channels):
                    print(f"   - '{channel}'")
        
        print("\n✅ Vector search test completed")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_vector_search())
