#!/usr/bin/env python3
"""
Search Quality Debug Script

Tests the search functionality to understand why queries return no results.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agentic.vectorstore.persistent_store import PersistentVectorStore
from agentic.services.channel_resolver import ChannelResolver

async def test_basic_search():
    """Test basic search without filters"""
    print("🔍 Testing Basic Search (No Filters)")
    print("=" * 50)
    
    config = {
        "collection_name": "discord_messages",
        "persist_directory": "./data/chromadb",
        "embedding_model": "text-embedding-3-small"
    }
    
    vector_store = PersistentVectorStore(config)
    
    # Test simple search
    results = await vector_store.similarity_search("AI", k=5)
    print(f"Search for 'AI': {len(results)} results")
    
    if results:
        for i, result in enumerate(results[:3]):
            print(f"  {i+1}. Channel: {result.get('channel_name', 'Unknown')}")
            print(f"      Content: {result.get('content', '')[:100]}...")
            print(f"      Score: {result.get('score', 0):.3f}")
    
    # Test another search
    results = await vector_store.similarity_search("agents", k=5)
    print(f"\nSearch for 'agents': {len(results)} results")
    
    return len(results) > 0

async def test_channel_filtering():
    """Test channel filtering specifically"""
    print("\n🏷️ Testing Channel Filtering")
    print("=" * 50)
    
    config = {
        "collection_name": "discord_messages",
        "persist_directory": "./data/chromadb",
        "embedding_model": "text-embedding-3-small"
    }
    
    vector_store = PersistentVectorStore(config)
    
    # Get available channels first
    print("📋 Available channels in vector store:")
    
    # Get sample of documents to see channel names
    if hasattr(vector_store, 'collection') and vector_store.collection:
        sample_results = vector_store.collection.get(
            limit=100,
            include=["metadatas"]
        )
        
        if sample_results and sample_results.get("metadatas"):
            channels = set()
            for meta in sample_results["metadatas"]:
                channel_name = meta.get("channel_name", "Unknown")
                if channel_name != "Unknown":
                    channels.add(channel_name)
            
            print(f"Found {len(channels)} unique channels:")
            for channel in sorted(list(channels)[:10]):
                print(f"   • {channel}")
            
            # Test filtering with a known channel
            if channels:
                test_channel = list(channels)[0]
                print(f"\n🔍 Testing search in channel: {test_channel}")
                
                # Test with filter
                results = await vector_store.similarity_search(
                    "discussion", 
                    k=5, 
                    filters={"channel_name": test_channel}
                )
                print(f"Results with channel filter: {len(results)}")
                
                if results:
                    for i, result in enumerate(results[:2]):
                        print(f"  {i+1}. Content: {result.get('content', '')[:100]}...")
                
                return len(results) > 0
    
    return False

async def test_channel_resolution():
    """Test channel ID to name resolution"""
    print("\n🔗 Testing Channel Resolution")
    print("=" * 50)
    
    # Test some channel IDs that were mentioned in the bot logs
    test_channel_ids = [
        "1371647370911154228",  # From the first query
        "1365732945859444767"   # From the second query
    ]
    
    print("Testing with channel IDs from Discord queries...")
    
    for channel_id in test_channel_ids:
        print(f"Channel ID {channel_id}:")
        
        # Check if this channel exists in vector store
        vector_config = {
            "collection_name": "discord_messages",
            "persist_directory": "./data/chromadb",
            "embedding_model": "text-embedding-3-small"
        }
        vector_store = PersistentVectorStore(vector_config)
        
        # Try searching with the channel ID as name
        results = await vector_store.filter_search(
            filters={"channel_name": channel_id},
            k=5
        )
        print(f"   Messages with channel_name='{channel_id}': {len(results)}")
        
        # Try searching with channel_id field
        results2 = await vector_store.filter_search(
            filters={"channel_id": channel_id},
            k=5
        )
        print(f"   Messages with channel_id='{channel_id}': {len(results2)}")

async def test_raw_collection_data():
    """Check raw collection data to understand the structure"""
    print("\n📊 Testing Raw Collection Data")
    print("=" * 50)
    
    config = {
        "collection_name": "discord_messages",
        "persist_directory": "./data/chromadb",
        "embedding_model": "text-embedding-3-small"
    }
    
    vector_store = PersistentVectorStore(config)
    
    if hasattr(vector_store, 'collection') and vector_store.collection:
        # Get a sample of documents with all metadata
        sample_results = vector_store.collection.get(
            limit=5,
            include=["documents", "metadatas", "ids"]
        )
        
        if sample_results:
            documents = sample_results.get("documents", [])
            metadatas = sample_results.get("metadatas", [])
            ids = sample_results.get("ids", [])
            
            print(f"Sample of {len(documents)} documents:")
            
            for i in range(min(3, len(documents))):
                print(f"\n📄 Document {i+1}:")
                print(f"   ID: {ids[i] if i < len(ids) else 'N/A'}")
                print(f"   Content: {documents[i][:100] if i < len(documents) else 'N/A'}...")
                
                if i < len(metadatas):
                    meta = metadatas[i]
                    print(f"   Channel ID: {meta.get('channel_id', 'N/A')}")
                    print(f"   Channel Name: {meta.get('channel_name', 'N/A')}")
                    print(f"   Author: {meta.get('author_username', 'N/A')}")
                    print(f"   Timestamp: {meta.get('timestamp', 'N/A')}")

async def test_search_with_debug():
    """Test search with detailed debugging"""
    print("\n🐛 Testing Search with Debug Info")
    print("=" * 50)
    
    config = {
        "collection_name": "discord_messages", 
        "persist_directory": "./data/chromadb",
        "embedding_model": "text-embedding-3-small"
    }
    
    vector_store = PersistentVectorStore(config)
    
    # Test the exact query pattern from the bot logs
    test_queries = [
        "what discussions have taken place",
        "fetch me the last 5 messages",
        "AI agents",
        "workshop"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Testing query: '{query}'")
        
        # Test without filters
        results = await vector_store.similarity_search(query, k=5)
        print(f"   No filters: {len(results)} results")
        
        # Test with a known good channel
        results_filtered = await vector_store.similarity_search(
            query, 
            k=5, 
            filters={"channel_name": "📚ai-philosophy-ethics"}
        )
        print(f"   With channel filter: {len(results_filtered)} results")
        
        if results and not results_filtered:
            print("   ⚠️ Results without filter but none with filter - filter issue!")
        elif not results:
            print("   ⚠️ No results even without filters - embedding/search issue!")

async def main():
    """Main debug function"""
    print("🔍 Search Quality Debug Script")
    print("=" * 60)
    
    success_count = 0
    
    # Test 1: Basic search
    if await test_basic_search():
        success_count += 1
        print("✅ Basic search working")
    else:
        print("❌ Basic search failing")
    
    # Test 2: Channel filtering
    if await test_channel_filtering():
        success_count += 1
        print("✅ Channel filtering working")
    else:
        print("❌ Channel filtering failing")
    
    # Test 3: Channel resolution
    await test_channel_resolution()
    
    # Test 4: Raw data inspection
    await test_raw_collection_data()
    
    # Test 5: Search debugging
    await test_search_with_debug()
    
    print(f"\n{'=' * 60}")
    print(f"📊 Debug Summary: {success_count}/2 basic tests passed")
    
    if success_count < 2:
        print("❌ Issues found with search functionality")
        print("💡 Recommendations:")
        print("   1. Check channel name mapping")
        print("   2. Verify embedding quality")
        print("   3. Test filter logic")
    else:
        print("✅ Basic search functionality working")
        print("💡 Issue might be in Discord query processing")

if __name__ == "__main__":
    asyncio.run(main())
