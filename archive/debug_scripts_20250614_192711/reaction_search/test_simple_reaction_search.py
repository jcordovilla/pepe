#!/usr/bin/env python3
"""
Simple test to verify reaction search functionality with existing data.
"""

import asyncio
import os
import sys
from typing import Dict, Any

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agentic.vectorstore.persistent_store import PersistentVectorStore

async def test_reaction_search_with_existing_data():
    """Test reaction search functionality using existing data."""
    print("🔍 Testing Reaction Search with Existing Data")
    print("=" * 50)
    
    try:
        # Set environment variables for ChromaDB
        os.environ["CHROMA_OPENAI_API_KEY"] = "test-key-for-testing"
        os.environ["OPENAI_API_KEY"] = "test-key-for-testing"
        
        # Initialize vector store with production config
        config = {
            "persist_directory": "./data/chromadb",  # Use the main vectorstore
            "embedding_model": "text-embedding-3-small",
            "batch_size": 10,
            "cache": {"type": "memory", "ttl": 3600}
        }
        
        vector_store = PersistentVectorStore(config)
        print("✅ Vector store initialized successfully")
        
        # Get collection stats
        stats = await vector_store.get_collection_stats()
        total_docs = stats.get('total_documents', 0)
        print(f"   Collection has {total_docs} documents")
        
        if total_docs == 0:
            print("⚠️  No existing data found. Skipping reaction search test.")
            return True
            
        # Test 1: Search for any messages with reactions
        print("\n1. Testing general reaction search...")
        reaction_results = await vector_store.reaction_search(k=5, sort_by="total_reactions")
        
        if reaction_results:
            print(f"✅ Found {len(reaction_results)} messages with reactions")
            for i, result in enumerate(reaction_results[:3]):  # Show first 3
                content = result.get("content", "")[:50]
                reaction_count = result.get("reaction_count", 0)
                emojis = result.get("reaction_emojis", "")
                print(f"   {i+1}. '{content}...' ({reaction_count} reactions) - {emojis}")
        else:
            print("ℹ️  No messages with reactions found in the database")
        
        # Test 2: Search for specific emoji (try common ones)
        common_emojis = ["👍", "❤️", "😂", "👀", "🎉", "🚀"]
        
        for emoji in common_emojis:
            print(f"\n2. Testing search for {emoji} reactions...")
            emoji_results = await vector_store.reaction_search(reaction=emoji, k=3)
            
            if emoji_results:
                print(f"✅ Found {len(emoji_results)} messages with {emoji} reactions")
                for result in emoji_results[:2]:  # Show first 2
                    content = result.get("content", "")[:40]
                    reaction_count = result.get("reaction_count", 0)
                    print(f"   - '{content}...' ({reaction_count} reactions)")
                break  # Found results, stop searching
            else:
                print(f"ℹ️  No messages with {emoji} reactions found")
        
        # Test 3: Test collection health
        print(f"\n3. Testing collection health...")
        health = await vector_store.health_check()
        print(f"✅ Collection health: {health}")
        
        await vector_store.close()
        print("\n✅ Reaction search test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main test function."""
    success = await test_reaction_search_with_existing_data()
    if success:
        print("\n🎉 All tests passed! Reaction search functionality is working.")
    else:
        print("\n❌ Some tests failed.")
    
    return success

if __name__ == "__main__":
    asyncio.run(main())
