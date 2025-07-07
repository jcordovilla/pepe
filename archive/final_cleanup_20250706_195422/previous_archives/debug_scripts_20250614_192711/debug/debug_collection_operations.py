#!/usr/bin/env python3
"""
Debug script to test collection operations step by step.
"""

import asyncio
import os
import sys
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agentic.vectorstore.persistent_store import PersistentVectorStore


async def debug_collection_operations():
    """Debug collection operations to find the exact issue."""
    print("🔍 Debugging Collection Operations")
    print("=" * 50)
    
    # Set test environment
    os.environ["CHROMA_OPENAI_API_KEY"] = "test-key-for-testing"
    os.environ["OPENAI_API_KEY"] = "test-key-for-testing"
    
    # Initialize vector store
    config = {
        "collection_name": "debug_test_collection",
        "persist_directory": "./data/debug_vectorstore",
        "embedding_model": "text-embedding-3-small",
        "batch_size": 2,
        "cache": {"type": "memory", "ttl": 3600}
    }
    
    try:
        vector_store = PersistentVectorStore(config)
        print("✅ Vector store initialized")
        
        # Check collection state
        print(f"Collection: {vector_store.collection}")
        print(f"Embedding function: {vector_store.embedding_function}")
        print(f"Collection count: {vector_store.collection.count()}")
        
        # Try to add a single simple message
        print("\n🧪 Testing single message addition...")
        
        simple_message = {
            "message_id": "debug_001",
            "content": "This is a simple test message for debugging",
            "channel_id": "debug_channel",
            "channel_name": "debug",
            "guild_id": "debug_guild",
            "author": {"id": "debug_author", "username": "debugger"},
            "timestamp": datetime.now().isoformat(),
            "jump_url": "https://discord.com/debug/001",
            "reactions": [
                {"emoji": "🔍", "count": 1},
                {"emoji": "🐛", "count": 2}
            ]
        }
        
        # Test the individual steps of add_messages
        print("Step 1: Processing message data...")
        content = simple_message.get("content", "")
        if not content.strip():
            print("❌ Empty content")
            return
        print(f"Content: {content[:50]}...")
        
        print("Step 2: Creating metadata...")
        reactions = simple_message.get("reactions", [])
        total_reactions = sum(r.get("count", 0) for r in reactions)
        reaction_emojis = [r.get("emoji", "") for r in reactions]
        
        metadata = {
            "message_id": str(simple_message.get("message_id", "")),
            "channel_id": str(simple_message.get("channel_id", "")),
            "channel_name": simple_message.get("channel_name", ""),
            "guild_id": str(simple_message.get("guild_id", "")),
            "author_id": str(simple_message.get("author", {}).get("id", "")),
            "author_username": simple_message.get("author", {}).get("username", ""),
            "timestamp": simple_message.get("timestamp", ""),
            "jump_url": simple_message.get("jump_url", ""),
            "content_length": len(content),
            "total_reactions": total_reactions,
            "reaction_emojis": ",".join(reaction_emojis),
            "indexed_at": datetime.utcnow().isoformat()
        }
        
        print(f"Metadata: {metadata}")
        
        print("Step 3: Testing collection upsert directly...")
        try:
            vector_store.collection.upsert(
                documents=[content],
                metadatas=[metadata],
                ids=[simple_message["message_id"]]
            )
            print("✅ Direct upsert successful")
        except Exception as e:
            print(f"❌ Direct upsert failed: {e}")
            import traceback
            traceback.print_exc()
            return
        
        print("Step 4: Checking collection count...")
        count_after = vector_store.collection.count()
        print(f"Collection count after upsert: {count_after}")
        
        print("Step 5: Testing through add_messages method...")
        success = await vector_store.add_messages([simple_message])
        print(f"add_messages result: {success}")
        
        final_count = vector_store.collection.count()
        print(f"Final collection count: {final_count}")
        
        # Test search
        print("\n🔍 Testing search...")
        try:
            results = await vector_store.reaction_search(k=1)
            print(f"Search results: {len(results)} found")
            if results:
                print(f"First result: {results[0]}")
        except Exception as e:
            print(f"Search failed: {e}")
        
        await vector_store.close()
        print("✅ Debug completed")
        
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(debug_collection_operations())
