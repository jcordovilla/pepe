#!/usr/bin/env python3
"""
Debug script to test batch operations specifically.
"""

import asyncio
import os
import sys
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agentic.vectorstore.persistent_store import PersistentVectorStore


async def debug_batch_operations():
    """Debug batch operations to find the exact issue."""
    print("🔍 Debugging Batch Operations")
    print("=" * 50)
    
    # Set test environment
    os.environ["CHROMA_OPENAI_API_KEY"] = "test-key-for-testing"
    os.environ["OPENAI_API_KEY"] = "test-key-for-testing"
    
    # Initialize vector store with small batch size
    config = {
        "collection_name": "debug_batch_collection",
        "persist_directory": "./data/debug_batch_vectorstore",
        "embedding_model": "text-embedding-3-small",
        "batch_size": 2,  # Small batch size for easier debugging
        "cache": {"type": "memory", "ttl": 3600}
    }
    
    try:
        vector_store = PersistentVectorStore(config)
        print("✅ Vector store initialized")
        
        # Create test messages exactly like in production test
        test_messages = [
            {
                "message_id": "msg_announcement_001",
                "content": "🎉 Big announcement! We're launching our new feature next week! Get ready for some amazing improvements to the Discord experience!",
                "channel_id": "channel_announcements",
                "channel_name": "announcements",
                "guild_id": "guild_main",
                "author": {"id": "admin_123", "username": "admin"},
                "timestamp": (datetime.now() - timedelta(days=1)).isoformat(),
                "jump_url": "https://discord.com/channels/guild/announcements/001",
                "reactions": [
                    {"emoji": "🎉", "count": 45},
                    {"emoji": "🚀", "count": 38},
                    {"emoji": "❤️", "count": 25},
                    {"emoji": "👍", "count": 52},
                    {"emoji": "🔥", "count": 33}
                ]
            },
            {
                "message_id": "msg_meme_002",
                "content": "When you finally fix that bug that's been haunting you for weeks 😂",
                "channel_id": "channel_general",
                "channel_name": "general",
                "guild_id": "guild_main",
                "author": {"id": "dev_456", "username": "developer"},
                "timestamp": (datetime.now() - timedelta(hours=6)).isoformat(),
                "jump_url": "https://discord.com/channels/guild/general/002",
                "reactions": [
                    {"emoji": "😂", "count": 89},
                    {"emoji": "💯", "count": 34},
                    {"emoji": "👀", "count": 18},
                    {"emoji": "🤣", "count": 67}
                ]
            },
            {
                "message_id": "msg_help_003",
                "content": "Can someone help me with this Python error? I've been stuck for hours...",
                "channel_id": "channel_help",
                "channel_name": "help",
                "guild_id": "guild_main",
                "author": {"id": "user_789", "username": "newbie"},
                "timestamp": (datetime.now() - timedelta(hours=3)).isoformat(),
                "jump_url": "https://discord.com/channels/guild/help/003",
                "reactions": [
                    {"emoji": "❤️", "count": 12},
                    {"emoji": "🤝", "count": 8},
                    {"emoji": "💪", "count": 5}
                ]
            }
        ]
        
        print(f"Testing with {len(test_messages)} messages (batch size: {config['batch_size']})")
        
        # Test add_messages with debug info
        print("\n🧪 Testing batch add_messages...")
        success = await vector_store.add_messages(test_messages)
        print(f"Batch add result: {success}")
        
        # Check final state
        count = vector_store.collection.count()
        print(f"Collection count after batch: {count}")
        
        # Test search
        print("\n🔍 Testing reaction search...")
        results = await vector_store.reaction_search(k=3, sort_by="total_reactions")
        print(f"Found {len(results)} results from reaction search")
        
        if results:
            for i, result in enumerate(results, 1):
                content = result.get("content", "")[:50] + "..." if len(result.get("content", "")) > 50 else result.get("content", "")
                reactions = result.get("reaction_count", 0)
                print(f"  {i}. '{content}' - {reactions} reactions")
        
        # Test individual operations for comparison
        print("\n🔬 Testing individual message additions...")
        for i, msg in enumerate(test_messages[:2]):  # Test first 2 individually
            result = await vector_store.add_messages([msg])
            print(f"Message {i+1} individual add: {result}")
            count = vector_store.collection.count()
            print(f"  Count after: {count}")
        
        await vector_store.close()
        print("✅ Batch debug completed")
        
    except Exception as e:
        print(f"❌ Batch debug failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(debug_batch_operations())
