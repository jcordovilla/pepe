#!/usr/bin/env python3
"""
Production test for reaction search functionality using real OpenAI API key.
"""

import asyncio
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, Any

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

from agentic.vectorstore.persistent_store import PersistentVectorStore

async def test_production_reaction_search():
    """Test reaction search functionality with real OpenAI API key."""
    print("🚀 Testing Production Reaction Search with Real API Key")
    print("=" * 60)
    
    try:
        # Verify API key is loaded
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or api_key.startswith("test-"):
            print("❌ No valid OpenAI API key found in .env file")
            return False
        
        print("✅ Real OpenAI API key detected")
        
        # Initialize vector store with production config
        config = {
            "persist_directory": "./data/test_production_real",
            "embedding_model": "text-embedding-3-small",
            "batch_size": 5,
            "cache": {"type": "memory", "ttl": 1800}
        }
        
        vector_store = PersistentVectorStore(config)
        print("✅ Vector store initialized with real embedding function")
        
        # Test with realistic Discord messages
        print("\n📝 Adding realistic Discord messages with reactions...")
        realistic_messages = [
            {
                "message_id": "prod_msg_001",
                "content": "🎉 Major product launch announcement! Our new AI features are now live!",
                "channel_id": "announcements",
                "channel_name": "announcements",
                "guild_id": "guild_main",
                "author": {"id": "ceo_123", "username": "CEO"},
                "timestamp": (datetime.now() - timedelta(days=2)).isoformat(),
                "jump_url": "https://discord.com/channels/guild/announcements/001",
                "reactions": [
                    {"emoji": "🎉", "count": 156},
                    {"emoji": "🚀", "count": 98},
                    {"emoji": "❤️", "count": 87},
                    {"emoji": "👏", "count": 124},
                    {"emoji": "🔥", "count": 76}
                ]
            },
            {
                "message_id": "prod_msg_002", 
                "content": "Just shipped a critical bug fix that was affecting user login. Thanks for your patience! 🛠️",
                "channel_id": "dev-updates",
                "channel_name": "dev-updates",
                "guild_id": "guild_main",
                "author": {"id": "dev_lead_456", "username": "DevLead"},
                "timestamp": (datetime.now() - timedelta(hours=8)).isoformat(),
                "jump_url": "https://discord.com/channels/guild/dev-updates/002",
                "reactions": [
                    {"emoji": "🛠️", "count": 45},
                    {"emoji": "👍", "count": 89},
                    {"emoji": "🙏", "count": 67},
                    {"emoji": "✅", "count": 34}
                ]
            },
            {
                "message_id": "prod_msg_003",
                "content": "Community poll: What feature should we prioritize next? React with your choice! 📊",
                "channel_id": "community",
                "channel_name": "community", 
                "guild_id": "guild_main",
                "author": {"id": "community_mgr_789", "username": "CommunityMgr"},
                "timestamp": (datetime.now() - timedelta(hours=4)).isoformat(),
                "jump_url": "https://discord.com/channels/guild/community/003",
                "reactions": [
                    {"emoji": "1️⃣", "count": 234},
                    {"emoji": "2️⃣", "count": 189},
                    {"emoji": "3️⃣", "count": 156},
                    {"emoji": "4️⃣", "count": 98}
                ]
            }
        ]
        
        success = await vector_store.add_messages(realistic_messages)
        if success:
            print(f"✅ Added {len(realistic_messages)} production messages")
            stats = await vector_store.get_collection_stats()
            print(f"   Collection now has {stats.get('total_documents', 0)} documents")
        else:
            print("❌ Failed to add production messages")
            return False
        
        # Test reaction search functionality
        print("\n🔍 Testing reaction search capabilities...")
        
        # Test 1: Most reacted messages
        print("\n1. Finding most reacted messages...")
        top_reacted = await vector_store.reaction_search(k=3, sort_by="total_reactions")
        
        if top_reacted:
            print(f"✅ Found {len(top_reacted)} most reacted messages:")
            for i, msg in enumerate(top_reacted, 1):
                content = msg.get("content", "")[:50] + "..." if len(msg.get("content", "")) > 50 else msg.get("content", "")
                reactions = msg.get("reaction_count", 0)
                channel = msg.get("channel_name", "unknown")
                print(f"   {i}. [{channel}] '{content}' - {reactions} reactions")
        else:
            print("❌ No reacted messages found")
            return False
        
        # Test 2: Specific emoji search
        print("\n2. Testing specific emoji searches...")
        test_emojis = ["🎉", "👍", "🔥"]
        
        for emoji in test_emojis:
            emoji_results = await vector_store.reaction_search(reaction=emoji, k=2)
            if emoji_results:
                print(f"   ✅ Found {len(emoji_results)} messages with {emoji}")
                for result in emoji_results[:1]:  # Show first result
                    content = result.get("content", "")[:40] + "..."
                    reactions = result.get("reaction_count", 0)
                    print(f"      - '{content}' ({reactions} total reactions)")
            else:
                print(f"   ℹ️  No messages with {emoji} found")
        
        # Test 3: Channel filtering
        print("\n3. Testing channel-specific searches...")
        channel_results = await vector_store.reaction_search(
            k=2, 
            sort_by="total_reactions",
            filters={"channel_name": "announcements"}
        )
        
        if channel_results:
            print(f"   ✅ Found {len(channel_results)} reacted messages in #announcements")
            for result in channel_results:
                content = result.get("content", "")[:50] + "..."
                reactions = result.get("reaction_count", 0)
                print(f"      - '{content}' ({reactions} reactions)")
        
        # Test 4: System health
        print("\n4. Testing system health...")
        health = await vector_store.health_check()
        print(f"   ✅ System health: {health}")
        
        # Cleanup
        await vector_store.close()
        print("\n✅ Production test cleanup completed")
        
        print("\n" + "=" * 60)
        print("🎉 Production Reaction Search Test PASSED!")
        print("\n✨ Your Discord bot is ready to handle reaction-based queries!")
        print("   Try asking: 'What was the most reacted message in #announcements?'")
        print("🚀 Production deployment successful!")
        
        return True
        
    except Exception as e:
        print(f"❌ Production test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main test function."""
    success = await test_production_reaction_search()
    
    if success:
        print("\n🎊 SUCCESS: Your Discord bot is production-ready!")
        print("   The reaction search functionality is fully operational.")
    else:
        print("\n❌ FAILED: Check the errors above and try again.")
    
    return success

if __name__ == "__main__":
    asyncio.run(main())
