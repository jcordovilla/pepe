#!/usr/bin/env python3
"""
Simplified Flow Test
Tests the Discord agent response flow with simplified approach.
"""

import asyncio
import sys
import os
import logging
from datetime import datetime

# Set test API key before imports
os.environ["OPENAI_API_KEY"] = "test-key-for-testing"

# Add project root to path
sys.path.insert(0, os.path.abspath('.'))

from agentic.interfaces.agent_api import AgentAPI
from agentic.interfaces.discord_interface import DiscordInterface, DiscordContext

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_discord_interface():
    """Test Discord interface directly"""
    
    print("=" * 80)
    print("🧪 TESTING DISCORD INTERFACE DIRECTLY")
    print("=" * 80)
    
    # Test the exact query that was failing
    test_query = "search for 5 messages in ❌💻non-coders-learning"
    
    print(f"📋 Test Query: {test_query}")
    print()
    
    try:
        # Initialize components
        config = {
            "vectorstore": {
                "persist_directory": "./data/chromadb",
                "collection_name": "discord_messages"
            },
            "vector_store": {  # Note: agent_api expects vector_store not vectorstore
                "persist_directory": "./data/chromadb", 
                "collection_name": "discord_messages"
            },
            "cache": {"enabled": True},
            "search_agent": {"default_k": 5},
            "analysis_agent": {"enabled": True},
            "memory": {"db_path": "./data/conversation_memory.db"}
        }
        
        agent_api = AgentAPI(config)
        discord_interface = DiscordInterface(config, agent_api)
        
        # Create Discord context
        discord_context = DiscordContext(
            user_id=12345,
            username="test_user", 
            channel_id=67890,
            guild_id=111,
            channel_name="test_channel",
            guild_name="test_guild",
            timestamp=datetime.now()
        )
        
        print("🔄 Processing query through Discord interface...")
        
        # Process query
        result = await discord_interface.process_query(test_query, discord_context)
        
        print(f"📊 Result type: {type(result)}")
        print(f"📊 Result: {result}")
        
        # Analyze result
        if isinstance(result, list):
            print(f"✅ Got list response with {len(result)} items")
            for i, item in enumerate(result[:3]):
                print(f"  Item {i+1}: {str(item)[:100]}...")
        elif isinstance(result, dict):
            print(f"✅ Got dict response with keys: {list(result.keys())}")
            if "response" in result:
                messages = result["response"].get("messages", [])
                print(f"📝 Found {len(messages)} messages")
                
                # Validation checks
                print("\n🔍 Validation Checks:")
                print("-" * 20)
                
                # Check count
                expected_count = 5
                actual_count = len(messages)
                if actual_count <= expected_count:
                    print(f"✅ Message count: {actual_count} (≤ {expected_count})")
                else:
                    print(f"⚠️  Message count: {actual_count} (expected ≤ {expected_count})")
                
                # Check authors
                if messages and len(messages) > 0:
                    unknown_authors = sum(1 for msg in list(messages) if msg.get("author") in [None, "", "Unknown"])
                    if unknown_authors == 0:
                        print(f"✅ Author information: All {len(messages)} messages have proper authors")
                    else:
                        print(f"❌ Author information: {unknown_authors}/{len(messages)} have unknown authors")
                    
                    # Sample messages
                    print(f"\n📝 Sample Messages:")
                    print("-" * 20)
                    for i, msg in enumerate(messages[:3]):
                        print(f"Message {i+1}:")
                        print(f"  Channel: {msg.get('channel', 'N/A')}")
                        print(f"  Author: {msg.get('author', 'N/A')}")
                        print(f"  Content: {msg.get('content', 'N/A')[:100]}...")
                        print()
        else:
            print(f"❌ Unexpected result type: {type(result)}")
            return False
        
        print("✅ Discord interface test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ ERROR in Discord interface test: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_agent_api_directly():
    """Test AgentAPI directly to understand the flow"""
    
    print("\n" + "=" * 80)
    print("🧪 TESTING AGENT API DIRECTLY")
    print("=" * 80)
    
    test_query = "search for 5 messages in ❌💻non-coders-learning"
    
    print(f"📋 Test Query: {test_query}")
    print()
    
    try:
        config = {
            "vector_store": {
                "persist_directory": "./data/chromadb",
                "collection_name": "discord_messages"
            },
            "cache": {"enabled": True},
            "search_agent": {"default_k": 5},
            "memory": {"db_path": "./data/conversation_memory.db"}
        }
        
        agent_api = AgentAPI(config)
        
        print("🔄 Querying AgentAPI...")
        
        result = await agent_api.query(
            query=test_query,
            user_id="test_user",
            context={
                "platform": "test",
                "timestamp": datetime.now().isoformat()
            }
        )
        
        print(f"📊 AgentAPI Result:")
        print(f"  Success: {result.get('success', False)}")
        print(f"  Answer: {result.get('answer', 'No answer')[:200]}...")
        print(f"  Sources count: {len(result.get('sources', []))}")
        
        if result.get('sources'):
            print(f"\n📝 Sample Sources:")
            for i, source in enumerate(result['sources'][:3]):
                channel = source.get('channel_name', 'unknown')
                author = source.get('author', {})
                if isinstance(author, dict):
                    author_name = author.get('username', 'unknown')
                else:
                    author_name = str(author)
                content = source.get('content', '')[:100]
                print(f"  {i+1}. Channel: {channel}")
                print(f"     Author: {author_name}")
                print(f"     Content: {content}...")
                print()
        
        return result.get('success', False)
        
    except Exception as e:
        print(f"❌ ERROR in AgentAPI test: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run tests"""
    
    print("🚀 STARTING SIMPLIFIED FLOW TESTING")
    print(f"⏰ Test started at: {datetime.now().isoformat()}")
    print()
    
    # Test 1: AgentAPI directly
    api_success = await test_agent_api_directly()
    
    # Test 2: Discord interface
    discord_success = await test_discord_interface()
    
    print("\n" + "=" * 80)
    print("🏁 FINAL TEST RESULTS")
    print("=" * 80)
    
    if api_success and discord_success:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Discord agent response fixes are working")
        exit_code = 0
    else:
        print("❌ SOME TESTS FAILED")
        print(f"  AgentAPI test: {'✅' if api_success else '❌'}")
        print(f"  Discord interface test: {'✅' if discord_success else '❌'}")
        exit_code = 1
    
    print(f"⏰ Test completed at: {datetime.now().isoformat()}")
    return exit_code


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
