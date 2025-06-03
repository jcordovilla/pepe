#!/usr/bin/env python3
"""
Complete End-to-End Flow Test
Tests the complete Discord agent response flow after all fixes have been applied.
"""

import asyncio
import sys
import os
import json
import logging
from datetime import datetime

# Set test API key before any imports
os.environ["OPENAI_API_KEY"] = "test-key-for-testing"

# Add project root to path
sys.path.insert(0, os.path.abspath('.'))

from agentic.interfaces.agent_api import AgentAPI
from agentic.interfaces.discord_interface import DiscordInterface

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_emoji_channel_query():
    """Test query with emoji-based channel name that was previously failing"""
    
    print("=" * 80)
    print("🧪 TESTING COMPLETE FLOW WITH EMOJI CHANNEL QUERY")
    print("=" * 80)
    
    # Test the exact query that was failing
    test_query = "search for 5 messages in ❌💻non-coders-learning"
    
    print(f"📋 Test Query: {test_query}")
    print()
    
    try:
        # Initialize agentic core
        config = {
            "vectorstore": {
                "persist_directory": "./data/chromadb",
                "collection_name": "discord_messages"
            },
            "cache": {"enabled": True},
            "agents": {
                "search_agent": {"default_k": 5},
                "analysis_agent": {"enabled": True}
            }
        }
        
        agent_api = AgentAPI(config)
        
        # Initialize Discord interface
        discord_interface = DiscordInterface(config, agent_api)
        
        print("🔍 Step 1: Query Analysis")
        print("-" * 40)
        
        # Test the flow
        result = await agent_api.query(
            query=test_query,
            user_id="test_user",
            context={}
        )
        
        print(f"✅ AgentAPI returned: {type(result)}")
        print(f"Response successful: {result.get('success', False)}")
        
        if result.get("success") and "sources" in result:
            sources = result["sources"]
            print(f"📊 Found {len(sources)} sources")
            
            # Validate sources
            for i, source in enumerate(sources[:3]):
                print(f"  {i+1}. Channel: {source.get('channel_name', 'Unknown')}")
                print(f"     Author: {source.get('author', {}).get('username', 'Unknown')}")
                print(f"     Content: {source.get('content', '')[:100]}...")
        
        return result
            
    except Exception as e:
        print(f"❌ ERROR in complete flow test: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_simple_search_query():
    """Test a simple search query without channel specification"""
    
    print("\n" + "=" * 80)
    print("🧪 TESTING SIMPLE SEARCH QUERY")
    print("=" * 80)
    
    test_query = "search for python programming discussions"
    
    print(f"📋 Test Query: {test_query}")
    print()
    
    try:
        config = {
            "vectorstore": {
                "persist_directory": "./data/chromadb",
                "collection_name": "discord_messages"
            },
            "cache": {"enabled": True},
            "agents": {
                "search_agent": {"default_k": 5}
            }
        }
        
        core = AgenticCore(config)
        discord_interface = DiscordInterface(config, core)
        
        # Test the flow
        discord_response = await discord_interface.process_query(test_query)
        
        if isinstance(discord_response, dict) and "response" in discord_response:
            messages = discord_response["response"].get("messages", [])
            print(f"✅ Simple search returned {len(messages)} messages")
            
            if messages:
                print("📝 Sample result:")
                msg = messages[0]
                print(f"  Author: {msg.get('author', 'N/A')}")
                print(f"  Channel: {msg.get('channel', 'N/A')}")
                print(f"  Content: {msg.get('content', 'N/A')[:100]}...")
            
            return True
        else:
            print("❌ Simple search failed")
            return False
            
    except Exception as e:
        print(f"❌ ERROR in simple search test: {e}")
        return False


async def main():
    """Run all tests"""
    
    print("🚀 STARTING COMPLETE FLOW TESTING")
    print(f"⏰ Test started at: {datetime.now().isoformat()}")
    print()
    
    # Test 1: Emoji channel query (the main issue)
    success1 = await test_emoji_channel_query()
    
    # Test 2: Simple search query  
    success2 = await test_simple_search_query()
    
    print("\n" + "=" * 80)
    print("🏁 FINAL TEST RESULTS")
    print("=" * 80)
    
    if success1 and success2:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Discord agent response issues have been resolved")
        exit_code = 0
    else:
        print("❌ SOME TESTS FAILED")
        print("🔧 Additional fixes may be needed")
        exit_code = 1
    
    print(f"⏰ Test completed at: {datetime.now().isoformat()}")
    return exit_code


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
