#!/usr/bin/env python3
"""
Debug Discord Response Issue

This script tests the Discord response flow to identify why users aren't receiving responses.
"""

import asyncio
import logging
import sys
import os
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agentic.interfaces.agent_api import AgentAPI
from agentic.interfaces.discord_interface import DiscordInterface, DiscordContext

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_query_processing():
    """Test the query processing without Discord interaction"""
    try:
        # Initialize agent API
        config = {
            "openai": {
                "api_key": os.getenv("OPENAI_API_KEY"),
                "model": "gpt-4-turbo"
            },
            "vector_store": {
                "type": "persistent",
                "path": "./data/chromadb"
            },
            "memory": {
                "database_path": "./data/conversation_memory.db"
            }
        }
        
        agent_api = AgentAPI(config)
        await agent_api.initialize()
        
        # Create Discord interface
        discord_config = {
            "discord": {
                "token": os.getenv("DISCORD_TOKEN"),
                "command_prefix": "!",
                "max_message_length": 2000
            }
        }
        
        discord_interface = DiscordInterface(discord_config, agent_api)
        
        # Create a test Discord context
        discord_context = DiscordContext(
            user_id=673901899837341736,
            username="Test User",
            channel_id=1361465528333369534,
            guild_id=1353058864810950737,
            channel_name="test-channel",
            guild_name="Test Guild",
            timestamp=datetime.now()
        )
        
        # Test query that was successful in logs
        query = "list the last 5 messages in general"
        
        print(f"🧪 Testing query processing: {query}")
        
        # Process the query without Discord interaction
        messages = await discord_interface.process_query(query, discord_context, None)
        
        print(f"✅ Query processed successfully!")
        print(f"📝 Number of response messages: {len(messages)}")
        
        for i, message in enumerate(messages):
            print(f"\n📄 Message {i+1}:")
            print(f"Length: {len(message)} characters")
            print(f"Content preview: {message[:200]}...")
            
        return True
        
    except Exception as e:
        logger.error(f"❌ Error in test: {e}", exc_info=True)
        return False


async def test_message_formatting():
    """Test the message formatting specifically"""
    try:
        # Create a mock result that should be formatted
        mock_result = {
            "status": "success",
            "response": {
                "messages": [
                    {
                        "content": "This is a test message",
                        "author": {"username": "TestUser"},
                        "timestamp": "2025-06-03T17:47:32",
                        "channel_name": "general",
                        "jump_url": "https://discord.com/channels/123/456/789"
                    }
                ],
                "total_count": 1
            },
            "execution_time": 0.5
        }
        
        config = {
            "discord": {
                "token": os.getenv("DISCORD_TOKEN"),
                "command_prefix": "!",
                "max_message_length": 2000
            }
        }
        
        discord_interface = DiscordInterface(config, None)
        
        discord_context = DiscordContext(
            user_id=673901899837341736,
            username="Test User",
            channel_id=1361465528333369534,
            guild_id=1353058864810950737,
            channel_name="test-channel",
            guild_name="Test Guild",
            timestamp=datetime.now()
        )
        
        query = "test query"
        
        print(f"🧪 Testing message formatting...")
        
        # Test the formatting method directly
        formatted_messages = await discord_interface._format_response(mock_result, query, discord_context)
        
        print(f"✅ Message formatting successful!")
        print(f"📝 Number of formatted messages: {len(formatted_messages)}")
        
        for i, message in enumerate(formatted_messages):
            print(f"\n📄 Formatted Message {i+1}:")
            print(f"Length: {len(message)} characters")
            print(f"Content: {message}")
            
        return True
        
    except Exception as e:
        logger.error(f"❌ Error in formatting test: {e}", exc_info=True)
        return False


async def main():
    """Main test function"""
    print("🔍 Debug Discord Response Issue")
    print("=" * 50)
    
    # Test 1: Message formatting
    print("\n1️⃣ Testing message formatting...")
    formatting_success = await test_message_formatting()
    
    # Test 2: Full query processing
    print("\n2️⃣ Testing full query processing...")
    query_success = await test_query_processing()
    
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    print(f"Message Formatting: {'✅ PASS' if formatting_success else '❌ FAIL'}")
    print(f"Query Processing: {'✅ PASS' if query_success else '❌ FAIL'}")
    
    if formatting_success and query_success:
        print("\n✅ All tests passed! The issue may be in Discord interaction handling.")
        print("💡 Next steps: Check Discord token, permissions, and network connectivity.")
    else:
        print("\n❌ Tests failed! The issue is in the response processing pipeline.")


if __name__ == "__main__":
    asyncio.run(main())
