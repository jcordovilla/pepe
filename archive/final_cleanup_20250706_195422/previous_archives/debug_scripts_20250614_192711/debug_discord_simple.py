#!/usr/bin/env python3
"""
Simple Discord Response Test

This script tests just the Discord message formatting without requiring API initialization.
"""

import asyncio
import logging
import sys
import os
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_message_formatting_simple():
    """Test the message formatting without full initialization"""
    try:
        # Import the minimal required components
        from agentic.interfaces.discord_interface import DiscordInterface, DiscordContext
        
        # Create a minimal config without OpenAI dependencies
        config = {
            "discord": {
                "token": "dummy_token",
                "command_prefix": "!",
                "max_message_length": 2000
            }
        }
        
        # Create the Discord interface with a None agent_api to avoid initialization
        # We'll manually create the needed attributes
        discord_interface = object.__new__(DiscordInterface)
        discord_interface.discord_config = config.get("discord", {})
        discord_interface.max_message_length = discord_interface.discord_config.get("max_message_length", 2000)
        discord_interface.enable_analytics = False
        
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
        
        # Create a mock result that matches the structure from the logs
        mock_result = {
            "status": "success",
            "response": {
                "messages": [
                    {
                        "content": "This is a test message from the search results",
                        "author": {"username": "TestUser1"},
                        "timestamp": "2025-06-03T17:47:32",
                        "channel_name": "general",
                        "jump_url": "https://discord.com/channels/1353058864810950737/1364361051830747156/123456"
                    },
                    {
                        "content": "Another test message with some longer content to see how it handles formatting",
                        "author": {"username": "TestUser2"},
                        "timestamp": "2025-06-03T17:45:15",
                        "channel_name": "general",
                        "jump_url": "https://discord.com/channels/1353058864810950737/1364361051830747156/789012"
                    }
                ],
                "total_count": 2
            },
            "execution_time": 0.402
        }
        
        query = "list the last 5 messages in general"
        
        print(f"🧪 Testing message formatting with mock result...")
        print(f"Query: {query}")
        print(f"Mock result status: {mock_result['status']}")
        print(f"Mock messages count: {len(mock_result['response']['messages'])}")
        
        # Test the formatting method directly
        formatted_messages = await discord_interface._format_response(mock_result, query, discord_context)
        
        print(f"\n✅ Message formatting successful!")
        print(f"📝 Number of formatted messages: {len(formatted_messages)}")
        
        for i, message in enumerate(formatted_messages):
            print(f"\n📄 Formatted Message {i+1}:")
            print(f"Length: {len(message)} characters")
            print(f"Discord limit check: {'✅ WITHIN LIMIT' if len(message) <= 2000 else '❌ TOO LONG'}")
            print("Content:")
            print("-" * 40)
            print(message)
            print("-" * 40)
            
        return True, formatted_messages
        
    except Exception as e:
        logger.error(f"❌ Error in formatting test: {e}", exc_info=True)
        return False, []


async def test_empty_result():
    """Test formatting with empty results"""
    try:
        from agentic.interfaces.discord_interface import DiscordInterface, DiscordContext
        
        config = {
            "discord": {
                "token": "dummy_token",
                "command_prefix": "!",
                "max_message_length": 2000
            }
        }
        
        discord_interface = object.__new__(DiscordInterface)
        discord_interface.discord_config = config.get("discord", {})
        discord_interface.max_message_length = discord_interface.discord_config.get("max_message_length", 2000)
        discord_interface.enable_analytics = False
        
        discord_context = DiscordContext(
            user_id=673901899837341736,
            username="Test User",
            channel_id=1361465528333369534,
            guild_id=1353058864810950737,
            channel_name="test-channel",
            guild_name="Test Guild",
            timestamp=datetime.now()
        )
        
        # Test with empty results
        empty_result = {
            "status": "success",
            "response": {
                "messages": [],
                "total_count": 0
            },
            "execution_time": 0.1
        }
        
        query = "test empty query"
        
        print(f"\n🧪 Testing with empty results...")
        formatted_messages = await discord_interface._format_response(empty_result, query, discord_context)
        
        print(f"✅ Empty result formatting successful!")
        print(f"📝 Number of formatted messages: {len(formatted_messages)}")
        
        for i, message in enumerate(formatted_messages):
            print(f"\nEmpty Result Message {i+1}: {message}")
            
        return True, formatted_messages
        
    except Exception as e:
        logger.error(f"❌ Error in empty result test: {e}", exc_info=True)
        return False, []


async def main():
    """Main test function"""
    print("🔍 Simple Discord Response Test")
    print("=" * 50)
    
    # Test 1: Normal message formatting
    print("\n1️⃣ Testing normal message formatting...")
    normal_success, normal_messages = await test_message_formatting_simple()
    
    # Test 2: Empty result formatting
    print("\n2️⃣ Testing empty result formatting...")
    empty_success, empty_messages = await test_empty_result()
    
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    print(f"Normal Message Formatting: {'✅ PASS' if normal_success else '❌ FAIL'}")
    print(f"Empty Result Formatting: {'✅ PASS' if empty_success else '❌ FAIL'}")
    
    if normal_success and empty_success:
        print("\n✅ Message formatting is working correctly!")
        print("💡 The issue is likely in the Discord interaction handling or network.")
        print("\n🔍 Next investigation areas:")
        print("   1. Discord interaction.followup.send() errors")
        print("   2. Discord bot permissions")
        print("   3. Network connectivity issues")
        print("   4. Exception handling in handle_slash_command")
    else:
        print("\n❌ Message formatting has issues!")
        
    return normal_success and empty_success


if __name__ == "__main__":
    asyncio.run(main())
