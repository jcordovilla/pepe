#!/usr/bin/env python3
"""
Test the Discord bot's ability to handle the problematic query that was returning no results
"""

import asyncio
import os
import sys
import logging
from datetime import datetime

# Add project root to path
sys.path.insert(0, '/Users/jose/Documents/apps/discord-bot-v2')

# Set up basic logging
logging.basicConfig(level=logging.INFO)

async def test_discord_bot_query():
    """Test the Discord bot with the specific query that was failing"""
    
    try:
        # Import the Discord interface
        from agentic.interfaces.discord_interface import DiscordInterface
        from agentic.interfaces.discord_interface import DiscordContext
        
        print("🤖 Initializing Discord interface...")
        discord_interface = DiscordInterface()
        await discord_interface.initialize()
        
        # Create a mock Discord context
        mock_context = DiscordContext(
            user_id=12345,
            username="test_user",
            guild_id=999,
            channel_id=1001,
            channel_name="agent-dev",
            message_content="list the last 3 messages in #agent-dev",
            timestamp=datetime.now(),
            is_dm=False
        )
        
        print("🔍 Testing query: 'list the last 3 messages in #agent-dev'")
        
        # Execute the problematic query
        result = await discord_interface.handle_query(
            "list the last 3 messages in #agent-dev",
            mock_context
        )
        
        print("📝 Bot Response:")
        print("=" * 50)
        if isinstance(result, dict):
            print(f"Status: {result.get('status', 'Unknown')}")
            print(f"Response: {result.get('response', 'No response')}")
            if result.get('agents_used'):
                print(f"Agents used: {', '.join(result['agents_used'])}")
        else:
            print(str(result))
        print("=" * 50)
        
        if result and "no results" not in str(result).lower() and "not found" not in str(result).lower():
            print("✅ SUCCESS: Bot returned actual search results!")
        else:
            print("❌ FAILURE: Bot still returning no results")
            
        return result
        
    except Exception as e:
        print(f"❌ Error testing Discord bot: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("🧪 Testing Discord bot with populated database...")
    result = asyncio.run(test_discord_bot_query())
    print("🎉 Test complete!")
