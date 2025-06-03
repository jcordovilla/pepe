# Test the Discord response transformation
import asyncio
import sys
import os
sys.path.append('/Users/jose/Documents/apps/discord-bot-v2')

from agentic.interfaces.discord_interface import DiscordInterface

async def test_transform():
    # Mock agent API response
    mock_response = {
        "success": True,
        "answer": "I found 3 relevant messages.",
        "sources": [
            {
                "author": {"username": "test_user"},
                "content": "Test message 1",
                "timestamp": "2025-06-03T18:00:00Z",
                "channel_name": "test-channel"
            },
            {
                "author": {"username": "test_user2"},
                "content": "Test message 2", 
                "timestamp": "2025-06-03T18:01:00Z",
                "channel_name": "test-channel"
            }
        ]
    }

    # Create Discord interface instance (minimal config)
    config = {"discord": {"token": "test"}}
    interface = DiscordInterface(config)
    
    # Test transformation
    transformed = interface._transform_agent_response(mock_response)
    print("Transformed response:")
    print(transformed)
    
    # Test with empty sources
    mock_empty = {
        "success": True,
        "answer": "No messages found.",
        "sources": []
    }
    
    transformed_empty = interface._transform_agent_response(mock_empty)
    print("\nTransformed empty response:")
    print(transformed_empty)

if __name__ == "__main__":
    asyncio.run(test_transform())
