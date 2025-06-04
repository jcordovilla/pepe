#!/usr/bin/env python3
"""
Comprehensive Discord Response Test

This test simulates the full Discord query flow to identify issues
with response formatting and delivery.
"""

import asyncio
import sys
import os
from datetime import datetime
from typing import Dict, Any, List

# Add the project root to Python path
sys.path.append('/Users/jose/Documents/apps/discord-bot-v2')

# Mock Discord context
class MockDiscordContext:
    def __init__(self):
        self.user_id = 123456789
        self.username = "test_user"
        self.channel_id = 987654321
        self.guild_id = 111222333
        self.timestamp = datetime.now()

async def test_full_response_flow():
    """Test the complete response flow from agent API to Discord formatting"""
    
    print("🧪 Testing Discord Response Flow")
    print("=" * 50)
    
    try:
        # Import the Discord interface
        from agentic.interfaces.discord_interface import DiscordInterface
        
        # Create minimal config
        config = {
            "discord": {
                "token": "test_token",
                "command_prefix": "!"
            },
            "orchestrator": {
                "memory_config": {
                    "db_path": "data/conversation_memory.db",
                    "max_history_length": 50,
                    "context_window_hours": 24
                }
            },
            "vectorstore": {
                "persist_directory": "data/chromadb",
                "collection_name": "discord_messages",
                "embedding_model": "text-embedding-3-small",
                "batch_size": 100
            }
        }
        
        # Initialize Discord interface
        print("📝 Initializing Discord interface...")
        
        # We'll need to mock the AgentAPI since we can't initialize it without OPENAI_API_KEY
        class MockAgentAPI:
            async def query(self, query: str, user_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
                """Mock agent API query that simulates finding messages"""
                print(f"🔍 Mock agent processing query: {query}")
                
                # Simulate search results with message-like objects
                mock_sources = [
                    {
                        "author": {"username": "jose_cordovilla", "id": "123"},
                        "content": "This is a test message about the query topic",
                        "timestamp": "2025-06-03T18:00:00Z",
                        "channel_name": "jose-test",
                        "jump_url": "https://discord.com/channels/123/456/789"
                    },
                    {
                        "author": {"username": "another_user", "id": "456"},
                        "content": "Here's another relevant message",
                        "timestamp": "2025-06-03T17:30:00Z",
                        "channel_name": "jose-test",
                        "jump_url": "https://discord.com/channels/123/456/790"
                    }
                ]
                
                return {
                    "success": True,
                    "answer": f"Found {len(mock_sources)} relevant messages.",
                    "sources": mock_sources,
                    "metadata": {
                        "response_time": 0.5,
                        "agents_used": ["searcher"],
                        "tokens_used": 100
                    }
                }
        
        # Create Discord interface with mocked agent API
        discord_interface = DiscordInterface.__new__(DiscordInterface)
        discord_interface.config = config
        discord_interface.agent_api = MockAgentAPI()
        discord_interface.cache_enabled = False
        discord_interface.enable_analytics = True
        discord_interface.max_message_length = 2000
        
        # Test query processing
        print("🚀 Testing query processing...")
        test_query = "list the last 5 messages in #jose-test"
        mock_context = MockDiscordContext()
        
        # Process query
        formatted_messages = await discord_interface.process_query(
            test_query, 
            mock_context,
            interaction=None
        )
        
        print(f"✅ Query processed successfully!")
        print(f"📤 Response contains {len(formatted_messages)} message(s)")
        
        # Display formatted messages
        for i, message in enumerate(formatted_messages, 1):
            print(f"\n📝 Message {i} (length: {len(message)}):")
            print("-" * 40)
            print(message[:500] + "..." if len(message) > 500 else message)
            print("-" * 40)
        
        # Test empty results
        print("\n🔍 Testing empty results...")
        
        class MockEmptyAgentAPI:
            async def query(self, query: str, user_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
                return {
                    "success": True,
                    "answer": "No messages found.",
                    "sources": [],
                    "metadata": {"response_time": 0.2}
                }
        
        discord_interface.agent_api = MockEmptyAgentAPI()
        empty_messages = await discord_interface.process_query(
            "find messages about nonexistent topic",
            mock_context,
            interaction=None
        )
        
        print(f"✅ Empty query processed!")
        print(f"📤 Empty response contains {len(empty_messages)} message(s)")
        for i, message in enumerate(empty_messages, 1):
            print(f"\n📝 Empty Message {i}:")
            print(message)
        
        # Test error case
        print("\n❌ Testing error handling...")
        
        class MockErrorAgentAPI:
            async def query(self, query: str, user_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
                return {
                    "success": False,
                    "error": "Mock error for testing"
                }
        
        discord_interface.agent_api = MockErrorAgentAPI()
        error_messages = await discord_interface.process_query(
            "trigger error",
            mock_context,
            interaction=None
        )
        
        print(f"✅ Error case processed!")
        print(f"📤 Error response contains {len(error_messages)} message(s)")
        for i, message in enumerate(error_messages, 1):
            print(f"\n📝 Error Message {i}:")
            print(message)
        
        print("\n🎉 All tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_full_response_flow())
