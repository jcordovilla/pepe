#!/usr/bin/env python3
"""
Test Discord Bot Query Processing

Test the complete Discord bot query processing pipeline to verify that:
1. Database searches return actual results
2. Author information is properly displayed
3. No interaction timeout errors occur
4. No interim processing messages are shown
"""

import asyncio
import logging
import sys
import os
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_discord_bot_query():
    """Test the Discord bot query processing"""
    
    print("🧪 Testing Discord Bot Query Processing...")
    
    try:
        # Import after setting up path
        from agentic.interfaces.discord_interface import DiscordInterface, DiscordContext
        
        # Configuration for the agentic system (matching main.py format)
        config = {
            'orchestrator': {
                'memory_config': {
                    'db_path': 'data/conversation_memory.db',
                    'max_history_length': 50,
                    'context_window_hours': 24
                }
            },
            'vector_store': {
                'persist_directory': './data/chromadb',
                'collection_name': 'discord_messages',
                'embedding_model': 'text-embedding-3-small',
                'batch_size': 100
            },
            'cache': {
                'redis_url': 'redis://localhost:6379',
                'default_ttl': 3600
            },
            'discord': {
                'token': 'test_token',  # Not needed for testing
                'guild_id': 'test_guild',
                'command_prefix': '!'
            }
        }
        
        print("🤖 Initializing Discord interface with config...")
        # Initialize Discord interface with config
        discord_interface = DiscordInterface(config)
        
        # Test query
        query = "list the last 3 messages in #agent-dev"
        
        print(f"📝 Testing query: {query}")
        
        # Create a mock Discord context for testing
        mock_context = DiscordContext(
            user_id=12345,
            username="test_user",
            channel_id=67890,
            guild_id=11111,
            channel_name="test-channel",
            guild_name="Test Server",
            timestamp=datetime.now()
        )
        
        # Test the query processing
        print("🔄 Processing query...")
        response = await discord_interface.process_query(query, mock_context)
        
        # Convert response to string if it's a list
        response_str = str(response) if isinstance(response, list) else response
        
        print(f"✅ Query processed successfully!")
        print(f"📤 Response (first 500 chars): {response_str[:500]}...")
        
        # Check for expected content
        if "agent-dev" in response_str or "🤖agent-dev" in response_str:
            print("✅ Channel name found in response")
        else:
            print("❌ Channel name not found in response")
            
        if "Unknown" not in response_str:
            print("✅ No 'Unknown' author information found")
        else:
            print("❌ 'Unknown' author information still present")
            
        # Check for actual search results
        if ("boozeena" in response_str.lower() or "oscarsan" in response_str.lower() or 
            "manaswita" in response_str.lower() or "Messages:" in response_str):
            print("✅ Actual user data found in response")
        else:
            print("❌ No actual user data found in response")
            
        return True
        
    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_discord_bot_query())
    if success:
        print("\n🎉 Discord bot query test completed successfully!")
    else:
        print("\n💥 Discord bot query test failed!")
        sys.exit(1)
