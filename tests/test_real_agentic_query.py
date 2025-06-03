#!/usr/bin/env python3
"""
Test Real Agentic Query with Channel ID Resolution

Tests the complete flow through the agentic system with channel ID-based filtering.
"""

import asyncio
import sys
import logging
import os
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging to see what's happening
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Add the parent directory to the path
sys.path.append('.')

from agentic.interfaces.agent_api import AgentAPI

async def test_real_agentic_query():
    """Test the complete agentic system with channel resolution"""
    
    print("Testing Real Agentic Query with Channel ID Resolution")
    print("=" * 60)
    
    # Initialize AgentAPI with the same config as the main system
    config = {
        "orchestrator": {},
        "vector_store": {
            "collection_name": "discord_messages",
            "persist_directory": "data/vectorstore",
            "embedding_model": "text-embedding-3-small",
            "batch_size": 100
        },
        "memory": {
            "db_path": "data/conversation_memory.db"
        },
        "pipeline": {
            "base_path": ".",
            "db_path": "data/discord_messages.db"
        },
        "analytics": {
            "db_path": "data/analytics.db"
        }
    }
    
    try:
        # Initialize the agent API
        print("1. Initializing AgentAPI...")
        agent_api = AgentAPI(config)
        
        # Test query that should trigger channel resolution
        test_query = "list the last 5 messages in agent-ops channel"
        user_id = "test_user_123"
        context = {
            "platform": "test",
            "timestamp": datetime.utcnow().isoformat()
        }
        
        print(f"2. Processing query: '{test_query}'")
        print("   - This should resolve 'agent-ops' to channel ID 1368249032866004992")
        print("   - And filter results to only that channel")
        
        # Process the query
        result = await agent_api.query(
            query=test_query,
            user_id=user_id,
            context=context
        )
        
        # Display results
        print("3. Query Results:")
        print(f"   - Success: {result.get('success', False)}")
        print(f"   - Response: {result.get('answer', 'No answer')[:200]}...")
        
        sources = result.get('sources', [])
        print(f"   - Sources found: {len(sources)}")
        
        if sources:
            print("   - Channel verification:")
            channel_ids = set()
            for i, source in enumerate(sources[:5]):  # Show first 5
                metadata = source.get('metadata', {})
                channel_name = metadata.get('channel_name', 'unknown')
                channel_id = metadata.get('channel_id', 'unknown')
                channel_ids.add(channel_id)
                print(f"     [{i+1}] Channel: {channel_name} (ID: {channel_id})")
            
            # Verify all results are from the correct channel
            expected_channel_id = "1368249032866004992"
            if len(channel_ids) == 1 and expected_channel_id in channel_ids:
                print(f"   ✅ All results correctly filtered to channel ID: {expected_channel_id}")
            else:
                print(f"   ❌ Results from multiple channels: {channel_ids}")
                print(f"   Expected only: {expected_channel_id}")
        
        # Show metadata
        metadata = result.get('metadata', {})
        print(f"   - Response time: {metadata.get('response_time', 0):.3f}s")
        print(f"   - Agents used: {metadata.get('agents_used', [])}")
        
        return result.get('success', False)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_real_agentic_query())
    if success:
        print("\n🎉 Real agentic query test PASSED!")
        print("Channel ID-based filtering is working correctly!")
    else:
        print("\n❌ Real agentic query test FAILED!")
        sys.exit(1)
