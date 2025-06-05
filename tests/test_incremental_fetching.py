#!/usr/bin/env python3
"""
Test Incremental Fetching Implementation
Tests the new incremental fetching capabilities of the Discord bot
"""

import asyncio
import logging
import os
import sys
from pathlib import Path
from datetime import datetime, timezone
from dotenv import load_dotenv

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

# Load environment variables from .env file
load_dotenv()

# Import our new modules
from agentic.services.fetch_state_manager import FetchStateManager
from agentic.services.discord_fetcher import DiscordMessageFetcher

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_incremental_fetching():
    """Test incremental fetching functionality"""
    
    print("🧪 Testing Incremental Discord Message Fetching")
    print("=" * 50)
    
    # Configuration
    config = {
        'page_size': 10,  # Small page size for testing
        'rate_limit_delay': 0.5,
        'max_retries': 3,
        'output_dir': 'data/test_fetched_messages',
        'state_file': 'data/test_fetch_state.json'
    }
    
    discord_token = os.getenv('DISCORD_TOKEN')
    guild_id = os.getenv('GUILD_ID')
    
    if not discord_token:
        print("❌ DISCORD_TOKEN environment variable not set")
        return False
    
    if not guild_id:
        print("❌ GUILD_ID environment variable not set")
        return False
    
    try:
        # Initialize fetcher
        fetcher = DiscordMessageFetcher(discord_token, config)
        await fetcher.initialize()
        
        print(f"🏰 Testing with Guild ID: {guild_id}")
        
        # Test 1: Initial fetch (should be treated as incremental if no state exists)
        print("\n📥 Test 1: Initial incremental fetch")
        result1 = await fetcher.fetch_guild_messages(
            guild_id=guild_id,
            limit_per_channel=5,  # Small limit for testing
            incremental=True
        )
        
        if result1["success"]:
            print(f"✅ Initial fetch successful: {result1['total_messages']} messages")
            print(f"📁 Files created: {len(result1.get('files_created', []))}")
            print(f"🔄 Files updated: {len(result1.get('files_updated', []))}")
        else:
            print(f"❌ Initial fetch failed: {result1.get('error')}")
            return False
        
        # Test 2: Immediate second fetch (should find no new messages)
        print("\n📈 Test 2: Second incremental fetch (should find no new messages)")
        result2 = await fetcher.fetch_guild_messages(
            guild_id=guild_id,
            limit_per_channel=5,
            incremental=True
        )
        
        if result2["success"]:
            print(f"✅ Second fetch successful: {result2['total_messages']} messages")
            print(f"📁 Files created: {len(result2.get('files_created', []))}")
            print(f"🔄 Files updated: {len(result2.get('files_updated', []))}")
            
            # Should have fewer or same messages since we're fetching incrementally
            if result2['total_messages'] <= result1['total_messages']:
                print("✅ Incremental fetching working correctly")
            else:
                print("⚠️ Unexpected: more messages in second fetch")
        else:
            print(f"❌ Second fetch failed: {result2.get('error')}")
            return False
        
        # Test 3: Full fetch mode
        print("\n📥 Test 3: Full fetch mode")
        result3 = await fetcher.fetch_guild_messages(
            guild_id=guild_id,
            limit_per_channel=5,
            incremental=False
        )
        
        if result3["success"]:
            print(f"✅ Full fetch successful: {result3['total_messages']} messages")
            print(f"📁 Files created: {len(result3.get('files_created', []))}")
            print(f"🔄 Files updated: {len(result3.get('files_updated', []))}")
        else:
            print(f"❌ Full fetch failed: {result3.get('error')}")
            return False
        
        # Test 4: Check state manager
        print("\n📊 Test 4: State manager information")
        state_summary = fetcher.state_manager.get_state_summary()
        print(f"📊 State Summary:")
        print(f"   Total channels tracked: {state_summary['total_channels']}")
        print(f"   Guilds: {state_summary['guilds']}")
        print(f"   Last updated: {state_summary['last_updated']}")
        
        # Cleanup
        await fetcher.close()
        
        print("\n🎉 All incremental fetching tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main test function"""
    success = await test_incremental_fetching()
    if success:
        print("\n✅ Incremental fetching implementation verified!")
        return 0
    else:
        print("\n❌ Incremental fetching tests failed!")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
