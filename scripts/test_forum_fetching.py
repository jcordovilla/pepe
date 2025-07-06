#!/usr/bin/env python3
"""
Test script for forum channel fetching functionality
"""

import asyncio
import json
import os
import sys
from pathlib import Path
import dotenv

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agentic.services.discord_fetcher import DiscordMessageFetcher
from agentic.config.modernized_config import get_modernized_config

dotenv.load_dotenv()

async def test_forum_fetching():
    """Test forum channel fetching functionality"""
    print("🧪 Testing Forum Channel Fetching")
    print("=" * 50)
    
    # Load configuration
    config = get_modernized_config()
    
    # Initialize Discord fetcher
    fetcher = DiscordMessageFetcher(
        token=config['discord']['token'],
        config=config.get('discord_fetcher', {})
    )
    
    try:
        # Initialize the fetcher
        await fetcher.initialize()
        print("✅ Discord fetcher initialized")
        
        # Test fetching from guild
        guild_id = config['discord']['guild_id']
        print(f"🏛️ Fetching from guild: {guild_id}")
        
        # Run a full fetch to get forum data
        result = await fetcher.fetch_guild_messages(
            guild_id=guild_id,
            incremental=False,  # Full fetch to get all data
            limit_per_channel=100  # Limit for testing
        )
        
        if result['success']:
            print("\n📊 Fetch Results:")
            print(f"   📝 Total Messages: {result['total_messages']}")
            print(f"   💬 Text Channels: {result['channels_processed']}")
            print(f"   📋 Forum Channels: {result['forum_channels_processed']}")
            print(f"   🧵 Threads Processed: {result['total_threads_processed']}")
            print(f"   📁 Files Created: {len(result['files_created'])}")
            print(f"   🔄 Files Updated: {len(result['files_updated'])}")
            
            if result['errors']:
                print(f"\n⚠️ Errors:")
                for error in result['errors']:
                    print(f"   ❌ {error['channel']}: {error['error']}")
            
            # Check for forum files
            output_dir = Path("./data/fetched_messages")
            forum_files = list(output_dir.glob("*_forum_messages.json"))
            thread_files = list(output_dir.glob("*_thread_messages.json"))
            
            print(f"\n📋 Forum Files Found:")
            print(f"   📁 Forum Summary Files: {len(forum_files)}")
            print(f"   🧵 Thread Files: {len(thread_files)}")
            
            if forum_files:
                print(f"\n📋 Forum Files:")
                for file in forum_files:
                    try:
                        with open(file, 'r') as f:
                            data = json.load(f)
                        print(f"   📄 {file.name}: {data.get('total_messages', 0)} messages from {data.get('threads_processed', 0)} threads")
                    except Exception as e:
                        print(f"   ❌ {file.name}: Error reading - {e}")
            
            if thread_files:
                print(f"\n🧵 Thread Files (showing first 5):")
                for file in thread_files[:5]:
                    try:
                        with open(file, 'r') as f:
                            data = json.load(f)
                        print(f"   📄 {file.name}: {data.get('message_count', 0)} messages in thread '{data.get('thread_name', 'Unknown')}'")
                    except Exception as e:
                        print(f"   ❌ {file.name}: Error reading - {e}")
                
                if len(thread_files) > 5:
                    print(f"   ... and {len(thread_files) - 5} more thread files")
            
        else:
            print(f"❌ Fetch failed: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"❌ Error during forum fetching test: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up
        await fetcher.close()
        print("\n🔌 Discord fetcher closed")

if __name__ == "__main__":
    asyncio.run(test_forum_fetching()) 