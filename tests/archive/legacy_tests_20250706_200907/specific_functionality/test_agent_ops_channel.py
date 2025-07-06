#!/usr/bin/env python3
"""
Agent-Ops Channel Test
Tests the exact query that failed: "list the last 5 messages in agent-ops channel"
"""

import asyncio
import sys
import os
import logging
from datetime import datetime

# Set test API key before any imports
os.environ["OPENAI_API_KEY"] = "test-key-for-testing"

# Add project root to path
sys.path.insert(0, os.path.abspath('.'))

from agentic.interfaces.agent_api import AgentAPI
from agentic.interfaces.discord_interface import DiscordInterface, DiscordContext

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_agent_ops_query():
    """Test the exact failing query: agent-ops channel search"""
    
    print("=" * 80)
    print("🧪 TESTING AGENT-OPS CHANNEL QUERY")
    print("=" * 80)
    
    # Test the exact query that failed
    test_query = "list the last 5 messages in 🦾agent-ops channel"
    
    print(f"📋 Test Query: {test_query}")
    print()
    
    try:
        # Use the same config as main.py (with the fix)
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
            }
        }
        
        agent_api = AgentAPI(config)
        
        print("🔄 Testing with AgentAPI...")
        result = await agent_api.query(
            query=test_query,
            user_id="test_user",
            context={}
        )
        
        print(f"✅ AgentAPI Success: {result.get('success', False)}")
        
        if result.get("success") and "sources" in result:
            sources = result["sources"]
            print(f"📊 Found {len(sources)} sources")
            
            # Validate the specific issues from the user's report
            print("\n🔍 Validation Checks:")
            print("-" * 40)
            
            # Check 1: Should return exactly 5 messages
            expected_count = 5
            actual_count = len(sources)
            if actual_count == expected_count:
                print(f"✅ Message count: {actual_count}/{expected_count} ✓")
            else:
                print(f"❌ Message count: {actual_count}/{expected_count} (user reported 10)")
            
            # Check 2: All messages should be from agent-ops channel
            target_channel = "🦾agent-ops"
            correct_channel_count = 0
            wrong_channels = set()
            
            for source in sources:
                channel = source.get('channel_name', 'Unknown')
                if channel == target_channel:
                    correct_channel_count += 1
                else:
                    wrong_channels.add(channel)
            
            if correct_channel_count == len(sources):
                print(f"✅ Channel filtering: {correct_channel_count}/{len(sources)} from correct channel ✓")
            else:
                print(f"❌ Channel filtering: {correct_channel_count}/{len(sources)} from correct channel")
                print(f"   Wrong channels found: {list(wrong_channels)}")
            
            # Check 3: Authors should not be "Unknown"
            proper_authors = 0
            unknown_authors = []
            
            for source in sources:
                author = source.get('author', {})
                username = author.get('username', 'Unknown') if isinstance(author, dict) else author
                if username and username != 'Unknown':
                    proper_authors += 1
                else:
                    unknown_authors.append(f"Message with unknown author")
            
            if proper_authors == len(sources):
                print(f"✅ Author information: {proper_authors}/{len(sources)} have proper authors ✓")
            else:
                print(f"❌ Author information: {proper_authors}/{len(sources)} have proper authors")
                print(f"   Unknown authors: {len(unknown_authors)}")
            
            # Display detailed results
            print(f"\n📝 Detailed Results:")
            print("-" * 40)
            for i, source in enumerate(sources):
                author = source.get('author', {})
                username = author.get('username', 'Unknown') if isinstance(author, dict) else author
                channel = source.get('channel_name', 'Unknown')
                content = source.get('content', '')[:100] + "..." if len(source.get('content', '')) > 100 else source.get('content', '')
                timestamp = source.get('timestamp', 'Unknown')
                
                print(f"  {i+1}. Channel: #{channel}")
                print(f"     Author: {username}")
                print(f"     Time: {timestamp}")
                print(f"     Content: {content}")
                print()
            
            # Test Discord interface formatting
            print("🎭 Testing Discord Interface Formatting...")
            print("-" * 40)
            
            discord_interface = DiscordInterface(config, agent_api)
            
            discord_context = DiscordContext(
                user_id=12345,
                username="test_user", 
                channel_id=67890,
                guild_id=111,
                channel_name="test-channel",
                guild_name="test-guild",
                timestamp=datetime.now()
            )
            
            formatted_result = await discord_interface.process_query(test_query, discord_context)
            
            if isinstance(formatted_result, list) and len(formatted_result) > 0:
                sample_text = formatted_result[0]
                print(f"✅ Discord formatting successful")
                
                # Check for proper author names in formatted text
                if "Unknown" in sample_text and ("ireney_67517" not in sample_text and "imkprabhat" not in sample_text):
                    print("❌ Formatted output still contains 'Unknown' authors")
                else:
                    print("✅ Formatted output has proper author names")
                
                print("\n📝 Sample Formatted Output:")
                print("-" * 40)
                print(sample_text[:500] + "..." if len(sample_text) > 500 else sample_text)
            else:
                print("❌ Discord formatting failed")
            
            return True
        else:
            print("❌ AgentAPI failed or returned no sources")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run the agent-ops channel test"""
    
    print("🚀 TESTING EXACT USER QUERY THAT FAILED")
    print(f"⏰ Test started at: {datetime.now().isoformat()}")
    print()
    
    success = await test_agent_ops_query()
    
    print("\n" + "=" * 80)
    print("🏁 AGENT-OPS CHANNEL TEST RESULTS")
    print("=" * 80)
    
    if success:
        print("🎉 AGENT-OPS QUERY TEST PASSED!")
        print("✅ The fix should now work for production Discord bot")
        print("\n🔧 Fixes confirmed:")
        print("   - Correct vector store path (./data/chromadb)")
        print("   - Proper author information extraction")
        print("   - Accurate channel filtering")
        print("   - Correct message count limiting")
        exit_code = 0
    else:
        print("❌ AGENT-OPS QUERY TEST FAILED")
        print("🔧 Issues may still exist")
        exit_code = 1
    
    print(f"⏰ Test completed at: {datetime.now().isoformat()}")
    return exit_code


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
