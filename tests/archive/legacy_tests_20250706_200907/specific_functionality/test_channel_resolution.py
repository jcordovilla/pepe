"""
Test channel resolution service.
"""

import os
import sys
sys.path.append('/Users/jose/Documents/apps/discord-bot-v2')

from agentic.services.channel_resolver import ChannelResolver

def test_channel_resolution():
    print("Testing Channel Resolution Service")
    print("=" * 50)
    
    # Initialize resolver
    resolver = ChannelResolver("./data/chromadb/chroma.sqlite3")
    
    # Test cases for channel resolution
    test_cases = [
        "agent-ops",
        "#agent-ops", 
        "🦾agent-ops",
        "agent ops",
        "agentops",
        "netarch-agents",
        "agent-dev",
        "general-chat",
        "non-existent-channel"
    ]
    
    print("Testing channel name resolution:")
    for test_name in test_cases:
        channel_id = resolver.resolve_channel_name(test_name)
        if channel_id:
            channel_info = resolver.get_channel_info(channel_id)
            print(f"✅ '{test_name}' -> {channel_id} ({channel_info.name if channel_info else 'Unknown'})")
        else:
            print(f"❌ '{test_name}' -> Not found")
    
    print("\nListing agent-related channels:")
    agent_channels = resolver.list_channels(pattern="agent")
    for channel in agent_channels:
        print(f"  {channel.name} (ID: {channel.id}, Messages: {channel.message_count})")
    
    print("\nTesting similar channels for 'agent':")
    similar = resolver.get_similar_channels("agent", limit=3)
    for channel_info, score in similar:
        print(f"  {channel_info.name} (Score: {score:.2f}, Messages: {channel_info.message_count})")

if __name__ == "__main__":
    test_channel_resolution()
