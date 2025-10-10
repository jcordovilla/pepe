#!/usr/bin/env python3
"""
Channel checking and analysis script for Discord bot maintenance
"""
import sys
import os
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from agentic.services.channel_resolver import ChannelResolver

def print_progress_bar(iteration, total, prefix='', suffix='', length=40, fill='█'):
    """Print a progress bar to the console"""
    percent = f"{100 * (iteration / float(total)):.1f}"
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='', flush=True)

def print_channel_header(title):
    """Print formatted channel header"""
    print(f"\n{'=' * 60}")
    print(f"📋 {title}")
    print('=' * 60)

def get_channels():
    """Get channel data using ChannelResolver"""
    try:
        resolver = ChannelResolver("./data/chromadb/chroma.sqlite3")
        channels = resolver.list_channels()
        
        # Convert to expected format
        channel_data = []
        for channel in channels:
            channel_data.append({
                "id": channel.id,
                "name": channel.name,
                "message_count": channel.message_count
            })
        
        return channel_data
    except Exception as e:
        print(f"Error fetching channels: {e}")
        return []

def analyze_channels():
    """Analyze Discord channels with progress tracking"""
    print_channel_header("Discord Channel Analysis")
    
    try:
        print("🔄 Fetching channel data...")
        channels = get_channels()
        
        if not channels:
            print("❌ No channels found or unable to fetch channel data")
            return
        
        print(f"✅ Found {len(channels)} channels")
        
        print("\n🔍 Analyzing channels...")
        for i, channel in enumerate(channels):
            time.sleep(0.1)  # Small delay for visual effect
            print_progress_bar(i + 1, len(channels), prefix='Progress:', suffix=f'Analyzing {channel["name"]}')
        
        print()  # New line after progress bar
        
        print_channel_header("Channel Summary")
        print('📋 Available channels:')
        for channel in channels:
            print(f'- #{channel["name"]} (ID: {channel["id"]}, Messages: {channel["message_count"]})')
        print(f'\n📊 Total channels: {len(channels)}')
        
        # Look for agent-dev channel specifically
        print_channel_header("Agent Development Channels")
        agent_dev_channels = [ch for ch in channels if 'agent' in ch['name'].lower() and 'dev' in ch['name'].lower()]
        if agent_dev_channels:
            print('🤖 Agent dev channels found:')
            for ch in agent_dev_channels:
                print(f'- #{ch["name"]} (ID: {ch["id"]}, Messages: {ch["message_count"]})')
        else:
            print('⚠️  No agent-dev like channels found')
            
        # Look for any channels with "agent" in the name
        agent_channels = [ch for ch in channels if 'agent' in ch['name'].lower()]
        if agent_channels:
            print_channel_header("Agent-Related Channels")
            print('🔍 Channels with "agent" in name:')
            for ch in agent_channels:
                print(f'- #{ch["name"]} (ID: {ch["id"]}, Messages: {ch["message_count"]})')
        
        # Channel statistics
        print_channel_header("Channel Statistics")
        total_messages = sum(ch["message_count"] for ch in channels)
        avg_messages = total_messages / len(channels) if channels else 0
        most_active = max(channels, key=lambda x: x["message_count"]) if channels else None
        
        print(f"📈 Total messages across all channels: {total_messages:,}")
        print(f"📊 Average messages per channel: {avg_messages:.1f}")
        if most_active:
            print(f"🏆 Most active channel: #{most_active['name']} ({most_active['message_count']:,} messages)")
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    analyze_channels()
