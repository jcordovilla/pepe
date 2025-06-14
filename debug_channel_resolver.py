#!/usr/bin/env python3
"""
Debug channel resolver issue.
"""

import sys
import sqlite3
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agentic.services.channel_resolver import ChannelResolver

def debug_channel_resolver():
    """Debug the channel resolver issue"""
    print("🔍 DEBUGGING CHANNEL RESOLVER")
    print("=" * 60)
    
    # Check if ChromaDB exists
    chromadb_path = "./data/chromadb/chroma.sqlite3"
    print(f"ChromaDB path: {chromadb_path}")
    print(f"ChromaDB exists: {Path(chromadb_path).exists()}")
    
    if Path(chromadb_path).exists():
        # Check database structure
        print("\n📋 DATABASE STRUCTURE")
        print("-" * 40)
        
        conn = sqlite3.connect(chromadb_path)
        
        # List all tables
        tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        print(f"Tables: {[t[0] for t in tables]}")
        
        # Check if embedding_metadata table exists
        if any('embedding_metadata' in str(t) for t in tables):
            print("✅ embedding_metadata table found")
            
            # Check sample data
            sample_query = '''
            SELECT DISTINCT key, COUNT(*) as count 
            FROM embedding_metadata 
            GROUP BY key 
            ORDER BY count DESC 
            LIMIT 10
            '''
            keys = conn.execute(sample_query).fetchall()
            print(f"Metadata keys: {keys}")
            
            # Check for channel_name specifically
            channel_query = '''
            SELECT DISTINCT string_value 
            FROM embedding_metadata 
            WHERE key = 'channel_name' 
            LIMIT 10
            '''
            channels = conn.execute(channel_query).fetchall()
            print(f"Sample channels: {[c[0] for c in channels]}")
            
        else:
            print("❌ embedding_metadata table not found")
            print(f"Available tables: {[t[0] for t in tables]}")
        
        conn.close()
    
    # Test resolver
    print("\n📋 TESTING RESOLVER")
    print("-" * 40)
    
    resolver = ChannelResolver()
    
    # Try to refresh cache
    cache_refreshed = resolver.refresh_cache()
    print(f"Cache refresh success: {cache_refreshed}")
    print(f"Cached channels: {len(resolver._channel_cache)}")
    print(f"Name mappings: {len(resolver._name_to_id_cache)}")
    
    if resolver._channel_cache:
        print("Sample cached channels:")
        for i, (channel_id, info) in enumerate(list(resolver._channel_cache.items())[:5]):
            print(f"  {channel_id}: {info.name} ({info.message_count} messages)")
    
    # Test specific resolution
    test_channel_id = "1363537366110703937"
    resolved_name = resolver.resolve_channel_name(test_channel_id)
    print(f"\nTest resolution: {test_channel_id} → {resolved_name}")
    
    # Try reverse lookup (by name)
    channel_info = resolver.get_channel_info(test_channel_id)
    if channel_info:
        print(f"Channel info: {channel_info.name} (ID: {channel_info.id})")
    else:
        print("No channel info found")

if __name__ == "__main__":
    debug_channel_resolver()
