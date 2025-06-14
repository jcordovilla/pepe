#!/usr/bin/env python3
"""
Debug script to check what vector store stats are actually returning
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from agentic.vectorstore.persistent_store import PersistentVectorStore

async def debug_vector_stats():
    """Debug the vector store statistics"""
    
    print("🔍 Debugging Vector Store Statistics")
    print("=" * 50)
    
    try:
        # Initialize the vector store
        config = {
            "collection_name": "discord_messages",
            "persist_directory": "./data/chromadb",
            "embedding_model": "text-embedding-3-small"
        }
        
        print("📝 Initializing vector store...")
        vector_store = PersistentVectorStore(config)
        
        print("📊 Getting collection statistics...")
        stats = await vector_store.get_collection_stats()
        
        print(f"📋 Raw Stats Object: {stats}")
        print()
        
        print("🔍 Detailed Analysis:")
        print(f"  📄 Total Documents: {stats.get('total_documents', 'N/A')}")
        print(f"  🏷️  Collection Name: {stats.get('collection_name', 'N/A')}")
        print(f"  🤖 Embedding Model: {stats.get('embedding_model', 'N/A')}")
        print()
        
        if "content_stats" in stats:
            content_stats = stats["content_stats"]
            print("📈 Content Statistics Found:")
            print(f"  📊 Raw content_stats: {content_stats}")
            print(f"  🔢 Total Tokens: {content_stats.get('total_tokens', 'NOT FOUND')}")
            print(f"  📏 Average Length: {content_stats.get('avg_length', 'NOT FOUND')}")
            print(f"  📐 Max Length: {content_stats.get('max_length', 'NOT FOUND')}")
            print(f"  📏 Min Length: {content_stats.get('min_length', 'NOT FOUND')}")
        else:
            print("❌ No content_stats found in response")
        
        if "top_channels" in stats:
            top_channels = stats["top_channels"]
            print(f"\n🏷️  Top Channels ({len(top_channels)} found):")
            for i, channel in enumerate(top_channels[:5]):
                print(f"  {i+1}. {channel.get('name', 'Unknown')}: {channel.get('count', 0)} messages")
        else:
            print("❌ No top_channels found in response")
        
        print(f"\n✅ Debug completed")
        
    except Exception as e:
        print(f"❌ Error during debugging: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_vector_stats())
