#!/usr/bin/env python3
"""
Quick script to check vector store status without rebuilding.
"""

import asyncio
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from agentic.vectorstore.persistent_store import PersistentVectorStore


async def check_vector_store():
    """Check the status of the vector store."""
    print("🔍 Checking Vector Store Status")
    print("=" * 50)
    
    try:
        # Initialize vector store with existing config
        config = {
            "collection_name": "discord_messages",
            "persist_directory": "./data/chromadb",
            "embedding_model": "text-embedding-3-small",
            "batch_size": 100,
            "cache": {"type": "memory", "ttl": 3600}
        }
        
        print("📝 Initializing vector store...")
        vector_store = PersistentVectorStore(config)
        print("✅ Vector store initialized")
        
        # Get collection stats
        print("\n📊 Getting collection statistics...")
        stats = await vector_store.get_collection_stats()
        
        print(f"\n📋 Vector Store Status:")
        print(f"   📦 Collection: {stats.get('collection_name', 'Unknown')}")
        print(f"   📄 Total Documents: {stats.get('total_documents', 0)}")
        print(f"   🤖 Embedding Model: {stats.get('embedding_model', 'Unknown')}")
        print(f"   🕒 Last Updated: {stats.get('last_updated', 'Unknown')}")
        
        # Show top channels if available
        if 'top_channels' in stats and stats['top_channels']:
            print(f"\n🏷️  Top Channels:")
            for channel in stats['top_channels'][:5]:
                print(f"   • {channel.get('name', 'Unknown')}: {channel.get('count', 0)} messages")
        
        # Show content stats if available
        if 'content_stats' in stats:
            content_stats = stats['content_stats']
            print(f"\n📝 Content Statistics:")
            print(f"   • Average length: {content_stats.get('avg_length', 0):.0f} characters")
            print(f"   • Total characters: {content_stats.get('total_characters', 0):,}")
        
        # Performance stats
        if 'performance_stats' in stats:
            perf_stats = stats['performance_stats']
            print(f"\n⚡ Performance Stats:")
            print(f"   • Total searches: {perf_stats.get('total_searches', 0)}")
            print(f"   • Cache hits: {perf_stats.get('cache_hits', 0)}")
            print(f"   • Cache misses: {perf_stats.get('cache_misses', 0)}")
        
        # Health check
        print("\n🏥 Health Check...")
        health = await vector_store.health_check()
        
        print(f"   Status: {health.get('status', 'Unknown')}")
        if 'checks' in health:
            for check_name, check_result in health['checks'].items():
                status = check_result.get('status', 'unknown')
                if status == 'healthy':
                    print(f"   ✅ {check_name}: {status}")
                elif status == 'warning' or status == 'degraded':
                    print(f"   ⚠️  {check_name}: {status}")
                else:
                    print(f"   ❌ {check_name}: {status}")
        
        # Close the vector store
        await vector_store.close()
        print("\n✅ Vector store check completed")
        
        # Summary
        total_docs = stats.get('total_documents', 0)
        if total_docs == 0:
            print("\n🚨 ISSUE: Vector store is empty!")
            print("   Need to run the embedding step to populate it with messages.")
        else:
            print(f"\n✅ Vector store has {total_docs:,} documents and is ready for searches.")
            
    except Exception as e:
        print(f"❌ Error checking vector store: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(check_vector_store())
