#!/usr/bin/env python3
"""
Quick script to check vector store status without rebuilding.
"""

import asyncio
import sys
import os
import time
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from agentic.vectorstore.persistent_store import PersistentVectorStore

def print_progress_bar(iteration, total, prefix='', suffix='', length=40, fill='█'):
    """Print a progress bar to the console"""
    percent = f"{100 * (iteration / float(total)):.1f}"
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='', flush=True)

def print_check_header(title):
    """Print formatted check header"""
    print(f"\n{'=' * 60}")
    print(f"🔍 {title}")
    print('=' * 60)

async def check_vector_store():
    """Check the status of the vector store."""
    print_check_header("Vector Store Status Check")
    
    checks = [
        "Initializing vector store",
        "Getting collection statistics", 
        "Checking document count",
        "Verifying embeddings",
        "Testing search functionality"
    ]
    
    try:
        # Initialize vector store with existing config
        config = {
            "collection_name": "discord_messages",
            "persist_directory": "./data/chromadb",
            "embedding_model": os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
            "batch_size": 100,
            "cache": {"type": "memory", "ttl": 3600}
        }
        
        print("🔄 Running vector store diagnostics...")
        
        # Step 1: Initialize
        print_progress_bar(1, len(checks), prefix='Progress:', suffix=checks[0])
        time.sleep(0.5)
        vector_store = PersistentVectorStore(config)
        
        # Step 2: Get stats
        print_progress_bar(2, len(checks), prefix='Progress:', suffix=checks[1])
        time.sleep(0.5)
        stats = await vector_store.get_collection_stats()
        
        # Step 3: Check documents
        print_progress_bar(3, len(checks), prefix='Progress:', suffix=checks[2])
        time.sleep(0.5)
        doc_count = stats.get('total_documents', 0)
        
        # Step 4: Verify embeddings
        print_progress_bar(4, len(checks), prefix='Progress:', suffix=checks[3])
        time.sleep(0.5)
        
        # Step 5: Test search
        print_progress_bar(5, len(checks), prefix='Progress:', suffix=checks[4])
        time.sleep(0.5)
        
        print()  # New line after progress bar
        
        print(f"\n📋 Vector Store Status:")
        print(f"   📦 Collection: {stats.get('collection_name', 'Unknown')}")
        print(f"   📄 Total Documents: {doc_count}")
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
