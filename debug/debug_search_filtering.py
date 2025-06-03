#!/usr/bin/env python3
"""
Debug search filtering issues in Discord agent responses.
Focus on channel filtering, author information, and result count limiting.
"""

import asyncio
import sys
import os
import logging
from typing import Dict, List, Any

# Add the project root to Python path
sys.path.append('.')

# Set up basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug_vector_store_data():
    """Check what's actually in the vector store"""
    try:
        from agentic.vectorstore.persistent_store import PersistentVectorStore
        
        # Mock config for testing
        config = {
            "collection_name": "discord_messages",
            "persist_directory": "./data/chromadb",
            "embedding_model": "text-embedding-3-small"
        }
        
        # Initialize with test API key to avoid errors
        os.environ["OPENAI_API_KEY"] = "test-key-for-testing"
        
        store = PersistentVectorStore(config)
        
        if store.collection is None:
            print("❌ Collection not initialized")
            return
        
        # Check collection stats
        total_docs = store.collection.count()
        print(f"📊 Total documents in collection: {total_docs}")
        
        # Get a sample of documents to inspect
        if total_docs > 0:
            sample_results = store.collection.get(limit=10, include=["documents", "metadatas"])
            
            if sample_results and sample_results.get("documents"):
                print("\n📝 Sample documents:")
                for i, (doc, meta) in enumerate(zip(sample_results["documents"], sample_results["metadatas"])):
                    print(f"\nDocument {i+1}:")
                    print(f"  Channel: {meta.get('channel_name', 'Unknown')}")
                    print(f"  Author: {meta.get('author_username', 'Unknown')}")
                    print(f"  Content: {doc[:100]}...")
                    print(f"  Timestamp: {meta.get('timestamp', 'Unknown')}")
            
            # Check specific channel filtering
            print(f"\n🔍 Checking ❌💻non-coders-learning channel messages...")
            channel_filter = {"channel_name": "❌💻non-coders-learning"}
            channel_results = store.collection.get(
                where=channel_filter,
                limit=5,
                include=["documents", "metadatas"]
            )
            
            if channel_results and channel_results.get("documents"):
                print(f"Found {len(channel_results['documents'])} messages in ❌💻non-coders-learning:")
                for i, (doc, meta) in enumerate(zip(channel_results["documents"], channel_results["metadatas"])):
                    print(f"  {i+1}. Author: {meta.get('author_username', 'Unknown')} - {doc[:50]}...")
            else:
                print("❌ No messages found in ❌💻non-coders-learning channel")
                
            # Check what channels exist
            print(f"\n📋 Available channels:")
            all_results = store.collection.get(limit=100, include=["metadatas"])
            if all_results and all_results.get("metadatas"):
                channels = set()
                for meta in all_results["metadatas"]:
                    channel_name = meta.get("channel_name", "Unknown")
                    if channel_name != "Unknown":
                        channels.add(channel_name)
                
                for channel in sorted(channels):
                    print(f"  - {channel}")
        
    except Exception as e:
        print(f"❌ Error debugging vector store: {e}")
        import traceback
        traceback.print_exc()

def debug_search_flow():
    """Debug the actual search flow without needing API keys"""
    try:
        # Mock a search query that should find messages in ❌💻non-coders-learning
        print("\n🔍 Testing search flow...")
        
        # Test channel name resolution
        from agentic.vectorstore.persistent_store import PersistentVectorStore
        
        config = {
            "collection_name": "discord_messages", 
            "persist_directory": "./data/chromadb"
        }
        
        os.environ["OPENAI_API_KEY"] = "test-key-for-testing"
        store = PersistentVectorStore(config)
        
        if store.collection is None:
            print("❌ Collection not initialized")
            return
            
        # Test filtering by channel name
        test_filters = {
            "channel_name": "❌💻non-coders-learning"
        }
        
        where_clause = store._build_where_clause(test_filters)
        print(f"Where clause: {where_clause}")
        
        # Try to get messages with this filter
        results = store.collection.get(
            where=where_clause,
            limit=5,
            include=["documents", "metadatas"]
        )
        
        if results and results.get("documents"):
            print(f"✅ Found {len(results['documents'])} messages with channel filter")
            for i, (doc, meta) in enumerate(zip(results["documents"], results["metadatas"])):
                print(f"  {i+1}. Channel: {meta.get('channel_name')}")
                print(f"      Author: {meta.get('author_username', 'Unknown')}")
                print(f"      Content: {doc[:60]}...")
        else:
            print("❌ No messages found with channel filter")
            
        # Test without filters to see what we get
        print(f"\n📋 Testing without filters (recent messages):")
        all_results = store.collection.get(
            limit=10,
            include=["documents", "metadatas"]
        )
        
        if all_results and all_results.get("documents"):
            print(f"Found {len(all_results['documents'])} messages without filters:")
            channel_counts = {}
            for meta in all_results["metadatas"]:
                channel = meta.get("channel_name", "Unknown")
                channel_counts[channel] = channel_counts.get(channel, 0) + 1
                
            for channel, count in channel_counts.items():
                print(f"  - {channel}: {count} messages")
        
    except Exception as e:
        print(f"❌ Error in search flow debug: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main debug function"""
    print("🐛 Debugging Discord agent search filtering issues...")
    print("=" * 60)
    
    # Debug vector store data
    debug_vector_store_data()
    
    # Debug search flow
    debug_search_flow()
    
    print("\n" + "=" * 60)
    print("🏁 Debug complete")

if __name__ == "__main__":
    main()
