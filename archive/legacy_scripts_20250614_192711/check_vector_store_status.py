#!/usr/bin/env python3
"""
Quick diagnostic script to check the current state of the vector store.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agentic.vectorstore.persistent_store import PersistentVectorStore

async def check_vector_store_status():
    """Check current vector store status and content"""
    print("🔍 VECTOR STORE DIAGNOSTIC")
    print("=" * 50)
    
    try:
        config = {
            "collection_name": "discord_messages",
            "persist_directory": "./data/chromadb",
            "embedding_model": "text-embedding-3-small"
        }
        
        vector_store = PersistentVectorStore(config)
        
        if vector_store.collection:
            count = vector_store.collection.count()
            print(f"📊 Total messages in vector store: {count}")
            
            if count > 0:
                # Get sample data
                sample_data = vector_store.collection.get(limit=3, include=["metadatas"])
                
                if sample_data and sample_data.get("metadatas"):
                    print(f"\n📋 Sample message metadata:")
                    for i, metadata in enumerate(sample_data["metadatas"][:3], 1):
                        print(f"   Message {i}:")
                        print(f"     Channel: {metadata.get('channel_name', 'N/A')}")
                        print(f"     Author: {metadata.get('author_display_name', metadata.get('author_username', 'N/A'))}")
                        print(f"     Timestamp: {metadata.get('timestamp', 'N/A')}")
                        print(f"     Fields: {len(metadata)} total")
                
                # Test a search
                print(f"\n🔍 Testing search functionality...")
                results = await vector_store.similarity_search("test query", k=3)
                print(f"   Search results: {len(results)} found")
                
                # Test filter search
                print(f"\n🎯 Testing filtered search...")
                filter_results = await vector_store.filter_search(
                    filters={}, 
                    k=3, 
                    sort_by="timestamp"
                )
                print(f"   Filter results: {len(filter_results)} found")
                
                if filter_results:
                    print(f"   Latest message timestamp: {filter_results[0].get('timestamp', 'N/A')}")
                
            else:
                print("❌ Vector store is EMPTY!")
                print("   This explains why the bot returns no results.")
                
        else:
            print("❌ Vector store collection not available!")
        
        print(f"\n💡 DIAGNOSIS:")
        if vector_store.collection and vector_store.collection.count() > 0:
            print("   ✅ Vector store is working correctly")
        else:
            print("   ❌ Vector store is empty - need to re-index data")
            print("   📝 SOLUTION: Run a complete data restoration")
            
    except Exception as e:
        print(f"❌ Error checking vector store: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(check_vector_store_status())
