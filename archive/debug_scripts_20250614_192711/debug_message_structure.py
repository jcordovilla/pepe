#!/usr/bin/env python3
"""
Debug script to check message data structure for username and timestamp issues.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agentic.vectorstore.persistent_store import PersistentVectorStore

async def debug_message_structure():
    """Check the actual structure of messages in the vector store"""
    print("🔍 Debugging Message Structure")
    print("=" * 60)
    
    try:
        config = {
            "collection_name": "discord_messages",
            "persist_directory": "./data/chromadb",
            "embedding_model": "text-embedding-3-small"
        }
        
        vector_store = PersistentVectorStore(config)
        
        # Get a few sample messages
        print("📋 Getting sample messages...")
        if vector_store.collection:
            sample_data = vector_store.collection.get(
                limit=3,
                include=["metadatas", "documents"]
            )
            
            if sample_data and sample_data.get("metadatas"):
                print(f"✅ Found {len(sample_data['metadatas'])} sample messages\n")
                
                for i, metadata in enumerate(sample_data["metadatas"][:3], 1):
                    print(f"📨 Message {i}:")
                    print(f"   Raw metadata keys: {list(metadata.keys())}")
                    
                    # Check author structure
                    if "author" in metadata:
                        author = metadata["author"]
                        print(f"   Author (type: {type(author)}): {author}")
                    elif "author_username" in metadata:
                        print(f"   Author username: {metadata['author_username']}")
                    elif "username" in metadata:
                        print(f"   Username: {metadata['username']}")
                    else:
                        print(f"   ❌ No author info found in metadata")
                    
                    # Check timestamp
                    if "timestamp" in metadata:
                        timestamp = metadata["timestamp"]
                        print(f"   Timestamp (type: {type(timestamp)}): {timestamp}")
                    
                    # Check other relevant fields
                    channel_name = metadata.get("channel_name", "Missing")
                    print(f"   Channel: {channel_name}")
                    
                    print("   ─────")
                
                # Also check document content
                if sample_data.get("documents"):
                    print(f"\n📄 Sample document content:")
                    print(f"   {sample_data['documents'][0][:100]}...")
            
        print("\n✅ Debug completed")
        
    except Exception as e:
        print(f"❌ Error during debugging: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_message_structure())
