#!/usr/bin/env python3
"""
Simple test for vector store initialization with test keys.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Set test environment variables BEFORE importing
os.environ["CHROMA_OPENAI_API_KEY"] = "test-key-for-testing"
os.environ["OPENAI_API_KEY"] = "test-key-for-testing"

from agentic.vectorstore.persistent_store import PersistentVectorStore

def test_vector_store_init():
    """Test vector store initialization."""
    print("🔧 Testing Vector Store Initialization")
    print("=" * 50)
    
    try:
        config = {
            "collection_name": "simple_test",
            "persist_directory": "./data/test_simple",
            "embedding_model": "text-embedding-3-small",
            "batch_size": 10
        }
        
        print("📝 Creating vector store...")
        vector_store = PersistentVectorStore(config)
        print("✅ Vector store created successfully")
        
        print(f"📋 Embedding function type: {type(vector_store.embedding_function)}")
        print(f"📋 Collection: {vector_store.collection}")
        
        # Test simple add
        print("\n📝 Testing simple message add...")
        test_message = [{
            "message_id": "test_001",
            "content": "This is a simple test message",
            "channel_id": "test_channel",
            "channel_name": "test",
            "guild_id": "test_guild",
            "author": {"id": "test_user", "username": "tester"},
            "timestamp": "2023-01-01T00:00:00Z",
            "reactions": []
        }]
        
        import asyncio
        result = asyncio.run(vector_store.add_messages(test_message))
        print(f"✅ Add result: {result}")
        
        # Check count
        count = vector_store.collection.count()
        print(f"📊 Collection count: {count}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_vector_store_init()
