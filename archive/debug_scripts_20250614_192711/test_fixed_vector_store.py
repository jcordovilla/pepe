#!/usr/bin/env python3
"""
Quick test of the fixed vector store to verify error handling works
"""

import asyncio
import os
import sys
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agentic.vectorstore.persistent_store import PersistentVectorStore

async def test_fixed_vector_store():
    """Test the fixed vector store with problematic files"""
    print("🧪 TESTING FIXED VECTOR STORE ERROR HANDLING")
    print("=" * 60)
    
    try:
        config = {
            "collection_name": "discord_messages_test",
            "persist_directory": "./data/chromadb_test",
            "embedding_model": "text-embedding-3-small"
        }
        
        vector_store = PersistentVectorStore(config)
        print("✅ Vector store initialized")
        
        # Test with one of the previously problematic files
        problem_files = [
            "data/fetched_messages/1353058864810950737_1371894277512364073_messages.json",
            "data/fetched_messages/1353058864810950737_1366389827221717002_messages.json"
        ]
        
        for file_path in problem_files:
            if Path(file_path).exists():
                print(f"\n📄 Testing: {Path(file_path).name}")
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if "messages" in data and data["messages"]:
                    messages = data["messages"]
                    print(f"   Found {len(messages)} messages")
                    
                    # Test the add_messages method
                    try:
                        success = await vector_store.add_messages(messages)
                        if success:
                            print(f"   ✅ Successfully processed all messages!")
                        else:
                            print(f"   ❌ Failed to process messages")
                    except Exception as e:
                        print(f"   ❌ Error: {e}")
            else:
                print(f"⚠️ File not found: {file_path}")
        
        # Clean up test collection
        if vector_store.collection:
            count = vector_store.collection.count()
            print(f"\n📊 Test results: {count} messages processed successfully")
            
            # Clean up
            try:
                vector_store.client.delete_collection("discord_messages_test")
                print("🗑️ Cleaned up test collection")
            except:
                pass
        
        print(f"\n✅ Vector store error handling test completed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_fixed_vector_store())
