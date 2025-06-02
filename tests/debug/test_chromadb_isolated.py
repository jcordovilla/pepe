#!/usr/bin/env python3
"""
Isolated test for ChromaDB document insertion with DefaultEmbeddingFunction.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Set test environment variables
os.environ["CHROMA_OPENAI_API_KEY"] = "test-key-for-testing"
os.environ["OPENAI_API_KEY"] = "test-key-for-testing"

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import tempfile
import shutil

def test_isolated_chromadb():
    """Test ChromaDB with DefaultEmbeddingFunction in isolation."""
    print("🔧 Testing ChromaDB with DefaultEmbeddingFunction")
    print("=" * 60)
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    print(f"📁 Using temp directory: {temp_dir}")
    
    try:
        # Test 1: Initialize client
        print("\n📝 Test 1: Initialize ChromaDB client...")
        client = chromadb.PersistentClient(
            path=temp_dir,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        print("✅ Client initialized successfully")
        
        # Test 2: Create embedding function
        print("\n📝 Test 2: Create DefaultEmbeddingFunction...")
        embedding_function = embedding_functions.DefaultEmbeddingFunction()
        print("✅ DefaultEmbeddingFunction created successfully")
        
        # Test 3: Create collection
        print("\n📝 Test 3: Create collection...")
        collection = client.create_collection(
            name="test_collection",
            embedding_function=embedding_function,
            metadata={"description": "Test collection"}
        )
        print("✅ Collection created successfully")
        
        # Test 4: Add a simple document
        print("\n📝 Test 4: Add simple document...")
        collection.add(
            documents=["This is a test document"],
            metadatas=[{"test": "metadata"}],
            ids=["test_id_1"]
        )
        print("✅ Document added successfully")
        
        # Test 5: Check collection count
        print("\n📝 Test 5: Check collection count...")
        count = collection.count()
        print(f"✅ Collection has {count} documents")
        
        # Test 6: Upsert document (this is what's failing)
        print("\n📝 Test 6: Upsert document...")
        collection.upsert(
            documents=["This is an upserted document"],
            metadatas=[{"test": "upsert_metadata"}],
            ids=["test_id_2"]
        )
        print("✅ Document upserted successfully")
        
        # Test 7: Final count
        print("\n📝 Test 7: Final count check...")
        final_count = collection.count()
        print(f"✅ Final collection has {final_count} documents")
        
        print("\n" + "=" * 60)
        print("✅ All tests passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        print("Full traceback:")
        traceback.print_exc()
        
    finally:
        # Clean up
        try:
            shutil.rmtree(temp_dir)
        except:
            pass

if __name__ == "__main__":
    test_isolated_chromadb()
