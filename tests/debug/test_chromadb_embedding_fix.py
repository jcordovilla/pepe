#!/usr/bin/env python3
"""
Test script to fix ChromaDB embedding function compatibility issue.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Set test environment variables
os.environ["CHROMA_OPENAI_API_KEY"] = "test-key-for-testing"
os.environ["OPENAI_API_KEY"] = "test-key-for-testing"

from chromadb.utils import embedding_functions
import chromadb
from chromadb.config import Settings

def test_embedding_function_fix():
    """Test different approaches to fix embedding function initialization."""
    print("🔧 Testing ChromaDB Embedding Function Fixes")
    print("=" * 60)
    
    # Test 1: Using default environment variable name
    print("\n📝 Test 1: Using CHROMA_OPENAI_API_KEY...")
    try:
        embedding_function = embedding_functions.OpenAIEmbeddingFunction(
            model_name="text-embedding-3-small"
        )
        print("✅ Test 1 PASSED: Default env var worked")
    except Exception as e:
        print(f"❌ Test 1 FAILED: {e}")
    
    # Test 2: Explicitly passing API key
    print("\n📝 Test 2: Explicitly passing API key...")
    try:
        embedding_function = embedding_functions.OpenAIEmbeddingFunction(
            api_key="test-key-for-testing",
            model_name="text-embedding-3-small"
        )
        print("✅ Test 2 PASSED: Explicit API key worked")
    except Exception as e:
        print(f"❌ Test 2 FAILED: {e}")
    
    # Test 3: Using different model
    print("\n📝 Test 3: Using text-embedding-ada-002...")
    try:
        embedding_function = embedding_functions.OpenAIEmbeddingFunction(
            api_key="test-key-for-testing",
            model_name="text-embedding-ada-002"
        )
        print("✅ Test 3 PASSED: ada-002 model worked")
    except Exception as e:
        print(f"❌ Test 3 FAILED: {e}")
    
    # Test 4: Create a test collection to see if the issue is in collection creation
    print("\n📝 Test 4: Creating test collection...")
    try:
        client = chromadb.Client(Settings(
            anonymized_telemetry=False,
            allow_reset=True
        ))
        
        embedding_function = embedding_functions.OpenAIEmbeddingFunction(
            api_key="test-key-for-testing",
            model_name="text-embedding-3-small"
        )
        
        collection = client.create_collection(
            name="test_collection_fix",
            embedding_function=embedding_function
        )
        print("✅ Test 4 PASSED: Collection creation worked")
        
        # Test adding a document
        collection.add(
            documents=["This is a test document"],
            ids=["test_id_1"]
        )
        print("✅ Test 4 PASSED: Document addition worked")
        
    except Exception as e:
        print(f"❌ Test 4 FAILED: {e}")
        import traceback
        print("Full traceback:")
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("🎯 ChromaDB Embedding Function Test Complete")

if __name__ == "__main__":
    test_embedding_function_fix()
