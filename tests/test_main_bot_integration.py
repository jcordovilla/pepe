#!/usr/bin/env python3
"""
Test script to validate main bot integration with ChromaDB after environment variable fix.
This tests that existing collections can be loaded properly with the user's API key.
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the current directory to Python path for imports
sys.path.append('/Users/jose/Documents/apps/discord-bot-v2')

def test_main_integration():
    """Test main bot integration with fixed environment variables"""
    
    print("🔧 Testing Main Bot Integration with ChromaDB")
    print("=" * 60)
    
    # Verify environment variables are set
    print("\n📋 Environment Variables Check:")
    openai_key = os.getenv('OPENAI_API_KEY')
    chroma_key = os.getenv('CHROMA_OPENAI_API_KEY')
    
    print(f"✅ OPENAI_API_KEY: {'Set' if openai_key else 'Missing'}")
    print(f"✅ CHROMA_OPENAI_API_KEY: {'Set' if chroma_key else 'Missing'}")
    
    if not openai_key or not chroma_key:
        print("❌ Environment variables not properly set!")
        return False
    
    if openai_key != chroma_key:
        print("⚠️  Warning: OPENAI_API_KEY and CHROMA_OPENAI_API_KEY are different")
    
    # Test ChromaDB collection loading
    print("\n🗃️  Testing ChromaDB Collection Loading:")
    try:
        from agentic.vectorstore.persistent_store import PersistentVectorStore
        
        # Initialize the store - this should now work with existing collections
        config = {
            "persist_directory": "./data/vectorstore",
            "embedding_model": "text-embedding-3-small"
        }
        store = PersistentVectorStore(config)
        print("✅ PersistentVectorStore initialized successfully")
        
        # Test reaction search capability
        print("\n🔍 Testing Reaction Search Capability:")
        
        # Use the async reaction_search method
        import asyncio
        async def test_search():
            # Test reaction search - this is an async method
            test_results = await store.reaction_search(k=5, sort_by="total_reactions")
            return test_results
        
        test_results = asyncio.run(test_search())
        print(f"✅ Reaction search executed successfully")
        print(f"📊 Search returned {len(test_results)} results")
        
        # Test collection health
        print("\n🏥 Testing Collection Health:")
        async def test_health():
            stats = await store.get_collection_stats()
            return stats
        
        stats = asyncio.run(test_health())
        collection_count = stats.get('total_documents', 0)
        print(f"✅ Collection contains {collection_count} documents")
        
        print("\n🎉 All Integration Tests Passed!")
        print("=" * 60)
        print("✅ Main bot is ready for production with reaction search!")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {str(e)}")
        print(f"📋 Error type: {type(e).__name__}")
        return False

if __name__ == "__main__":
    success = test_main_integration()
    if success:
        print("\n🚀 Ready to deploy reaction search functionality!")
    else:
        print("\n🔧 Please check the error above and retry.")
    
    sys.exit(0 if success else 1)
