#!/usr/bin/env python3
"""
Debug Discord Message Search
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import logging
from core.rag_engine import LocalVectorStore
from core.config import get_config

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def debug_discord_search():
    """Debug the Discord message search step by step."""
    print("🔍 Debugging Discord Message Search")
    print("=" * 50)
    
    try:
        # Load config
        config = get_config()
        print(f"Index path: {config.faiss_index_path}")
        
        # Initialize vector store
        store = LocalVectorStore(config.faiss_index_path)
        
        # Load the store
        print("Loading vector store...")
        store.load()
        print(f"Loaded {store._index.ntotal} vectors")
        print(f"Metadata entries: {len(store._metadata)}")
        print(f"Message order entries: {len(store._message_order)}")
        
        # Try a simple search
        print("\nTesting search...")
        query = "AI agents"
        results = store.search(query, k=3)
        print(f"Search returned {len(results)} results")
        
        for i, result in enumerate(results):
            print(f"Result {i+1}: {result.get('content', '')[:100]}...")
            print(f"  Score: {result.get('score', 'N/A')}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_discord_search()
