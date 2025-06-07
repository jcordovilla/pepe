#!/usr/bin/env python3
"""
Fix embedding model dimension mismatch and rebuild FAISS index.

This script resolves the issue where the FAISS index contains 1536-dimensional 
vectors (from OpenAI embeddings) but the current configuration specifies a 
768-dimensional model (msmarco-distilbert-base-v4).
"""

import os
import shutil
import time
from datetime import datetime
from core.ai_client import get_ai_client
from core.config import get_config

def backup_current_index():
    """Backup the current FAISS index before rebuilding."""
    config = get_config()
    index_dir = config.faiss_index_path
    
    if os.path.exists(index_dir):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = f"{index_dir}_backup_{timestamp}"
        
        print(f"🔄 Backing up current index to {backup_dir}")
        shutil.copytree(index_dir, backup_dir)
        print(f"✅ Backup created: {backup_dir}")
        return backup_dir
    else:
        print("ℹ️  No existing index to backup")
        return None

def verify_embedding_model():
    """Verify the current embedding model configuration and index compatibility."""
    import faiss
    
    config = get_config()
    ai_client = get_ai_client()
    
    print("=== Current Configuration ===")
    print(f"Model: {config.models.embedding_model}")
    print(f"Expected dimension: {config.models.embedding_dimension}")
    
    # Test embedding creation
    test_text = "Test embedding dimension"
    embedding = ai_client.create_embeddings(test_text)
    actual_dimension = embedding.shape[1] if len(embedding.shape) > 1 else embedding.shape[0]
    
    print(f"Actual dimension: {actual_dimension}")
    print(f"Test embedding shape: {embedding.shape}")
    
    # Check if model configuration is correct
    model_config_correct = actual_dimension == config.models.embedding_dimension
    if model_config_correct:
        print("✅ Model configuration is correct")
    else:
        print("❌ Model dimension mismatch detected")
        return False
    
    # Check if FAISS index exists and has correct dimensions
    index_path = os.path.join(config.faiss_index_path, "index.faiss")
    if os.path.exists(index_path):
        index = faiss.read_index(index_path)
        print(f"Current index dimension: {index.d}")
        
        if index.d == config.models.embedding_dimension:
            print("✅ Index dimension matches model")
            return True
        else:
            print("❌ Index dimension mismatch - rebuild required")
            return False
    else:
        print("⚠️  No existing index found - rebuild required")
        return False

def rebuild_index():
    """Rebuild the FAISS index with the correct embedding model."""
    from core.embed_store import build_faiss_index
    
    print("\n=== Rebuilding FAISS Index ===")
    print("This will recreate the index using msmarco-distilbert-base-v4 (768 dimensions)")
    
    # Remove existing index to force full rebuild
    config = get_config()
    index_dir = config.faiss_index_path
    if os.path.exists(index_dir):
        print(f"🗑️  Removing existing index: {index_dir}")
        shutil.rmtree(index_dir)
    
    # Rebuild index
    start_time = time.time()
    build_faiss_index()
    end_time = time.time()
    
    print(f"⏱️  Index rebuild completed in {end_time - start_time:.1f} seconds")

def verify_new_index():
    """Verify the new index has the correct dimensions."""
    import faiss
    import pickle
    
    config = get_config()
    index_path = os.path.join(config.faiss_index_path, "index.faiss")
    pkl_path = os.path.join(config.faiss_index_path, "index.pkl")
    
    if not os.path.exists(index_path):
        print("❌ Index file not found after rebuild")
        return False
    
    print("\n=== Verifying New Index ===")
    
    # Check FAISS index
    index = faiss.read_index(index_path)
    print(f"Index type: {type(index)}")
    print(f"Total vectors: {index.ntotal}")
    print(f"Vector dimension: {index.d}")
    
    # Check if dimension matches config
    config = get_config()
    if index.d == config.models.embedding_dimension:
        print("✅ Index dimension matches configuration")
        return True
    else:
        print(f"❌ Index dimension ({index.d}) doesn't match config ({config.models.embedding_dimension})")
        return False

def test_search_functionality():
    """Test that search works with the new index."""
    print("\n=== Testing Search Functionality ===")
    
    try:
        from core.rag_engine import load_vectorstore
        
        # Load vector store
        vector_store = load_vectorstore()
        
        # Test search
        query = "Python async programming help"
        results = vector_store.search(query, k=3)
        
        print(f"Search query: '{query}'")
        print(f"Results found: {len(results)}")
        
        for i, result in enumerate(results[:2]):  # Show top 2 results
            content_preview = result.get('content', '')[:100] + "..." if len(result.get('content', '')) > 100 else result.get('content', '')
            score = result.get('score', 0)
            print(f"  {i+1}. Score: {score:.3f}")
            print(f"     Content: {content_preview}")
        
        print("✅ Search functionality working")
        return True
        
    except Exception as e:
        print(f"❌ Search test failed: {e}")
        return False

def main():
    """Main execution function."""
    print("=== Discord Bot Embedding Model Fix ===")
    print("This script will fix the dimension mismatch between the FAISS index")
    print("and the current embedding model configuration.\n")
    
    # Step 1: Verify current model and index compatibility
    if not verify_embedding_model():
        print("\nProceeding with index rebuild to fix dimension mismatch...")
    else:
        print("\nModel and index are already compatible!")
        print("✅ No rebuild necessary")
        return
    
    # Step 2: Backup current index
    backup_dir = backup_current_index()
    
    # Step 3: Rebuild index
    try:
        rebuild_index()
    except Exception as e:
        print(f"❌ Index rebuild failed: {e}")
        if backup_dir:
            print(f"You can restore from backup: {backup_dir}")
        return
    
    # Step 4: Verify new index
    if not verify_new_index():
        print("❌ Index verification failed")
        return
    
    # Step 5: Test search functionality
    if not test_search_functionality():
        print("⚠️  Search test failed - index may have issues")
        return
    
    print("\n🎉 Successfully fixed embedding model configuration!")
    print("✅ FAISS index now uses msmarco-distilbert-base-v4 with 768 dimensions")
    print("✅ Search functionality verified")
    
    if backup_dir:
        print(f"\n💡 Backup location: {backup_dir}")
        print("You can delete the backup once you're confident the new index works well.")

if __name__ == "__main__":
    main()
