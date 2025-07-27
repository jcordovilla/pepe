#!/usr/bin/env python3
"""
Rebuild the vector store from scratch
This script will delete the corrupted ChromaDB collection and rebuild it from the SQLite database
"""

import asyncio
import os
import shutil
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

from agentic.vectorstore.persistent_store import PersistentVectorStore

async def rebuild_vectorstore():
    """Rebuild the vector store from scratch."""
    print("🔧 Starting vector store rebuild...")
    
    # Step 1: Backup current ChromaDB directory
    chroma_dir = Path("./data/chromadb")
    backup_dir = Path("./data/chromadb_corrupted_backup")
    
    if chroma_dir.exists():
        print(f"📦 Backing up current ChromaDB directory to {backup_dir}")
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        shutil.copytree(chroma_dir, backup_dir)
        print("✅ Backup created")
    
    # Step 2: Delete the corrupted ChromaDB directory
    print("🗑️ Removing corrupted ChromaDB directory")
    if chroma_dir.exists():
        shutil.rmtree(chroma_dir)
    print("✅ Corrupted directory removed")
    
    # Step 3: Recreate ChromaDB directory
    print("📁 Creating fresh ChromaDB directory")
    chroma_dir.mkdir(parents=True, exist_ok=True)
    print("✅ Fresh directory created")
    
    # Step 4: Test that we can create a new collection
    print("🧪 Testing new collection creation")
    try:
        # Initialize ChromaDB client
        client = chromadb.PersistentClient(
            path=str(chroma_dir),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=False
            )
        )
        
        # Initialize embedding function
        embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="msmarco-distilbert-base-v4"
        )
        
        # Create new collection
        collection = client.create_collection(
            name="discord_messages",
            embedding_function=embedding_function,
            metadata={"description": "Discord messages vector store"}
        )
        
        print(f"✅ Successfully created new collection: {collection.name}")
        print(f"📊 Collection count: {collection.count()}")
        
        # Clean up test collection
        client.delete_collection(name="discord_messages")
        print("✅ Test collection cleaned up")
        
    except Exception as e:
        print(f"❌ Error testing collection creation: {e}")
        return False
    
    # Step 5: Run the indexing script to rebuild from SQLite
    print("🔄 Running message indexing to rebuild vector store")
    try:
        from scripts.index_database_messages import DatabaseMessageIndexer
        
        indexer = DatabaseMessageIndexer()
        await indexer.index_all_messages()
        
        print("✅ Vector store rebuild completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error during indexing: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(rebuild_vectorstore())
    if success:
        print("\n🎉 Vector store rebuild completed successfully!")
        print("📊 You can now run the digest agent to test it.")
    else:
        print("\n❌ Vector store rebuild failed!")
        print("🔍 Check the logs above for more details.") 