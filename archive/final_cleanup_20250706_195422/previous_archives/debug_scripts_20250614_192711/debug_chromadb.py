#!/usr/bin/env python3
"""
Debug ChromaDB to understand what's happening with the collections
"""

import os
import logging
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug_chromadb():
    """Debug ChromaDB collections and data"""
    try:
        # Initialize ChromaDB client
        persist_directory = "./data/chromadb"
        client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=False
            )
        )
        
        print("=" * 50)
        print("🔍 ChromaDB Debug Information")
        print("=" * 50)
        
        # List all collections
        collections = client.list_collections()
        print(f"📁 Total Collections: {len(collections)}")
        
        for i, collection in enumerate(collections):
            print(f"\n📦 Collection {i+1}: {collection.name}")
            print(f"   🆔 ID: {collection.id}")
            print(f"   📊 Metadata: {collection.metadata}")
            
            # Try to get collection stats without embedding function first
            try:
                count = collection.count()
                print(f"   📄 Document Count: {count}")
                
                if count > 0:
                    # Try to peek at a few documents
                    try:
                        sample = collection.peek(limit=3)
                        print(f"   👀 Sample IDs: {sample.get('ids', [])[:3]}")
                        if 'metadatas' in sample and sample['metadatas']:
                            print(f"   📋 Sample Metadata Keys: {list(sample['metadatas'][0].keys()) if sample['metadatas'][0] else 'None'}")
                    except Exception as e:
                        print(f"   ⚠️  Error peeking at documents: {e}")
                
            except Exception as e:
                print(f"   ❌ Error getting count: {e}")
        
        # Try to recreate the collection with proper embedding function
        print(f"\n🔄 Testing Collection Recreation...")
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            embedding_function = embedding_functions.OpenAIEmbeddingFunction(
                api_key=api_key,
                model_name="text-embedding-3-small"
            )
            
            try:
                # Try to get existing collection with embedding function
                collection = client.get_collection(
                    name="discord_messages",
                    embedding_function=embedding_function
                )
                print(f"✅ Successfully accessed collection with embedding function")
                print(f"   📄 Document Count: {collection.count()}")
                
            except Exception as e:
                print(f"❌ Error accessing collection with embedding function: {e}")
                
                # Try to create collection
                try:
                    collection = client.create_collection(
                        name="discord_messages_debug",
                        embedding_function=embedding_function,
                        metadata={"description": "Debug test collection"}
                    )
                    print(f"✅ Successfully created debug collection")
                    
                    # Clean up debug collection
                    client.delete_collection("discord_messages_debug")
                    print(f"🧹 Cleaned up debug collection")
                    
                except Exception as create_error:
                    print(f"❌ Error creating debug collection: {create_error}")
        
        print(f"\n✅ ChromaDB debug completed")
        
    except Exception as e:
        print(f"❌ Error debugging ChromaDB: {e}")
        logger.error(f"Error debugging ChromaDB: {e}")

if __name__ == "__main__":
    debug_chromadb()
