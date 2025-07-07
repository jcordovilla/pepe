#!/usr/bin/env python3
"""
Fix Vector Store Embedding Dimension Mismatch

The current vector store was created with text-embedding-ada-002 (1536 dimensions)
but we're trying to query with text-embedding-3-small (384 dimensions).

This script will:
1. Backup the current vector store
2. Recreate it with the correct embedding model
3. Re-process the messages with consistent embeddings
"""

import asyncio
import json
import os
import shutil
from pathlib import Path
from datetime import datetime
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agentic.vectorstore.persistent_store import PersistentVectorStore

def print_header(title: str):
    """Print formatted header"""
    print(f"\n{'=' * 60}")
    print(f"🔧 {title}")
    print('=' * 60)

def backup_vector_store():
    """Backup the current vector store"""
    print_header("Backing Up Current Vector Store")
    
    chromadb_path = Path("data/chromadb")
    if not chromadb_path.exists():
        print("❌ No vector store to backup")
        return False
    
    backup_path = Path(f"data/chromadb_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    try:
        shutil.copytree(chromadb_path, backup_path)
        print(f"✅ Vector store backed up to: {backup_path}")
        return True
    except Exception as e:
        print(f"❌ Backup failed: {e}")
        return False

def remove_vector_store():
    """Remove the current vector store"""
    print_header("Removing Current Vector Store")
    
    chromadb_path = Path("data/chromadb")
    if chromadb_path.exists():
        try:
            shutil.rmtree(chromadb_path)
            print("✅ Current vector store removed")
            return True
        except Exception as e:
            print(f"❌ Failed to remove vector store: {e}")
            return False
    else:
        print("✅ No vector store to remove")
        return True

async def recreate_vector_store():
    """Recreate vector store with correct embedding model"""
    print_header("Recreating Vector Store with Correct Embedding Model")
    
    try:
        # Create new vector store with consistent embedding model
        config = {
            "collection_name": "discord_messages",
            "persist_directory": "./data/chromadb",
            "embedding_model": os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),  # Consistent model
            "batch_size": 100,
            "cache": {"type": "memory", "ttl": 3600}
        }
        
        print("🔄 Creating new vector store...")
        vector_store = PersistentVectorStore(config)
        print("✅ New vector store created")
        
        # Load and process message files
        data_dir = Path("data/fetched_messages")
        message_files = list(data_dir.glob("*.json"))
        message_files = [f for f in message_files if not f.name.startswith('fetch_stats')]
        
        print(f"🔄 Re-processing {len(message_files)} message files...")
        
        total_messages = 0
        processed_messages = 0
        
        for i, file_path in enumerate(message_files):
            try:
                progress = (i + 1) / len(message_files) * 100
                print(f"\r   Progress: {progress:.1f}% ({i+1}/{len(message_files)}) - {file_path.name}", end='', flush=True)
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                messages = data.get('messages', [])
                total_messages += len(messages)
                
                if messages:
                    # Filter messages with actual content
                    valid_messages = []
                    for msg in messages:
                        content = msg.get('content', '').strip()
                        if content and len(content) > 10:
                            valid_messages.append(msg)
                    
                    # Process in batches
                    batch_size = config['batch_size']
                    for j in range(0, len(valid_messages), batch_size):
                        batch = valid_messages[j:j + batch_size]
                        success = await vector_store.add_messages(batch)
                        if success:
                            processed_messages += len(batch)
                        
                        # Small delay to prevent overwhelming
                        await asyncio.sleep(0.02)
                
            except Exception as e:
                print(f"\n❌ Error processing {file_path}: {e}")
                continue
        
        print(f"\n✅ Vector store recreated: {processed_messages}/{total_messages} messages processed")
        
        # Verify final count
        try:
            if vector_store.collection is not None:
                final_count = vector_store.collection.count()
                print(f"📊 Final vector store count: {final_count}")
                return final_count > 0
            else:
                print(f"⚠️ Collection not available for count verification")
                return processed_messages > 0
        except Exception as e:
            print(f"⚠️ Could not verify final count: {e}")
            return processed_messages > 0
        
    except Exception as e:
        print(f"❌ Error recreating vector store: {e}")
        return False

async def test_fixed_vector_store():
    """Test the fixed vector store"""
    print_header("Testing Fixed Vector Store")
    
    try:
        config = {
            "collection_name": "discord_messages",
            "persist_directory": "./data/chromadb",
            "embedding_model": os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        }
        
        vector_store = PersistentVectorStore(config)
        
        # Test basic search
        print("🔍 Testing semantic search...")
        results = await vector_store.similarity_search("machine learning", k=3)
        print(f"   Found {len(results)} results")
        
        if results:
            print("✅ Semantic search working!")
            for i, result in enumerate(results, 1):
                content = result.get('content', '')[:80] + "..." if len(result.get('content', '')) > 80 else result.get('content', '')
                channel = result.get('channel_name', 'Unknown')
                score = result.get('score', 0)
                print(f"   {i}. [{channel}] Score: {score:.3f}")
                print(f"      {content}")
        else:
            print("❌ Still no results found")
        
        # Test channel filtering
        print("\n🔍 Testing channel filtering...")
        results = await vector_store.similarity_search(
            "discussion", 
            k=3, 
            filters={"channel_name": "📚ai-philosophy-ethics"}
        )
        print(f"   Found {len(results)} results with channel filter")
        
        return len(results) > 0
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

async def main():
    """Main function to fix the vector store"""
    print("🔧 Fix Vector Store Embedding Dimension Mismatch")
    print(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\n🎯 Problem: Collection expects 1536 dimensions, getting 384")
    print("   Solution: Recreate vector store with consistent embedding model")
    
    # Step 1: Backup current vector store
    if not backup_vector_store():
        print("❌ Failed to backup vector store")
        return False
    
    # Step 2: Remove current vector store
    if not remove_vector_store():
        print("❌ Failed to remove current vector store")
        return False
    
    # Step 3: Recreate with correct embedding model
    if not await recreate_vector_store():
        print("❌ Failed to recreate vector store")
        return False
    
    # Step 4: Test the fixed vector store
    if await test_fixed_vector_store():
        print("\n🎉 Vector store fix completed successfully!")
        print("✅ Semantic search now working")
        print("✅ Bot should now return proper results")
        return True
    else:
        print("\n❌ Vector store fix failed")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
