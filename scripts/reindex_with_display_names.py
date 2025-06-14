#!/usr/bin/env python3
"""
Re-index all Discord messages to include author display names.
This script will rebuild the vector store with the updated metadata structure.
"""

import asyncio
import os
import sys
import json
import shutil
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agentic.vectorstore.persistent_store import PersistentVectorStore

async def reindex_with_display_names():
    """Re-index all messages to include author display names"""
    print("🔄 RE-INDEXING DISCORD MESSAGES WITH DISPLAY NAMES")
    print("=" * 70)
    
    try:
        # Backup current vector store
        backup_dir = f"data/backup_vectorstore_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        if Path("data/chromadb").exists():
            print(f"📦 Creating backup: {backup_dir}")
            shutil.copytree("data/chromadb", backup_dir)
            print("✅ Backup created successfully")
        
        # Initialize vector store (this will create a fresh one)
        config = {
            "collection_name": "discord_messages",
            "persist_directory": "./data/chromadb",
            "embedding_model": "text-embedding-3-small"
        }
        
        # Remove the directory to ensure fresh start
        chromadb_path = Path("data/chromadb")
        if chromadb_path.exists():
            print("🗑️ Removing ChromaDB directory for fresh start...")
            shutil.rmtree(chromadb_path)
        
        # Recreate collection
        print("🆕 Creating fresh collection...")
        vector_store = PersistentVectorStore(config)
        
        # Process all message files
        message_files = list(Path("data/fetched_messages").glob("*.json"))
        print(f"📁 Found {len(message_files)} message files to process")
        
        total_messages_processed = 0
        files_processed = 0
        
        for message_file in message_files:
            print(f"\n📄 Processing: {message_file.name}")
            
            try:
                with open(message_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Extract messages from the file structure
                messages = []
                if isinstance(data, dict) and "messages" in data:
                    # Standard format: {"messages": [...]}
                    messages = data["messages"]
                elif isinstance(data, list):
                    # Direct list format: [...]
                    messages = data
                else:
                    print(f"   ⚠️ Unknown format in {message_file.name}")
                    continue
                
                if not messages:
                    print(f"   ⚠️ No messages found in {message_file.name}")
                    continue
                
                print(f"   📨 Adding {len(messages)} messages...")
                
                # Add messages to vector store (this will now include display names)
                success = await vector_store.add_messages(messages)
                
                if success:
                    total_messages_processed += len(messages)
                    files_processed += 1
                    print(f"   ✅ Successfully processed {len(messages)} messages")
                else:
                    print(f"   ❌ Failed to process {message_file.name}")
                    
            except Exception as e:
                print(f"   ❌ Error processing {message_file.name}: {e}")
                continue
        
        # Verify the re-indexing
        print(f"\n🔍 VERIFICATION:")
        print("─" * 50)
        
        # Check total count
        if vector_store.collection:
            total_count = vector_store.collection.count()
            print(f"   Total messages in vector store: {total_count}")
            
            # Check sample for display names
            sample_data = vector_store.collection.get(limit=3, include=["metadatas"])
            if sample_data and sample_data.get("metadatas"):
                print(f"   Sample messages with display names:")
                for i, metadata in enumerate(sample_data["metadatas"][:3], 1):
                    username = metadata.get("author_username", "N/A")
                    display_name = metadata.get("author_display_name", "N/A")
                    print(f"     {i}. Username: {username} | Display Name: {display_name}")
        
        print(f"\n✅ RE-INDEXING COMPLETED SUCCESSFULLY!")
        print("─" * 50)
        print(f"   Files processed: {files_processed}/{len(message_files)}")
        print(f"   Total messages: {total_messages_processed}")
        print(f"   Backup location: {backup_dir}")
        print("\n💡 The bot will now display author display names instead of usernames!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during re-indexing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("⚠️  WARNING: This will rebuild the entire vector store!")
    print("   A backup will be created automatically.")
    
    response = input("\nDo you want to proceed? (y/N): ").strip().lower()
    if response == 'y':
        success = asyncio.run(reindex_with_display_names())
        if success:
            print("\n🎉 Re-indexing completed! Restart the bot to see display names.")
        else:
            print("\n💥 Re-indexing failed! Check the logs above.")
    else:
        print("❌ Re-indexing cancelled.")
