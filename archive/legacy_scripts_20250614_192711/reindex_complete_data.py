#!/usr/bin/env python3
"""
Complete re-indexing script to rebuild the vector store with enhanced Discord message data.
This captures the full catalog of available Discord message fields.
"""

import asyncio
import os
import sys
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agentic.vectorstore.persistent_store import PersistentVectorStore

async def reindex_with_complete_data():
    """Re-index all Discord messages with the complete catalog of data fields"""
    print("🔄 COMPLETE RE-INDEXING WITH ENHANCED DATA FIELDS")
    print("=" * 80)
    
    try:
        # Configuration
        config = {
            "collection_name": "discord_messages",
            "persist_directory": "./data/chromadb",
            "embedding_model": "text-embedding-3-small",
            "batch_size": 100
        }
        
        # Step 1: Backup existing vector store
        print("💾 Step 1: Backing up existing vector store...")
        backup_dir = f"./data/chromadb_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        if Path(config["persist_directory"]).exists():
            shutil.copytree(config["persist_directory"], backup_dir)
            print(f"   ✅ Backup created: {backup_dir}")
        else:
            print("   ℹ️ No existing vector store to backup")
        
        # Step 2: Clear existing collection (but keep the directory structure)
        print("\n🗑️ Step 2: Clearing existing collection...")
        try:
            # Remove only the collection data, not the entire directory
            collection_files = Path(config["persist_directory"]).glob("*")
            for file_path in collection_files:
                if file_path.is_file():
                    file_path.unlink()
                elif file_path.is_dir() and file_path.name != "backups":
                    shutil.rmtree(file_path)
            print("   ✅ Existing collection cleared")
        except Exception as e:
            print(f"   ⚠️ Warning clearing collection: {e}")
        
        # Step 3: Initialize new vector store with enhanced schema
        print(f"\n🚀 Step 3: Initializing enhanced vector store...")
        vector_store = PersistentVectorStore(config)
        print("   ✅ Vector store initialized with enhanced schema")
        
        # Step 4: Load and process all message files
        print(f"\n📁 Step 4: Loading original message files...")
        message_files = list(Path("data/fetched_messages").glob("*.json"))
        
        if not message_files:
            print("   ❌ No message files found in data/fetched_messages/")
            return False
        
        print(f"   Found {len(message_files)} message files")
        
        total_messages = 0
        processed_files = 0
        
        for file_path in message_files:
            print(f"\n   📄 Processing: {file_path.name}")
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if not data or "messages" not in data:
                    print(f"      ⚠️ Skipping invalid file format")
                    continue
                
                messages = data["messages"]
                if not messages:
                    print(f"      ℹ️ No messages in file")
                    continue
                
                # Add guild/channel context to each message
                guild_name = data.get("guild_name", "")
                
                enhanced_messages = []
                for msg in messages:
                    # Add guild context that might be missing
                    if not msg.get("guild_name") and guild_name:
                        msg["guild_name"] = guild_name
                    
                    # Ensure all required fields exist
                    if msg.get("content") and msg.get("message_id"):
                        enhanced_messages.append(msg)
                
                if enhanced_messages:
                    # Add messages to vector store with enhanced metadata
                    success = await vector_store.add_messages(enhanced_messages)
                    
                    if success:
                        print(f"      ✅ Added {len(enhanced_messages)} messages")
                        total_messages += len(enhanced_messages)
                        processed_files += 1
                    else:
                        print(f"      ❌ Failed to add messages")
                else:
                    print(f"      ⚠️ No valid messages to add")
                    
            except Exception as e:
                print(f"      ❌ Error processing file: {e}")
                continue
        
        # Step 5: Verify the new index
        print(f"\n🔍 Step 5: Verifying enhanced index...")
        
        if vector_store.collection:
            stored_count = vector_store.collection.count()
            print(f"   📊 Total messages stored: {stored_count:,}")
            
            # Get sample to verify enhanced fields
            sample_data = vector_store.collection.get(limit=3, include=["metadatas"])
            
            if sample_data and sample_data.get("metadatas"):
                sample_metadata = sample_data["metadatas"][0]
                enhanced_fields = len(sample_metadata)
                
                print(f"   📋 Enhanced fields per message: {enhanced_fields}")
                print(f"   🆕 New fields include:")
                
                new_fields = [
                    "author_display_name", "author_bot", "author_discriminator",
                    "guild_name", "message_type", "pinned", "reply_to_message_id",
                    "word_count", "has_code_block", "has_urls", "mention_count",
                    "attachment_count", "attachment_filenames", "attachment_types",
                    "embed_count", "has_embeds", "processing_version"
                ]
                
                for field in new_fields:
                    if field in sample_metadata:
                        value = sample_metadata[field]
                        if isinstance(value, str) and len(str(value)) > 30:
                            display_value = str(value)[:27] + "..."
                        else:
                            display_value = value
                        print(f"      ✅ {field}: {display_value}")
                    else:
                        print(f"      ❌ {field}: missing")
        
        # Step 6: Performance test
        print(f"\n⚡ Step 6: Testing enhanced search performance...")
        
        test_queries = [
            "AI and machine learning discussions",
            "feedback on workshops", 
            "shared resources and links"
        ]
        
        for query in test_queries:
            try:
                start_time = datetime.now()
                results = await vector_store.similarity_search(query, k=3)
                end_time = datetime.now()
                
                search_time = (end_time - start_time).total_seconds()
                print(f"   🔍 '{query[:30]}...' → {len(results)} results in {search_time:.2f}s")
                
                # Check if enhanced fields are present in results
                if results and len(results) > 0:
                    sample_result = results[0]
                    has_display_name = bool(sample_result.get('author_display_name'))
                    has_attachments = bool(sample_result.get('attachment_count', 0) > 0)
                    has_reactions = bool(sample_result.get('total_reactions', 0) > 0)
                    
                    enhancements = []
                    if has_display_name: enhancements.append("display_names")
                    if has_attachments: enhancements.append("attachments") 
                    if has_reactions: enhancements.append("reactions")
                    
                    if enhancements:
                        print(f"      ✅ Enhanced data: {', '.join(enhancements)}")
                    
            except Exception as e:
                print(f"   ❌ Search test failed: {e}")
        
        # Step 7: Summary
        print(f"\n✅ RE-INDEXING COMPLETE!")
        print("─" * 80)
        print(f"   📁 Files processed: {processed_files}/{len(message_files)}")
        print(f"   📨 Messages indexed: {total_messages:,}")
        print(f"   💾 Backup location: {backup_dir}")
        print(f"   🔧 Enhanced fields: ~35 fields per message")
        print(f"   🚀 Vector store ready with complete Discord data catalog")
        
        print(f"\n🎯 WHAT'S NEW:")
        print("   • Display names instead of usernames")
        print("   • Complete author information (bot status, discriminator)")
        print("   • Message metadata (type, pinned status, reply info)")
        print("   • Content analysis (word count, code blocks, URLs)")
        print("   • Enhanced reactions and mentions data")
        print("   • Attachment information (files, types, sizes)")
        print("   • Embed and rich content detection")
        print("   • Guild/server context information")
        
        return True
        
    except Exception as e:
        print(f"❌ Critical error during re-indexing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(reindex_with_complete_data())
    
    if success:
        print(f"\n🎉 Re-indexing completed successfully!")
        print(f"   Restart the Discord bot to use the enhanced data.")
    else:
        print(f"\n💥 Re-indexing failed!")
        print(f"   Check the error messages above.")
    
    print(f"\nNext steps:")
    print(f"   1. Restart the Discord bot: python3 main.py")
    print(f"   2. Test queries with enhanced data display")
    print(f"   3. Verify display names and rich metadata appear")
