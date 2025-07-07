#!/usr/bin/env python3
"""
Fixed re-indexing script that handles data format variations robustly.
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

async def reindex_with_robust_data_handling():
    """Re-index with robust handling of data format variations"""
    print("🔄 ROBUST RE-INDEXING WITH DATA FORMAT ERROR HANDLING")
    print("=" * 80)
    
    try:
        # Configuration
        config = {
            "collection_name": "discord_messages",
            "persist_directory": "./data/chromadb",
            "embedding_model": "text-embedding-3-small",
            "batch_size": 50  # Smaller batches for better error handling
        }
        
        # Step 1: Initialize vector store
        print("🚀 Initializing vector store...")
        vector_store = PersistentVectorStore(config)
        print("   ✅ Vector store initialized")
        
        # Step 2: Load and process message files with error handling
        print(f"\n📁 Loading message files...")
        message_files = list(Path("data/fetched_messages").glob("*.json"))
        
        if not message_files:
            print("   ❌ No message files found")
            return False
        
        print(f"   Found {len(message_files)} message files")
        
        total_messages = 0
        processed_files = 0
        error_files = []
        
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
                
                # Add guild/channel context and validate messages
                guild_name = data.get("guild_name", "")
                
                valid_messages = []
                invalid_count = 0
                
                for msg in messages:
                    try:
                        # Add guild context if missing
                        if not msg.get("guild_name") and guild_name:
                            msg["guild_name"] = guild_name
                        
                        # More inclusive validation - allow messages with:
                        # 1. message_id (required)
                        # 2. AND any of: content, attachments, embeds, or system message types
                        has_message_id = bool(msg.get("message_id"))
                        has_content = bool(msg.get("content", "").strip())
                        has_attachments = bool(msg.get("attachments", []))
                        has_embeds = bool(msg.get("embeds", []))
                        is_system_message = str(msg.get("type", "")).startswith("MessageType.")
                        
                        # Message is valid if it has ID and any meaningful content
                        is_valid = has_message_id and (has_content or has_attachments or has_embeds or is_system_message)
                        
                        if is_valid:
                            valid_messages.append(msg)
                        else:
                            invalid_count += 1
                            # Log detailed skip reason for debugging
                            skip_reason = []
                            if not has_message_id:
                                skip_reason.append("no_id")
                            if not has_content:
                                skip_reason.append("no_content")
                            if not has_attachments:
                                skip_reason.append("no_attachments")
                            if not has_embeds:
                                skip_reason.append("no_embeds")
                            if not is_system_message:
                                skip_reason.append("not_system")
                                
                    except Exception as msg_error:
                        print(f"      ⚠️ Invalid message format: {msg_error}")
                        invalid_count += 1
                        continue
                
                if invalid_count > 0:
                    print(f"      ⚠️ Skipped {invalid_count} invalid messages")
                
                if valid_messages:
                    # Add messages with comprehensive error handling
                    try:
                        success = await vector_store.add_messages(valid_messages)
                        
                        if success:
                            print(f"      ✅ Added {len(valid_messages)} messages")
                            total_messages += len(valid_messages)
                            processed_files += 1
                        else:
                            print(f"      ❌ Failed to add messages (vector store error)")
                            error_files.append(file_path.name)
                    except Exception as add_error:
                        print(f"      ❌ Error adding messages: {add_error}")
                        error_files.append(file_path.name)
                        
                        # Log detailed error for debugging
                        if "int" in str(add_error) and "len" in str(add_error):
                            print(f"         💡 Data type issue detected - this file needs manual inspection")
                else:
                    print(f"      ⚠️ No valid messages to add")
                    
            except Exception as e:
                print(f"      ❌ Error processing file: {e}")
                error_files.append(file_path.name)
                continue
        
        # Step 3: Summary and verification
        print(f"\n📊 PROCESSING SUMMARY:")
        print("─" * 60)
        print(f"   📁 Total files: {len(message_files)}")
        print(f"   ✅ Successfully processed: {processed_files}")
        print(f"   ❌ Failed files: {len(error_files)}")
        print(f"   📨 Messages indexed: {total_messages:,}")
        
        if error_files:
            print(f"\n❌ FILES WITH ERRORS:")
            for error_file in error_files[:10]:  # Show max 10 error files
                print(f"   • {error_file}")
            if len(error_files) > 10:
                print(f"   ... and {len(error_files) - 10} more")
        
        # Step 4: Verify the index
        if vector_store.collection:
            stored_count = vector_store.collection.count()
            print(f"\n✅ VERIFICATION:")
            print(f"   📊 Total messages in vector store: {stored_count:,}")
            
            if stored_count > 0:
                # Test search functionality
                test_results = await vector_store.similarity_search("test", k=3)
                print(f"   🔍 Search test: {len(test_results)} results")
                
                # Show sample enhanced data
                sample_data = vector_store.collection.get(limit=1, include=["metadatas"])
                if sample_data and sample_data.get("metadatas"):
                    sample_metadata = sample_data["metadatas"][0]
                    enhanced_fields = len(sample_metadata)
                    print(f"   📋 Enhanced fields per message: {enhanced_fields}")
                    
                    # Check for display names
                    display_name = sample_metadata.get("author_display_name")
                    if display_name:
                        print(f"   👤 Display names working: '{display_name}'")
                    else:
                        print(f"   ⚠️ Display names not found")
                
                success_rate = (processed_files / len(message_files)) * 100
                print(f"   📈 Success rate: {success_rate:.1f}%")
                
                if success_rate >= 50:
                    print(f"   🎉 Re-indexing successful!")
                    return True
                else:
                    print(f"   ⚠️ Low success rate - may need data format fixes")
                    return False
            else:
                print(f"   ❌ No messages were successfully indexed")
                return False
        else:
            print(f"   ❌ Vector store verification failed")
            return False
            
    except Exception as e:
        print(f"❌ Critical error during re-indexing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(reindex_with_robust_data_handling())
    
    if success:
        print(f"\n🎉 Robust re-indexing completed!")
        print(f"   The bot should now work with enhanced data.")
    else:
        print(f"\n💥 Re-indexing had issues!")
        print(f"   Check the error files for data format problems.")
    
    print(f"\nNext steps:")
    print(f"   1. Test bot with: 'fetch the last 3 messages from <#channel>'")
    print(f"   2. Verify display names and enhanced metadata appear")
    print(f"   3. If still issues, check individual error files")
