#!/usr/bin/env python3
"""
Comprehensive analysis of all data fields retrieved and stored by the Discord bot.
"""

import asyncio
import os
import sys
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agentic.vectorstore.persistent_store import PersistentVectorStore

async def analyze_all_data_fields():
    """Analyze all data fields stored in the vector store and used by the bot"""
    print("📊 COMPLETE DATA FIELD ANALYSIS")
    print("=" * 80)
    
    try:
        config = {
            "collection_name": "discord_messages",
            "persist_directory": "./data/chromadb",
            "embedding_model": "text-embedding-3-small"
        }
        
        vector_store = PersistentVectorStore(config)
        
        # Get sample messages to analyze all fields
        print("🔍 Analyzing vector store data structure...")
        if vector_store.collection:
            sample_data = vector_store.collection.get(
                limit=10,  # Get more samples for comprehensive analysis
                include=["metadatas", "documents", "embeddings"]
            )
            
            if sample_data and sample_data.get("metadatas"):
                print(f"✅ Analyzing {len(sample_data['metadatas'])} messages\n")
                
                # Collect all unique field names
                all_fields = set()
                field_types = {}
                field_examples = {}
                
                for metadata in sample_data["metadatas"]:
                    for key, value in metadata.items():
                        all_fields.add(key)
                        field_types[key] = type(value).__name__
                        if key not in field_examples:
                            field_examples[key] = value
                
                # Sort fields alphabetically
                sorted_fields = sorted(all_fields)
                
                print("📋 METADATA FIELDS (stored in vector store):")
                print("─" * 60)
                for field in sorted_fields:
                    field_type = field_types.get(field, "unknown")
                    example = field_examples.get(field, "N/A")
                    
                    # Truncate long examples
                    if isinstance(example, str) and len(example) > 50:
                        example = example[:47] + "..."
                    
                    print(f"   {field:<20} | {field_type:<8} | {example}")
                
                # Check document content structure
                if sample_data.get("documents"):
                    print(f"\n📄 DOCUMENT CONTENT:")
                    print("─" * 60)
                    sample_doc = sample_data["documents"][0]
                    print(f"   Type: {type(sample_doc).__name__}")
                    print(f"   Length: {len(sample_doc)} characters")
                    print(f"   Sample: {sample_doc[:100]}...")
                
                # Check embedding structure
                embeddings = sample_data.get("embeddings")
                if embeddings is not None and len(embeddings) > 0:
                    print(f"\n🔢 EMBEDDING VECTORS:")
                    print("─" * 60)
                    sample_embedding = embeddings[0]
                    print(f"   Type: {type(sample_embedding).__name__}")
                    print(f"   Dimensions: {len(sample_embedding)}")
                    print(f"   Sample values: {sample_embedding[:5]}...")
                
                print(f"\n📊 SUMMARY:")
                print("─" * 60)
                print(f"   Total metadata fields: {len(sorted_fields)}")
                print(f"   Total messages analyzed: {len(sample_data['metadatas'])}")
                embeddings = sample_data.get("embeddings")
                vector_dims = len(embeddings[0]) if embeddings is not None and len(embeddings) > 0 else "N/A"
                print(f"   Vector dimensions: {vector_dims}")
                
        # Also check what fields are used in the original message files
        print(f"\n📁 ORIGINAL MESSAGE FILE STRUCTURE:")
        print("─" * 60)
        
        message_files = list(Path("data/fetched_messages").glob("*.json"))
        if message_files:
            # Read a sample message file
            sample_file = message_files[0]
            try:
                with open(sample_file, 'r') as f:
                    original_data = json.load(f)
                
                if original_data and len(original_data) > 0:
                    # Check if it's a list or dict
                    if isinstance(original_data, list):
                        sample_message = original_data[0]
                    elif isinstance(original_data, dict):
                        sample_message = original_data
                    else:
                        print(f"   Unexpected data format: {type(original_data)}")
                        sample_message = None
                    
                    if sample_message:
                        print(f"   Sample file: {sample_file.name}")
                        print(f"   Data type: {type(original_data).__name__}")
                        print(f"   Fields in original messages:")
                        
                        def print_nested_fields(obj, prefix="", max_depth=3, current_depth=0):
                            """Recursively print nested field structure"""
                            if current_depth >= max_depth:
                                print(f"      {prefix}... | (nested too deep)")
                                return
                                
                            if isinstance(obj, dict):
                                for key, value in obj.items():
                                    field_path = f"{prefix}.{key}" if prefix else key
                                    if isinstance(value, dict):
                                        print(f"      {field_path:<25} | dict")
                                        print_nested_fields(value, field_path, max_depth, current_depth + 1)
                                    elif isinstance(value, list) and value and isinstance(value[0], dict):
                                        print(f"      {field_path:<25} | list[dict]")
                                        print_nested_fields(value[0], f"{field_path}[0]", max_depth, current_depth + 1)
                                    else:
                                        value_type = type(value).__name__
                                        if isinstance(value, str) and len(value) > 30:
                                            display_value = value[:27] + "..."
                                        else:
                                            display_value = value
                                        print(f"      {field_path:<25} | {value_type:<8} | {display_value}")
                        
                        print_nested_fields(sample_message)
                else:
                    print(f"   File {sample_file.name} is empty or invalid")
            except Exception as e:
                print(f"   Error reading {sample_file.name}: {e}")
        else:
            print("   No message files found in data/fetched_messages/")
        
        # Check what fields are actually used by the Discord interface
        print(f"\n🤖 FIELDS USED BY DISCORD BOT INTERFACE:")
        print("─" * 60)
        interface_fields = [
            "author_username",      # Primary username field
            "author.username",      # Fallback username field  
            "timestamp",            # Message timestamp
            "content",              # Message text content
            "channel_name",         # Channel name with emojis
            "jump_url",            # Direct link to message
            "author_id",           # User ID for identification
            "channel_id",          # Channel ID for filtering
            "guild_id",            # Server/guild ID
            "message_id",          # Unique message identifier
            "reaction_emojis",     # Reaction data
            "total_reactions",     # Reaction count
            "content_length",      # Length of message content
            "indexed_at"           # When message was indexed
        ]
        
        for field in interface_fields:
            print(f"   {field:<20} | Used in bot responses and filtering")
        
        print(f"\n✅ Complete data field analysis finished!")
        print(f"\n💡 KEY INSIGHTS:")
        print("   • Messages are stored with comprehensive metadata")
        print("   • Both original Discord API fields and derived fields are preserved")
        print("   • Vector embeddings enable semantic search capabilities")
        print("   • Timestamps allow chronological sorting and filtering")
        print("   • Channel and user information enables precise filtering")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(analyze_all_data_fields())
