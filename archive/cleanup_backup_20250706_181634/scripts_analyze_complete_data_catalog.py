#!/usr/bin/env python3
"""
Comprehensive analysis of Discord message data fields:
1. What Discord API provides
2. What we currently store 
3. What we're missing
4. Recommendations for complete data capture
"""

import asyncio
import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, Set

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agentic.vectorstore.persistent_store import PersistentVectorStore

async def analyze_complete_message_data():
    """Complete analysis of Discord message data availability vs storage"""
    print("🔍 COMPREHENSIVE DISCORD MESSAGE DATA ANALYSIS")
    print("=" * 80)
    
    # 1. Analyze what Discord API provides (from original message files)
    print("\n📡 DISCORD API FIELDS (from fetched message files):")
    print("─" * 80)
    
    discord_api_fields = set()
    nested_fields = set()
    
    message_files = list(Path("data/fetched_messages").glob("*.json"))
    if message_files:
        sample_file = message_files[0]
        try:
            with open(sample_file, 'r') as f:
                data = json.load(f)
            
            if data and "messages" in data and data["messages"]:
                sample_message = data["messages"][0]
                
                def collect_fields(obj, prefix="", max_depth=4, current_depth=0):
                    """Recursively collect all field paths"""
                    if current_depth >= max_depth:
                        return
                        
                    if isinstance(obj, dict):
                        for key, value in obj.items():
                            field_path = f"{prefix}.{key}" if prefix else key
                            discord_api_fields.add(field_path)
                            
                            if isinstance(value, dict):
                                nested_fields.add(field_path)
                                collect_fields(value, field_path, max_depth, current_depth + 1)
                            elif isinstance(value, list) and value:
                                if isinstance(value[0], dict):
                                    nested_fields.add(f"{field_path}[]")
                                    collect_fields(value[0], f"{field_path}[0]", max_depth, current_depth + 1)
                
                collect_fields(sample_message)
                
                print(f"   Total fields available from Discord API: {len(discord_api_fields)}")
                print(f"   Nested object fields: {len(nested_fields)}")
                
                # Print all available fields
                for field in sorted(discord_api_fields):
                    field_type = "nested" if field in nested_fields else "primitive"
                    print(f"   {field:<35} | {field_type}")
                    
        except Exception as e:
            print(f"   Error analyzing Discord API data: {e}")
    
    # 2. Analyze what we currently store in vector store
    print(f"\n💾 CURRENTLY STORED FIELDS (in vector store):")
    print("─" * 80)
    
    current_stored_fields = set()
    
    try:
        config = {
            "collection_name": "discord_messages", 
            "persist_directory": "./data/chromadb",
            "embedding_model": "text-embedding-3-small"
        }
        
        vector_store = PersistentVectorStore(config)
        
        if vector_store.collection:
            sample_data = vector_store.collection.get(limit=5, include=["metadatas"])
            
            if sample_data and sample_data.get("metadatas"):
                for metadata in sample_data["metadatas"]:
                    current_stored_fields.update(metadata.keys())
                
                print(f"   Total fields currently stored: {len(current_stored_fields)}")
                for field in sorted(current_stored_fields):
                    print(f"   {field:<35} | stored")
                    
    except Exception as e:
        print(f"   Error analyzing stored data: {e}")
    
    # 3. Calculate missing fields
    print(f"\n❌ MISSING FIELDS (available but not stored):")
    print("─" * 80)
    
    # Convert Discord API fields to what should be stored as metadata
    expected_stored_fields = set()
    
    for field in discord_api_fields:
        if not any(nested in field for nested in ["[0]", "[]"]):  # Skip array element examples
            # Convert nested field paths to flattened storage format
            if field.startswith("author."):
                expected_stored_fields.add(f"author_{field[7:]}")  # author.username -> author_username
            elif field.startswith("reactions[0]."):
                expected_stored_fields.add("reaction_emojis")
                expected_stored_fields.add("total_reactions")
            elif field.startswith("mentions[0]."):
                expected_stored_fields.add("mentioned_users")
            elif field.startswith("attachments[0]."):
                expected_stored_fields.add("attachment_urls")
                expected_stored_fields.add("attachment_types")
            elif "." not in field:
                expected_stored_fields.add(field)
    
    missing_fields = expected_stored_fields - current_stored_fields
    
    if missing_fields:
        print(f"   Found {len(missing_fields)} missing fields:")
        for field in sorted(missing_fields):
            print(f"   {field:<35} | MISSING")
    else:
        print("   ✅ No missing fields detected")
    
    # 4. Recommendations
    print(f"\n💡 RECOMMENDATIONS FOR COMPLETE DATA CAPTURE:")
    print("─" * 80)
    
    recommendations = []
    
    if "author_display_name" in missing_fields:
        recommendations.append("Add author_display_name to capture user-friendly names")
    
    if "author_bot" in missing_fields:
        recommendations.append("Add author_bot to distinguish bot messages")
        
    if "message_type" in missing_fields:
        recommendations.append("Add message_type to handle different Discord message types")
        
    if "pinned" in missing_fields:
        recommendations.append("Add pinned status for important messages")
        
    if "reference" in missing_fields:
        recommendations.append("Add reference data for reply threads")
        
    if "attachment_urls" in missing_fields:
        recommendations.append("Add attachment metadata for file searches")
        
    if "mentioned_users" in missing_fields:
        recommendations.append("Add mentioned users for @mention searches")
        
    if "embeds" in missing_fields:
        recommendations.append("Add embed data for rich content")
    
    # Always recommend these improvements
    recommendations.extend([
        "Store thread information for threaded conversations",
        "Capture edit history timestamps for modified messages", 
        "Store message flags and system message data",
        "Include user roles and permissions context",
        "Add message analytics (word count, sentiment, etc.)"
    ])
    
    for i, rec in enumerate(recommendations, 1):
        print(f"   {i:2d}. {rec}")
    
    # 5. Generate updated field mapping
    print(f"\n🗺️  SUGGESTED COMPLETE FIELD MAPPING:")
    print("─" * 80)
    
    field_mapping = {
        # Core message fields (currently stored)
        "message_id": "message_id",
        "channel_id": "channel_id", 
        "guild_id": "guild_id",
        "content": "content (document)",
        "timestamp": "timestamp",
        "jump_url": "jump_url",
        
        # Author fields (enhanced)
        "author.id": "author_id",
        "author.username": "author_username", 
        "author.display_name": "author_display_name",  # NEW
        "author.discriminator": "author_discriminator",  # NEW
        "author.bot": "author_bot",  # NEW
        
        # Channel/Guild context
        "channel_name": "channel_name",
        "guild_name": "guild_name",  # NEW
        
        # Message metadata
        "type": "message_type",  # NEW
        "pinned": "pinned",  # NEW
        "reference": "reply_to_message_id",  # NEW
        
        # Content analysis (derived)
        "content_length": "content_length",
        "word_count": "word_count",  # NEW
        "has_code": "has_code_block",  # NEW
        "has_links": "has_urls",  # NEW
        
        # Reactions (enhanced)
        "reactions[].emoji": "reaction_emojis",
        "reactions[].count": "total_reactions", 
        "reaction_details": "reaction_details",  # NEW - full reaction data
        
        # Mentions (enhanced)
        "mentions[].id": "mentioned_user_ids",  # NEW
        "mentions[].display_name": "mentioned_user_names",  # NEW
        
        # Attachments (enhanced) 
        "attachments[].url": "attachment_urls",  # NEW
        "attachments[].filename": "attachment_filenames",  # NEW
        "attachments[].content_type": "attachment_types",  # NEW
        "attachments[].size": "attachment_sizes",  # NEW
        
        # Rich content
        "embeds": "embed_count",  # NEW
        "embed_titles": "embed_titles",  # NEW
        "embed_urls": "embed_urls",  # NEW
        
        # System metadata
        "indexed_at": "indexed_at",
        "last_updated": "last_updated",  # NEW
        "processing_version": "processing_version"  # NEW
    }
    
    print("   Current → Enhanced Storage Mapping:")
    for discord_field, storage_field in field_mapping.items():
        status = "NEW" if storage_field.endswith("# NEW") else "exists"
        clean_field = storage_field.replace("  # NEW", "")
        print(f"   {discord_field:<25} → {clean_field:<25} | {status}")
    
    print(f"\n✅ Analysis complete!")
    print(f"   Discord API provides: {len(discord_api_fields)} fields")
    print(f"   Currently storing: {len(current_stored_fields)} fields") 
    print(f"   Missing opportunities: {len(missing_fields)} fields")
    print(f"   Recommended total: {len(field_mapping)} fields")
    
    return {
        "discord_api_fields": discord_api_fields,
        "current_stored_fields": current_stored_fields,
        "missing_fields": missing_fields,
        "recommendations": recommendations,
        "field_mapping": field_mapping
    }

if __name__ == "__main__":
    asyncio.run(analyze_complete_message_data())
