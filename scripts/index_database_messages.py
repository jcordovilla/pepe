#!/usr/bin/env python3
"""
Index messages from SQLite database into vector store
Reads messages from discord_messages.db and creates embeddings for semantic search
"""

import asyncio
import json
import sqlite3
import sys
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agentic.vectorstore.persistent_store import PersistentVectorStore
from agentic.config.modernized_config import get_modernized_config
from tqdm import tqdm

class DatabaseMessageIndexer:
    def __init__(self):
        self.db_path = project_root / 'data' / 'discord_messages.db'
        self.config = get_modernized_config()
        
        # Initialize vector store
        vector_config = {
            "collection_name": "discord_messages",
            "persist_directory": "./data/chromadb",
            "embedding_model": "msmarco-distilbert-base-v4",
        "embedding_type": "sentence_transformers",
            "batch_size": 100
        }
        
        # Ensure ChromaDB directory exists
        chroma_dir = Path("./data/chromadb")
        chroma_dir.mkdir(parents=True, exist_ok=True)
        
        self.vector_store = PersistentVectorStore(vector_config)
    
    async def index_all_messages(self):
        """Index all messages from database into vector store"""
        print("🚀 Starting message indexing from database...", flush=True)
        print(f"📊 Database: {self.db_path}", flush=True)
        
        # Get total message count
        total_messages = self.get_message_count()
        print(f"📝 Total messages to index: {total_messages:,}", flush=True)
        
        if total_messages == 0:
            print("❌ No messages found in database", flush=True)
            return
        
        # Process messages in batches
        batch_size = 100
        processed = 0
        indexed = 0
        errors = 0
        
        print(f"🔄 Processing in batches of {batch_size}...", flush=True)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row  # Enable column access by name
            
            # Get all messages
            cursor = conn.execute("""
                SELECT * FROM messages 
                ORDER BY timestamp DESC
            """)
            
            batch = []
            
            # Create progress bar for overall indexing
            with tqdm(total=total_messages, desc="🔍 Indexing messages", unit="msg") as pbar:
                for row in cursor:
                    try:
                        # Convert SQLite row to message dict
                        message_dict = self.convert_row_to_message_dict(row)
                        batch.append(message_dict)
                        
                        # Process batch when full
                        if len(batch) >= batch_size:
                            success = await self.index_batch(batch)
                            if success:
                                indexed += len(batch)
                            else:
                                errors += len(batch)
                            
                            processed += len(batch)
                            pbar.update(len(batch))
                            pbar.set_postfix({
                                "indexed": f"{indexed:,}",
                                "errors": errors,
                                "batch": f"{processed//batch_size}"
                            })
                            batch = []
                            
                    except Exception as e:
                        print(f"\n❌ Error processing message {row['message_id']}: {e}", flush=True)
                        errors += 1
                        pbar.update(1)
                
                # Process remaining batch
                if batch:
                    try:
                        success = await self.index_batch(batch)
                        if success:
                            indexed += len(batch)
                        else:
                            errors += len(batch)
                        
                        processed += len(batch)
                        pbar.update(len(batch))
                        pbar.set_postfix({
                            "indexed": f"{indexed:,}",
                            "errors": errors,
                            "final": len(batch)
                        })
                    except Exception as e:
                        print(f"\n❌ Error processing final batch: {e}", flush=True)
                        errors += len(batch)
                        pbar.update(len(batch))
        
        # Print final results
        print("\n📊 Indexing Complete!", flush=True)
        print(f"   📝 Total Processed: {processed:,}", flush=True)
        print(f"   ✅ Successfully Indexed: {indexed:,}", flush=True)
        print(f"   ❌ Errors: {errors}", flush=True)
        print(f"   📊 Success Rate: {(indexed/processed*100):.1f}%" if processed > 0 else "   📊 Success Rate: 0%", flush=True)
    
    def get_message_count(self) -> int:
        """Get total number of messages in database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM messages")
            return cursor.fetchone()[0]
    
    def convert_row_to_message_dict(self, row) -> Dict[str, Any]:
        """Convert SQLite row to message dictionary format expected by vector store"""
        # Parse JSON fields with proper NULL handling
        try:
            mentions = json.loads(row['mentions']) if row['mentions'] is not None else []
        except (json.JSONDecodeError, TypeError):
            mentions = []
            
        try:
            reactions = json.loads(row['reactions']) if row['reactions'] is not None else []
        except (json.JSONDecodeError, TypeError):
            reactions = []
            
        try:
            attachments = json.loads(row['attachments']) if row['attachments'] is not None else []
        except (json.JSONDecodeError, TypeError):
            attachments = []
            
        try:
            reference = json.loads(row['reference']) if row['reference'] is not None else None
        except (json.JSONDecodeError, TypeError):
            reference = None
        
        # Create message dict in the format expected by vector store with NULL handling
        message_dict = {
            "message_id": str(row['message_id']) if row['message_id'] is not None else "",
            "channel_id": str(row['channel_id']) if row['channel_id'] is not None else "",
            "channel_name": str(row['channel_name']) if row['channel_name'] is not None else "Unknown",
            "guild_id": str(row['guild_id']) if row['guild_id'] is not None else None,
            "guild_name": str(row['guild_name']) if row['guild_name'] is not None else None,
            "content": str(row['content']) if row['content'] is not None else "",
            "timestamp": str(row['timestamp']) if row['timestamp'] is not None else "",
            "jump_url": str(row['jump_url']) if row['jump_url'] is not None else "",
            "author": {
                "id": str(row['author_id']) if row['author_id'] is not None else "",
                "username": str(row['author_username']) if row['author_username'] is not None else "Unknown",
                "display_name": str(row['author_display_name']) if row['author_display_name'] is not None else str(row['author_username']) if row['author_username'] is not None else "Unknown"
            },
            "mentions": mentions,
            "reactions": reactions,
            "attachments": attachments,
            "embeds": int(row['embeds_count']) if row['embeds_count'] is not None else 0,
            "pinned": bool(row['pinned']) if row['pinned'] is not None else False,
            "type": str(row['message_type']) if row['message_type'] is not None else "MessageType.default",
            "reference": reference
        }
        
        # Add thread metadata if this is a forum thread message
        if row['is_forum_thread']:
            message_dict.update({
                "thread_id": str(row['thread_id']) if row['thread_id'] is not None else "",
                "thread_name": str(row['thread_name']) if row['thread_name'] is not None else "",
                "forum_channel_id": str(row['forum_channel_id']) if row['forum_channel_id'] is not None else "",
                "forum_channel_name": str(row['forum_channel_name']) if row['forum_channel_name'] is not None else "",
                "is_forum_thread": True
            })
        
        return message_dict
    
    async def index_batch(self, messages: List[Dict[str, Any]]) -> bool:
        """Index a batch of messages into the vector store"""
        try:
            success = await self.vector_store.add_messages(messages)
            return success
        except Exception as e:
            print(f"   ❌ Error indexing batch: {e}", flush=True)
            return False
    
    async def get_index_stats(self):
        """Get statistics about the indexed data"""
        try:
            stats = await self.vector_store.get_stats()
            print("\n📈 Vector Store Statistics:", flush=True)
            print(f"   📦 Total Documents: {stats.get('total_documents', 0):,}", flush=True)
            print(f"   💾 Collection Size: {stats.get('collection_size_mb', 0):.1f} MB", flush=True)
            print(f"   🤖 Embedding Model: {stats.get('embedding_model', 'Unknown')}", flush=True)
            
            # Get channel breakdown
            channel_stats = await self.vector_store.get_channel_stats()
            if channel_stats:
                print(f"\n📊 Channel Breakdown (Top 10):", flush=True)
                for channel in channel_stats[:10]:
                    print(f"   • #{channel.get('name', 'Unknown')}: {channel.get('count', 0):,} messages", flush=True)
            
        except Exception as e:
            print(f"   ❌ Error getting stats: {e}", flush=True)

async def main():
    """Main function"""
    try:
        indexer = DatabaseMessageIndexer()
        await indexer.index_all_messages()
        await indexer.get_index_stats()
        print("\n🎉 Database indexing completed successfully!", flush=True)
        
    except Exception as e:
        print(f"❌ Error: {e}", flush=True)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main()) 