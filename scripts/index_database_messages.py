#!/usr/bin/env python3
"""
Index messages from SQLite database into vector store
Reads messages from discord_messages.db and creates embeddings for semantic search
"""

import asyncio
import json
import os
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
        self.checkpoint_path = project_root / 'data' / 'indexing_checkpoint.json'
        self.config = get_modernized_config()
        
        # Initialize vector store
        vector_config = {
            "collection_name": "discord_messages",
            "persist_directory": "./data/chromadb",
            "embedding_model": os.getenv("EMBEDDING_MODEL", "msmarco-distilbert-base-v4"),
            "embedding_type": "sentence_transformers",
            "batch_size": 100
        }
        
        # Ensure ChromaDB directory exists
        chroma_dir = Path("./data/chromadb")
        chroma_dir.mkdir(parents=True, exist_ok=True)
        
        self.vector_store = PersistentVectorStore(vector_config)
        
        # Load checkpoint for incremental indexing
        self.checkpoint = self.load_checkpoint()
    
    def load_checkpoint(self) -> Dict[str, Any]:
        """Load checkpoint data for incremental indexing"""
        if self.checkpoint_path.exists():
            try:
                with open(self.checkpoint_path, 'r') as f:
                    checkpoint = json.load(f)
                print(f"📋 Loaded indexing checkpoint from {checkpoint.get('last_updated', 'unknown')}")
                return checkpoint
            except Exception as e:
                print(f"⚠️ Warning: Could not load indexing checkpoint: {e}")
        return {
            'last_indexed_message_id': None,
            'last_updated': None,
            'total_messages_indexed': 0
        }
    
    def save_checkpoint(self, last_message_id: str):
        """Save checkpoint data for incremental indexing"""
        try:
            self.checkpoint['last_indexed_message_id'] = last_message_id
            self.checkpoint['last_updated'] = asyncio.get_event_loop().time()
            
            with open(self.checkpoint_path, 'w') as f:
                json.dump(self.checkpoint, f, indent=2)
        except Exception as e:
            print(f"⚠️ Warning: Could not save indexing checkpoint: {e}")
    
    async def index_all_messages(self):
        """Index messages from database into vector store with incremental support"""
        print("🚀 Starting message indexing from database...", flush=True)
        print(f"📊 Database: {self.db_path}", flush=True)
        
        # Get total message count and new message count
        total_messages, new_messages = self.get_message_counts()
        print(f"📝 Total messages in database: {total_messages:,}", flush=True)
        
        if new_messages == 0:
            print("✅ No new messages to index - database is up to date", flush=True)
            return
        
        print(f"🆕 New messages to index: {new_messages:,}", flush=True)
        
        # Process messages in batches
        batch_size = 100
        processed = 0
        indexed = 0
        errors = 0
        
        print(f"🔄 Processing in batches of {batch_size}...", flush=True)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row  # Enable column access by name
            
            # Get new messages only
            query = """
                SELECT * FROM messages 
                WHERE message_id > ?
                ORDER BY timestamp ASC
            """ if self.checkpoint['last_indexed_message_id'] else """
                SELECT * FROM messages 
                ORDER BY timestamp ASC
            """
            
            params = [self.checkpoint['last_indexed_message_id']] if self.checkpoint['last_indexed_message_id'] else []
            
            cursor = conn.execute(query, params)
            
            batch = []
            last_message_id = None
            
            # Create progress bar for indexing
            with tqdm(total=new_messages, desc="🔍 Indexing messages", unit="msg") as pbar:
                for row in cursor:
                    try:
                        # Convert SQLite row to message dict
                        message_dict = self.convert_row_to_message_dict(row)
                        batch.append(message_dict)
                        last_message_id = row['message_id']
                        
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
                
                # Process remaining batch
                if batch:
                    success = await self.index_batch(batch)
                    if success:
                        indexed += len(batch)
                    else:
                        errors += len(batch)
                    
                    processed += len(batch)
                    pbar.update(len(batch))
            
            # Save checkpoint with last processed message ID
            if last_message_id:
                self.save_checkpoint(last_message_id)
        
        print(f"\n✅ Indexing complete!")
        print(f"📊 Results: {indexed:,} messages indexed, {errors:,} errors")
        
        if errors > 0:
            print(f"⚠️  {errors:,} messages failed to index")
    
    def get_message_counts(self) -> tuple[int, int]:
        """Get total messages and new messages count"""
        with sqlite3.connect(self.db_path) as conn:
            # Get total messages
            cursor = conn.execute("SELECT COUNT(*) as count FROM messages")
            total_messages = cursor.fetchone()[0]
            
            # Get new messages (after last indexed message)
            if self.checkpoint['last_indexed_message_id']:
                cursor = conn.execute("""
                    SELECT COUNT(*) as count 
                    FROM messages 
                    WHERE message_id > ?
                """, [self.checkpoint['last_indexed_message_id']])
                new_messages = cursor.fetchone()[0]
            else:
                new_messages = total_messages
            
            return total_messages, new_messages
    
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
        
        # Extract bot information from raw_data JSON
        bot_info = False
        try:
            if row['raw_data']:
                raw_data = json.loads(row['raw_data'])
                if 'author' in raw_data and 'bot' in raw_data['author']:
                    bot_info = bool(raw_data['author']['bot'])
        except (json.JSONDecodeError, TypeError, KeyError):
            bot_info = False
        
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
                "display_name": str(row['author_display_name']) if row['author_display_name'] is not None else str(row['author_username']) if row['author_username'] is not None else "Unknown",
                "bot": bot_info  # Include bot information extracted from raw_data
            },
            "mentions": mentions,
            "reactions": reactions,
            "attachments": attachments,
            "embeds": int(row['embeds_count']) if row['embeds_count'] is not None else 0,
            "pinned": bool(row['pinned']) if row['pinned'] is not None else False,
            "type": str(row['message_type']) if row['message_type'] is not None else "MessageType.default",
            "reference": reference,
            # Always include forum_channel_id and forum_channel_name
            "forum_channel_id": str(row['forum_channel_id']) if row['forum_channel_id'] is not None else "",
            "forum_channel_name": str(row['forum_channel_name']) if row['forum_channel_name'] is not None else ""
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