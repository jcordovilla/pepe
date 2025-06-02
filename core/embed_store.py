#!/usr/bin/env python3
"""
Discord Message Embedder and Store

Processes fetched Discord messages and stores them in the vector database
with embeddings for semantic search. Handles reactions data and metadata.
"""

import os
import sys
import json
import logging
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentic.vectorstore.persistent_store import PersistentVectorStore

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
FETCHED_MESSAGES_DIR = Path("data/fetched_messages")
PROCESSED_MARKER_DIR = Path("data/processing_markers")

# Ensure directories exist
PROCESSED_MARKER_DIR.mkdir(parents=True, exist_ok=True)


class MessageEmbedder:
    """
    Processes fetched Discord messages and stores them in the vector database.
    
    Handles batch processing, deduplication, and progress tracking.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the message embedder"""
        if config is None:
            config = {
                'persist_directory': 'data/vectorstore',
                'collection_name': 'discord_messages',
                'embedding_model': 'text-embedding-3-small',
                'batch_size': 100
            }
        
        self.config = config
        self.vector_store = PersistentVectorStore(config)
        self.stats = {
            "start_time": datetime.utcnow().isoformat(),
            "files_processed": 0,
            "messages_processed": 0,
            "messages_added": 0,
            "messages_updated": 0,
            "messages_skipped": 0,
            "errors": []
        }
        
        logger.info("Message embedder initialized")
    
    async def process_all_messages(self):
        """Process all fetched message files"""
        logger.info("🔄 Starting message embedding and storage process...")
        logger.info(f"📂 Looking for message files in: {FETCHED_MESSAGES_DIR}")
        
        if not FETCHED_MESSAGES_DIR.exists():
            logger.warning(f"❌ Messages directory not found: {FETCHED_MESSAGES_DIR}")
            return
        
        # Find all JSON message files
        message_files = list(FETCHED_MESSAGES_DIR.glob("*_messages.json"))
        
        if not message_files:
            logger.warning("❌ No message files found to process")
            return
        
        logger.info(f"📄 Found {len(message_files)} message files to process")
        
        for file_path in message_files:
            try:
                await self.process_message_file(file_path)
                self.stats["files_processed"] += 1
            except Exception as e:
                logger.error(f"❌ Error processing {file_path}: {e}")
                self.stats["errors"].append(f"File {file_path.name}: {str(e)}")
        
        # Save processing stats
        await self.save_stats()
        
        # Print summary
        logger.info("✅ Message processing completed!")
        logger.info(f"📊 Summary: {self.stats['messages_added']} added, "
                   f"{self.stats['messages_updated']} updated, "
                   f"{self.stats['messages_skipped']} skipped, "
                   f"{len(self.stats['errors'])} errors")
    
    async def process_message_file(self, file_path: Path):
        """
        Process a single message file.
        
        Args:
            file_path: Path to the JSON message file
        """
        logger.info(f"📄 Processing: {file_path.name}")
        
        # Check if already processed
        if await self.is_file_processed(file_path):
            logger.info(f"⏭️ Skipping already processed file: {file_path.name}")
            return
        
        try:
            # Load message data
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            messages = data.get('messages', [])
            if not messages:
                logger.info(f"📭 No messages in {file_path.name}")
                await self.mark_file_processed(file_path)
                return
            
            logger.info(f"📝 Found {len(messages)} messages in {file_path.name}")
            
            # Process messages in batches
            batch_size = self.config.get('batch_size', 100)
            for i in range(0, len(messages), batch_size):
                batch = messages[i:i + batch_size]
                await self.process_message_batch(batch, file_path.name)
            
            # Mark file as processed
            await self.mark_file_processed(file_path)
            
        except Exception as e:
            logger.error(f"❌ Error processing file {file_path}: {e}")
            raise
    
    async def process_message_batch(self, messages: List[Dict[str, Any]], source_file: str):
        """
        Process a batch of messages.
        
        Args:
            messages: List of message dictionaries
            source_file: Name of source file for logging
        """
        try:
            # Filter valid messages
            valid_messages = []
            for msg in messages:
                if self.is_valid_message(msg):
                    valid_messages.append(msg)
                    self.stats["messages_processed"] += 1
                else:
                    self.stats["messages_skipped"] += 1
            
            if not valid_messages:
                logger.debug(f"No valid messages in batch from {source_file}")
                return
            
            # Add to vector store
            success = await self.vector_store.add_messages(valid_messages)
            
            if success:
                self.stats["messages_added"] += len(valid_messages)
                logger.debug(f"✅ Added {len(valid_messages)} messages from {source_file}")
            else:
                logger.error(f"❌ Failed to add messages from {source_file}")
                self.stats["errors"].append(f"Failed to add batch from {source_file}")
                
        except Exception as e:
            logger.error(f"❌ Error processing batch from {source_file}: {e}")
            self.stats["errors"].append(f"Batch from {source_file}: {str(e)}")
            raise
    
    def is_valid_message(self, message: Dict[str, Any]) -> bool:
        """
        Check if a message is valid for processing.
        
        Args:
            message: Message dictionary
            
        Returns:
            True if message is valid
        """
        # Check required fields
        required_fields = ['message_id', 'content', 'author', 'timestamp']
        
        for field in required_fields:
            if not message.get(field):
                return False
        
        # Check content length
        content = message.get('content', '').strip()
        if len(content) < 3:  # Skip very short messages
            return False
        
        # Skip bot messages if configured
        author = message.get('author', {})
        if author.get('bot', False):
            return False  # Skip bot messages for now
        
        return True
    
    async def is_file_processed(self, file_path: Path) -> bool:
        """
        Check if a file has already been processed.
        
        Args:
            file_path: Path to the message file
            
        Returns:
            True if file has been processed
        """
        marker_file = PROCESSED_MARKER_DIR / f"{file_path.stem}.processed"
        return marker_file.exists()
    
    async def mark_file_processed(self, file_path: Path):
        """
        Mark a file as processed.
        
        Args:
            file_path: Path to the message file
        """
        marker_file = PROCESSED_MARKER_DIR / f"{file_path.stem}.processed"
        
        try:
            with open(marker_file, 'w') as f:
                json.dump({
                    "file_path": str(file_path),
                    "processed_at": datetime.utcnow().isoformat(),
                    "file_size": file_path.stat().st_size,
                    "file_modified": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                }, f, indent=2)
            
            logger.debug(f"✅ Marked {file_path.name} as processed")
            
        except Exception as e:
            logger.warning(f"⚠️ Could not create processing marker for {file_path}: {e}")
    
    async def save_stats(self):
        """Save processing statistics"""
        self.stats["end_time"] = datetime.utcnow().isoformat()
        
        stats_file = PROCESSED_MARKER_DIR / f"processing_stats_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(self.stats, f, indent=2, ensure_ascii=False)
            
            logger.info(f"📊 Processing statistics saved to {stats_file}")
            
        except Exception as e:
            logger.error(f"❌ Error saving statistics: {e}")
    
    async def reset_processing_markers(self):
        """Reset all processing markers (for reprocessing)"""
        logger.warning("🔄 Resetting all processing markers...")
        
        try:
            for marker_file in PROCESSED_MARKER_DIR.glob("*.processed"):
                marker_file.unlink()
            
            logger.info("✅ All processing markers reset")
            
        except Exception as e:
            logger.error(f"❌ Error resetting markers: {e}")


async def main():
    """Main function"""
    logger.info("🚀 Starting Discord message embedding process...")
    
    # Configuration for vector store
    config = {
        'persist_directory': 'data/vectorstore',
        'collection_name': 'discord_messages',
        'embedding_model': 'text-embedding-3-small',
        'batch_size': 50  # Smaller batch size for more reliable processing
    }
    
    embedder = MessageEmbedder(config)
    
    # Check for reset flag
    if len(sys.argv) > 1 and sys.argv[1] == "--reset":
        await embedder.reset_processing_markers()
        logger.info("🔄 Processing markers reset. Run again without --reset to process messages.")
        return
    
    try:
        await embedder.process_all_messages()
    except Exception as e:
        logger.error(f"❌ Fatal error in message processing: {e}")
        sys.exit(1)
    
    logger.info("🎉 Message embedding process completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())
