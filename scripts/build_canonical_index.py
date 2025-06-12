#!/usr/bin/env python3
"""
Canonical Discord Message Index Builder

Builds the main FAISS index for Discord message search with preprocessed content
and comprehensive metadata. This is the primary search index for the Discord bot.
"""

import json
import logging
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Add project root to path for imports
project_root = Path(__file__).parent.parent
import sys
sys.path.insert(0, str(project_root))

from scripts.content_preprocessor import ContentPreprocessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CanonicalIndexBuilder:
    """Canonical FAISS index builder for Discord message search."""
    
    def __init__(self, 
                 model_name: str = "msmarco-distilbert-base-v4",
                 db_path: str = "data/discord_messages.db",
                 batch_size: int = 100):
        """
        Initialize the canonical index builder.
        
        Args:
            model_name: Name of the sentence transformer model
            db_path: Path to the SQLite database
            batch_size: Batch size for encoding
        """
        self.model_name = model_name
        self.db_path = db_path
        self.batch_size = batch_size
        
        # Initialize sentence transformer
        logger.info(f"Loading sentence transformer model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        # Initialize preprocessor
        self.preprocessor = ContentPreprocessor()
        
        # Index components
        self.index = None
        self.metadata = []
        self.id_mapping = {}  # FAISS index -> message ID mapping
        
    def load_messages_from_db(self, limit: Optional[int] = None) -> List[Dict]:
        """Load messages from database."""
        logger.info("Loading messages from database...")
        
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        
        query = """
        SELECT 
            id, content, author, channel_id, 
            timestamp, edited_at, type as message_type, tts, pinned,
            embeds, attachments, stickers, components, reference,
            thread, webhook_id, application_id, poll, raw_mentions,
            clean_content, guild_id, flags, message_id, channel_name,
            reactions
        FROM messages 
        ORDER BY timestamp DESC
        """
        
        if limit:
            query += f" LIMIT {limit}"
            
        cursor = conn.execute(query)
        messages = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        logger.info(f"Loaded {len(messages)} messages from database")
        return messages
        
    def preprocess_messages(self, messages: List[Dict]) -> List[Dict]:
        """Preprocess messages using the content preprocessor."""
        logger.info("Preprocessing messages for canonical indexing...")
        
        processed_messages = []
        filtered_count = 0
        
        # Create SQLAlchemy session for preprocessor
        from db.db import SessionLocal
        db_session = SessionLocal()
        
        try:
            for message in messages:
                # Parse author JSON field
                author_data = json.loads(message['author']) if message['author'] else {}
                
                # Convert database row to message object for preprocessor
                class MockMessage:
                    def __init__(self, data):
                        self.message_id = data['message_id']
                        self.content = data['content'] or ''
                        self.embeds = json.loads(data['embeds']) if data['embeds'] else []
                        self.attachments = json.loads(data['attachments']) if data['attachments'] else []
                        self.stickers = json.loads(data['stickers']) if data['stickers'] else []
                        self.reference = json.loads(data['reference']) if data['reference'] else None
                        self.author = author_data
                        self.timestamp = data['timestamp']
                        self.channel_id = data['channel_id']
                        self.guild_id = data['guild_id']
                        self.type = data.get('message_type', 'default')
                        self.pinned = bool(data.get('pinned', False))
                        self.reactions = json.loads(data.get('reactions', '[]')) if data.get('reactions') else []
                        
                mock_message = MockMessage(message)
                
                # Preprocess the message
                processed = self.preprocessor.preprocess_message(mock_message, db_session)
                
                if processed:  # Not filtered out
                    # Combine original metadata with processed content
                    enhanced_message = {
                        **message,  # Original database fields
                        'processed_content': processed['searchable_text'],
                        'extracted_urls': processed['extracted_urls'],
                        'has_embed_content': processed['has_embeds'],
                        'embed_content': [],  # Will be populated separately if needed
                        'reply_context': '',  # Will be populated separately if needed  
                        'content_length': processed['content_length']
                    }
                    processed_messages.append(enhanced_message)
                else:
                    filtered_count += 1
                    
        finally:
            db_session.close()
            
        logger.info(f"Canonical preprocessing complete. {len(processed_messages)} messages processed, {filtered_count} filtered")
        return processed_messages
        
    def create_embeddings(self, messages: List[Dict]) -> np.ndarray:
        """Create embeddings for preprocessed messages."""
        logger.info("Creating embeddings for canonical index...")
        
        # Extract content for embedding
        texts = [msg['processed_content'] for msg in messages]
        
        # Create embeddings in batches
        embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            batch_embeddings = self.model.encode(
                batch_texts,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            embeddings.append(batch_embeddings)
            
            if (i // self.batch_size + 1) % 10 == 0:
                logger.info(f"Processed {i + len(batch_texts)} / {len(texts)} messages")
                
        embeddings = np.vstack(embeddings)
        logger.info(f"Created embeddings for canonical index: {embeddings.shape}")
        return embeddings
        
    def build_faiss_index(self, embeddings: np.ndarray) -> faiss.Index:
        """Build optimized FAISS index for canonical search."""
        logger.info("Building canonical FAISS index...")
        
        n_vectors, dim = embeddings.shape
        
        if n_vectors < 1000:
            # For small datasets, use simple flat index
            index = faiss.IndexFlatIP(dim)  # Inner product for normalized vectors
            logger.info("Using IndexFlatIP for canonical index (small dataset)")
        elif n_vectors < 10000:
            # For medium datasets, use IVF with reasonable number of clusters
            n_clusters = min(100, n_vectors // 50)
            quantizer = faiss.IndexFlatIP(dim)
            index = faiss.IndexIVFFlat(quantizer, dim, n_clusters)
            logger.info(f"Using IndexIVFFlat with {n_clusters} clusters for canonical index")
        else:
            # For large datasets, use more sophisticated index
            n_clusters = min(1000, n_vectors // 100)
            quantizer = faiss.IndexFlatIP(dim)
            index = faiss.IndexIVFFlat(quantizer, dim, n_clusters)
            logger.info(f"Using IndexIVFFlat with {n_clusters} clusters for canonical index (large dataset)")
            
        # Train index if needed
        if hasattr(index, 'train'):
            logger.info("Training canonical index...")
            index.train(embeddings)
            
        # Add vectors to index
        logger.info("Adding vectors to canonical index...")
        index.add(embeddings)
        
        logger.info(f"Canonical FAISS index built successfully. Total vectors: {index.ntotal}")
        return index
        
    def create_metadata(self, messages: List[Dict]) -> List[Dict]:
        """Create metadata for canonical index."""
        logger.info("Creating metadata for canonical index...")
        
        metadata = []
        for i, msg in enumerate(messages):
            # Parse JSON fields safely
            embeds = json.loads(msg['embeds']) if msg['embeds'] else []
            attachments = json.loads(msg['attachments']) if msg['attachments'] else []
            reference = json.loads(msg['reference']) if msg['reference'] else None
            author_data = json.loads(msg['author']) if msg['author'] else {}
            
            # Create canonical metadata
            meta = {
                # Core identifiers
                'message_id': msg['message_id'],
                'author_id': author_data.get('id', 'unknown'),
                'author_name': author_data.get('username', 'Unknown'),
                'channel_id': msg['channel_id'],
                'channel_name': msg['channel_name'],
                'guild_id': msg['guild_id'],
                
                # Temporal data
                'timestamp': msg['timestamp'],
                'edited_at': msg['edited_at'],
                'date': msg['timestamp'][:10] if msg['timestamp'] else None,
                
                # Content metadata
                'original_content': msg['content'],
                'processed_content': msg['processed_content'],
                'content_length': msg['content_length'],
                'has_embed_content': msg['has_embed_content'],
                'extracted_urls': msg['extracted_urls'],
                
                # Message properties
                'message_type': msg['message_type'],
                'is_tts': bool(msg['tts']),
                'is_pinned': bool(msg['pinned']),
                'has_embeds': len(embeds) > 0,
                'has_attachments': len(attachments) > 0,
                'has_reply': reference is not None,
                'is_thread_message': msg['thread'] is not None,
                'is_webhook': msg['webhook_id'] is not None,
                
                # Rich content details
                'embed_count': len(embeds),
                'attachment_count': len(attachments),
                'url_count': len(msg['extracted_urls']),
                
                # FAISS index position
                'faiss_index': i
            }
            
            metadata.append(meta)
            
        logger.info(f"Created canonical metadata for {len(metadata)} messages")
        return metadata
        
    def save_index_and_metadata(self, 
                               index: faiss.Index, 
                               metadata: List[Dict],
                               base_filename: str = None) -> Tuple[str, str]:
        """Save canonical FAISS index and metadata to files."""
        if base_filename is None:
            # Use canonical filename
            base_filename = "discord_messages_index"
            
            # Ensure we save to data/indices directory
            os.makedirs("data/indices", exist_ok=True)
            
            # Backup existing index if it exists
            existing_index = f"data/indices/{base_filename}.index"
            if os.path.exists(existing_index):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_name = f"data/indices/{base_filename}_backup_{timestamp}"
                logger.info(f"Backing up existing canonical index to: {backup_name}")
                os.rename(existing_index, f"{backup_name}.index")
                if os.path.exists(f"data/indices/{base_filename}_metadata.json"):
                    os.rename(f"data/indices/{base_filename}_metadata.json", f"{backup_name}_metadata.json")
        
        index_path = f"data/indices/{base_filename}.index"
        metadata_path = f"data/indices/{base_filename}_metadata.json"
        
        # Save FAISS index
        logger.info(f"Saving canonical FAISS index to: {index_path}")
        faiss.write_index(index, index_path)
        
        # Save metadata
        logger.info(f"Saving canonical metadata to: {metadata_path}")
        with open(metadata_path, 'w') as f:
            json.dump({
                'metadata': metadata,
                'index_info': {
                    'total_vectors': index.ntotal,
                    'dimension': self.embedding_dim,
                    'model_name': self.model_name,
                    'created_at': datetime.now().isoformat(),
                    'index_type': 'canonical_search',
                    'preprocessing_config': {
                        'min_content_length': self.preprocessor.config.min_content_length,
                        'include_embed_content': self.preprocessor.config.include_embed_content,
                        'include_reply_context': self.preprocessor.config.include_reply_context,
                        'normalize_urls': self.preprocessor.config.normalize_urls,
                        'filter_bot_messages': self.preprocessor.config.filter_bot_messages,
                        'max_embed_fields_per_message': self.preprocessor.config.max_embed_fields_per_message,
                        'max_reply_context_length': self.preprocessor.config.max_reply_context_length
                    }
                }
            }, f, indent=2)
            
        return index_path, metadata_path
        
    def build_canonical_index(self, 
                            limit: Optional[int] = None,
                            force_rebuild: bool = False) -> Dict:
        """Build complete canonical FAISS index with all steps."""
        logger.info("Starting canonical Discord message index build...")
        
        # Step 1: Load messages
        messages = self.load_messages_from_db(limit)
        
        # Step 2: Preprocess messages
        processed_messages = self.preprocess_messages(messages)
        
        if not processed_messages:
            if force_rebuild:
                raise ValueError("No messages available for canonical indexing. Check your database or preprocessing filters.")
            
            logger.info("✅ Canonical index is up to date - no new messages to process")
            
            # Check if existing index exists
            existing_index_path = "data/indices/discord_messages_index.index"
            existing_metadata_path = "data/indices/discord_messages_index_metadata.json"
            
            if os.path.exists(existing_index_path) and os.path.exists(existing_metadata_path):
                # Return existing index info
                with open(existing_metadata_path, 'r') as f:
                    existing_metadata = json.load(f)
                
                stats = {
                    "messages_processed": 0,
                    "messages_indexed": 0,
                    "existing_index_entries": existing_metadata.get('index_info', {}).get('total_vectors', 0),
                    "status": "up_to_date",
                    "message": "Canonical index is current - no new messages to process"
                }
                
                return stats
            else:
                # No existing index and no messages to process
                raise ValueError("No existing canonical index found and no messages to process. Run with --force to rebuild.")
        
        # Step 3: Create embeddings
        embeddings = self.create_embeddings(processed_messages)
        
        # Step 4: Build FAISS index
        self.index = self.build_faiss_index(embeddings)
        
        # Step 5: Create metadata
        self.metadata = self.create_metadata(processed_messages)
        
        # Step 6: Create ID mapping
        self.id_mapping = {i: msg['id'] for i, msg in enumerate(processed_messages)}
        
        # Step 7: Save everything
        index_path, metadata_path = self.save_index_and_metadata(
            self.index, self.metadata
        )
        
        # Generate summary statistics
        messages_with_attachments = 0
        for msg in processed_messages:
            attachments = json.loads(msg['attachments']) if msg['attachments'] else []
            if len(attachments) > 0:
                messages_with_attachments += 1
        
        stats = {
            'total_messages_processed': len(processed_messages),
            'total_messages_loaded': len(messages),
            'filter_rate': (len(messages) - len(processed_messages)) / len(messages) if messages else 0,
            'embedding_dimension': self.embedding_dim,
            'index_type': type(self.index).__name__,
            'model_name': self.model_name,
            'avg_content_length': np.mean([msg['content_length'] for msg in processed_messages]),
            'messages_with_embeds': sum(1 for msg in processed_messages if msg['has_embed_content']),
            'messages_with_attachments': messages_with_attachments,
            'total_urls_extracted': sum(len(msg['extracted_urls']) for msg in processed_messages),
            'index_path': index_path,
            'metadata_path': metadata_path
        }
        
        logger.info("Canonical FAISS index build complete!")
        logger.info(f"Index saved to: {index_path}")
        logger.info(f"Metadata saved to: {metadata_path}")
        logger.info(f"Processed {stats['total_messages_processed']} messages")
        logger.info(f"Filter rate: {stats['filter_rate']:.1%}")
        
        return stats


def main():
    """Main function to build canonical FAISS index."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Build canonical Discord message FAISS index")
    parser.add_argument("--model", default="msmarco-distilbert-base-v4", 
                       help="Sentence transformer model name")
    parser.add_argument("--db", default="data/discord_messages.db",
                       help="Path to Discord messages database")
    parser.add_argument("--batch-size", type=int, default=100,
                       help="Batch size for processing")
    parser.add_argument("--limit", type=int,
                       help="Limit number of messages to process (for testing)")
    parser.add_argument("--force", action="store_true",
                       help="Force complete rebuild")
    
    args = parser.parse_args()
    
    try:
        # Initialize builder
        builder = CanonicalIndexBuilder(
            model_name=args.model,
            db_path=args.db,
            batch_size=args.batch_size
        )
        
        # Build canonical index
        stats = builder.build_canonical_index(
            limit=args.limit,
            force_rebuild=args.force
        )
        
        # Save build report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f"data/indices/canonical_index_build_report_{timestamp}.json"
        
        build_report = {
            'build_timestamp': datetime.now().isoformat(),
            'statistics': stats,
            'configuration': {
                'model_name': args.model,
                'db_path': args.db,
                'batch_size': args.batch_size,
                'limit': args.limit,
                'force_rebuild': args.force
            }
        }
        
        os.makedirs("data/indices", exist_ok=True)
        with open(report_path, 'w') as f:
            json.dump(build_report, f, indent=2)
            
        logger.info(f"Build report saved to: {report_path}")
        
        print("\n" + "="*60)
        print("🎉 CANONICAL INDEX BUILD COMPLETE!")
        print("="*60)
        print(f"📁 Index: {stats.get('index_path', 'N/A')}")
        print(f"📄 Metadata: {stats.get('metadata_path', 'N/A')}")
        print(f"📊 Report: {report_path}")
        print(f"🔍 Messages indexed: {stats.get('total_messages_processed', 0)}")
        print(f"🎯 Embedding dimension: {stats.get('embedding_dimension', 0)}")
        print(f"⚡ Filter rate: {stats.get('filter_rate', 0):.1%}")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Failed to build canonical index: {e}")
        raise


if __name__ == "__main__":
    main()