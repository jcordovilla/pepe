#!/usr/bin/env python3
"""
Community-Focused FAISS Index Builder

Builds an optimized FAISS index using enhanced community preprocessing 
with comprehensive metadata for Discord agent capabilities including:
- Expert identification and skill mining
- Conversation threading and context analysis  
- Community engagement and activity metrics
- Temporal event detection and deadline extraction
- Question/answer pattern detection
- Resource and tutorial classification
"""

import json
import logging
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import asdict

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Add project root to path for imports
project_root = Path(__file__).parent.parent
import sys
sys.path.insert(0, str(project_root))

from scripts.content_preprocessor import ContentPreprocessor
from scripts.enhanced_community_preprocessor import CommunityPreprocessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CommunityFAISSIndexBuilder:
    """Community-focused FAISS index builder with rich metadata and expert identification."""
    
    def __init__(self, 
                 model_name: str = "msmarco-distilbert-base-v4",
                 db_path: str = "data/discord_messages.db",
                 batch_size: int = 100):
        """
        Initialize the community FAISS index builder.
        
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
        
        # Initialize preprocessors
        self.content_preprocessor = ContentPreprocessor()
        self.community_preprocessor = CommunityPreprocessor()
        
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
        """Preprocess messages using both content and community preprocessors."""
        logger.info("Preprocessing messages with community focus...")
        
        processed_messages = []
        filtered_count = 0
        
        # Create SQLAlchemy session for preprocessor (fixed: was using raw sqlite3.connect)
        from db.db import SessionLocal
        session = SessionLocal()
        
        try:
            for message in messages:
                # Parse author JSON field
                author_data = json.loads(message['author']) if message['author'] else {}
                author_id = author_data.get('id', 'unknown')
                
                # Convert database row to message object for preprocessors
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
                        self.mention_ids = json.loads(data.get('raw_mentions', '[]')) if data.get('raw_mentions') else []
                        
                mock_message = MockMessage(message)
                
                # Basic content preprocessing (now with proper SQLAlchemy session)
                content_processed = self.content_preprocessor.preprocess_message(mock_message, session)
                
                # Debug: Log first few results
                if len(processed_messages) < 3:
                    logger.info(f"DEBUG: Message {message['message_id']} - Content: '{(mock_message.content or '')[:50]}...' - Processed: {content_processed is not None}")
            
            if content_processed:  # Not filtered out
                # Enhanced community preprocessing
                try:
                    community_metadata = self.community_preprocessor.process_message(mock_message)
                    
                    # Combine original metadata with processed content and community analysis
                    enhanced_message = {
                        **message,  # Original database fields
                        'processed_content': content_processed['searchable_text'],
                        'extracted_urls': content_processed['extracted_urls'],
                        'has_embed_content': content_processed['has_embeds'],
                        'embed_content': [],  # Will be populated separately if needed
                        'reply_context': '',  # Will be populated separately if needed  
                        'content_length': content_processed['content_length'],
                        
                        # Community-focused metadata
                        'community_metadata': asdict(community_metadata)
                    }
                    processed_messages.append(enhanced_message)
                except Exception as e:
                    logger.warning(f"Community preprocessing failed for message {message['message_id']}: {e}")
                    # Fall back to basic processing
                    enhanced_message = {
                        **message,  # Original database fields
                        'processed_content': content_processed['searchable_text'],
                        'extracted_urls': content_processed['extracted_urls'],
                        'has_embed_content': content_processed['has_embeds'],
                        'embed_content': [],
                        'reply_context': '',  
                        'content_length': content_processed['content_length'],
                        'community_metadata': None
                    }
                    processed_messages.append(enhanced_message)
            else:
                filtered_count += 1
                
        finally:
            session.close()
            
        logger.info(f"Community preprocessing complete. {len(processed_messages)} messages processed, {filtered_count} filtered")
        return processed_messages
        
    def create_embeddings(self, messages: List[Dict]) -> np.ndarray:
        """Create embeddings for preprocessed messages."""
        logger.info("Creating embeddings...")
        
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
        logger.info(f"Created embeddings: {embeddings.shape}")
        return embeddings
        
    def build_faiss_index(self, embeddings: np.ndarray) -> faiss.Index:
        """Build optimized FAISS index."""
        logger.info("Building FAISS index...")
        
        n_vectors, dim = embeddings.shape
        
        if n_vectors < 1000:
            # For small datasets, use simple flat index
            index = faiss.IndexFlatIP(dim)  # Inner product for normalized vectors
            logger.info("Using IndexFlatIP for small dataset")
        elif n_vectors < 10000:
            # For medium datasets, use IVF with reasonable number of clusters
            n_clusters = min(100, n_vectors // 50)
            quantizer = faiss.IndexFlatIP(dim)
            index = faiss.IndexIVFFlat(quantizer, dim, n_clusters)
            logger.info(f"Using IndexIVFFlat with {n_clusters} clusters")
        else:
            # For large datasets, use more sophisticated index
            n_clusters = min(1000, n_vectors // 100)
            quantizer = faiss.IndexFlatIP(dim)
            index = faiss.IndexIVFFlat(quantizer, dim, n_clusters)
            logger.info(f"Using IndexIVFFlat with {n_clusters} clusters for large dataset")
            
        # Train index if needed
        if hasattr(index, 'train'):
            logger.info("Training index...")
            index.train(embeddings)
            
        # Add vectors to index
        logger.info("Adding vectors to index...")
        index.add(embeddings)
        
        logger.info(f"FAISS index built successfully. Total vectors: {index.ntotal}")
        return index
        
    def create_community_metadata(self, messages: List[Dict]) -> List[Dict]:
        """Create rich community-focused metadata for enhanced retrieval."""
        logger.info("Creating community-focused metadata...")
        
        metadata = []
        for i, msg in enumerate(messages):
            # Parse JSON fields safely
            embeds = json.loads(msg['embeds']) if msg['embeds'] else []
            attachments = json.loads(msg['attachments']) if msg['attachments'] else []
            reference = json.loads(msg['reference']) if msg['reference'] else None
            author_data = json.loads(msg['author']) if msg['author'] else {}
            
            # Extract community metadata
            community_meta = msg.get('community_metadata', {})
            
            # Create comprehensive metadata
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
                'embed_content': msg['embed_content'],
                'reply_context': msg['reply_context'],
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
                'has_application': msg['application_id'] is not None,
                'has_poll': msg['poll'] is not None,
                
                # Rich content details
                'embed_count': len(embeds),
                'attachment_count': len(attachments),
                'url_count': len(msg['extracted_urls']),
                
                # Attachment details
                'attachment_types': [att.get('content_type', '') for att in attachments] if attachments else [],
                'has_images': any('image' in att.get('content_type', '') for att in attachments) if attachments else False,
                'has_videos': any('video' in att.get('content_type', '') for att in attachments) if attachments else False,
                
                # FAISS index position
                'faiss_index': i,
                
                # === COMMUNITY-FOCUSED ENHANCEMENTS ===
                
                # Expert identification & skills
                'skill_keywords': community_meta.get('skill_keywords', []),
                'expertise_indicators': community_meta.get('expertise_indicators', {}),
                'has_technical_skills': len(community_meta.get('skill_keywords', [])) > 0,
                'expertise_confidence': max(community_meta.get('expertise_indicators', {}).values()) if community_meta.get('expertise_indicators') else 0.0,
                
                # Question/Answer patterns
                'question_indicators': community_meta.get('question_indicators', False),
                'solution_indicators': community_meta.get('solution_indicators', False),
                'help_seeking': community_meta.get('help_seeking', False),
                'help_providing': community_meta.get('help_providing', False),
                
                # Conversation threading
                'conversation_thread_id': community_meta.get('conversation_thread_id'),
                'reply_depth': community_meta.get('reply_depth', 0),
                'thread_participants': community_meta.get('thread_participants', []),
                'question_resolved': community_meta.get('question_resolved'),
                'resolution_confidence': community_meta.get('resolution_confidence', 0.0),
                
                # Community engagement
                'reaction_sentiment': community_meta.get('reaction_sentiment', {}),
                'engagement_score': community_meta.get('engagement_score', 0.0),
                'influence_score': community_meta.get('influence_score', 0.0),
                'topic_category': community_meta.get('topic_category', 'general'),
                
                # Temporal & events
                'event_mentions': community_meta.get('event_mentions', []),
                'deadline_indicators': community_meta.get('deadline_indicators', []),
                'time_sensitive': community_meta.get('time_sensitive', False),
                'has_deadlines': len(community_meta.get('deadline_indicators', [])) > 0,
                
                # Content classification
                'content_type': community_meta.get('content_type', 'general'),
                'resource_quality': community_meta.get('resource_quality', 0.0),
                'code_snippets': community_meta.get('code_snippets', []),
                'tutorial_steps': community_meta.get('tutorial_steps', []),
                'has_code': len(community_meta.get('code_snippets', [])) > 0,
                'is_tutorial': community_meta.get('content_type') == 'tutorial',
                'is_resource': community_meta.get('content_type') == 'resource',
                'is_question': community_meta.get('content_type') == 'question',
                
                # Enhanced search capabilities
                'search_tags': self._generate_search_tags(community_meta),
                'difficulty_level': self._assess_difficulty_level(community_meta),
                'interaction_type': self._classify_interaction_type(community_meta)
            }
            
            metadata.append(meta)
            
        logger.info(f"Created community metadata for {len(metadata)} messages")
        return metadata
    
    def _generate_search_tags(self, community_meta: Dict) -> List[str]:
        """Generate searchable tags based on community metadata"""
        tags = []
        
        # Add skill-based tags
        skills = community_meta.get('skill_keywords', [])
        tags.extend([f"skill:{skill}" for skill in skills[:5]])  # Limit to top 5
        
        # Add content type tags
        content_type = community_meta.get('content_type', 'general')
        if content_type != 'general':
            tags.append(f"type:{content_type}")
        
        # Add topic category tags
        topic = community_meta.get('topic_category', 'general')
        if topic != 'general':
            tags.append(f"topic:{topic}")
        
        # Add behavior tags
        if community_meta.get('help_seeking'):
            tags.append("behavior:help_seeking")
        if community_meta.get('help_providing'):
            tags.append("behavior:help_providing")
        if community_meta.get('time_sensitive'):
            tags.append("urgent")
        if community_meta.get('code_snippets'):
            tags.append("has_code")
        
        return tags
    
    def _assess_difficulty_level(self, community_meta: Dict) -> str:
        """Assess the technical difficulty level of the content"""
        expertise_scores = community_meta.get('expertise_indicators', {})
        avg_expertise = sum(expertise_scores.values()) / len(expertise_scores) if expertise_scores else 0
        
        if avg_expertise > 0.8:
            return 'expert'
        elif avg_expertise > 0.6:
            return 'intermediate'
        elif avg_expertise > 0.3:
            return 'beginner'
        else:
            return 'general'
    
    def _classify_interaction_type(self, community_meta: Dict) -> str:
        """Classify the type of community interaction"""
        if community_meta.get('help_seeking') and community_meta.get('help_providing'):
            return 'collaborative'
        elif community_meta.get('help_seeking'):
            return 'question'
        elif community_meta.get('help_providing'):
            return 'answer'
        elif community_meta.get('content_type') == 'tutorial':
            return 'educational'
        elif community_meta.get('content_type') == 'resource':
            return 'informational'
        elif community_meta.get('event_mentions'):
            return 'event'
        else:
            return 'discussion'
        
    def save_index_and_metadata(self, 
                               index: faiss.Index, 
                               metadata: List[Dict],
                               base_filename: str = None) -> Tuple[str, str]:
        """Save FAISS index and metadata to files."""
        if base_filename is None:
            # Use canonical community index name
            base_filename = "community_faiss_index"
            
            # Backup existing index if it exists
            existing_index = f"data/indices/{base_filename}.index"
            if os.path.exists(existing_index):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_name = f"data/indices/{base_filename}_backup_{timestamp}"
                logger.info(f"Backing up existing community index to: {backup_name}")
                os.rename(existing_index, f"{backup_name}.index")
                if os.path.exists(f"data/indices/{base_filename}_metadata.json"):
                    os.rename(f"data/indices/{base_filename}_metadata.json", f"{backup_name}_metadata.json")
            
        # Ensure we save to data/indices directory
        os.makedirs("data/indices", exist_ok=True)
        index_path = f"data/indices/{base_filename}.index"
        metadata_path = f"data/indices/{base_filename}_metadata.json"
        
        # Save FAISS index
        logger.info(f"Saving FAISS index to: {index_path}")
        faiss.write_index(index, index_path)
        
        # Save metadata
        logger.info(f"Saving metadata to: {metadata_path}")
        with open(metadata_path, 'w') as f:
            json.dump({
                'metadata': metadata,
                'index_info': {
                    'total_vectors': index.ntotal,
                    'dimension': self.embedding_dim,
                    'model_name': self.model_name,
                    'created_at': datetime.now().isoformat(),
                    'index_type': 'community_focused',
                    'preprocessing_config': {
                        'min_content_length': self.content_preprocessor.config.min_content_length,
                        'include_embed_content': self.content_preprocessor.config.include_embed_content,
                        'include_reply_context': self.content_preprocessor.config.include_reply_context,
                        'normalize_urls': self.content_preprocessor.config.normalize_urls,
                        'filter_bot_messages': self.content_preprocessor.config.filter_bot_messages,
                        'max_embed_fields_per_message': self.content_preprocessor.config.max_embed_fields_per_message,
                        'max_reply_context_length': self.content_preprocessor.config.max_reply_context_length
                    },
                    'community_features': {
                        'expert_identification': True,
                        'skill_mining': True,
                        'conversation_threading': True,
                        'engagement_analysis': True,
                        'temporal_event_extraction': True,
                        'qa_pattern_detection': True,
                        'resource_classification': True
                    }
                }
            }, f, indent=2)
        
        return index_path, metadata_path
        
    def build_complete_index(self, 
                           limit: Optional[int] = None,
                           save_filename: str = None,
                           force_rebuild: bool = False) -> Tuple[str, str, Dict]:
        """Build complete community-focused FAISS index with all steps."""
        logger.info("Starting complete community FAISS index build...")
        
        # Set default filename if not provided
        if save_filename is None:
            save_filename = "community_faiss_index"
        
        # Step 1: Load messages
        messages = self.load_messages_from_db(limit)
        
        # Step 2: Preprocess messages with community focus
        processed_messages = self.preprocess_messages(messages)
        
        if not processed_messages:
            if force_rebuild:
                # Force rebuild was requested but no messages to process
                raise ValueError("No messages available for processing. Check your database or preprocessing filters.")
            
            logger.info("✅ Community index is up to date - no new messages to process")
            
            # Check if existing index exists (use correct .index extension)
            existing_index_path = f"data/indices/{save_filename}.index"
            existing_metadata_path = f"data/indices/{save_filename}_metadata.json"
            
            if os.path.exists(existing_index_path) and os.path.exists(existing_metadata_path):
                # Return existing index info
                with open(existing_metadata_path, 'r') as f:
                    existing_metadata = json.load(f)
                
                stats = {
                    "messages_processed": 0,
                    "messages_indexed": 0,
                    "existing_index_entries": existing_metadata.get('total_messages', 0),
                    "status": "up_to_date",
                    "message": "Index is current - no new messages to process"
                }
                
                return existing_index_path, existing_metadata_path, stats
            else:
                # No existing index and no messages to process
                raise ValueError("No existing index found and no messages to process. Run with --force to rebuild.")
            
        # Step 3: Create embeddings
        embeddings = self.create_embeddings(processed_messages)
        
        # Step 4: Build FAISS index
        self.index = self.build_faiss_index(embeddings)
        
        # Step 5: Create rich community metadata
        self.metadata = self.create_community_metadata(processed_messages)
        
        # Step 6: Create ID mapping
        self.id_mapping = {i: msg['id'] for i, msg in enumerate(processed_messages)}
        
        # Step 7: Save everything
        index_path, metadata_path = self.save_index_and_metadata(
            self.index, self.metadata, save_filename
        )
        
        # Generate summary statistics
        stats = self._generate_build_stats(messages, processed_messages)
        
        logger.info("Community FAISS index build complete!")
        logger.info(f"Index saved to: {index_path}")
        logger.info(f"Metadata saved to: {metadata_path}")
        logger.info(f"Processed {stats['total_messages_processed']} messages")
        logger.info(f"Community features extracted: {stats['community_features_extracted']}")
        
        return index_path, metadata_path, stats
    
    def _generate_build_stats(self, original_messages: List[Dict], processed_messages: List[Dict]) -> Dict:
        """Generate comprehensive build statistics"""
        # Calculate basic stats
        messages_with_attachments = 0
        messages_with_community_data = 0
        skill_distribution = {}
        content_type_distribution = {}
        topic_distribution = {}
        
        for msg in processed_messages:
            attachments = json.loads(msg['attachments']) if msg['attachments'] else []
            if len(attachments) > 0:
                messages_with_attachments += 1
            
            community_meta = msg.get('community_metadata', {})
            if community_meta:
                messages_with_community_data += 1
                
                # Collect skill distribution
                for skill in community_meta.get('skill_keywords', []):
                    skill_distribution[skill] = skill_distribution.get(skill, 0) + 1
                
                # Collect content type distribution
                content_type = community_meta.get('content_type', 'general')
                content_type_distribution[content_type] = content_type_distribution.get(content_type, 0) + 1
                
                # Collect topic distribution
                topic = community_meta.get('topic_category', 'general')
                topic_distribution[topic] = topic_distribution.get(topic, 0) + 1
        
        return {
            'total_messages_processed': len(processed_messages),
            'total_messages_loaded': len(original_messages),
            'filter_rate': (len(original_messages) - len(processed_messages)) / len(original_messages) if original_messages else 0,
            'embedding_dimension': self.embedding_dim,
            'index_type': type(self.index).__name__,
            'model_name': self.model_name,
            'avg_content_length': np.mean([msg['content_length'] for msg in processed_messages]),
            'messages_with_embeds': sum(1 for msg in processed_messages if msg['has_embed_content']),
            'messages_with_attachments': messages_with_attachments,
            'total_urls_extracted': sum(len(msg['extracted_urls']) for msg in processed_messages),
            
            # Community-specific stats
            'community_features_extracted': messages_with_community_data,
            'community_extraction_rate': messages_with_community_data / len(processed_messages) if processed_messages else 0,
            'top_skills': dict(sorted(skill_distribution.items(), key=lambda x: x[1], reverse=True)[:10]),
            'content_type_distribution': content_type_distribution,
            'topic_distribution': topic_distribution,
            'messages_with_code': sum(1 for msg in processed_messages 
                                    if msg.get('community_metadata', {}).get('code_snippets')),
            'help_seeking_messages': sum(1 for msg in processed_messages 
                                       if msg.get('community_metadata', {}).get('help_seeking')),
            'help_providing_messages': sum(1 for msg in processed_messages 
                                         if msg.get('community_metadata', {}).get('help_providing')),
            'time_sensitive_messages': sum(1 for msg in processed_messages 
                                         if msg.get('community_metadata', {}).get('time_sensitive'))
        }

def main():
    """Main function to build community FAISS index."""
    # Initialize builder
    builder = CommunityFAISSIndexBuilder(
        model_name="msmarco-distilbert-base-v4",
        db_path="data/discord_messages.db",
        batch_size=50
    )
    
    try:
        # Build complete index
        index_path, metadata_path, stats = builder.build_complete_index()
        
        # Save summary report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f"data/reports/community_faiss_build_report_{timestamp}.json"
        os.makedirs("data/reports", exist_ok=True)
        
        report = {
            'build_completed_at': datetime.now().isoformat(),
            'index_path': index_path,
            'metadata_path': metadata_path,
            'statistics': stats,
            'builder_config': {
                'model_name': builder.model_name,
                'batch_size': builder.batch_size,
                'embedding_dimension': builder.embedding_dim
            }
        }
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Build report saved to: {report_path}")
        
        return index_path, metadata_path, stats
        
    except Exception as e:
        logger.error(f"Error building community FAISS index: {e}")
        raise

if __name__ == "__main__":
    main()
