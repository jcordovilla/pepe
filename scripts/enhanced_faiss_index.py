#!/usr/bin/env python3
"""
Enhanced FAISS Indexing System for Discord Messages

This module creates FAISS indexes with rich metadata support for Discord messages.
Builds on preprocessed content to create optimized vector search capabilities.

Features:
- Multiple index types (flat, IVF, HNSW)
- Rich metadata storage and filtering
- Batch processing for large datasets
- Index persistence and loading
- Search result ranking and filtering
"""

import os
import sys
import json
import pickle
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import faiss
from dataclasses import dataclass

# Add project root to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from scripts.content_preprocessor import ContentPreprocessor, PreprocessingConfig
from utils.logger import setup_logging

setup_logging()
import logging
log = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    log.error("sentence-transformers not installed. Run: pip install sentence-transformers")
    sys.exit(1)

@dataclass
class IndexConfig:
    """Configuration for FAISS index creation"""
    model_name: str = "all-MiniLM-L6-v2"  # Lightweight, good performance
    index_type: str = "flat"  # Options: flat, ivf, hnsw
    dimension: int = 384  # Dimension for all-MiniLM-L6-v2
    nlist: int = 100  # For IVF index
    m: int = 16  # For HNSW index
    ef_construction: int = 200  # For HNSW index
    ef_search: int = 100  # For HNSW search
    batch_size: int = 32
    normalize_embeddings: bool = True

@dataclass
class SearchResult:
    """Search result with rich metadata"""
    message_id: int
    score: float
    content: str
    metadata: Dict[str, Any]
    
class EnhancedFAISSIndex:
    """Enhanced FAISS index with rich metadata support"""
    
    def __init__(self, config: IndexConfig = None):
        self.config = config or IndexConfig()
        self.model = None
        self.index = None
        self.metadata = {}  # message_id -> metadata dict
        self.id_mapping = []  # index position -> message_id
        self.dimension = self.config.dimension
        
    def _load_model(self):
        """Load the sentence transformer model"""
        if self.model is None:
            log.info(f"Loading embedding model: {self.config.model_name}")
            self.model = SentenceTransformer(self.config.model_name)
            # Update dimension from model
            self.dimension = self.model.get_sentence_embedding_dimension()
            self.config.dimension = self.dimension
            log.info(f"Model loaded with dimension: {self.dimension}")
    
    def _create_index(self, num_vectors: int):
        """Create appropriate FAISS index based on configuration"""
        if self.config.index_type == "flat":
            if self.config.normalize_embeddings:
                index = faiss.IndexFlatIP(self.dimension)  # Inner product for normalized vectors
            else:
                index = faiss.IndexFlatL2(self.dimension)
                
        elif self.config.index_type == "ivf":
            # Use IVF (Inverted File) for faster search on large datasets
            nlist = min(self.config.nlist, max(1, num_vectors // 50))  # Adaptive nlist
            quantizer = faiss.IndexFlatL2(self.dimension)
            index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
            
        elif self.config.index_type == "hnsw":
            # Use HNSW for very fast approximate search
            index = faiss.IndexHNSWFlat(self.dimension, self.config.m)
            index.hnsw.efConstruction = self.config.ef_construction
            
        else:
            raise ValueError(f"Unknown index type: {self.config.index_type}")
            
        log.info(f"Created {self.config.index_type} index for {num_vectors} vectors")
        return index
    
    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        """Convert texts to embeddings using the model"""
        self._load_model()
        
        log.info(f"Generating embeddings for {len(texts)} texts")
        embeddings = self.model.encode(
            texts, 
            batch_size=self.config.batch_size, 
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        if self.config.normalize_embeddings:
            # Normalize for cosine similarity (using inner product)
            faiss.normalize_L2(embeddings)
            
        return embeddings.astype(np.float32)
    
    def build_index(self, preprocessed_messages: List[Dict]):
        """Build FAISS index from preprocessed messages"""
        if not preprocessed_messages:
            raise ValueError("No preprocessed messages provided")
            
        log.info(f"Building index for {len(preprocessed_messages)} messages")
        
        # Extract texts and prepare metadata
        texts = []
        for msg in preprocessed_messages:
            texts.append(msg['searchable_text'])
            self.metadata[msg['message_id']] = {
                'content': msg['original_content'],
                'cleaned_content': msg['cleaned_content'],
                'timestamp': msg['timestamp'],
                'channel_id': msg['channel_id'],
                'guild_id': msg['guild_id'],
                'author_id': msg['author_id'],
                'message_type': msg['message_type'],
                'is_pinned': msg['is_pinned'],
                'has_embeds': msg['has_embeds'],
                'has_attachments': msg['has_attachments'],
                'has_reply_context': msg['has_reply_context'],
                'content_length': msg['content_length'],
                'reaction_count': msg['reaction_count'],
                'extracted_urls': msg['extracted_urls'],
            }
            self.id_mapping.append(msg['message_id'])
        
        # Generate embeddings
        embeddings = self._embed_texts(texts)
        
        # Create and populate index
        self.index = self._create_index(len(embeddings))
        
        if self.config.index_type == "ivf":
            # Train IVF index
            log.info("Training IVF index...")
            self.index.train(embeddings)
        
        # Add vectors to index
        log.info("Adding vectors to index...")
        self.index.add(embeddings)
        
        if self.config.index_type == "hnsw":
            # Set search parameters for HNSW
            self.index.hnsw.efSearch = self.config.ef_search
        
        log.info(f"Index built successfully with {self.index.ntotal} vectors")
    
    def search(self, 
               query: str, 
               k: int = 10, 
               filters: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """Search the index with optional metadata filtering"""
        
        if self.index is None:
            raise ValueError("Index not built yet. Call build_index() first.")
        
        # Generate query embedding
        query_embedding = self._embed_texts([query])
        
        # Search the index
        scores, indices = self.index.search(query_embedding, k * 2)  # Get more for filtering
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # Invalid index
                continue
                
            message_id = self.id_mapping[idx]
            metadata = self.metadata[message_id]
            
            # Apply filters if provided
            if filters and not self._passes_filters(metadata, filters):
                continue
                
            result = SearchResult(
                message_id=message_id,
                score=float(score),
                content=metadata['content'],
                metadata=metadata
            )
            results.append(result)
            
            if len(results) >= k:  # Got enough results
                break
        
        return results
    
    def _passes_filters(self, metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check if metadata passes the provided filters"""
        for key, value in filters.items():
            if key == 'min_timestamp':
                if metadata['timestamp'] < value:
                    return False
            elif key == 'max_timestamp':
                if metadata['timestamp'] > value:
                    return False
            elif key == 'channel_ids':
                if metadata['channel_id'] not in value:
                    return False
            elif key == 'author_ids':
                if metadata['author_id'] not in value:
                    return False
            elif key == 'has_embeds':
                if metadata['has_embeds'] != value:
                    return False
            elif key == 'has_attachments':
                if metadata['has_attachments'] != value:
                    return False
            elif key == 'is_pinned':
                if metadata['is_pinned'] != value:
                    return False
            elif key == 'min_reactions':
                if metadata['reaction_count'] < value:
                    return False
            elif key == 'min_content_length':
                if metadata['content_length'] < value:
                    return False
        
        return True
    
    def save_index(self, directory: str):
        """Save index and metadata to disk"""
        os.makedirs(directory, exist_ok=True)
        
        # Save FAISS index
        index_path = os.path.join(directory, "faiss_index.index")
        faiss.write_index(self.index, index_path)
        
        # Save metadata and mappings
        metadata_path = os.path.join(directory, "metadata.json")
        with open(metadata_path, 'w') as f:
            # Convert datetime objects to strings for JSON serialization
            serializable_metadata = {}
            for msg_id, meta in self.metadata.items():
                serializable_meta = meta.copy()
                if 'timestamp' in serializable_meta:
                    serializable_meta['timestamp'] = serializable_meta['timestamp'].isoformat()
                serializable_metadata[str(msg_id)] = serializable_meta
            
            json.dump({
                'metadata': serializable_metadata,
                'id_mapping': self.id_mapping,
                'config': {
                    'model_name': self.config.model_name,
                    'index_type': self.config.index_type,
                    'dimension': self.config.dimension,
                    'normalize_embeddings': self.config.normalize_embeddings
                }
            }, f, indent=2)
        
        log.info(f"Index saved to {directory}")
    
    def load_index(self, directory: str):
        """Load index and metadata from disk"""
        # Load FAISS index
        index_path = os.path.join(directory, "faiss_index.index")
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Index file not found: {index_path}")
        
        self.index = faiss.read_index(index_path)
        
        # Load metadata and mappings
        metadata_path = os.path.join(directory, "metadata.json")
        with open(metadata_path, 'r') as f:
            data = json.load(f)
            
            # Restore metadata with proper types
            self.metadata = {}
            for msg_id_str, meta in data['metadata'].items():
                msg_id = int(msg_id_str)
                meta_copy = meta.copy()
                if 'timestamp' in meta_copy:
                    meta_copy['timestamp'] = datetime.fromisoformat(meta_copy['timestamp'])
                self.metadata[msg_id] = meta_copy
            
            self.id_mapping = data['id_mapping']
            
            # Restore config
            config_data = data['config']
            self.config.model_name = config_data['model_name']
            self.config.index_type = config_data['index_type']
            self.config.dimension = config_data['dimension']
            self.config.normalize_embeddings = config_data['normalize_embeddings']
            self.dimension = self.config.dimension
        
        log.info(f"Index loaded from {directory} with {self.index.ntotal} vectors")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics"""
        if self.index is None:
            return {"status": "not_built"}
        
        return {
            "status": "ready",
            "total_vectors": self.index.ntotal,
            "dimension": self.dimension,
            "index_type": self.config.index_type,
            "model_name": self.config.model_name,
            "total_messages": len(self.metadata),
            "normalize_embeddings": self.config.normalize_embeddings
        }

def main():
    """Main function for testing enhanced indexing"""
    # Initialize preprocessor and index
    preprocessor = ContentPreprocessor()
    index_config = IndexConfig(
        model_name="all-MiniLM-L6-v2",
        index_type="flat",  # Start with flat for testing
        batch_size=32
    )
    
    faiss_index = EnhancedFAISSIndex(index_config)
    
    # Preprocess a sample of messages
    log.info("Preprocessing messages for indexing...")
    preprocessed_messages = preprocessor.preprocess_batch(limit=1000)
    
    if not preprocessed_messages:
        log.error("No messages to index after preprocessing")
        return
    
    # Build index
    log.info("Building FAISS index...")
    faiss_index.build_index(preprocessed_messages)
    
    # Test search
    log.info("Testing search functionality...")
    test_queries = [
        "hello world",
        "discord bot",
        "python code",
        "help with programming"
    ]
    
    for query in test_queries:
        results = faiss_index.search(query, k=5)
        log.info(f"\nQuery: '{query}' - Found {len(results)} results:")
        for i, result in enumerate(results[:3]):  # Show top 3
            content_preview = result.content[:100] + "..." if len(result.content) > 100 else result.content
            log.info(f"  {i+1}. Score: {result.score:.3f} - {content_preview}")
    
    # Save index
    index_dir = f"index_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    faiss_index.save_index(index_dir)
    log.info(f"Index saved to: {index_dir}")
    
    # Print stats
    stats = faiss_index.get_stats()
    log.info(f"Index stats: {json.dumps(stats, indent=2)}")

if __name__ == "__main__":
    main()
