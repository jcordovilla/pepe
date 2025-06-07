#!/usr/bin/env python3
"""
Resource-Focused FAISS Index Builder

Builds an optimized FAISS index specifically for resources (links, documents, tools)
with comprehensive metadata for enhanced retrieval and agent resource discovery.
"""

import json
import logging
import os
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ResourceFAISSIndexBuilder:
    """Resource-focused FAISS index builder with rich metadata for semantic resource discovery."""
    
    def __init__(self, 
                 model_name: str = "msmarco-distilbert-base-v4",
                 resources_path: str = "data/resources/test_resources.json",
                 batch_size: int = 100):
        """
        Initialize the resource FAISS index builder.
        
        Args:
            model_name: Name of the sentence transformer model
            resources_path: Path to the resources JSON file
            batch_size: Batch size for encoding
        """
        self.model_name = model_name
        self.resources_path = resources_path
        self.batch_size = batch_size
        
        # Initialize sentence transformer
        logger.info(f"Loading sentence transformer model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        # Index components
        self.index = None
        self.metadata = []
        self.id_mapping = {}  # FAISS index -> resource ID mapping
        
    def load_resources_from_json(self) -> List[Dict]:
        """Load resources from JSON file."""
        logger.info(f"Loading resources from: {self.resources_path}")
        
        if not os.path.exists(self.resources_path):
            raise FileNotFoundError(f"Resources file not found: {self.resources_path}")
            
        with open(self.resources_path, 'r', encoding='utf-8') as f:
            resources = json.load(f)
            
        logger.info(f"Loaded {len(resources)} resources from JSON")
        return resources
    
    def preprocess_resources(self, resources: List[Dict]) -> List[Dict]:
        """Preprocess resources for embedding generation."""
        logger.info("Preprocessing resources...")
        
        processed_resources = []
        
        for resource in resources:
            # Create searchable text combining title, description, and context
            searchable_parts = []
            
            # Add title
            title = resource.get('title', '').strip()
            if title:
                searchable_parts.append(title)
                
            # Add description
            description = resource.get('description', '').strip()
            if description:
                searchable_parts.append(description)
                
            # Add tag/category context
            tag = resource.get('tag', '').strip()
            if tag:
                searchable_parts.append(f"Category: {tag}")
                
            # Add domain context
            url = resource.get('resource_url', '')
            if url:
                try:
                    from urllib.parse import urlparse
                    domain = urlparse(url).netloc.replace('www.', '')
                    if domain:
                        searchable_parts.append(f"Domain: {domain}")
                except:
                    pass
                    
            # Add author context
            author = resource.get('author', '').strip()
            if author:
                searchable_parts.append(f"Shared by: {author}")
                
            # Create combined searchable text
            searchable_text = ' | '.join(searchable_parts)
            
            # Enhanced resource with searchable text
            enhanced_resource = {
                **resource,
                'searchable_text': searchable_text,
                'searchable_length': len(searchable_text)
            }
            
            processed_resources.append(enhanced_resource)
            
        logger.info(f"Preprocessed {len(processed_resources)} resources")
        return processed_resources
    
    def create_embeddings(self, resources: List[Dict]) -> np.ndarray:
        """Create embeddings for preprocessed resources."""
        logger.info("Creating embeddings...")
        
        # Extract searchable text for embedding
        texts = [resource['searchable_text'] for resource in resources]
        
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
                logger.info(f"Processed {i + len(batch_texts)} / {len(texts)} resources")
                
        embeddings = np.vstack(embeddings)
        logger.info(f"Created embeddings: {embeddings.shape}")
        return embeddings
    
    def build_faiss_index(self, embeddings: np.ndarray) -> faiss.Index:
        """Build optimized FAISS index for resources."""
        logger.info("Building FAISS index...")
        
        n_vectors, dim = embeddings.shape
        
        if n_vectors < 1000:
            # For small datasets, use simple flat index
            index = faiss.IndexFlatIP(dim)  # Inner product for normalized vectors
            logger.info("Using IndexFlatIP for resource dataset")
        elif n_vectors < 10000:
            # For medium datasets, use IVF with reasonable number of clusters
            n_clusters = min(100, n_vectors // 50)
            quantizer = faiss.IndexFlatIP(dim)
            index = faiss.IndexIVFFlat(quantizer, dim, n_clusters)
            logger.info(f"Using IndexIVFFlat with {n_clusters} clusters")
            
            # Train index
            logger.info("Training index...")
            index.train(embeddings)
        else:
            # For large datasets, use more sophisticated index
            n_clusters = min(1000, n_vectors // 100)
            quantizer = faiss.IndexFlatIP(dim)
            index = faiss.IndexIVFFlat(quantizer, dim, n_clusters)
            logger.info(f"Using IndexIVFFlat with {n_clusters} clusters for large dataset")
            
            # Train index
            logger.info("Training index...")
            index.train(embeddings)
            
        # Add vectors to index
        logger.info("Adding vectors to index...")
        index.add(embeddings)
        
        logger.info(f"FAISS index built successfully. Total vectors: {index.ntotal}")
        return index
    
    def create_resource_metadata(self, resources: List[Dict]) -> List[Dict]:
        """Create rich metadata for enhanced resource retrieval."""
        logger.info("Creating resource metadata...")
        
        metadata = []
        for i, resource in enumerate(resources):
            # Parse URL for additional metadata
            domain = ""
            url_path = ""
            if resource.get('resource_url'):
                try:
                    from urllib.parse import urlparse
                    parsed = urlparse(resource['resource_url'])
                    domain = parsed.netloc.replace('www.', '')
                    url_path = parsed.path
                except:
                    pass
            
            # Create comprehensive metadata
            meta = {
                # Core identifiers
                'resource_id': resource.get('id'),
                'title': resource.get('title', 'Untitled Resource'),
                'description': resource.get('description', ''),
                'resource_url': resource.get('resource_url', ''),
                'discord_url': resource.get('discord_url', ''),
                
                # Content classification
                'tag': resource.get('tag', 'Other'),
                'type': resource.get('type', 'link'),
                'domain': domain,
                'url_path': url_path,
                
                # Social metadata
                'author': resource.get('author', 'Unknown'),
                'channel': resource.get('channel', ''),
                'date': resource.get('date', ''),
                'message_id': resource.get('message_id', ''),
                'guild_id': resource.get('guild_id', ''),
                'channel_id': resource.get('channel_id', ''),
                
                # Search metadata
                'searchable_text': resource.get('searchable_text', ''),
                'searchable_length': resource.get('searchable_length', 0),
                
                # Classification features
                'is_article': resource.get('tag', '').lower() in ['news/article', 'article'],
                'is_tool': resource.get('tag', '').lower() == 'tool',
                'is_academic': domain in ['arxiv.org', 'scholar.google.com', 'pubmed.ncbi.nlm.nih.gov'],
                'is_github': domain == 'github.com',
                'is_youtube': domain in ['youtube.com', 'youtu.be'],
                'is_social': domain in ['twitter.com', 'linkedin.com', 'medium.com'],
                'is_documentation': 'doc' in url_path.lower() or 'docs' in domain,
                
                # Temporal features
                'is_recent': resource.get('date', '') >= '2025-01-01',
                'year': resource.get('date', '')[:4] if resource.get('date') else '',
                
                # FAISS index position
                'faiss_index': i
            }
            
            metadata.append(meta)
            
        logger.info(f"Created metadata for {len(metadata)} resources")
        return metadata
    
    def save_index_and_metadata(self, 
                               index: faiss.Index, 
                               metadata: List[Dict],
                               base_filename: str = None) -> Tuple[str, str]:
        """Save FAISS index and metadata to files."""
        if base_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_filename = f"resource_faiss_{timestamp}"
            
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
                    'index_type': 'resource_focused',
                    'data_source': self.resources_path,
                    'features': {
                        'semantic_search': True,
                        'domain_classification': True,
                        'temporal_filtering': True,
                        'content_type_classification': True,
                        'social_context_preservation': True
                    }
                }
            }, f, indent=2)
        
        return index_path, metadata_path
    
    def _generate_build_stats(self, resources: List[Dict]) -> Dict:
        """Generate comprehensive build statistics"""
        
        # Content type distribution
        tag_distribution = {}
        domain_distribution = {}
        author_distribution = {}
        
        for resource in resources:
            # Tag distribution
            tag = resource.get('tag', 'Other')
            tag_distribution[tag] = tag_distribution.get(tag, 0) + 1
            
            # Domain distribution
            url = resource.get('resource_url', '')
            if url:
                try:
                    from urllib.parse import urlparse
                    domain = urlparse(url).netloc.replace('www.', '')
                    if domain:
                        domain_distribution[domain] = domain_distribution.get(domain, 0) + 1
                except:
                    pass
                    
            # Author distribution
            author = resource.get('author', 'Unknown')
            author_distribution[author] = author_distribution.get(author, 0) + 1
        
        return {
            'total_resources_processed': len(resources),
            'embedding_dimension': self.embedding_dim,
            'index_type': type(self.index).__name__,
            'model_name': self.model_name,
            'avg_searchable_length': np.mean([r.get('searchable_length', 0) for r in resources]),
            
            # Resource type analysis
            'tag_distribution': dict(sorted(tag_distribution.items(), key=lambda x: x[1], reverse=True)),
            'top_domains': dict(sorted(domain_distribution.items(), key=lambda x: x[1], reverse=True)[:10]),
            'top_contributors': dict(sorted(author_distribution.items(), key=lambda x: x[1], reverse=True)[:10]),
            
            # Content analysis
            'articles': sum(1 for r in resources if r.get('tag', '').lower() in ['news/article', 'article']),
            'tools': sum(1 for r in resources if r.get('tag', '').lower() == 'tool'),
            'github_repos': sum(1 for r in resources if 'github.com' in r.get('resource_url', '')),
            'youtube_videos': sum(1 for r in resources if any(domain in r.get('resource_url', '') 
                                 for domain in ['youtube.com', 'youtu.be'])),
            'academic_papers': sum(1 for r in resources if any(domain in r.get('resource_url', '') 
                                  for domain in ['arxiv.org', 'scholar.google.com'])),
            
            # Temporal analysis
            'recent_resources': sum(1 for r in resources if r.get('date', '') >= '2025-01-01'),
            'total_unique_domains': len(domain_distribution),
            'total_unique_authors': len(author_distribution)
        }
    
    def build_complete_index(self, 
                           save_filename: str = None) -> Tuple[str, str, Dict]:
        """Build complete resource-focused FAISS index with all steps."""
        logger.info("Starting complete resource FAISS index build...")
        
        # Step 1: Load resources
        resources = self.load_resources_from_json()
        
        # Step 2: Preprocess resources
        processed_resources = self.preprocess_resources(resources)
        
        if not processed_resources:
            raise ValueError("No resources found after preprocessing")
            
        # Step 3: Create embeddings
        embeddings = self.create_embeddings(processed_resources)
        
        # Step 4: Build FAISS index
        self.index = self.build_faiss_index(embeddings)
        
        # Step 5: Create rich metadata
        self.metadata = self.create_resource_metadata(processed_resources)
        
        # Step 6: Create ID mapping
        self.id_mapping = {i: resource['id'] for i, resource in enumerate(processed_resources)}
        
        # Step 7: Save everything
        index_path, metadata_path = self.save_index_and_metadata(
            self.index, self.metadata, save_filename
        )
        
        # Generate summary statistics
        stats = self._generate_build_stats(processed_resources)
        
        logger.info("Resource FAISS index build complete!")
        logger.info(f"Index saved to: {index_path}")
        logger.info(f"Metadata saved to: {metadata_path}")
        logger.info(f"Processed {stats['total_resources_processed']} resources")
        logger.info(f"Top resource types: {list(stats['tag_distribution'].keys())[:3]}")
        
        return index_path, metadata_path, stats

def main():
    """Main function to build resource FAISS index."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Build FAISS index for resources")
    parser.add_argument("--model", default="msmarco-distilbert-base-v4", 
                       help="Sentence transformer model name")
    parser.add_argument("--resources", default="data/resources/test_resources.json",
                       help="Path to resources JSON file")
    parser.add_argument("--batch-size", type=int, default=100,
                       help="Batch size for processing")
    parser.add_argument("--output", 
                       help="Output filename prefix (optional)")
    
    args = parser.parse_args()
    
    try:
        # Initialize builder
        builder = ResourceFAISSIndexBuilder(
            model_name=args.model,
            resources_path=args.resources,
            batch_size=args.batch_size
        )
        
        # Build complete index
        index_path, metadata_path, stats = builder.build_complete_index(
            save_filename=args.output
        )
        
        # Save build report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f"data/indices/resource_faiss_build_report_{timestamp}.json"
        
        build_report = {
            'build_timestamp': datetime.now().isoformat(),
            'index_path': index_path,
            'metadata_path': metadata_path,
            'statistics': stats,
            'configuration': {
                'model_name': args.model,
                'resources_path': args.resources,
                'batch_size': args.batch_size
            }
        }
        
        with open(report_path, 'w') as f:
            json.dump(build_report, f, indent=2)
            
        logger.info(f"Build report saved to: {report_path}")
        
        print("\n" + "="*60)
        print("🎉 RESOURCE FAISS INDEX BUILD COMPLETE!")
        print("="*60)
        print(f"📁 Index: {index_path}")
        print(f"📄 Metadata: {metadata_path}")
        print(f"📊 Report: {report_path}")
        print(f"🔍 Resources indexed: {stats['total_resources_processed']}")
        print(f"🎯 Embedding dimension: {stats['embedding_dimension']}")
        print(f"🏷️ Top categories: {', '.join(list(stats['tag_distribution'].keys())[:3])}")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Failed to build resource index: {e}")
        raise

if __name__ == "__main__":
    main()
