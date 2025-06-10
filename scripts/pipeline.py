#!/usr/bin/env python3
"""
Integrated Discord Message Processing Pipeline

This script orchestrates the complete pipeline from raw Discord messages
to searchable FAISS index with rich metadata support.

Pipeline stages:
1. Content preprocessing and filtering
2. Embed extraction and text enhancement
3. FAISS index creation with metadata
4. Search functionality testing
5. Performance optimization
"""

import os
import sys
import json
import argparse
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

# Add project root to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from scripts.content_preprocessor import ContentPreprocessor, PreprocessingConfig
from scripts.enhanced_faiss_index import EnhancedFAISSIndex, IndexConfig, SearchResult
from utils.logger import setup_logging

setup_logging()
import logging
log = logging.getLogger(__name__)

class DiscordMessagePipeline:
    """Complete pipeline for Discord message processing and indexing"""
    
    def __init__(self, 
                 preprocessing_config: PreprocessingConfig = None,
                 index_config: IndexConfig = None):
        self.preprocessing_config = preprocessing_config or PreprocessingConfig()
        self.index_config = index_config or IndexConfig()
        self.preprocessor = ContentPreprocessor(self.preprocessing_config)
        self.faiss_index = EnhancedFAISSIndex(self.index_config)
        self.pipeline_stats = {}
        
    def run_full_pipeline(self, 
                         limit: int = None, 
                         save_index: bool = True,
                         index_name: str = None) -> Dict[str, Any]:
        """Run the complete pipeline from messages to searchable index"""
        
        pipeline_start = datetime.now()
        log.info("Starting complete Discord message processing pipeline")
        
        # Stage 1: Preprocessing
        log.info("Stage 1: Preprocessing messages...")
        preprocessing_start = datetime.now()
        
        preprocessed_messages = self.preprocessor.preprocess_batch(limit=limit)
        
        preprocessing_time = (datetime.now() - preprocessing_start).total_seconds()
        
        if not preprocessed_messages:
            raise ValueError("No messages survived preprocessing filters")
        
        log.info(f"Preprocessing completed: {len(preprocessed_messages)} messages in {preprocessing_time:.1f}s")
        
        # Stage 2: Index building
        log.info("Stage 2: Building FAISS index...")
        indexing_start = datetime.now()
        
        self.faiss_index.build_index(preprocessed_messages)
        
        indexing_time = (datetime.now() - indexing_start).total_seconds()
        log.info(f"Index building completed in {indexing_time:.1f}s")
        
        # Stage 3: Save index if requested
        if save_index:
            log.info("Stage 3: Saving index...")
            save_start = datetime.now()
            
            if index_name is None:
                index_name = f"discord_index_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Ensure index is saved in the correct location
            index_path = os.path.join("data", "indices", index_name)
            self.faiss_index.save_index(index_path)
            save_time = (datetime.now() - save_start).total_seconds()
            log.info(f"Index saved to '{index_path}' in {save_time:.1f}s")
        
        # Generate pipeline statistics
        total_time = (datetime.now() - pipeline_start).total_seconds()
        
        self.pipeline_stats = {
            'pipeline_completed': datetime.now().isoformat(),
            'total_processing_time_seconds': total_time,
            'preprocessing_time_seconds': preprocessing_time,
            'indexing_time_seconds': indexing_time,
            'messages_processed': len(preprocessed_messages),
            'processing_rate_messages_per_second': len(preprocessed_messages) / total_time,
            'index_stats': self.faiss_index.get_stats(),
            'preprocessing_config': {
                'min_content_length': self.preprocessing_config.min_content_length,
                'include_embed_content': self.preprocessing_config.include_embed_content,
                'include_reply_context': self.preprocessing_config.include_reply_context,
                'normalize_urls': self.preprocessing_config.normalize_urls,
                'filter_bot_messages': self.preprocessing_config.filter_bot_messages
            },
            'index_config': {
                'model_name': self.index_config.model_name,
                'index_type': self.index_config.index_type,
                'dimension': self.index_config.dimension,
                'batch_size': self.index_config.batch_size,
                'normalize_embeddings': self.index_config.normalize_embeddings
            }
        }
        
        if save_index:
            self.pipeline_stats['index_saved_to'] = index_path
        
        log.info(f"Pipeline completed successfully in {total_time:.1f}s")
        log.info(f"Processing rate: {self.pipeline_stats['processing_rate_messages_per_second']:.1f} messages/second")
        
        return self.pipeline_stats
    
    def test_search_functionality(self, test_queries: List[str] = None) -> Dict[str, List[SearchResult]]:
        """Test search functionality with various queries"""
        
        if test_queries is None:
            test_queries = [
                "discord bot setup",
                "python programming",
                "help with code",
                "error debugging",
                "how to install",
                "API documentation",
                "database connection",
                "web development",
                "machine learning",
                "git repository"
            ]
        
        log.info(f"Testing search functionality with {len(test_queries)} queries")
        
        search_results = {}
        total_search_time = 0
        
        for query in test_queries:
            search_start = datetime.now()
            results = self.faiss_index.search(query, k=5)
            search_time = (datetime.now() - search_start).total_seconds()
            total_search_time += search_time
            
            search_results[query] = results
            
            log.info(f"Query: '{query}' - {len(results)} results in {search_time*1000:.1f}ms")
            for i, result in enumerate(results[:2]):  # Show top 2
                content_preview = result.content[:80] + "..." if len(result.content) > 80 else result.content
                log.info(f"  {i+1}. Score: {result.score:.3f} - {content_preview}")
        
        avg_search_time = total_search_time / len(test_queries)
        log.info(f"Average search time: {avg_search_time*1000:.1f}ms")
        
        return search_results
    
    def analyze_content_distribution(self) -> Dict[str, Any]:
        """Analyze the distribution of content in the index"""
        
        if not self.faiss_index.metadata:
            return {"error": "No metadata available"}
        
        analysis = {
            'total_messages': len(self.faiss_index.metadata),
            'content_types': {
                'with_embeds': 0,
                'with_attachments': 0,
                'with_reply_context': 0,
                'pinned_messages': 0
            },
            'content_length_stats': {
                'lengths': [],
                'min': float('inf'),
                'max': 0,
                'avg': 0
            },
            'temporal_distribution': {},
            'channel_distribution': {},
            'author_distribution': {}
        }
        
        lengths = []
        for metadata in self.faiss_index.metadata.values():
            # Content type analysis
            if metadata['has_embeds']:
                analysis['content_types']['with_embeds'] += 1
            if metadata['has_attachments']:
                analysis['content_types']['with_attachments'] += 1
            if metadata['has_reply_context']:
                analysis['content_types']['with_reply_context'] += 1
            if metadata['is_pinned']:
                analysis['content_types']['pinned_messages'] += 1
            
            # Content length analysis
            length = metadata['content_length']
            lengths.append(length)
            analysis['content_length_stats']['min'] = min(analysis['content_length_stats']['min'], length)
            analysis['content_length_stats']['max'] = max(analysis['content_length_stats']['max'], length)
            
            # Temporal distribution
            if metadata['timestamp']:
                date_key = metadata['timestamp'].date().isoformat()
                analysis['temporal_distribution'][date_key] = analysis['temporal_distribution'].get(date_key, 0) + 1
            
            # Channel distribution
            channel_id = str(metadata['channel_id'])
            analysis['channel_distribution'][channel_id] = analysis['channel_distribution'].get(channel_id, 0) + 1
            
            # Author distribution
            if metadata['author_id']:
                author_id = str(metadata['author_id'])
                analysis['author_distribution'][author_id] = analysis['author_distribution'].get(author_id, 0) + 1
        
        # Finalize content length stats
        if lengths:
            analysis['content_length_stats']['avg'] = sum(lengths) / len(lengths)
            analysis['content_length_stats']['lengths'] = lengths  # For further analysis
        
        # Sort distributions by count
        analysis['channel_distribution'] = dict(sorted(
            analysis['channel_distribution'].items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:10])  # Top 10 channels
        
        analysis['author_distribution'] = dict(sorted(
            analysis['author_distribution'].items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:10])  # Top 10 authors
        
        return analysis
    
    def save_pipeline_report(self, output_path: str = None):
        """Save comprehensive pipeline report"""
        
        if output_path is None:
            output_path = f"pipeline_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Generate content analysis
        content_analysis = self.analyze_content_distribution()
        
        # Combine all statistics
        report = {
            'pipeline_stats': self.pipeline_stats,
            'content_analysis': content_analysis,
            'report_generated': datetime.now().isoformat()
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        log.info(f"Pipeline report saved to: {output_path}")
        return output_path

def main():
    """Main function with command-line interface"""
    parser = argparse.ArgumentParser(description="Discord Message Processing Pipeline")
    parser.add_argument("--limit", type=int, help="Limit number of messages to process")
    parser.add_argument("--index-type", choices=["flat", "ivf", "hnsw"], default="flat", 
                       help="FAISS index type")
    parser.add_argument("--model", default="msmarco-distilbert-base-v4", 
                       help="Sentence transformer model name")
    parser.add_argument("--no-save", action="store_true", help="Don't save index to disk")
    parser.add_argument("--index-name", help="Custom name for saved index")
    parser.add_argument("--test-search", action="store_true", help="Run search functionality tests")
    parser.add_argument("--min-content-length", type=int, default=10, 
                       help="Minimum content length for preprocessing")
    
    args = parser.parse_args()
    
    # Configure preprocessing
    preprocessing_config = PreprocessingConfig(
        min_content_length=args.min_content_length,
        include_embed_content=True,
        include_reply_context=True,
        normalize_urls=True,
        filter_bot_messages=True
    )
    
    # Configure indexing
    index_config = IndexConfig(
        model_name=args.model,
        index_type=args.index_type,
        batch_size=32,
        normalize_embeddings=True
    )
    
    # Initialize pipeline
    pipeline = DiscordMessagePipeline(preprocessing_config, index_config)
    
    try:
        # Run pipeline
        stats = pipeline.run_full_pipeline(
            limit=args.limit,
            save_index=not args.no_save,
            index_name=args.index_name
        )
        
        # Test search if requested
        if args.test_search:
            search_results = pipeline.test_search_functionality()
            log.info(f"Search testing completed for {len(search_results)} queries")
        
        # Save comprehensive report
        report_path = pipeline.save_pipeline_report()
        
        # Print summary
        print(f"\n{'='*60}")
        print("PIPELINE SUMMARY")
        print(f"{'='*60}")
        print(f"Messages processed: {stats['messages_processed']}")
        print(f"Total time: {stats['total_processing_time_seconds']:.1f}s")
        print(f"Processing rate: {stats['processing_rate_messages_per_second']:.1f} msg/s")
        print(f"Index type: {stats['index_config']['index_type']}")
        print(f"Model: {stats['index_config']['model_name']}")
        print(f"Report saved: {report_path}")
        if not args.no_save:
            print(f"Index saved: {stats.get('index_saved_to', 'N/A')}")
        
    except Exception as e:
        log.error(f"Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()
