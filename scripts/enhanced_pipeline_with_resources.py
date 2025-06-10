#!/usr/bin/env python3
"""
Enhanced Discord Message Pipeline WITH Resource Detection

This enhanced version adds resource detection to the existing pipeline,
specifically filtering out internal meeting summaries and recordings.
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
from core.resource_detector import detect_resources, simple_vet_resource
from utils.logger import setup_logging

setup_logging()
import logging
log = logging.getLogger(__name__)

class EnhancedDiscordMessagePipeline:
    """Enhanced pipeline that includes resource detection with meeting filtering"""
    
    def __init__(self, 
                 preprocessing_config: PreprocessingConfig = None,
                 index_config: IndexConfig = None,
                 enable_resource_detection: bool = True):
        self.preprocessing_config = preprocessing_config or PreprocessingConfig()
        self.index_config = index_config or IndexConfig()
        self.enable_resource_detection = enable_resource_detection
        
        # Initialize components
        self.preprocessor = ContentPreprocessor(self.preprocessing_config)
        self.faiss_index = EnhancedFAISSIndex(self.index_config)
        
        self.pipeline_stats = {}
    
    def run_full_pipeline_with_resources(self, 
                                       limit: int = None, 
                                       save_index: bool = True,
                                       index_name: str = None) -> Dict[str, Any]:
        """Enhanced pipeline that includes resource detection"""
        
        pipeline_start = datetime.now()
        log.info("Starting enhanced Discord message processing pipeline with resource detection")
        
        # Stage 1: Preprocessing
        log.info("Stage 1: Preprocessing messages...")
        preprocessing_start = datetime.now()
        
        preprocessed_messages = self.preprocessor.preprocess_batch(limit=limit)
        
        preprocessing_time = (datetime.now() - preprocessing_start).total_seconds()
        
        if not preprocessed_messages:
            raise ValueError("No messages survived preprocessing filters")
        
        log.info(f"Preprocessing completed: {len(preprocessed_messages)} messages in {preprocessing_time:.1f}s")
        
        # Stage 2: Resource Detection (NEW)
        resource_time = 0
        if self.enable_resource_detection:
            log.info("Stage 2: Running resource detection (filtering out internal meetings)...")
            resource_start = datetime.now()
            
            enriched_messages = []
            total_resources = 0
            filtered_meetings = 0
            
            for message in preprocessed_messages:
                content = message.get('content', '')
                
                # Create a simple message object for resource detection
                class SimpleMessage:
                    def __init__(self, content):
                        self.content = content
                        self.author = None  # No author to avoid bot filtering
                
                simple_msg = SimpleMessage(content)
                
                # Detect resources in message content using the existing function
                detected_resources = detect_resources(simple_msg)
                
                # Filter resources using enhanced vetting (this will exclude meeting recordings/summaries)
                valuable_resources = []
                for resource in detected_resources:
                    # Apply enhanced vetting that includes meeting content filtering
                    vetting_result = simple_vet_resource(resource)
                    if vetting_result.get('is_valuable', False):
                        # Update resource with vetting results
                        resource.update(vetting_result)
                        valuable_resources.append(resource)
                    else:
                        # Count filtered meeting content
                        if any(keyword in resource.get('context_snippet', '').lower() 
                               for keyword in ['meeting', 'recap', 'admin', 'zoom', 'fathom']):
                            filtered_meetings += 1
                
                # Add resource metadata to message
                message['detected_resources'] = valuable_resources
                message['has_resources'] = len(valuable_resources) > 0
                message['resource_count'] = len(valuable_resources)
                message['resource_types'] = list(set(r.get('type', 'unknown') for r in valuable_resources))
                
                total_resources += len(valuable_resources)
                enriched_messages.append(message)
            
            resource_time = (datetime.now() - resource_start).total_seconds()
            log.info(f"Resource detection completed in {resource_time:.1f}s")
            log.info(f"Found {total_resources} valuable resources, filtered out {filtered_meetings} meeting-related items")
            
            preprocessed_messages = enriched_messages
        
        # Stage 3: Index building (now with resource metadata)
        log.info("Stage 3: Building FAISS index...")
        indexing_start = datetime.now()
        
        self.faiss_index.build_index(preprocessed_messages)
        
        indexing_time = (datetime.now() - indexing_start).total_seconds()
        log.info(f"Index building completed in {indexing_time:.1f}s")
        
        # Stage 4: Save index if requested
        if save_index:
            log.info("Stage 4: Saving index...")
            save_start = datetime.now()
            
            if index_name is None:
                index_name = f"enhanced_discord_index_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            self.faiss_index.save_index(index_name)
            save_time = (datetime.now() - save_start).total_seconds()
            log.info(f"Index saved to '{index_name}' in {save_time:.1f}s")
        
        # Generate enhanced pipeline statistics
        total_time = (datetime.now() - pipeline_start).total_seconds()
        
        self.pipeline_stats = {
            'pipeline_completed': datetime.now().isoformat(),
            'total_processing_time_seconds': total_time,
            'preprocessing_time_seconds': preprocessing_time,
            'resource_detection_time_seconds': resource_time,
            'indexing_time_seconds': indexing_time,
            'messages_processed': len(preprocessed_messages),
            'processing_rate_messages_per_second': len(preprocessed_messages) / total_time,
            'index_stats': self.faiss_index.get_stats(),
            'resource_detection_enabled': self.enable_resource_detection,
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
            self.pipeline_stats['index_saved_to'] = index_name
        
        log.info(f"Enhanced pipeline completed successfully in {total_time:.1f}s")
        log.info(f"Processing rate: {self.pipeline_stats['processing_rate_messages_per_second']:.1f} messages/second")
        
        return self.pipeline_stats
    
    def analyze_resource_distribution(self) -> Dict[str, Any]:
        """Analyze the distribution of detected resources"""
        
        if not self.faiss_index.metadata:
            return {"error": "No metadata available"}
        
        analysis = {
            'total_messages': len(self.faiss_index.metadata),
            'messages_with_resources': 0,
            'total_resources': 0,
            'resource_types': {},
            'resource_domains': {},
            'filtered_meeting_content': 0
        }
        
        for metadata in self.faiss_index.metadata.values():
            if metadata.get('has_resources', False):
                analysis['messages_with_resources'] += 1
                
                resources = metadata.get('detected_resources', [])
                analysis['total_resources'] += len(resources)
                
                # Analyze resource types
                for resource in resources:
                    resource_type = resource.get('type', 'unknown')
                    analysis['resource_types'][resource_type] = analysis['resource_types'].get(resource_type, 0) + 1
                    
                    # Analyze domains
                    url = resource.get('url', '')
                    if url:
                        try:
                            from urllib.parse import urlparse
                            domain = urlparse(url).netloc.replace('www.', '')
                            analysis['resource_domains'][domain] = analysis['resource_domains'].get(domain, 0) + 1
                        except:
                            pass
        
        # Sort by frequency
        analysis['resource_types'] = dict(sorted(
            analysis['resource_types'].items(), 
            key=lambda x: x[1], 
            reverse=True
        ))
        
        analysis['resource_domains'] = dict(sorted(
            analysis['resource_domains'].items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:10])  # Top 10 domains
        
        return analysis

def main():
    """Main function with enhanced command-line interface"""
    parser = argparse.ArgumentParser(description="Enhanced Discord Message Processing Pipeline with Resource Detection")
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
    parser.add_argument("--disable-resource-detection", action="store_true", 
                       help="Disable resource detection (use basic pipeline)")
    
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
    
    # Initialize enhanced pipeline
    pipeline = EnhancedDiscordMessagePipeline(
        preprocessing_config, 
        index_config,
        enable_resource_detection=not args.disable_resource_detection
    )
    
    try:
        # Run enhanced pipeline
        stats = pipeline.run_full_pipeline_with_resources(
            limit=args.limit,
            save_index=not args.no_save,
            index_name=args.index_name
        )
        
        # Test search if requested
        if args.test_search:
            # Reuse test functionality from base pipeline
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
            for query in test_queries:
                results = pipeline.faiss_index.search(query, k=3)
                log.info(f"Query: '{query}' - {len(results)} results")
        
        # Generate resource analysis
        if pipeline.enable_resource_detection:
            resource_analysis = pipeline.analyze_resource_distribution()
            log.info(f"Resource analysis: {resource_analysis['total_resources']} resources in {resource_analysis['messages_with_resources']} messages")
        
        # Print summary
        print(f"\n{'='*60}")
        print("ENHANCED PIPELINE SUMMARY")
        print(f"{'='*60}")
        print(f"Messages processed: {stats['messages_processed']}")
        print(f"Total time: {stats['total_processing_time_seconds']:.1f}s")
        print(f"Processing rate: {stats['processing_rate_messages_per_second']:.1f} msg/s")
        print(f"Resource detection: {'Enabled' if stats['resource_detection_enabled'] else 'Disabled'}")
        if stats['resource_detection_enabled']:
            print(f"Resource detection time: {stats['resource_detection_time_seconds']:.1f}s")
        print(f"Index type: {stats['index_config']['index_type']}")
        print(f"Model: {stats['index_config']['model_name']}")
        if not args.no_save:
            print(f"Index saved: {stats.get('index_saved_to', 'N/A')}")
        
    except Exception as e:
        log.error(f"Enhanced pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()
