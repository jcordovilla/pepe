#!/usr/bin/env python3
"""
Discord Bot Preprocessing Pipeline Runner

This script runs all preprocessing routines in the correct order to prepare
Discord messages for enhanced RAG agent capabilities. It orchestrates:

1. Content Preprocessing - Basic content cleaning and standardization
2. Community Preprocessing - Advanced community-focused analysis
3. Enhanced FAISS Index Building - Standard semantic search index
4. Community FAISS Index Building - Community-focused semantic search index

The script provides comprehensive logging, error handling, and progress tracking
for the complete preprocessing pipeline.
"""

import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Add project root to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

# Import preprocessing modules
from scripts.content_preprocessor import ContentPreprocessor
from scripts.enhanced_community_preprocessor import CommunityPreprocessor
from scripts.build_enhanced_faiss_index import EnhancedFAISSIndexBuilder
from scripts.build_community_faiss_index import CommunityFAISSIndexBuilder
from db import SessionLocal, Message
from utils.logger import setup_logging

# Configure logging
setup_logging()
logger = logging.getLogger(__name__)

class PreprocessingPipeline:
    """
    Complete preprocessing pipeline for Discord bot RAG capabilities.
    
    This class orchestrates all preprocessing steps in the correct order
    and provides comprehensive reporting and error handling.
    """
    
    def __init__(self, 
                 db_path: str = "data/discord_messages.db",
                 model_name: str = "all-MiniLM-L6-v2",
                 batch_size: int = 50):
        """
        Initialize the preprocessing pipeline.
        
        Args:
            db_path: Path to the SQLite database
            model_name: Sentence transformer model name
            batch_size: Batch size for processing
        """
        self.db_path = db_path
        self.model_name = model_name
        self.batch_size = batch_size
        
        # Pipeline statistics
        self.stats = {
            'pipeline_start_time': None,
            'pipeline_end_time': None,
            'total_duration_minutes': 0,
            'content_preprocessing': {},
            'community_preprocessing': {},
            'enhanced_index_build': {},
            'community_index_build': {},
            'errors': [],
            'warnings': []
        }
        
        logger.info(f"Preprocessing pipeline initialized")
        logger.info(f"Database: {self.db_path}")
        logger.info(f"Model: {self.model_name}")
        logger.info(f"Batch size: {self.batch_size}")
    
    def check_prerequisites(self) -> bool:
        """
        Check if all prerequisites are met before starting preprocessing.
        
        Returns:
            True if prerequisites are met, False otherwise
        """
        logger.info("Checking preprocessing prerequisites...")
        
        # Check database exists and has messages
        if not os.path.exists(self.db_path):
            logger.error(f"Database not found: {self.db_path}")
            return False
        
        # Check message count
        session = SessionLocal()
        try:
            message_count = session.query(Message).count()
            if message_count == 0:
                logger.error("No messages found in database")
                return False
            
            logger.info(f"✅ Found {message_count:,} messages in database")
            
            # Get date range
            oldest = session.query(Message.timestamp).order_by(Message.timestamp.asc()).first()
            newest = session.query(Message.timestamp).order_by(Message.timestamp.desc()).first()
            
            if oldest and newest:
                logger.info(f"✅ Message date range: {oldest[0]} to {newest[0]}")
            
        except Exception as e:
            logger.error(f"Error checking database: {e}")
            return False
        finally:
            session.close()
        
        # Check output directories exist
        os.makedirs("data/indices", exist_ok=True)
        os.makedirs("data/reports", exist_ok=True)
        
        logger.info("✅ All prerequisites met")
        return True
    
    def run_content_preprocessing(self) -> Dict[str, Any]:
        """
        Run basic content preprocessing.
        
        Returns:
            Dictionary with preprocessing results and statistics
        """
        logger.info("=" * 60)
        logger.info("STEP 1: CONTENT PREPROCESSING")
        logger.info("=" * 60)
        
        start_time = datetime.now()
        
        try:
            # Initialize preprocessor
            preprocessor = ContentPreprocessor()
            
            # Generate preprocessing report
            logger.info("Analyzing content preprocessing requirements...")
            report = preprocessor.generate_preprocessing_report(sample_size=1000)
            
            # Save report
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = f"data/reports/content_preprocessing_report_{timestamp}.json"
            
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            duration = (datetime.now() - start_time).total_seconds() / 60
            
            results = {
                'status': 'completed',
                'duration_minutes': round(duration, 2),
                'report_path': report_path,
                'statistics': {
                    'total_messages_analyzed': report.get('total_messages', 0),
                    'filter_rate': report.get('filter_rate', 0),
                    'avg_content_length': report.get('avg_content_length', 0),
                    'messages_with_embeds': report.get('messages_with_embeds', 0),
                    'messages_with_attachments': report.get('messages_with_attachments', 0),
                    'total_urls_extracted': report.get('total_urls_extracted', 0)
                }
            }
            
            logger.info(f"✅ Content preprocessing completed in {duration:.1f} minutes")
            logger.info(f"📄 Report saved to: {report_path}")
            logger.info(f"📊 Filter rate: {report.get('filter_rate', 0):.1%}")
            logger.info(f"📊 Avg content length: {report.get('avg_content_length', 0):.0f} chars")
            
            return results
            
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds() / 60
            error_msg = f"Content preprocessing failed: {e}"
            logger.error(error_msg)
            
            return {
                'status': 'failed',
                'duration_minutes': round(duration, 2),
                'error': str(e),
                'statistics': {}
            }
    
    def run_community_preprocessing(self) -> Dict[str, Any]:
        """
        Run advanced community-focused preprocessing.
        
        Returns:
            Dictionary with preprocessing results and statistics
        """
        logger.info("=" * 60)
        logger.info("STEP 2: COMMUNITY PREPROCESSING")
        logger.info("=" * 60)
        
        start_time = datetime.now()
        
        try:
            # Initialize community preprocessor
            preprocessor = CommunityPreprocessor()
            
            # Test with sample messages to validate
            session = SessionLocal()
            try:
                sample_messages = session.query(Message).limit(10).all()
                processed_count = 0
                
                for msg in sample_messages:
                    try:
                        metadata = preprocessor.process_message(msg)
                        processed_count += 1
                    except Exception as e:
                        logger.warning(f"Failed to process message {msg.message_id}: {e}")
                
                logger.info(f"✅ Community preprocessor validated with {processed_count}/10 sample messages")
                
            finally:
                session.close()
            
            duration = (datetime.now() - start_time).total_seconds() / 60
            
            results = {
                'status': 'completed',
                'duration_minutes': round(duration, 2),
                'statistics': {
                    'sample_messages_processed': processed_count,
                    'validation_success_rate': processed_count / 10 if processed_count > 0 else 0,
                    'features_available': [
                        'expert_identification',
                        'skill_mining', 
                        'qa_pattern_detection',
                        'conversation_threading',
                        'engagement_analysis',
                        'temporal_event_extraction',
                        'resource_classification'
                    ]
                }
            }
            
            logger.info(f"✅ Community preprocessing validated in {duration:.1f} minutes")
            logger.info(f"📊 Sample validation rate: {processed_count}/10 messages")
            
            return results
            
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds() / 60
            error_msg = f"Community preprocessing failed: {e}"
            logger.error(error_msg)
            
            return {
                'status': 'failed',
                'duration_minutes': round(duration, 2),
                'error': str(e),
                'statistics': {}
            }
    
    def build_enhanced_faiss_index(self, limit: Optional[int] = None) -> Dict[str, Any]:
        """
        Build enhanced FAISS index with rich metadata.
        
        Args:
            limit: Optional limit on number of messages to process
            
        Returns:
            Dictionary with build results and statistics
        """
        logger.info("=" * 60)
        logger.info("STEP 3: ENHANCED FAISS INDEX BUILD")
        logger.info("=" * 60)
        
        start_time = datetime.now()
        
        try:
            # Initialize builder
            builder = EnhancedFAISSIndexBuilder(
                model_name=self.model_name,
                db_path=self.db_path,
                batch_size=self.batch_size
            )
            
            # Build complete index
            index_path, metadata_path, stats = builder.build_complete_index(limit=limit)
            
            duration = (datetime.now() - start_time).total_seconds() / 60
            
            results = {
                'status': 'completed',
                'duration_minutes': round(duration, 2),
                'index_path': index_path,
                'metadata_path': metadata_path,
                'statistics': stats
            }
            
            logger.info(f"✅ Enhanced FAISS index built in {duration:.1f} minutes")
            logger.info(f"📁 Index: {index_path}")
            logger.info(f"📁 Metadata: {metadata_path}")
            logger.info(f"📊 Messages processed: {stats.get('total_messages_processed', 0):,}")
            
            return results
            
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds() / 60
            error_msg = f"Enhanced FAISS index build failed: {e}"
            logger.error(error_msg)
            
            return {
                'status': 'failed',
                'duration_minutes': round(duration, 2),
                'error': str(e),
                'statistics': {}
            }
    
    def build_community_faiss_index(self, limit: Optional[int] = None) -> Dict[str, Any]:
        """
        Build community-focused FAISS index with expert identification.
        
        Args:
            limit: Optional limit on number of messages to process
            
        Returns:
            Dictionary with build results and statistics
        """
        logger.info("=" * 60)
        logger.info("STEP 4: COMMUNITY FAISS INDEX BUILD")
        logger.info("=" * 60)
        
        start_time = datetime.now()
        
        try:
            # Initialize builder
            builder = CommunityFAISSIndexBuilder(
                model_name=self.model_name,
                db_path=self.db_path,
                batch_size=self.batch_size
            )
            
            # Build complete index
            index_path, metadata_path, stats = builder.build_complete_index(limit=limit)
            
            duration = (datetime.now() - start_time).total_seconds() / 60
            
            results = {
                'status': 'completed',
                'duration_minutes': round(duration, 2),
                'index_path': index_path,
                'metadata_path': metadata_path,
                'statistics': stats
            }
            
            logger.info(f"✅ Community FAISS index built in {duration:.1f} minutes")
            logger.info(f"📁 Index: {index_path}")
            logger.info(f"📁 Metadata: {metadata_path}")
            logger.info(f"📊 Messages processed: {stats.get('total_messages_processed', 0):,}")
            logger.info(f"📊 Community features extracted: {stats.get('community_features_extracted', 0):,}")
            
            return results
            
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds() / 60
            error_msg = f"Community FAISS index build failed: {e}"
            logger.error(error_msg)
            
            return {
                'status': 'failed',
                'duration_minutes': round(duration, 2),
                'error': str(e),
                'statistics': {}
            }
    
    def run_complete_pipeline(self, 
                             limit: Optional[int] = None,
                             skip_content_preprocessing: bool = False,
                             skip_community_preprocessing: bool = False,
                             skip_enhanced_index: bool = False,
                             skip_community_index: bool = False) -> Dict[str, Any]:
        """
        Run the complete preprocessing pipeline.
        
        Args:
            limit: Optional limit on number of messages to process for indexing
            skip_content_preprocessing: Skip content preprocessing step
            skip_community_preprocessing: Skip community preprocessing step  
            skip_enhanced_index: Skip enhanced FAISS index build
            skip_community_index: Skip community FAISS index build
            
        Returns:
            Dictionary with complete pipeline results
        """
        logger.info("🚀 STARTING COMPLETE PREPROCESSING PIPELINE")
        logger.info("=" * 80)
        
        self.stats['pipeline_start_time'] = datetime.now()
        
        # Check prerequisites
        if not self.check_prerequisites():
            logger.error("❌ Prerequisites not met. Aborting pipeline.")
            return {
                'status': 'failed',
                'error': 'Prerequisites not met',
                'statistics': self.stats
            }
        
        # Step 1: Content Preprocessing
        if not skip_content_preprocessing:
            self.stats['content_preprocessing'] = self.run_content_preprocessing()
        else:
            logger.info("⏭️  Skipping content preprocessing")
            self.stats['content_preprocessing'] = {'status': 'skipped'}
        
        # Step 2: Community Preprocessing
        if not skip_community_preprocessing:
            self.stats['community_preprocessing'] = self.run_community_preprocessing()
        else:
            logger.info("⏭️  Skipping community preprocessing")
            self.stats['community_preprocessing'] = {'status': 'skipped'}
        
        # Step 3: Enhanced FAISS Index
        if not skip_enhanced_index:
            self.stats['enhanced_index_build'] = self.build_enhanced_faiss_index(limit=limit)
        else:
            logger.info("⏭️  Skipping enhanced FAISS index build")
            self.stats['enhanced_index_build'] = {'status': 'skipped'}
        
        # Step 4: Community FAISS Index  
        if not skip_community_index:
            self.stats['community_index_build'] = self.build_community_faiss_index(limit=limit)
        else:
            logger.info("⏭️  Skipping community FAISS index build")
            self.stats['community_index_build'] = {'status': 'skipped'}
        
        # Finalize pipeline
        self.stats['pipeline_end_time'] = datetime.now()
        duration = (self.stats['pipeline_end_time'] - self.stats['pipeline_start_time']).total_seconds() / 60
        self.stats['total_duration_minutes'] = round(duration, 2)
        
        # Generate final report
        self._generate_pipeline_report()
        
        # Determine overall status
        statuses = [
            self.stats['content_preprocessing'].get('status', 'skipped'),
            self.stats['community_preprocessing'].get('status', 'skipped'),
            self.stats['enhanced_index_build'].get('status', 'skipped'),
            self.stats['community_index_build'].get('status', 'skipped')
        ]
        
        if any(status == 'failed' for status in statuses):
            overall_status = 'partial_failure'
        elif all(status in ['completed', 'skipped'] for status in statuses):
            overall_status = 'completed'
        else:
            overall_status = 'unknown'
        
        logger.info("=" * 80)
        logger.info(f"🏁 PREPROCESSING PIPELINE {overall_status.upper()}")
        logger.info(f"⏱️  Total duration: {duration:.1f} minutes")
        logger.info("=" * 80)
        
        return {
            'status': overall_status,
            'total_duration_minutes': duration,
            'statistics': self.stats
        }
    
    def _generate_pipeline_report(self) -> str:
        """
        Generate a comprehensive pipeline report.
        
        Returns:
            Path to the generated report file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f"data/reports/preprocessing_pipeline_report_{timestamp}.json"
        
        # Add summary statistics
        summary = {
            'pipeline_summary': {
                'start_time': self.stats['pipeline_start_time'].isoformat(),
                'end_time': self.stats['pipeline_end_time'].isoformat(),
                'total_duration_minutes': self.stats['total_duration_minutes'],
                'steps_completed': sum(1 for step in [
                    self.stats['content_preprocessing'],
                    self.stats['community_preprocessing'], 
                    self.stats['enhanced_index_build'],
                    self.stats['community_index_build']
                ] if step.get('status') == 'completed'),
                'steps_failed': sum(1 for step in [
                    self.stats['content_preprocessing'],
                    self.stats['community_preprocessing'],
                    self.stats['enhanced_index_build'], 
                    self.stats['community_index_build']
                ] if step.get('status') == 'failed'),
                'overall_success': all(step.get('status') in ['completed', 'skipped'] for step in [
                    self.stats['content_preprocessing'],
                    self.stats['community_preprocessing'],
                    self.stats['enhanced_index_build'],
                    self.stats['community_index_build']
                ])
            },
            'detailed_statistics': self.stats
        }
        
        # Convert datetime objects to strings for JSON serialization
        def serialize_datetime(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            return obj
        
        with open(report_path, 'w') as f:
            json.dump(summary, f, indent=2, default=serialize_datetime)
        
        logger.info(f"📄 Pipeline report saved to: {report_path}")
        return report_path


def main():
    """
    Main entry point for the preprocessing pipeline.
    
    This function can be called from command line or imported and used programmatically.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Discord Bot Preprocessing Pipeline")
    parser.add_argument("--limit", type=int, help="Limit number of messages to process")
    parser.add_argument("--skip-content", action="store_true", help="Skip content preprocessing")
    parser.add_argument("--skip-community", action="store_true", help="Skip community preprocessing")
    parser.add_argument("--skip-enhanced", action="store_true", help="Skip enhanced FAISS index")
    parser.add_argument("--skip-community-index", action="store_true", help="Skip community FAISS index")
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="Sentence transformer model")
    parser.add_argument("--batch-size", type=int, default=50, help="Processing batch size")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = PreprocessingPipeline(
        model_name=args.model,
        batch_size=args.batch_size
    )
    
    # Run complete pipeline
    results = pipeline.run_complete_pipeline(
        limit=args.limit,
        skip_content_preprocessing=args.skip_content,
        skip_community_preprocessing=args.skip_community,
        skip_enhanced_index=args.skip_enhanced,
        skip_community_index=args.skip_community_index
    )
    
    # Exit with appropriate code
    if results['status'] == 'completed':
        logger.info("✅ Preprocessing pipeline completed successfully!")
        sys.exit(0)
    elif results['status'] == 'partial_failure':
        logger.warning("⚠️ Preprocessing pipeline completed with some failures")
        sys.exit(1)
    else:
        logger.error("❌ Preprocessing pipeline failed")
        sys.exit(2)


if __name__ == "__main__":
    main()
