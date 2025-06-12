#!/usr/bin/env python3
"""
Unified Discord Message Processing Pipeline

Single entry point for all Discord message processing and indexing tasks.
This replaces multiple scattered pipeline scripts with one clean, unified system.

Features:
- Canonical index building and updating
- Community expert search index
- Resource discovery index  
- Content preprocessing
- Incremental updates
- Comprehensive reporting

Usage:
    python core/pipeline.py --help                    # Show all options
    python core/pipeline.py --build-all              # Build all indices
    python core/pipeline.py --update                 # Incremental update
    python core/pipeline.py --build-canonical        # Just canonical index
    python core/pipeline.py --build-community        # Just community index
    python core/pipeline.py --build-resources        # Just resource index
    python core/pipeline.py --limit 1000             # Test with limited data
"""

import os
import sys
import json
import argparse
import logging
import os
import sys
import json
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.logger import setup_logging

# Configure logging
setup_logging()
log = logging.getLogger(__name__)

class UnifiedPipeline:
    """
    Unified pipeline for all Discord message processing tasks.
    
    Replaces the scattered scripts with a single, clean interface.
    """
    
    def __init__(self):
        self.stats = {}
        self.start_time = datetime.now()
        
    def ensure_dependencies(self):
        """Ensure all required modules are available."""
        try:
            # Import all required builders
            from scripts.build_canonical_index import CanonicalIndexBuilder
            from scripts.build_community_faiss_index import CommunityFAISSIndexBuilder  
            from scripts.build_resource_faiss_index import ResourceFAISSIndexBuilder
            from scripts.content_preprocessor import ContentPreprocessor
            
            log.info("✅ All dependencies verified")
            return True
            
        except ImportError as e:
            log.error(f"❌ Missing dependency: {e}")
            log.error("Install required packages: pip install faiss-cpu sentence-transformers")
            return False
    
    def check_prerequisites(self) -> bool:
        """Check if system is ready for pipeline execution."""
        log.info("🔍 Checking prerequisites...")
        
        # Check database exists
        db_path = "data/discord_messages.db"
        if not os.path.exists(db_path):
            log.error(f"❌ Database not found: {db_path}")
            log.error("Run 'python core/fetch_messages.py' first to collect Discord data")
            return False
            
        # Check message count
        try:
            from db.db import SessionLocal, Message
            session = SessionLocal()
            count = session.query(Message).count()
            session.close()
            
            if count == 0:
                log.error("❌ No messages in database")
                log.error("Run 'python core/fetch_messages.py' to fetch messages first")
                return False
            
            log.info(f"✅ Database ready: {count:,} messages available")
            
        except Exception as e:
            log.error(f"❌ Database connection failed: {e}")
            return False
            
        # Ensure output directories exist
        os.makedirs("data/indices", exist_ok=True)
        os.makedirs("data/reports", exist_ok=True)
        
        log.info("✅ All prerequisites satisfied")
        return True
    
    def build_canonical_index(self, force_rebuild: bool = False) -> Dict[str, Any]:
        """Build or update the canonical Discord messages index."""
        log.info("🔨 Building canonical index...")
        start_time = datetime.now()
        
        try:
            from scripts.build_canonical_index import CanonicalIndexBuilder
            
            builder = CanonicalIndexBuilder()
            builder.build_canonical_index(force_rebuild=force_rebuild)
            
            duration = (datetime.now() - start_time).total_seconds()
            
            return {
                "status": "success",
                "duration_seconds": duration,
                "index_path": "data/indices/discord_messages_index",
                "type": "canonical"
            }
            
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            log.error(f"❌ Canonical index build failed: {e}")
            
            return {
                "status": "failed", 
                "duration_seconds": duration,
                "error": str(e),
                "type": "canonical"
            }
    
    def build_community_index(self, force_rebuild: bool = False) -> Dict[str, Any]:
        """Build the community expert search index."""
        log.info("👥 Building community index...")
        start_time = datetime.now()
        
        try:
            from scripts.build_community_faiss_index import CommunityFAISSIndexBuilder
            
            builder = CommunityFAISSIndexBuilder()
            # Pass force_rebuild flag to the builder
            index_path, metadata_path, stats = builder.build_complete_index(
                save_filename="community_faiss_index", 
                force_rebuild=force_rebuild
            )
            
            duration = (datetime.now() - start_time).total_seconds()
            
            # Check if this was a graceful "up to date" response
            if stats.get("status") == "up_to_date":
                log.info(f"✅ Community index is up to date ({stats.get('existing_index_entries', 0)} entries)")
                return {
                    "status": "up_to_date",
                    "duration_seconds": duration,
                    "index_path": index_path,
                    "metadata_path": metadata_path,
                    "statistics": stats,
                    "type": "community",
                    "message": stats.get("message", "Index is current")
                }
            
            return {
                "status": "success",
                "duration_seconds": duration,
                "index_path": index_path,
                "metadata_path": metadata_path,
                "statistics": stats,
                "type": "community"
            }
            
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            log.error(f"❌ Community index build failed: {e}")
            
            return {
                "status": "failed",
                "duration_seconds": duration, 
                "error": str(e),
                "type": "community"
            }
    
    def build_resource_index(self, force_rebuild: bool = False) -> Dict[str, Any]:
        """Build the resource discovery index."""
        log.info("📚 Building resource index...")
        start_time = datetime.now()
        
        try:
            from scripts.build_resource_faiss_index import ResourceFAISSIndexBuilder
            
            builder = ResourceFAISSIndexBuilder()
            # Pass force_rebuild flag to the builder  
            index_path, metadata_path, stats = builder.build_complete_index(
                save_filename="resource_faiss_index",
                force_rebuild=force_rebuild
            )
            
            duration = (datetime.now() - start_time).total_seconds()
            
            # Check if this was a graceful "up to date" response
            if stats.get("status") == "up_to_date":
                log.info(f"✅ Resource index is up to date ({stats.get('existing_index_entries', 0)} entries)")
                return {
                    "status": "up_to_date",
                    "duration_seconds": duration,
                    "index_path": index_path,
                    "metadata_path": metadata_path,
                    "statistics": stats,
                    "type": "resource",
                    "message": stats.get("message", "Index is current")
                }
            
            return {
                "status": "success",
                "duration_seconds": duration,
                "index_path": index_path,
                "metadata_path": metadata_path,
                "statistics": stats,
                "type": "resource"
            }
            
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            log.error(f"❌ Resource index build failed: {e}")
            
            return {
                "status": "failed",
                "duration_seconds": duration,
                "error": str(e),
                "type": "resource"
            }
    
    def run_full_pipeline(self, 
                         build_canonical: bool = True,
                         build_community: bool = True, 
                         build_resources: bool = True,
                         force_rebuild: bool = False) -> Dict[str, Any]:
        """
        Run the complete pipeline with all index building.
        
        Args:
            build_canonical: Build main message search index
            build_community: Build expert/skill search index  
            build_resources: Build resource discovery index
            force_rebuild: Force complete rebuild vs incremental
        """
        
        log.info("🚀 Starting unified Discord pipeline...")
        pipeline_start = datetime.now()
        
        # Check prerequisites
        if not self.check_prerequisites():
            return {"status": "failed", "error": "Prerequisites not met"}
        
        results = {
            "pipeline_started": pipeline_start.isoformat(),
            "indices_built": [],
            "errors": [],
            "total_duration_seconds": 0
        }
        
        # Build canonical index (main search)
        if build_canonical:
            result = self.build_canonical_index(force_rebuild=force_rebuild)
            results["indices_built"].append(result)
            if result["status"] == "failed":
                results["errors"].append(f"Canonical index: {result['error']}")
        
        # Build community index (expert search)
        if build_community:
            result = self.build_community_index(force_rebuild=force_rebuild)
            results["indices_built"].append(result)
            if result["status"] == "failed":
                results["errors"].append(f"Community index: {result['error']}")
        
        # Build resource index (resource discovery)
        if build_resources:
            result = self.build_resource_index(force_rebuild=force_rebuild)
            results["indices_built"].append(result)
            if result["status"] == "failed":
                results["errors"].append(f"Resource index: {result['error']}")
        
        # Calculate totals
        total_duration = (datetime.now() - pipeline_start).total_seconds()
        results["total_duration_seconds"] = total_duration
        results["pipeline_completed"] = datetime.now().isoformat()
        
        # Determine overall status
        successful_builds = [r for r in results["indices_built"] if r["status"] in ["success", "up_to_date"]]
        failed_builds = [r for r in results["indices_built"] if r["status"] == "failed"]
        
        if len(failed_builds) == 0:
            results["status"] = "success"
            log.info(f"✅ Pipeline completed successfully in {total_duration:.1f}s")
        elif len(successful_builds) > 0:
            results["status"] = "partial_success" 
            log.warning(f"⚠️ Pipeline completed with {len(failed_builds)} failures in {total_duration:.1f}s")
        else:
            results["status"] = "failed"
            log.error(f"❌ Pipeline failed completely in {total_duration:.1f}s")
        
        # Save pipeline report
        self.save_pipeline_report(results)
        
        return results
    
    def update_indices(self) -> Dict[str, Any]:
        """
        Perform incremental updates to all indices.
        Faster than full rebuild for regular maintenance.
        """
        log.info("🔄 Performing incremental index updates...")
        
        return self.run_full_pipeline(
            build_canonical=True,  # Canonical supports incremental updates
            build_community=True,  # Community needs rebuild (no incremental yet)
            build_resources=True,  # Resources need rebuild (no incremental yet)
            force_rebuild=False    # Use incremental where possible
        )
    
    def save_pipeline_report(self, results: Dict[str, Any]) -> str:
        """Save comprehensive pipeline execution report."""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f"data/reports/unified_pipeline_report_{timestamp}.json"
        
        # Add system info to report
        enhanced_results = {
            **results,
            "system_info": {
                "python_version": sys.version,
                "working_directory": os.getcwd(),
                "pipeline_script": __file__
            },
            "summary": {
                "total_indices": len(results.get("indices_built", [])),
                "successful_indices": len([r for r in results.get("indices_built", []) if r["status"] == "success"]),
                "failed_indices": len([r for r in results.get("indices_built", []) if r["status"] == "failed"]),
                "total_errors": len(results.get("errors", []))
            }
        }
        
        with open(report_path, 'w') as f:
            json.dump(enhanced_results, f, indent=2, default=str)
        
        log.info(f"📄 Pipeline report saved: {report_path}")
        return report_path
    
    def print_summary(self, results: Dict[str, Any]):
        """Print a human-readable summary of pipeline execution."""
        
        print(f"\n{'='*60}")
        print("🎯 UNIFIED PIPELINE SUMMARY")
        print(f"{'='*60}")
        
        print(f"Status: {results['status'].upper()}")
        print(f"Duration: {results['total_duration_seconds']:.1f}s")
        print(f"Indices Built: {len(results.get('indices_built', []))}")
        
        if results.get("errors"):
            print(f"Errors: {len(results['errors'])}")
            for error in results["errors"][:3]:  # Show first 3 errors
                print(f"  ❌ {error}")
        
        print(f"\n📊 Index Build Results:")
        for result in results.get("indices_built", []):
            if result["status"] == "success":
                status_emoji = "✅"
                status_text = f"{result['duration_seconds']:.1f}s"
            elif result["status"] == "up_to_date":
                status_emoji = "🔄"
                entries = result.get("statistics", {}).get("existing_index_entries", 0)
                status_text = f"up to date ({entries} entries)"
            else:
                status_emoji = "❌"
                status_text = f"failed ({result['duration_seconds']:.1f}s)"
            
            index_type = result["type"].title()
            print(f"  {status_emoji} {index_type}: {status_text}")
        
        print(f"\n🗂️ Index System Status:")
        print(f"  📁 Main Search: data/indices/discord_messages_index")
        print(f"  👥 Expert Search: data/indices/community_faiss_index") 
        print(f"  📚 Resource Search: data/indices/resource_faiss_index")
        
        if results["status"] in ["success", "up_to_date"]:
            print(f"\n🎉 All systems ready! Your Discord bot can now:")
            print(f"  • Search messages with enhanced accuracy")
            print(f"  • Find experts by skills and expertise")
            print(f"  • Discover relevant resources and links")
        elif any(r["status"] == "up_to_date" for r in results.get("indices_built", [])):
            print(f"\n✅ Indices are current! Your Discord bot is ready to use.")
            print(f"  💡 Tip: Run 'python core/fetch_messages.py' to get new messages,")
            print(f"       then 'python core/pipeline.py --update' to refresh indices.")
            print(f"\n🎉 All systems ready! Your Discord bot can now:")
            print(f"  • Search messages with enhanced accuracy")
            print(f"  • Find experts by skills and expertise")
            print(f"  • Discover relevant resources and links")
        
        print(f"{'='*60}\n")
    
    def preprocess_messages_with_resources(self, limit: Optional[int] = None) -> Dict[str, Any]:
        """
        Complete message preprocessing including resource detection and classification.
        
        This method:
        1. Preprocesses message content (cleaning, URL extraction, etc.)
        2. Detects and classifies valuable resources
        3. Filters out meeting recordings and admin content
        4. Enriches messages with resource metadata
        
        Args:
            limit: Maximum number of messages to process (for testing)
            
        Returns:
            Dict containing preprocessing results and statistics
        """
        log.info("📝 Starting comprehensive message preprocessing...")
        start_time = datetime.now()
        
        try:
            from scripts.content_preprocessor import ContentPreprocessor, PreprocessingConfig
            from core.resource_detector import detect_resources, simple_vet_resource
            
            # Initialize preprocessor with standard config
            config = PreprocessingConfig(
                min_content_length=10,
                include_embed_content=True,
                include_reply_context=True,
                normalize_urls=True,
                filter_bot_messages=True
            )
            preprocessor = ContentPreprocessor(config)
            
            # Stage 1: Basic preprocessing
            log.info("  Stage 1: Content preprocessing...")
            preprocessing_start = datetime.now()
            preprocessed_messages = preprocessor.preprocess_batch(limit=limit)
            preprocessing_time = (datetime.now() - preprocessing_start).total_seconds()
            
            if not preprocessed_messages:
                return {
                    "status": "failed",
                    "error": "No messages survived preprocessing filters",
                    "duration_seconds": (datetime.now() - start_time).total_seconds()
                }
            
            log.info(f"  ✅ Preprocessed {len(preprocessed_messages)} messages in {preprocessing_time:.1f}s")
            
            # Stage 2: Resource detection and classification
            log.info("  Stage 2: Resource detection and classification...")
            resource_start = datetime.now()
            
            enriched_messages = []
            total_resources = 0
            filtered_meetings = 0
            resource_types = {}
            
            for message in preprocessed_messages:
                # Create message object for resource detection
                class SimpleMessage:
                    def __init__(self, content):
                        self.content = content
                        self.author = None  # No author to avoid bot filtering
                
                simple_msg = SimpleMessage(message.get('searchable_text', message.get('original_content', '')))
                
                # Detect resources in message content
                detected_resources = detect_resources(simple_msg)
                
                # Filter resources using enhanced vetting (excludes meeting recordings/summaries)
                valuable_resources = []
                for resource in detected_resources:
                    vetting_result = simple_vet_resource(resource)
                    if vetting_result.get('is_valuable', False):
                        resource.update(vetting_result)
                        valuable_resources.append(resource)
                        
                        # Track resource types
                        resource_type = resource.get('type', 'unknown')
                        resource_types[resource_type] = resource_types.get(resource_type, 0) + 1
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
            total_time = (datetime.now() - start_time).total_seconds()
            
            log.info(f"  ✅ Resource detection completed in {resource_time:.1f}s")
            log.info(f"  📊 Found {total_resources} valuable resources, filtered {filtered_meetings} meeting items")
            
            # Return comprehensive results
            return {
                "status": "success",
                "duration_seconds": total_time,
                "preprocessing_time_seconds": preprocessing_time,
                "resource_detection_time_seconds": resource_time,
                "messages_processed": len(enriched_messages),
                "resources_detected": total_resources,
                "meetings_filtered": filtered_meetings,
                "resource_types": resource_types,
                "messages": enriched_messages,
                "processing_rate_msg_per_sec": len(enriched_messages) / total_time if total_time > 0 else 0
            }
            
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            log.error(f"❌ Message preprocessing failed: {e}")
            
            return {
                "status": "failed",
                "error": str(e),
                "duration_seconds": duration
            }
    
    def run_preprocessing_only(self, limit: Optional[int] = None) -> Dict[str, Any]:
        """
        Run only the preprocessing and resource detection phase.
        
        This is useful for:
        - Testing preprocessing logic
        - Analyzing resource detection results
        - Preparing data before index building
        
        Args:
            limit: Maximum number of messages to process
            
        Returns:
            Preprocessing results and statistics
        """
        log.info("🔍 Running preprocessing and resource detection only...")
        
        result = self.preprocess_messages_with_resources(limit=limit)
        
        if result["status"] == "success":
            log.info("✅ Preprocessing completed successfully")
            log.info(f"📊 Summary:")
            log.info(f"  • Messages processed: {result['messages_processed']}")
            log.info(f"  • Resources detected: {result['resources_detected']}")
            log.info(f"  • Meetings filtered: {result['meetings_filtered']}")
            log.info(f"  • Processing rate: {result['processing_rate_msg_per_sec']:.1f} msg/sec")
            log.info(f"  • Resource types found: {result['resource_types']}")
        else:
            log.error(f"❌ Preprocessing failed: {result.get('error', 'Unknown error')}")
        
        return result
def main():
    """Main entry point with comprehensive CLI interface."""
    
    parser = argparse.ArgumentParser(
        description="Unified Discord Message Processing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --build-all              # Build all indices (recommended)
  %(prog)s --update                 # Incremental updates only
  %(prog)s --build-canonical        # Just rebuild main search index
  %(prog)s --build-community        # Just rebuild expert search index
  %(prog)s --build-resources        # Just rebuild resource search index
  %(prog)s --preprocess-only        # Test preprocessing and resource detection only
  %(prog)s --build-all --force      # Force complete rebuild of all indices
        """
    )
    
    # Main action arguments
    action_group = parser.add_mutually_exclusive_group(required=True)
    action_group.add_argument("--build-all", action="store_true",
                             help="Build all indices (canonical, community, resources)")
    action_group.add_argument("--update", action="store_true", 
                             help="Perform incremental updates to all indices")
    action_group.add_argument("--build-canonical", action="store_true",
                             help="Build only the canonical message search index")
    action_group.add_argument("--build-community", action="store_true",
                             help="Build only the community expert search index")
    action_group.add_argument("--build-resources", action="store_true",
                             help="Build only the resource discovery index")
    action_group.add_argument("--preprocess-only", action="store_true",
                             help="Run only preprocessing and resource detection (no index building)")
    
    # Options
    parser.add_argument("--force", action="store_true",
                       help="Force complete rebuild instead of incremental update")
    parser.add_argument("--limit", type=int, 
                       help="Limit processing to N messages (for testing)")
    parser.add_argument("--quiet", action="store_true",
                       help="Suppress detailed output")
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    
    # Initialize pipeline
    pipeline = UnifiedPipeline()
    
    # Check dependencies first
    if not pipeline.ensure_dependencies():
        sys.exit(1)
    
    try:
        # Execute requested action
        if args.build_all:
            results = pipeline.run_full_pipeline(force_rebuild=args.force)
            
        elif args.update:
            results = pipeline.update_indices()
            
        elif args.build_canonical:
            results = {"indices_built": [pipeline.build_canonical_index(args.force)]}
            results["status"] = "success" if results["indices_built"][0]["status"] == "success" else "failed"
            results["total_duration_seconds"] = results["indices_built"][0]["duration_seconds"]
            
        elif args.build_community:
            results = {"indices_built": [pipeline.build_community_index()]}
            results["status"] = "success" if results["indices_built"][0]["status"] == "success" else "failed"
            results["total_duration_seconds"] = results["indices_built"][0]["duration_seconds"]
            
        elif args.build_resources:
            results = {"indices_built": [pipeline.build_resource_index()]}
            results["status"] = "success" if results["indices_built"][0]["status"] == "success" else "failed" 
            results["total_duration_seconds"] = results["indices_built"][0]["duration_seconds"]
            
        elif args.preprocess_only:
            result = pipeline.run_preprocessing_only(limit=args.limit)
            results = {
                "status": result["status"],
                "preprocessing_results": result,
                "total_duration_seconds": result["duration_seconds"]
            }
        
        # Print summary
        if not args.quiet:
            pipeline.print_summary(results)
        
        # Exit with appropriate code
        if results["status"] == "failed":
            sys.exit(1)
        elif results["status"] == "partial_success":
            sys.exit(2)
        else:
            sys.exit(0)
            
    except KeyboardInterrupt:
        log.info("🛑 Pipeline interrupted by user")
        sys.exit(130)
        
    except Exception as e:
        log.error(f"💥 Pipeline failed with unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
