#!/usr/bin/env python3
"""
Unified Discord Message Processing Pipeline

Complete end-to-end pipeline for Discord message processing and indexing.
This replaces multiple scattered pipeline scripts with one comprehensive system.

Features:
- Discord message fetching (auto-fetch if database empty)
- Resource detection and extraction from messages
- Content preprocessing and cleaning
- Canonical index building (main message search)
- Community expert search index (skill/expertise matching)
- Resource discovery index (links, tools, tutorials)
- Incremental updates and intelligent rebuilds
- Comprehensive reporting and monitoring

Full End-to-End Workflow:
1. Fetch Discord messages (if needed)
2. Detect and extract resources from messages
3. Sync resources to JSON for indexing
4. Build canonical message search index
5. Build community expert search index
6. Build resource discovery index

Usage:
    python core/pipeline.py --help                    # Show all options
    python core/pipeline.py --build-all              # Complete end-to-end pipeline
    python core/pipeline.py --build-all --auto-fetch # Auto-fetch if no messages
    python core/pipeline.py --update                 # Incremental update
    python core/pipeline.py --build-canonical        # Just canonical index
    python core/pipeline.py --build-community        # Just community index
    python core/pipeline.py --build-resources        # Resource detection + index
    python core/pipeline.py --build-all --force      # Force complete rebuild
    python core/pipeline.py --limit 1000             # Test with limited data
"""

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
    
    def check_prerequisites(self, auto_fetch: bool = False) -> bool:
        """Check if system is ready for pipeline execution."""
        log.info("🔍 Checking prerequisites...")
        
        # Check database exists and has messages
        db_path = "data/discord_messages.db"
        needs_fetch = False
        
        if not os.path.exists(db_path):
            log.warning(f"⚠️ Database not found: {db_path}")
            needs_fetch = True
        else:
            # Check message count
            try:
                from db.db import SessionLocal, Message
                session = SessionLocal()
                count = session.query(Message).count()
                session.close()
                
                if count == 0:
                    log.warning("⚠️ No messages in database")
                    needs_fetch = True
                else:
                    log.info(f"✅ Database ready: {count:,} messages available")
                    
            except Exception as e:
                log.error(f"❌ Database connection failed: {e}")
                return False
        
        # Auto-fetch if requested and needed
        if needs_fetch:
            if auto_fetch:
                log.info("🚀 Auto-fetching Discord messages...")
                fetch_result = self.fetch_discord_messages()
                if fetch_result["status"] != "success":
                    log.error("❌ Auto-fetch failed")
                    return False
                log.info(f"✅ Fetched {fetch_result.get('messages_fetched', 0):,} messages")
            else:
                log.error("❌ No messages available")
                log.error("Use --auto-fetch flag or run 'python core/fetch_messages.py' first")
                return False
            
        # Ensure output directories exist
        os.makedirs("data/indices", exist_ok=True)
        os.makedirs("data/reports", exist_ok=True)
        os.makedirs("data/resources", exist_ok=True)
        
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
            index_path, metadata_path, stats = builder.build_complete_index()
            
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
            # Step 1: Sync resources from database to JSON
            from core.repo_sync import sync_resources_to_json, check_resource_sync_needed
            
            json_path = "data/resources/detected_resources.json"
            
            # Check if resource sync is needed
            if force_rebuild or check_resource_sync_needed(json_path):
                log.info("🔄 Syncing resources from database to JSON...")
                resource_count = sync_resources_to_json(json_path)
                log.info(f"✅ Synced {resource_count} resources to JSON")
            else:
                log.info("📄 Resource JSON is already up to date")
            
            # Step 2: Build FAISS index from the JSON
            from scripts.build_resource_faiss_index import ResourceFAISSIndexBuilder
            
            builder = ResourceFAISSIndexBuilder()
            index_path, metadata_path, stats = builder.build_complete_index()
            
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
                         force_rebuild: bool = False,
                         detect_resources: bool = True,
                         auto_fetch: bool = False) -> Dict[str, Any]:
        """
        Run the complete pipeline with all index building.
        
        Args:
            build_canonical: Build main message search index
            build_community: Build expert/skill search index  
            build_resources: Build resource discovery index
            force_rebuild: Force complete rebuild vs incremental
            detect_resources: Run resource detection on messages
            auto_fetch: Automatically fetch messages if database is empty
        """
        
        log.info("🚀 Starting unified Discord pipeline...")
        pipeline_start = datetime.now()
        
        # Check prerequisites (with optional auto-fetch)
        if not self.check_prerequisites(auto_fetch=auto_fetch):
            return {"status": "failed", "error": "Prerequisites not met"}
        
        results = {
            "pipeline_started": pipeline_start.isoformat(),
            "indices_built": [],
            "errors": [],
            "total_duration_seconds": 0
        }
        
        # Step 1: Detect and store resources (if requested)
        if detect_resources:
            log.info("🔍 Phase 1: Resource Detection")
            result = self.detect_and_store_resources(force_reprocess=force_rebuild)
            results["indices_built"].append(result)
            if result["status"] == "failed":
                results["errors"].append(f"Resource detection: {result['error']}")
        
        # Step 2: Sync resources to JSON (needed for resource index)
        if build_resources:
            log.info("📄 Phase 2: Resource JSON Sync")
            result = self.sync_resources_to_json(force_rebuild=force_rebuild)
            results["indices_built"].append(result)
            if result["status"] == "failed":
                results["errors"].append(f"Resource sync: {result['error']}")
        
        # Step 3: Build canonical index (main search)
        if build_canonical:
            log.info("🔨 Phase 3: Canonical Index")
            result = self.build_canonical_index(force_rebuild=force_rebuild)
            results["indices_built"].append(result)
            if result["status"] == "failed":
                results["errors"].append(f"Canonical index: {result['error']}")
        
        # Step 4: Build community index (expert search)
        if build_community:
            log.info("👥 Phase 4: Community Index") 
            result = self.build_community_index(force_rebuild=force_rebuild)
            results["indices_built"].append(result)
            if result["status"] == "failed":
                results["errors"].append(f"Community index: {result['error']}")
        
        # Step 5: Build resource index (resource discovery)
        if build_resources:
            log.info("📚 Phase 5: Resource Index")
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
    
    def fetch_discord_messages(self, limit: Optional[int] = None) -> Dict[str, Any]:
        """Fetch Discord messages if database is empty or missing."""
        log.info("📥 Fetching Discord messages...")
        start_time = datetime.now()
        
        try:
            import asyncio
            import discord
            from core.fetch_messages import DiscordFetcher
            from db.db import SessionLocal, Message
            import os
            from dotenv import load_dotenv
            
            # Load Discord token
            load_dotenv()
            DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
            
            if not DISCORD_TOKEN:
                raise Exception("DISCORD_TOKEN not found in environment variables")
            
            # Get initial message count
            session = SessionLocal()
            initial_count = session.query(Message).count()
            session.close()
            
            async def run_fetch():
                intents = discord.Intents.default()
                intents.message_content = True
                intents.guilds = True
                intents.members = True
                
                client = DiscordFetcher(intents=intents)
                try:
                    await client.start(DISCORD_TOKEN)
                finally:
                    await client.close()
            
            # Run the async fetch
            log.info("🚀 Starting Discord message fetch...")
            asyncio.run(run_fetch())
            
            # Get final message count
            session = SessionLocal()
            final_count = session.query(Message).count()
            session.close()
            
            messages_fetched = final_count - initial_count
            duration = (datetime.now() - start_time).total_seconds()
            
            log.info(f"✅ Fetched {messages_fetched:,} new messages")
            
            return {
                "status": "success",
                "duration_seconds": duration,
                "messages_fetched": messages_fetched,
                "total_messages": final_count,
                "type": "message_fetch"
            }
            
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            log.error(f"❌ Message fetch failed: {e}")
            
            return {
                "status": "failed",
                "duration_seconds": duration,
                "error": str(e),
                "type": "message_fetch"
            }

    def detect_and_store_resources(self, force_reprocess: bool = False) -> Dict[str, Any]:
        """Detect resources in messages and store them in database."""
        log.info("🔍 Detecting and storing resources...")
        start_time = datetime.now()
        
        try:
            from core.resource_detector import detect_resources
            from db.db import SessionLocal, Message, Resource
            from sqlalchemy import text
            
            session = SessionLocal()
            
            try:
                if force_reprocess:
                    log.info("🔄 Force reprocessing all messages for resources")
                    # Clear existing resources if force reprocessing
                    session.execute(text("DELETE FROM resources"))
                    session.commit()
                    messages = session.query(Message).all()
                else:
                    log.info("📈 Processing messages for new resources")
                    # Get messages that haven't been processed for resources yet
                    processed_message_ids = session.query(Resource.message_id).distinct()
                    messages = session.query(Message).filter(
                        ~Message.id.in_(processed_message_ids)
                    ).all()
                
                new_resources = 0
                messages_processed = 0
                
                for message in messages:
                    try:
                        # Detect resources in this message
                        detected = detect_resources(message)
                        
                        # Store each resource in the database
                        for resource_data in detected:
                            # Check if this resource already exists
                            existing = session.query(Resource).filter_by(
                                url=resource_data.get('url', ''),
                                message_id=message.id
                            ).first()
                            
                            if not existing:
                                resource = Resource(
                                    message_id=message.id,
                                    guild_id=message.guild_id,
                                    channel_id=message.channel_id,
                                    url=resource_data.get('url', ''),
                                    name=resource_data.get('name', ''),
                                    description=resource_data.get('description', ''),
                                    type=resource_data.get('type', ''),
                                    tag=resource_data.get('tag', 'Other'),
                                    author=resource_data.get('author'),  # This should be JSON
                                    author_display=resource_data.get('author', ''),
                                    channel_name=getattr(message, 'channel_name', ''),
                                    timestamp=message.timestamp,
                                    jump_url=resource_data.get('jump_url', '')
                                )
                                session.add(resource)
                                new_resources += 1
                        
                        messages_processed += 1
                        
                        # Commit every 100 messages to avoid memory issues
                        if messages_processed % 100 == 0:
                            session.commit()
                            log.info(f"📊 Processed {messages_processed} messages, found {new_resources} new resources")
                    
                    except Exception as e:
                        log.warning(f"⚠️ Failed to process message {message.id}: {e}")
                        continue
                
                # Final commit
                session.commit()
                
            finally:
                session.close()
            
            duration = (datetime.now() - start_time).total_seconds()
            
            return {
                "status": "success",
                "duration_seconds": duration,
                "resources_detected": new_resources,
                "messages_processed": messages_processed,
                "type": "resource_detection"
            }
            
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            log.error(f"❌ Resource detection failed: {e}")
            
            return {
                "status": "failed",
                "duration_seconds": duration,
                "error": str(e),
                "type": "resource_detection"
            }

    def sync_resources_to_json(self, force_rebuild: bool = False) -> Dict[str, Any]:
        """Sync detected resources from database to JSON file."""
        log.info("📄 Syncing resources to JSON...")
        start_time = datetime.now()
        
        try:
            from core.repo_sync import sync_resources_to_json, check_resource_sync_needed
            
            json_path = "data/resources/detected_resources.json"
            
            # Check if sync is needed
            if force_rebuild or check_resource_sync_needed(json_path):
                log.info("🔄 Syncing resources from database to JSON...")
                resource_count = sync_resources_to_json(json_path)
                message = f"Synced {resource_count} resources to JSON"
                status = "success"
            else:
                log.info("📄 Resource JSON is already up to date")
                # Get current resource count
                if os.path.exists(json_path):
                    with open(json_path, 'r') as f:
                        data = json.load(f)
                        resource_count = len(data)
                else:
                    resource_count = 0
                message = "Resource JSON is current"
                status = "up_to_date"
            
            duration = (datetime.now() - start_time).total_seconds()
            
            return {
                "status": status,
                "duration_seconds": duration,
                "resources_synced": resource_count,
                "json_path": json_path,
                "message": message,
                "type": "resource_sync"
            }
            
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            log.error(f"❌ Resource sync failed: {e}")
            
            return {
                "status": "failed",
                "duration_seconds": duration,
                "error": str(e),
                "type": "resource_sync"
            }
    
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
                "successful_indices": len([r for r in results.get("indices_built", []) if r["status"] in ["success", "up_to_date"]]),
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
        
        print(f"\n📊 Pipeline Phase Results:")
        for result in results.get("indices_built", []):
            if result["status"] == "success":
                status_emoji = "✅"
                if result["type"] == "resource_detection":
                    status_text = f"{result.get('resources_detected', 0)} resources detected"
                elif result["type"] == "resource_sync":
                    status_text = f"{result.get('resources_synced', 0)} resources synced"
                elif result["type"] == "message_fetch":
                    status_text = f"{result.get('messages_fetched', 0)} messages fetched"
                else:
                    status_text = f"{result['duration_seconds']:.1f}s"
            elif result["status"] == "up_to_date":
                status_emoji = "🔄"
                if result["type"] == "resource_sync":
                    status_text = f"current ({result.get('resources_synced', 0)} resources)"
                else:
                    entries = result.get("statistics", {}).get("existing_index_entries", 0)
                    status_text = f"up to date ({entries} entries)"
            elif result["status"] == "skipped":
                status_emoji = "⏭️"
                status_text = "skipped"
            else:
                status_emoji = "❌"
                status_text = f"failed ({result['duration_seconds']:.1f}s)"
            
            # Format phase names nicely
            phase_names = {
                "message_fetch": "Message Fetch",
                "resource_detection": "Resource Detection", 
                "resource_sync": "Resource JSON Sync",
                "canonical": "Canonical Index",
                "community": "Community Index",
                "resource": "Resource Index"
            }
            phase_name = phase_names.get(result["type"], result["type"].title())
            print(f"  {status_emoji} {phase_name}: {status_text}")
        
        print(f"\n🗂️ Index System Status:")
        print(f"  📁 Main Search: data/indices/discord_messages_index")
        print(f"  👥 Expert Search: data/indices/community_faiss_index") 
        print(f"  📚 Resource Search: data/indices/resource_faiss_index")
        
        up_to_date_count = len([r for r in results.get("indices_built", []) if r["status"] == "up_to_date"])
        
        if results["status"] == "success":
            print(f"\n🎉 Complete pipeline success! Your Discord bot now has:")
            print(f"  📡 Fresh Discord message data")
            print(f"  🔍 Intelligent resource detection and extraction") 
            print(f"  🔎 Enhanced message search with semantic matching")
            print(f"  👥 Expert discovery by skills and expertise")
            print(f"  📚 Comprehensive resource discovery system")
            print(f"\n💡 The bot can now provide rich, contextual answers by combining:")
            print(f"     • Historical Discord conversations")
            print(f"     • Community expert knowledge")
            print(f"     • Curated external resources")
        elif up_to_date_count > 0:
            print(f"\n✅ All systems current! Your Discord bot is ready to use.")
            print(f"  💡 Tip: Run with --auto-fetch to automatically refresh message data")
            print(f"       or 'python core/fetch_messages.py' + 'python core/pipeline.py --update'")
            print(f"\n🎉 The bot provides intelligent search across:")
            print(f"  📝 Message history with semantic understanding")
            print(f"  🧠 Expert knowledge and community insights")
            print(f"  🔗 Curated resources, tools, and tutorials")
        
        print(f"{'='*60}\n")

def main():
    """Main entry point with comprehensive CLI interface."""
    
    parser = argparse.ArgumentParser(
        description="Unified Discord Message Processing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --build-all                    # Complete end-to-end pipeline
  %(prog)s --build-all --auto-fetch       # Auto-fetch Discord messages if needed
  %(prog)s --build-all --force            # Force complete rebuild of everything
  %(prog)s --build-all --skip-resources   # Skip resource detection phase
  %(prog)s --update                       # Incremental updates only
  %(prog)s --build-canonical              # Just rebuild main search index
  %(prog)s --build-community              # Just rebuild expert search index
  %(prog)s --build-resources              # Resource detection + indexing
  %(prog)s --build-resources --auto-fetch # Fetch messages + detect resources
  %(prog)s --limit 1000 --build-all       # Test with limited data
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
    
    # Options
    parser.add_argument("--force", action="store_true",
                       help="Force complete rebuild instead of incremental update")
    parser.add_argument("--auto-fetch", action="store_true",
                       help="Automatically fetch Discord messages if database is empty")
    parser.add_argument("--skip-resources", action="store_true",
                       help="Skip resource detection phase")
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
            results = pipeline.run_full_pipeline(
                force_rebuild=args.force,
                detect_resources=not args.skip_resources,
                auto_fetch=args.auto_fetch
            )
            
        elif args.update:
            results = pipeline.update_indices()
            
        elif args.build_canonical:
            if not pipeline.check_prerequisites(auto_fetch=args.auto_fetch):
                sys.exit(1)
            results = {"indices_built": [pipeline.build_canonical_index(args.force)]}
            results["status"] = "success" if results["indices_built"][0]["status"] in ["success", "up_to_date"] else "failed"
            results["total_duration_seconds"] = results["indices_built"][0]["duration_seconds"]
            
        elif args.build_community:
            if not pipeline.check_prerequisites(auto_fetch=args.auto_fetch):
                sys.exit(1)
            results = {"indices_built": [pipeline.build_community_index(args.force)]}
            results["status"] = "success" if results["indices_built"][0]["status"] in ["success", "up_to_date"] else "failed"
            results["total_duration_seconds"] = results["indices_built"][0]["duration_seconds"]
            
        elif args.build_resources:
            if not pipeline.check_prerequisites(auto_fetch=args.auto_fetch):
                sys.exit(1)
            # Resource building includes both detection and indexing
            detection_result = pipeline.detect_and_store_resources(force_reprocess=args.force) if not args.skip_resources else {"status": "skipped", "type": "resource_detection"}
            sync_result = pipeline.sync_resources_to_json(force_rebuild=args.force)
            index_result = pipeline.build_resource_index(args.force)
            
            results = {
                "indices_built": [detection_result, sync_result, index_result],
                "status": "success" if all(r["status"] in ["success", "up_to_date", "skipped"] for r in [detection_result, sync_result, index_result]) else "failed",
                "total_duration_seconds": sum(r.get("duration_seconds", 0) for r in [detection_result, sync_result, index_result])
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
