#!/usr/bin/env python3
"""
Standalone Data Processing Pipeline
Modern implementation using unified services architecture

This pipeline can be run independently of the Discord bot for:
- Fetching messages from Discord API
- Processing and analyzing content
- Generating embeddings and storing in vector database
- Resource identification and classification
- Manual data maintenance and batch operations
"""

import asyncio
import logging
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import sys
from tqdm import tqdm

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import modern services
from agentic.services.unified_data_manager import UnifiedDataManager
from agentic.services.content_processor import ContentProcessingService
from agentic.services.sync_service import DataSynchronizationService
from agentic.services.discord_fetcher import DiscordMessageFetcher
from agentic.interfaces.agent_api import AgentAPI
from agentic.vectorstore.persistent_store import PersistentVectorStore

logger = logging.getLogger(__name__)

class StandalonePipelineRunner:
    """
    Standalone data processing pipeline using modern services architecture
    
    Features:
    - Discord message fetching via API
    - Content processing and classification  
    - Vector embedding generation and storage
    - Resource identification and analysis
    - Manual data maintenance operations
    - Integration with existing modernized services
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize core services
        self.data_manager = UnifiedDataManager(config.get('data_manager', {}))
        self.vector_store = PersistentVectorStore(config.get('vector_store', {}))
        self.sync_service = DataSynchronizationService(config.get('sync', {}))
        
        # Initialize agent API for advanced operations
        self.agent_api = AgentAPI(config)
        
        # Initialize Discord fetcher if token provided
        discord_token = config.get('discord', {}).get('token')
        if discord_token:
            self.discord_fetcher = DiscordMessageFetcher(
                token=discord_token,
                config=config.get('discord_fetcher', {})
            )
        else:
            self.discord_fetcher = None
            logger.warning("🚫 No Discord token provided - fetching disabled")
        
        # Initialize content processor (requires OpenAI client)
        try:
            from openai import OpenAI
            openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
            self.content_processor = ContentProcessingService(openai_client)
        except Exception as e:
            logger.warning(f"⚠️ Content processor initialization failed: {e}")
            self.content_processor = None
        
        logger.info("🔄 Standalone pipeline runner initialized")
    
    async def initialize(self):
        """Initialize all async services"""
        try:
            await self.data_manager.initialize()
            
            if self.discord_fetcher:
                await self.discord_fetcher.initialize()
            
            logger.info("✅ Pipeline services initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize pipeline services: {e}")
            raise
    
    async def run_complete_pipeline(
        self, 
        mode: str = "process_existing",
        guild_id: Optional[str] = None,
        channel_ids: Optional[List[str]] = None,
        fetch_limit: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Run the complete pipeline
        
        Args:
            mode: Pipeline mode ('fetch_and_process', 'process_existing', 'fetch_only')
            guild_id: Discord guild ID for fetching (if applicable)
            channel_ids: Specific channel IDs to fetch (if applicable)
            fetch_limit: Limit messages per channel (if applicable)
            
        Returns:
            Dictionary with pipeline results
        """
        results = {
            "success": False,
            "timestamp": datetime.now().isoformat(),
            "mode": mode,
            "stages": {},
            "statistics": {}
        }
        
        try:
            print(f"🚀 Starting standalone pipeline in '{mode}' mode...")
            
            # Define pipeline stages for progress tracking
            stages = []
            if mode in ["fetch_and_process", "fetch_only"]:
                stages.append("fetch")
            if mode != "fetch_only":
                stages.extend(["discovery", "processing", "embeddings", "resources", "synchronization", "validation"])
            
            # Create overall pipeline progress bar
            with tqdm(total=len(stages), desc="🔄 Pipeline Progress", unit="stage") as pipeline_pbar:
                
                # Stage 1: Data Acquisition (if needed)
                if mode in ["fetch_and_process", "fetch_only"]:
                    print("\n🔍 Stage 1: Fetching Discord messages...")
                    fetch_results = await self._fetch_discord_messages(guild_id, channel_ids, fetch_limit)
                    results["stages"]["fetch"] = fetch_results
                    pipeline_pbar.update(1)
                    pipeline_pbar.set_postfix({"Current": "Fetching"})
                    
                    if not fetch_results["success"]:
                        return results
                    
                    if mode == "fetch_only":
                        results["success"] = True
                        return results
                
                # Stage 2: Data Discovery
                print("\n📁 Stage 2: Discovering data files...")
                discovery_results = await self._discover_data_files()
                results["stages"]["discovery"] = discovery_results
                pipeline_pbar.update(1)
                pipeline_pbar.set_postfix({"Current": "Discovery"})
                
                if not discovery_results["success"]:
                    return results
            
                # Stage 3: Content Processing and Classification
                print("\n🔍 Stage 3: Processing and classifying content...")
                processing_results = await self._process_content(discovery_results["files"])
                results["stages"]["processing"] = processing_results
                pipeline_pbar.update(1)
                pipeline_pbar.set_postfix({"Current": "Processing"})
                
                # Stage 4: Vector Store Operations
                print("\n🧠 Stage 4: Generating embeddings and updating vector store...")
                embedding_results = await self._process_embeddings(discovery_results["files"])
                results["stages"]["embeddings"] = embedding_results
                pipeline_pbar.update(1)
                pipeline_pbar.set_postfix({"Current": "Embeddings"})
                
                # Stage 5: Resource Identification
                print("\n📚 Stage 5: Identifying and classifying resources...")
                resource_results = await self._identify_resources(discovery_results["files"])
                results["stages"]["resources"] = resource_results
                pipeline_pbar.update(1)
                pipeline_pbar.set_postfix({"Current": "Resources"})
                
                # Stage 6: Data Synchronization
                print("\n🔄 Stage 6: Synchronizing data across systems...")
                sync_results = await self._synchronize_data()
                results["stages"]["synchronization"] = sync_results
                pipeline_pbar.update(1)
                pipeline_pbar.set_postfix({"Current": "Sync"})
                
                # Stage 7: Validation and Statistics
                print("\n✅ Stage 7: Validating results and generating statistics...")
                validation_results = await self._validate_and_analyze()
                results["stages"]["validation"] = validation_results
                pipeline_pbar.update(1)
                pipeline_pbar.set_postfix({"Current": "Validation"})
                
                # Determine overall success
                results["success"] = all([
                    discovery_results["success"],
                    processing_results["success"],
                    embedding_results["success"],
                    resource_results["success"],
                    sync_results["success"],
                    validation_results["success"]
                ])
                
                # Generate final statistics
                results["statistics"] = await self._generate_statistics()
                pipeline_pbar.set_postfix({"Current": "Complete"})
                
                if results["success"]:
                    print("\n🎉 Standalone pipeline completed successfully!")
                else:
                    print("\n⚠️ Pipeline completed with some issues - check results")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Pipeline failed: {e}")
            results["error"] = str(e)
            return results
    
    async def _fetch_discord_messages(
        self, 
        guild_id: Optional[str], 
        channel_ids: Optional[List[str]], 
        fetch_limit: Optional[int]
    ) -> Dict[str, Any]:
        """Fetch messages from Discord API"""
        try:
            if not self.discord_fetcher:
                return {
                    "success": False,
                    "error": "Discord fetcher not initialized - no token provided"
                }
            
            if not guild_id:
                return {
                    "success": False,
                    "error": "Guild ID required for fetching messages"
                }
            
            fetch_result = await self.discord_fetcher.fetch_guild_messages(
                guild_id=guild_id,
                channel_ids=channel_ids,
                limit_per_channel=fetch_limit
            )
            
            if fetch_result["success"]:
                print(f"✅ Fetched messages from {fetch_result['channels_processed']} channels")
                print(f"📊 Total messages: {fetch_result['total_messages']}")
            
            return fetch_result
            
        except Exception as e:
            logger.error(f"❌ Discord message fetching failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _discover_data_files(self) -> Dict[str, Any]:
        """Discover available data files"""
        try:
            json_dir = Path("data/fetched_messages")
            
            if not json_dir.exists():
                return {
                    "success": False,
                    "error": "No data/fetched_messages directory found",
                    "files": []
                }
            
            # Find message files
            json_files = [f for f in json_dir.iterdir() if f.suffix == '.json' and f.name.endswith('_messages.json')]
            
            if not json_files:
                return {
                    "success": False,
                    "error": "No message JSON files found", 
                    "files": []
                }
            
            # Analyze files
            total_messages = 0
            file_info = []
            
            for json_file in tqdm(json_files, desc="🔍 Analyzing files", unit="file"):
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        message_count = len(data.get('messages', []))
                        total_messages += message_count
                        
                        file_info.append({
                            "path": str(json_file),
                            "name": json_file.name,
                            "message_count": message_count,
                            "size_mb": json_file.stat().st_size / (1024 * 1024),
                            "guild_id": data.get("guild_id"),
                            "guild_name": data.get("guild_name"),
                            "channel_id": data.get("channel_id"),
                            "channel_name": data.get("channel_name")
                        })
                except Exception as e:
                    logger.warning(f"Could not analyze {json_file}: {e}")
            
            print(f"📊 Found {len(json_files)} files with {total_messages:,} messages")
            
            return {
                "success": True,
                "files": file_info,
                "total_files": len(json_files),
                "total_messages": total_messages
            }
            
        except Exception as e:
            logger.error(f"❌ Data discovery failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "files": []
            }
    
    async def _process_content(self, file_info_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process and classify content"""
        try:
            total_processed = 0
            total_classified = 0
            classifications = {}
            
            # Calculate total messages for progress bar
            total_messages = sum(file_info['message_count'] for file_info in file_info_list)
            
            # Create progress bar for overall progress
            with tqdm(total=total_messages, desc="🔍 Processing content", unit="msg") as pbar:
                for file_info in file_info_list:
                    # Load messages
                    with open(file_info['path'], 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        messages = data.get('messages', [])
                    
                    # Process each message
                    for msg in messages:
                        if self.content_processor:
                            try:
                                analysis = await self.content_processor.analyze_message_content(msg)
                                
                                # Track classifications
                                content_type = analysis.get("content_type", "unknown")
                                classifications[content_type] = classifications.get(content_type, 0) + 1
                                total_classified += 1
                                
                            except Exception as e:
                                logger.warning(f"Content processing failed for message {msg.get('message_id')}: {e}")
                        
                        total_processed += 1
                        pbar.update(1)
                        
                        # Update progress bar description with current stats
                        if total_processed % 50 == 0:  # Update every 50 messages
                            pbar.set_postfix({
                                'classified': total_classified,
                                'rate': f"{(total_classified/total_processed)*100:.1f}%" if total_processed > 0 else "0%"
                            })
            
            print(f"✅ Processed {total_processed:,} messages")
            print(f"🏷️ Classified {total_classified:,} messages")
            
            # Show top classifications
            if classifications:
                print("\n📋 Content type distribution:")
                for content_type, count in sorted(classifications.items(), key=lambda x: x[1], reverse=True)[:10]:
                    print(f"  {content_type}: {count:,}")
            
            return {
                "success": True,
                "messages_processed": total_processed,
                "messages_classified": total_classified,
                "classifications": classifications,
                "success_rate": (total_classified / total_processed) * 100 if total_processed > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"❌ Content processing failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "messages_processed": 0,
                "messages_classified": 0
            }
    
    async def _process_embeddings(self, file_info_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate embeddings and update vector store"""
        try:
            total_processed = 0
            total_added = 0
            
            # Calculate total messages for progress bar
            total_messages = sum(file_info['message_count'] for file_info in file_info_list)
            
            with tqdm(total=total_messages, desc="🧠 Processing embeddings", unit="msg") as pbar:
                for file_info in file_info_list:
                    # Load messages
                    with open(file_info['path'], 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        messages = data.get('messages', [])
                    
                    # Filter valid messages for embeddings
                    valid_messages = []
                    for msg in messages:
                        content = msg.get('content', '').strip()
                        if content and len(content) > 10:  # Basic content validation
                            valid_messages.append(msg)
                    
                    # Add to vector store in batches
                    batch_size = 50
                    for i in range(0, len(valid_messages), batch_size):
                        batch = valid_messages[i:i + batch_size]
                        
                        try:
                            result = await self.vector_store.add_messages(batch)
                            if result:
                                total_added += len(batch)
                            
                            # Small delay to avoid overwhelming the system
                            await asyncio.sleep(0.1)
                            
                        except Exception as e:
                            logger.warning(f"Batch processing failed: {e}")
                    
                    # Update progress bar with processed messages from this file
                    total_processed += len(messages)
                    pbar.update(len(messages))
                    pbar.set_postfix({
                        'added': total_added,
                        'rate': f"{(total_added/total_processed)*100:.1f}%" if total_processed > 0 else "0%"
                    })
            
            print(f"✅ Processed {total_processed:,} messages")
            print(f"🧠 Added {total_added:,} messages to vector store")
            
            return {
                "success": True,
                "messages_processed": total_processed,
                "messages_added": total_added,
                "success_rate": (total_added / total_processed) * 100 if total_processed > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"❌ Embedding processing failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "messages_processed": 0,
                "messages_added": 0
            }
    
    async def _identify_resources(self, file_info_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Identify and classify resources from messages"""
        try:
            total_messages = 0
            resources_found = {
                "urls": 0,
                "code_snippets": 0,
                "attachments": 0,
                "learning_resources": 0,
                "tools_mentioned": 0
            }
            
            # Calculate total messages for progress bar
            total_expected = sum(file_info['message_count'] for file_info in file_info_list)
            
            with tqdm(total=total_expected, desc="📚 Identifying resources", unit="msg") as pbar:
                for file_info in file_info_list:
                    # Load messages
                    with open(file_info['path'], 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        messages = data.get('messages', [])
                    
                    for msg in messages:
                        total_messages += 1
                        content = msg.get('content', '')
                        
                        # Count URLs
                        if 'http' in content:
                            resources_found["urls"] += content.count('http')
                        
                        # Count code snippets
                        if '```' in content or '`' in content:
                            resources_found["code_snippets"] += 1
                        
                        # Count attachments
                        attachments = msg.get('attachments', [])
                        if attachments:
                            resources_found["attachments"] += len(attachments)
                        
                        # Identify learning resources (simple keyword matching)
                        learning_keywords = ['tutorial', 'course', 'guide', 'documentation', 'learn', 'training']
                        if any(keyword in content.lower() for keyword in learning_keywords):
                            resources_found["learning_resources"] += 1
                        
                        # Identify tools mentioned
                        tool_keywords = ['langchain', 'openai', 'pytorch', 'tensorflow', 'huggingface', 'anthropic']
                        if any(keyword in content.lower() for keyword in tool_keywords):
                            resources_found["tools_mentioned"] += 1
                        
                        pbar.update(1)
                        
                        # Update progress bar description with current stats
                        if total_messages % 100 == 0:  # Update every 100 messages
                            total_resources = sum(resources_found.values())
                            pbar.set_postfix({
                                'found': total_resources,
                                'rate': f"{(total_resources/total_messages)*100:.1f}%" if total_messages > 0 else "0%"
                            })
            
            print(f"✅ Analyzed {total_messages:,} messages for resources")
            print("\n📚 Resources identified:")
            for resource_type, count in resources_found.items():
                print(f"  {resource_type}: {count:,}")
            
            return {
                "success": True,
                "messages_analyzed": total_messages,
                "resources_found": resources_found,
                "total_resources": sum(resources_found.values())
            }
            
        except Exception as e:
            logger.error(f"❌ Resource identification failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "messages_analyzed": 0
            }
    
    async def _synchronize_data(self) -> Dict[str, Any]:
        """Synchronize data using modern sync service"""
        try:
            sync_result = await self.sync_service.sync_discord_data()
            
            print(f"✅ Data synchronization completed")
            
            return {
                "success": True,
                "sync_result": sync_result
            }
            
        except Exception as e:
            logger.error(f"❌ Data synchronization failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _validate_and_analyze(self) -> Dict[str, Any]:
        """Validate pipeline results and generate analysis"""
        try:
            # Get system stats using agent API
            stats = await self.agent_api.get_system_stats()
            
            if not stats.get("success"):
                return {
                    "success": False,
                    "error": "Failed to get system statistics"
                }
            
            vector_stats = stats.get("vector_store", {})
            total_docs = vector_stats.get("total_documents", 0)
            
            # Get additional vector store statistics
            collection_stats = await self.vector_store.get_collection_stats()
            
            print(f"✅ Validation complete")
            print(f"📊 Vector store documents: {total_docs:,}")
            print(f"🏛️ Collection statistics available")
            
            return {
                "success": total_docs > 0,
                "total_documents": total_docs,
                "system_stats": stats,
                "collection_stats": collection_stats
            }
            
        except Exception as e:
            logger.error(f"❌ Results validation failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _generate_statistics(self) -> Dict[str, Any]:
        """Generate comprehensive pipeline statistics"""
        try:
            # Get system stats
            stats = await self.agent_api.get_system_stats()
            
            return {
                "pipeline_version": "standalone_v2",
                "architecture": "unified_agentic_services",
                "timestamp": datetime.now().isoformat(),
                "system_stats": stats,
                "features": [
                    "discord_api_fetching",
                    "content_classification",
                    "vector_embeddings",
                    "resource_identification",
                    "data_synchronization",
                    "modern_services_architecture"
                ]
            }
            
        except Exception as e:
            logger.error(f"❌ Statistics generation failed: {e}")
            return {
                "error": str(e)
            }
    
    async def cleanup(self):
        """Cleanup resources"""
        try:
            if self.discord_fetcher:
                await self.discord_fetcher.close()
            logger.info("🧹 Pipeline cleanup completed")
        except Exception as e:
            logger.error(f"❌ Cleanup failed: {e}")


async def main():
    """Main function to run the standalone pipeline"""
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("🚀 Standalone Data Processing Pipeline")
    print("=====================================")
    
    # Configuration
    config = {
        'data_manager': {
            'vector_config': {
                'collection_name': 'discord_messages',
                'persist_directory': './data/chromadb',
                'embedding_model': 'text-embedding-3-small',
                'batch_size': 100
            }
        },
        'vector_store': {
            'collection_name': 'discord_messages',
            'persist_directory': './data/chromadb',
            'embedding_model': 'text-embedding-3-small',
            'batch_size': 100
        },
        'orchestrator': {
            'memory_config': {
                'db_path': 'data/conversation_memory.db',
                'max_history_length': 50,
                'context_window_hours': 24
            }
        },
        'cache': {
            'redis_url': 'redis://localhost:6379',
            'default_ttl': 3600
        },
        'sync': {
            'enable_validation': True,
            'backup_enabled': True
        },
        'discord': {
            'token': os.getenv('DISCORD_BOT_TOKEN')
        },
        'discord_fetcher': {
            'page_size': 100,
            'rate_limit_delay': 1.0,
            'max_retries': 3,
            'output_dir': 'data/fetched_messages'
        }
    }
    
    try:
        # Initialize pipeline
        pipeline = StandalonePipelineRunner(config)
        await pipeline.initialize()
        
        # Get mode from command line arguments or default to processing existing data
        import argparse
        parser = argparse.ArgumentParser(description='Standalone Data Processing Pipeline')
        parser.add_argument('--mode', choices=['fetch_and_process', 'process_existing', 'fetch_only'], 
                          default='process_existing', help='Pipeline mode')
        parser.add_argument('--guild-id', help='Discord guild ID for fetching')
        parser.add_argument('--channel-ids', nargs='+', help='Specific channel IDs to fetch')
        parser.add_argument('--fetch-limit', type=int, help='Limit messages per channel')
        
        args = parser.parse_args()
        
        # Run pipeline
        results = await pipeline.run_complete_pipeline(
            mode=args.mode,
            guild_id=args.guild_id,
            channel_ids=args.channel_ids,
            fetch_limit=args.fetch_limit
        )
        
        # Save results
        results_file = Path("data/pipeline_results_standalone.json")
        results_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n📊 Results saved to {results_file}")
        
        # Cleanup
        await pipeline.cleanup()
        
        if results["success"]:
            print("🎉 Standalone pipeline completed successfully!")
            return 0
        else:
            print("⚠️ Pipeline completed with issues")
            return 1
            
    except Exception as e:
        logger.error(f"❌ Pipeline execution failed: {e}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
