#!/usr/bin/env python3
"""
Discord Message Database Update Pipeline with Progress Tracking
"""

import asyncio
import logging
import time
import sys
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from agentic.interfaces.agent_api import AgentAPI

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProgressBar:
    """Simple progress bar for terminal output"""
    
    def __init__(self, total: int, description: str = "", width: int = 50):
        self.total = total
        self.current = 0
        self.description = description
        self.width = width
        self.start_time = time.time()
        
    def update(self, amount: int = 1):
        """Update progress by amount"""
        self.current = min(self.current + amount, self.total)
        self._display()
        
    def set_progress(self, current: int):
        """Set current progress directly"""
        self.current = min(current, self.total)
        self._display()
        
    def _display(self):
        """Display the progress bar"""
        if self.total == 0:
            percent = 100
        else:
            percent = (self.current / self.total) * 100
            
        filled = int(self.width * self.current // self.total) if self.total > 0 else self.width
        bar = '█' * filled + '░' * (self.width - filled)
        
        elapsed = time.time() - self.start_time
        if self.current > 0 and elapsed > 0:
            rate = self.current / elapsed
            eta = (self.total - self.current) / rate if rate > 0 else 0
            eta_str = f" ETA: {eta:.1f}s" if eta > 0 else ""
        else:
            eta_str = ""
            
        print(f"\r{self.description} |{bar}| {percent:.1f}% ({self.current}/{self.total}){eta_str}", end='', flush=True)
        
        if self.current >= self.total:
            print()  # New line when complete
            
    def finish(self):
        """Mark progress as complete"""
        self.current = self.total
        self._display()

class PipelineProgress:
    """Track progress across multiple pipeline stages"""
    
    def __init__(self):
        self.stages = {
            "initialization": {"total": 1, "current": 0, "description": "🔧 Initializing"},
            "fetch": {"total": 100, "current": 0, "description": "📥 Fetching messages"},
            "embed": {"total": 100, "current": 0, "description": "🧠 Creating embeddings"},
            "detect": {"total": 100, "current": 0, "description": "🔍 Detecting resources"},
            "sync": {"total": 100, "current": 0, "description": "🔄 Syncing data"}
        }
        self.current_stage = None
        self.progress_bar = None
        
    def start_stage(self, stage_name: str, total: Optional[int] = None):
        """Start a new pipeline stage"""
        if stage_name in self.stages:
            if total:
                self.stages[stage_name]["total"] = total
            
            self.current_stage = stage_name
            stage_info = self.stages[stage_name]
            
            print(f"\n{stage_info['description']}...")
            self.progress_bar = ProgressBar(
                total=stage_info["total"],
                description=stage_info["description"],
                width=40
            )
            
    def update_progress(self, amount: int = 1):
        """Update current stage progress"""
        if self.progress_bar:
            self.progress_bar.update(amount)
            
    def set_progress(self, current: int):
        """Set current stage progress directly"""
        if self.progress_bar:
            self.progress_bar.set_progress(current)
            
    def finish_stage(self):
        """Mark current stage as complete"""
        if self.progress_bar:
            self.progress_bar.finish()
            if self.current_stage:
                self.stages[self.current_stage]["current"] = self.stages[self.current_stage]["total"]
        
    def get_overall_progress(self) -> float:
        """Get overall pipeline progress percentage"""
        total_possible = sum(stage["total"] for stage in self.stages.values())
        total_completed = sum(stage["current"] for stage in self.stages.values())
        return (total_completed / total_possible) * 100 if total_possible > 0 else 0

async def run_pipeline_with_progress(agent_api: AgentAPI, progress: PipelineProgress):
    """Run the real pipeline with progress tracking"""
    import os
    import json
    from pathlib import Path
    from core.embed_store import MessageEmbedder
    
    # Stage 1: Initialization
    progress.start_stage("initialization")
    
    # Initialize vector store and message embedder
    config = {
        "collection_name": "discord_messages",
        "persist_directory": "./data/vectorstore",
        "embedding_model": "text-embedding-3-small",
        "batch_size": 100
    }
    
    embedder = MessageEmbedder(config)
    vector_store = agent_api.vector_store
    
    progress.finish_stage()
    
    # Stage 2: Check for JSON files and count them
    json_dir = Path("data/fetched_messages")
    if not json_dir.exists():
        print("⚠️  No JSON files found in data/fetched_messages/")
        print("📥 Please run the message fetcher first: python core/fetch_messages.py")
        return {"success": False, "error": "No data to process"}
        
    json_files = [f for f in json_dir.iterdir() if f.suffix == '.json' and f.name.endswith('_messages.json')]
    total_files = len(json_files)
    
    if total_files == 0:
        print("⚠️  No message JSON files found in data/fetched_messages/")
        print("📥 Please run the message fetcher first: python core/fetch_messages.py")
        return {"success": False, "error": "No data to process"}
    
    # Count total messages to process
    total_messages = 0
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                messages = data.get('messages', [])
                total_messages += len(messages)
        except Exception as e:
            logger.warning(f"Could not count messages in {json_file}: {e}")
    
    # Stage 3: Processing JSON files (real embedding and storage)
    progress.start_stage("embed", total_files)
    print(f"📁 Found {total_files} JSON files with ~{total_messages} messages to process")
    
    messages_processed = 0
    messages_added = 0
    
    for i, json_file in enumerate(json_files):
        try:
            # Load messages from file
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                messages = data.get('messages', [])
            
            if messages:
                # Process messages in batches
                batch_size = 50
                for j in range(0, len(messages), batch_size):
                    batch = messages[j:j + batch_size]
                    
                    # Filter valid messages
                    valid_messages = [msg for msg in batch if embedder.is_valid_message(msg)]
                    
                    if valid_messages:
                        # Add to vector store
                        success = await vector_store.add_messages(valid_messages)
                        if success:
                            messages_added += len(valid_messages)
                        
                    messages_processed += len(batch)
            
            progress.update_progress(1)
            
            # Log progress every 10 files
            if (i + 1) % 10 == 0 or i == total_files - 1:
                logger.info(f"Processed {i + 1}/{total_files} files ({messages_processed} messages)")
                
        except Exception as e:
            logger.error(f"Error processing {json_file}: {e}")
            continue
    
    progress.finish_stage()
    
    # Stage 4: Database optimization and indexing
    progress.start_stage("detect", 4)
    
    operations = [
        ("Building search indices", vector_store.optimize),
        ("Collecting statistics", vector_store.get_collection_stats),
        ("Running health check", vector_store.health_check),
        ("Finalizing database", lambda: True)
    ]
    
    for operation_name, operation_func in operations:
        try:
            logger.info(f"🔄 {operation_name}...")
            if asyncio.iscoroutinefunction(operation_func):
                result = await operation_func()
            else:
                result = operation_func()
            logger.info(f"✅ {operation_name} complete")
        except Exception as e:
            logger.warning(f"⚠️ {operation_name} failed: {e}")
        
        progress.update_progress(1)
    
    progress.finish_stage()
    
    # Stage 5: Final validation and statistics
    progress.start_stage("sync", 1)
    
    try:
        # Get final collection stats
        final_stats = await vector_store.get_collection_stats()
        logger.info("📊 Final statistics collected")
    except Exception as e:
        logger.warning(f"Could not collect final stats: {e}")
        final_stats = {"total_documents": messages_added}
    
    progress.finish_stage()
    
    return {
        "success": True,
        "stats": {
            "total_messages": messages_added,
            "total_files": total_files,
            "messages_processed": messages_processed,
            "overall_progress": progress.get_overall_progress(),
            "collection_stats": final_stats
        }
    }

async def main():
    """Run the pipeline update with progress tracking"""
    print("🚀 Starting Discord message database update...")
    print("=" * 60)
    
    # Initialize progress tracker
    progress = PipelineProgress()
    
    # Initialize the API
    config = {
        "orchestrator": {},
        "vector_store": {  # Changed from "vectorstore" to "vector_store" 
            "persist_directory": "data/vectorstore",
            "collection_name": "discord_messages", 
            "embedding_model": "text-embedding-3-small",
            "batch_size": 100
        },
        "memory": {"db_path": "data/conversation_memory.db"},
        "pipeline": {"base_path": "."},
        "analytics": {"db_path": "data/analytics.db"}
    }
    
    start_time = time.time()
    agent_api = None
    
    try:
        agent_api = AgentAPI(config)
        
        # Run pipeline with progress tracking
        result = await run_pipeline_with_progress(agent_api, progress)
        
        elapsed_time = time.time() - start_time
        
        print("\n" + "=" * 60)
        if result.get("success"):
            print("✅ Pipeline completed successfully!")
            stats = result.get("stats", {})
            print(f"📈 Processing Summary:")
            print(f"   • Files processed: {stats.get('total_files', 'N/A')}")
            print(f"   • Messages processed: {stats.get('messages_processed', 'N/A')}")
            print(f"   • Messages added to vector store: {stats.get('total_messages', 'N/A')}")
            print(f"   • Overall progress: {stats.get('overall_progress', 0):.1f}%")
            print(f"   • Time elapsed: {elapsed_time:.1f} seconds")
            
            # Display vector store statistics
            collection_stats = stats.get("collection_stats", {})
            if collection_stats:
                print("\n📊 Vector Store Status:")
                print(f"   • Total documents: {collection_stats.get('total_documents', 'N/A')}")
                print(f"   • Collection name: {collection_stats.get('collection_name', 'N/A')}")
                if "top_channels" in collection_stats:
                    top_channels = collection_stats["top_channels"][:3]
                    channel_names = []
                    for ch in top_channels:
                        name = ch.get('name', 'unknown')
                        count = ch.get('count', 0)
                        channel_names.append(f'#{name} ({count})')
                    print(f"   • Top channels: {', '.join(channel_names)}")
            
            # Display legacy database status (if exists)
            print("\n📊 Legacy Database Status:")
            try:
                import sqlite3
                import os
                db_path = "data/discord_bot.db"
                if os.path.exists(db_path):
                    conn = sqlite3.connect(db_path)
                    cursor = conn.cursor()
                    cursor.execute('SELECT COUNT(*) FROM messages')
                    count = cursor.fetchone()[0]
                    conn.close()
                    print(f"   • Messages in legacy database: {count}")
                else:
                    print("   • Legacy database file not found")
            except Exception as e:
                print(f"   • Could not check legacy database: {e}")
                
        else:
            print(f"❌ Pipeline failed: {result.get('error')}")
            
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"\n❌ Error after {elapsed_time:.1f}s: {e}")
        logger.exception("Pipeline error details:")
    finally:
        if agent_api is not None:
            try:
                await agent_api.close()
            except Exception:
                pass  # Ignore cleanup errors

if __name__ == "__main__":
    asyncio.run(main())
