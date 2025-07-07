#!/usr/bin/env python3
"""
Fast System Initialization Script

Quickly initializes the system by:
1. Creating missing databases
2. Processing messages ONLY for vector embeddings (no AI classification)
3. Setting up conversation memory and analytics
4. Applying performance optimizations

This is optimized for speed and minimal API costs.
"""

import asyncio
import json
import os
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/fast_init.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def print_progress_bar(iteration, total, prefix='', suffix='', length=50, fill='█'):
    """Print a progress bar"""
    percent = f"{100 * (iteration / float(total)):.1f}"
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='', flush=True)
    if iteration == total:
        print()

def print_header(title: str):
    """Print formatted header"""
    print(f"\n{'=' * 60}")
    print(f"⚡ {title}")
    print('=' * 60)

async def check_prerequisites() -> bool:
    """Check if all prerequisites are met"""
    print_header("Prerequisites Check")
    
    # Check environment variables
    required_vars = ['OPENAI_API_KEY', 'DISCORD_TOKEN', 'GUILD_ID']
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
        else:
            print(f"✅ {var}: Present")
    
    if missing_vars:
        print(f"❌ Missing environment variables: {missing_vars}")
        return False
    
    # Check data directory
    data_dir = Path("data/fetched_messages")
    if not data_dir.exists():
        print(f"❌ Missing data directory: {data_dir}")
        return False
    
    message_files = list(data_dir.glob("*.json"))
    message_files = [f for f in message_files if not f.name.startswith('fetch_stats')]
    
    if not message_files:
        print("❌ No message files found")
        return False
    
    print(f"✅ Found {len(message_files)} message files")
    return True

async def fast_vector_store_init() -> bool:
    """Initialize vector store with embeddings ONLY (no AI classification)"""
    print_header("Fast Vector Store Initialization")
    
    try:
        from agentic.vectorstore.persistent_store import PersistentVectorStore
        
        # Configuration for vector store
        config = {
            "collection_name": "discord_messages",
            "persist_directory": "./data/chromadb", 
            "embedding_model": os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
            "batch_size": 100,  # Larger batches for speed
            "cache": {"type": "memory", "ttl": 3600}
        }
        
        print("🔄 Initializing vector store...")
        vector_store = PersistentVectorStore(config)
        
        # Check if already populated
        try:
            if vector_store.collection is not None:
                count = vector_store.collection.count()
                if count > 0:
                    print(f"✅ Vector store already has {count} documents")
                    return True
        except:
            pass
        
        # Load message files
        data_dir = Path("data/fetched_messages")
        message_files = list(data_dir.glob("*.json"))
        message_files = [f for f in message_files if not f.name.startswith('fetch_stats')]
        
        print(f"🔄 Processing {len(message_files)} files for vector embeddings...")
        
        total_messages = 0
        processed_messages = 0
        
        for i, file_path in enumerate(message_files):
            try:
                print_progress_bar(i + 1, len(message_files), 
                                 prefix='Files:', 
                                 suffix=f'{file_path.name}')
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                messages = data.get('messages', [])
                total_messages += len(messages)
                
                if messages:
                    # Filter messages with actual content
                    valid_messages = []
                    for msg in messages:
                        content = msg.get('content', '').strip()
                        if content and len(content) > 10:  # Basic filter
                            valid_messages.append(msg)
                    
                    # Process in batches for speed
                    batch_size = config['batch_size']
                    for j in range(0, len(valid_messages), batch_size):
                        batch = valid_messages[j:j + batch_size]
                        success = await vector_store.add_messages(batch)
                        if success:
                            processed_messages += len(batch)
                        
                        # Small delay to prevent overwhelming
                        await asyncio.sleep(0.05)  # Reduced delay for speed
                
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                continue
        
        print(f"\n✅ Vector store initialized: {processed_messages}/{total_messages} messages processed")
        
        # Verify final count
        try:
            if vector_store.collection is not None:
                final_count = vector_store.collection.count()
                print(f"📊 Final vector store count: {final_count}")
                return final_count > 0
            else:
                logger.warning("Collection not initialized")
                return processed_messages > 0
        except Exception as e:
            logger.warning(f"Could not verify final count: {e}")
            return processed_messages > 0
        
    except Exception as e:
        logger.error(f"Error initializing vector store: {e}")
        return False

async def initialize_conversation_memory() -> bool:
    """Initialize conversation memory database"""
    print_header("Conversation Memory Setup")
    
    try:
        from agentic.memory.conversation_memory import ConversationMemory
        
        config = {
            "db_path": "data/conversation_memory.db",
            "max_history_length": 50,
            "context_window_hours": 24
        }
        
        memory = ConversationMemory(config)
        
        # Wait for initialization
        if hasattr(memory, '_init_task') and memory._init_task:
            await memory._init_task
        
        print("✅ Conversation memory initialized")
        return True
        
    except Exception as e:
        logger.error(f"Error initializing conversation memory: {e}")
        return False

async def initialize_analytics() -> bool:
    """Initialize analytics database"""
    print_header("Analytics Setup")
    
    try:
        from agentic.analytics.query_answer_repository import QueryAnswerRepository
        
        config = {
            "db_path": "data/analytics.db",
            "retention_days": 90
        }
        
        repository = QueryAnswerRepository(config)
        print("✅ Analytics database initialized")
        return True
        
    except Exception as e:
        logger.error(f"Error initializing analytics: {e}")
        return False

async def apply_performance_optimizations():
    """Apply performance optimizations"""
    print_header("Performance Optimizations")
    
    # Create cache directories
    cache_dirs = [
        "data/cache/embeddings",
        "data/cache/responses", 
        "data/cache/queries",
        "data/cache/memory"
    ]
    
    for cache_dir in cache_dirs:
        os.makedirs(cache_dir, exist_ok=True)
        print(f"✅ Cache directory: {cache_dir}")
    
    # Create optimized config
    perf_config = {
        "vector_store": {
            "batch_size": 100,
            "cache_embeddings": True,
            "connection_pool_size": 5
        },
        "memory": {
            "cache_conversations": True,
            "max_cache_size": 1000
        },
        "analytics": {
            "batch_metrics": True,
            "async_recording": True
        }
    }
    
    config_path = "data/performance_config.json"
    with open(config_path, "w") as f:
        json.dump(perf_config, f, indent=2)
    
    print(f"✅ Performance config saved: {config_path}")

async def verify_system() -> bool:
    """Quick system verification"""
    print_header("System Verification")
    
    try:
        # Test vector store
        from agentic.vectorstore.persistent_store import PersistentVectorStore
        vector_config = {
            "collection_name": "discord_messages",
            "persist_directory": "./data/chromadb",
            "embedding_model": os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        }
        vector_store = PersistentVectorStore(vector_config)
        if vector_store.collection is not None:
            count = vector_store.collection.count()
            print(f"✅ Vector store: {count} documents")
        else:
            print("✅ Vector store: Initialized (count unavailable)")
        
        # Test conversation memory
        from agentic.memory.conversation_memory import ConversationMemory
        memory_config = {"db_path": "data/conversation_memory.db"}
        memory = ConversationMemory(memory_config)
        if hasattr(memory, '_init_task') and memory._init_task:
            await memory._init_task
        print("✅ Conversation memory: Ready")
        
        # Test analytics
        from agentic.analytics.query_answer_repository import QueryAnswerRepository
        analytics_config = {"db_path": "data/analytics.db"}
        analytics = QueryAnswerRepository(analytics_config)
        print("✅ Analytics: Ready")
        
        return True
        
    except Exception as e:
        logger.error(f"System verification failed: {e}")
        return False

async def main():
    """Main initialization function"""
    print("⚡ Fast Discord Bot System Initialization")
    print(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Ensure directories exist
    os.makedirs("logs", exist_ok=True)
    os.makedirs("data/chromadb", exist_ok=True)
    os.makedirs("data/cache", exist_ok=True)
    
    success_count = 0
    total_steps = 5
    
    print("\n🎯 This script focuses on SPEED and MINIMAL API COSTS")
    print("   - Vector embeddings only (no expensive AI classification)")
    print("   - Batch processing for efficiency")
    print("   - Essential databases only")
    
    # Step 1: Prerequisites
    if await check_prerequisites():
        success_count += 1
    else:
        print("❌ Prerequisites failed")
        return False
    
    # Step 2: Fast vector store setup
    if await fast_vector_store_init():
        success_count += 1
        print("✅ Vector store setup completed")
    else:
        print("❌ Vector store setup failed")
    
    # Step 3: Conversation memory
    if await initialize_conversation_memory():
        success_count += 1
        print("✅ Conversation memory completed")
    else:
        print("❌ Conversation memory failed")
    
    # Step 4: Analytics
    if await initialize_analytics():
        success_count += 1
        print("✅ Analytics completed")
    else:
        print("❌ Analytics failed")
    
    # Step 5: Performance optimizations and verification
    await apply_performance_optimizations()
    if await verify_system():
        success_count += 1
        print("✅ System verification passed")
    else:
        print("❌ System verification failed")
    
    print(f"\n{'=' * 60}")
    print(f"📊 Fast Initialization Summary: {success_count}/{total_steps} steps completed")
    
    if success_count >= 4:  # Allow for some non-critical failures
        print("🎉 Fast initialization successful!")
        print("\n🚀 Next steps:")
        print("   1. Run: python3 scripts/system_status.py")
        print("   2. Run: python3 scripts/validate_deployment.py") 
        print("   3. Start bot: python3 main.py")
        print("\n💡 Note: AI classification was skipped for speed.")
        print("   The bot will work for search and basic queries.")
        print("   Run full pipeline later if you need AI features.")
        return True
    else:
        print("❌ Fast initialization failed")
        print("🔧 Check logs for details")
        return False

if __name__ == "__main__":
    result = asyncio.run(main())
    sys.exit(0 if result else 1)
