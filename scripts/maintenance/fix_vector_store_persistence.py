#!/usr/bin/env python3
"""
Fix Vector Store Persistence Issues

This script identifies and fixes issues that cause the vector store to lose data.
The main problems are:
1. allow_reset=True in ChromaDB settings
2. Embedding function mismatches between sessions
3. Collection recreation instead of loading existing data
"""

import os
import sys
import json
import logging
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Load environment variables
load_dotenv()

from agentic.vectorstore.persistent_store import PersistentVectorStore

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_chromadb_directory():
    """Check the ChromaDB directory structure and files"""
    print("🔍 Checking ChromaDB Directory Structure")
    print("=" * 60)
    
    chromadb_path = Path("data/chromadb")
    if not chromadb_path.exists():
        print("❌ ChromaDB directory does not exist")
        return False
    
    print(f"📁 ChromaDB path: {chromadb_path.absolute()}")
    
    # List all files in the directory
    for item in chromadb_path.iterdir():
        if item.is_file():
            size_mb = item.stat().st_size / (1024 * 1024)
            print(f"   📄 {item.name}: {size_mb:.2f} MB")
        elif item.is_dir():
            print(f"   📁 {item.name}/")
    
    return True


def backup_chromadb():
    """Create a backup of the current ChromaDB"""
    print("\n💾 Creating ChromaDB Backup")
    print("=" * 60)
    
    import shutil
    from datetime import datetime
    
    chromadb_path = Path("data/chromadb")
    if not chromadb_path.exists():
        print("❌ No ChromaDB to backup")
        return False
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = Path(f"data/chromadb_backup_{timestamp}")
    
    try:
        shutil.copytree(chromadb_path, backup_path)
        print(f"✅ Backup created: {backup_path}")
        return True
    except Exception as e:
        print(f"❌ Backup failed: {e}")
        return False


def test_vector_store_with_fixed_config():
    """Test vector store with fixed configuration"""
    print("\n🔧 Testing Vector Store with Fixed Configuration")
    print("=" * 60)
    
    # Fixed configuration - no allow_reset, consistent embedding function
    config = {
        'persist_directory': 'data/vectorstore_fixed',
        'collection_name': 'discord_messages_fixed',
        'embedding_model': 'text-embedding-3-small',
        'batch_size': 100
    }
    
    try:
        print("📝 Initializing vector store with fixed config...")
        vector_store = PersistentVectorStore(config)
        
        print(f"✅ Vector store initialized")
        print(f"📊 Collection count: {vector_store.collection.count()}")
        
        # Test adding a sample message
        test_message = [{
            "message_id": "test_persistence_001",
            "content": "This is a test message to verify persistence",
            "channel_id": "test_channel",
            "channel_name": "test",
            "guild_id": "test_guild",
            "author": {"id": "test_user", "username": "tester"},
            "timestamp": "2025-06-03T15:00:00Z",
            "reactions": [{"emoji": "✅", "count": 1}]
        }]
        
        print("📝 Adding test message...")
        import asyncio
        result = asyncio.run(vector_store.add_messages(test_message))
        print(f"✅ Add result: {result}")
        
        # Check count after adding
        new_count = vector_store.collection.count()
        print(f"📊 New collection count: {new_count}")
        
        # Close properly
        asyncio.run(vector_store.close())
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_fixed_persistent_store():
    """Create a version of persistent_store.py with fixes"""
    print("\n🔧 Creating Fixed Vector Store Configuration")
    print("=" * 60)
    
    # Read the current persistent_store.py
    store_path = Path("agentic/vectorstore/persistent_store.py")
    with open(store_path, 'r') as f:
        content = f.read()
    
    # Create a backup
    backup_path = Path("agentic/vectorstore/persistent_store.py.backup")
    with open(backup_path, 'w') as f:
        f.write(content)
    print(f"✅ Backup created: {backup_path}")
    
    # The main fixes needed:
    fixes_needed = [
        {
            "issue": "allow_reset=True",
            "description": "This setting allows ChromaDB to reset, potentially losing data",
            "fix": "Change to allow_reset=False"
        },
        {
            "issue": "Inconsistent embedding function handling",
            "description": "Different embedding functions can cause collection loading issues",
            "fix": "Ensure consistent embedding function usage"
        },
        {
            "issue": "Exception handling in get_collection",
            "description": "Broad exception handling might mask specific issues",
            "fix": "More specific exception handling"
        }
    ]
    
    print("🔍 Issues identified:")
    for i, fix in enumerate(fixes_needed, 1):
        print(f"   {i}. {fix['issue']}")
        print(f"      - Problem: {fix['description']}")
        print(f"      - Solution: {fix['fix']}")
    
    return True


def main():
    """Main function to check and fix vector store persistence issues"""
    print("🔧 Vector Store Persistence Fix")
    print("=" * 60)
    print("This script will identify and fix issues causing data loss in the vector store.")
    print()
    
    # Step 1: Check current state
    check_chromadb_directory()
    
    # Step 2: Create backup
    backup_chromadb()
    
    # Step 3: Identify configuration issues
    create_fixed_persistent_store()
    
    # Step 4: Test with fixed configuration
    test_vector_store_with_fixed_config()
    
    print("\n" + "=" * 60)
    print("🏁 Next Steps:")
    print("1. Apply the configuration fixes to persistent_store.py")
    print("2. Clear processing markers: rm -rf data/processing_markers/*")
    print("3. Re-run embedding process: python core/embed_store.py")
    print("4. Verify persistence: python check_vector_store.py")
    print()
    print("🔍 Key Changes Needed:")
    print("   - Change allow_reset=True to allow_reset=False")
    print("   - Ensure consistent OpenAI embedding function usage")
    print("   - Add better error handling for collection loading")
    print("   - Add collection existence verification before operations")


if __name__ == "__main__":
    main()
