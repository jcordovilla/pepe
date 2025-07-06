#!/usr/bin/env python3
"""
Quick fix for embedding function mismatch by ensuring consistent configuration.
"""

import json
import shutil
from pathlib import Path
from datetime import datetime

def fix_embedding_config():
    """Fix the embedding configuration mismatch"""
    print("🔧 Fixing Embedding Function Configuration")
    print("=" * 60)
    
    # Backup current vector store
    chromadb_path = Path("data/chromadb")
    backup_path = Path(f"data/chromadb_conflict_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    try:
        if chromadb_path.exists():
            shutil.copytree(chromadb_path, backup_path)
            print(f"✅ Created backup: {backup_path}")
        
        # Remove conflicting collection
        if chromadb_path.exists():
            shutil.rmtree(chromadb_path)
            print("✅ Removed conflicting vector store")
        
        # Create a consistent config file for the main bot
        main_config = {
            "vector_store": {
                "collection_name": "discord_messages",
                "persist_directory": "./data/chromadb",
                "embedding_model": os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),  # Consistent model
                "batch_size": 100
            },
            "embedding_function": "text-embedding-3-small"  # Ensure consistency
        }
        
        config_path = "data/bot_config.json"
        with open(config_path, "w") as f:
            json.dump(main_config, f, indent=2)
        
        print(f"✅ Created consistent config: {config_path}")
        print("✅ Vector store conflict resolved")
        
        return True
        
    except Exception as e:
        print(f"❌ Error fixing config: {e}")
        return False

if __name__ == "__main__":
    success = fix_embedding_config()
    if success:
        print("\n🎉 Configuration fixed!")
        print("   You can now run: python3 main.py")
    else:
        print("\n❌ Fix failed")
