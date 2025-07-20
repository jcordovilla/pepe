# Scripts Directory

This directory contains utility scripts for maintenance, testing, and system management of the Discord Bot.

## 🚀 Core Scripts

### **`discord_message_fetcher.py`**
Fetch messages from Discord and store them in SQLite database.
```bash
# Full fetch (all messages)
poetry run python scripts/discord_message_fetcher.py --full

# Incremental fetch (new messages only)
poetry run python scripts/discord_message_fetcher.py --incremental

# Show help
poetry run python scripts/discord_message_fetcher.py --help
```

**Note**: This script automatically ignores channels that have "test" in their name (case-insensitive) to avoid fetching test data.

### **`index_database_messages.py`**
Index messages from SQLite database into vector store for semantic search.
```bash
# Full indexing (all messages)
poetry run python scripts/index_database_messages.py --full

# Incremental indexing (new messages only)
poetry run python scripts/index_database_messages.py --incremental
```

### **`resource_detector.py`**
CLI-based resource detection using local Llama models.
```bash
# Detect resources from messages
poetry run python scripts/resource_detector.py

# Use fast model for quicker processing
poetry run python scripts/resource_detector.py --fast-model

# Reset cache and reprocess all resources
poetry run python scripts/resource_detector.py --reset-cache
```

## 🛠️ Maintenance Scripts

### **`delete_and_recreate_collection.py`**
Delete and recreate ChromaDB collection (useful for fixing embedding model issues).
```bash
poetry run python scripts/delete_and_recreate_collection.py
```

## 🎯 Quick Usage Guide

### First Time Setup
```bash
# 1. Ensure Poetry environment is active
poetry shell

# 2. Setup system
poetry run ./pepe-admin setup

# 3. Fetch Discord messages
poetry run ./pepe-admin sync --full

# 4. Start bot
poetry run python main.py
```

### Regular Maintenance
```bash
# Check system status
poetry run ./pepe-admin info

# Incremental sync (new messages only)
poetry run ./pepe-admin sync

# Run system validation
poetry run ./pepe-admin test

# Detect resources
poetry run ./pepe-admin resources detect
```

## ⚠️ Important Notes

- **Always use Poetry**: All commands should be run with `poetry run` or within `poetry shell`
- **Test channel filtering**: All scripts automatically ignore channels with "test" in the name
- **Real-time processing** - Messages are handled as they arrive
- **Unified architecture** - All functionality integrated into main bot
- **Local Llama models** - Resource detection uses local models for privacy
- **OpenAI embeddings only** - Vector search uses OpenAI for embeddings

## Running Scripts

All scripts should be run from the project root directory using Poetry:

```bash
# From project root
cd /Users/jose/Documents/apps/discord-bot-agentic

# Option 1: Use poetry run for each command
poetry run python scripts/discord_message_fetcher.py --full
poetry run python scripts/index_database_messages.py --incremental
poetry run python scripts/resource_detector.py --fast-model

# Option 2: Activate Poetry shell and run commands directly
poetry shell
python scripts/discord_message_fetcher.py --full
python scripts/index_database_messages.py --incremental
python scripts/resource_detector.py --fast-model
```

## Integration with Main System

These scripts work with the agentic framework and will:
- Automatically create necessary data directories
- Use the same configuration as the main application
- Provide detailed logging and error reporting
- Maintain compatibility with the production environment
- Always use the Poetry-managed dependencies
