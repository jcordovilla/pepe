# Scripts Directory

This directory contains utility scripts for maintenance, testing, and system management of the modernized Discord Bot.

## 🚀 Core Scripts

### **`run_pipeline.py`** (DEPRECATED)
Legacy pipeline script - **DO NOT USE**. The Discord bot now handles all data processing automatically.

### **`system_status.py`**
System health monitoring and status reporting.
```bash
python scripts/system_status.py
```

### **`test_system.py`**
Comprehensive system testing and validation.
```bash
python scripts/test_system.py
```

### **`validate_deployment.py`**
Pre-deployment validation and readiness checks.
```bash
python scripts/validate_deployment.py
```

## 🔧 Maintenance Scripts

### **`create_snapshot.py`**
Create system snapshots for backup and rollback.
```bash
python scripts/create_snapshot.py
```

### **`restore_snapshot.py`**
Restore system from previous snapshots.
```bash
python scripts/restore_snapshot.py
```

## 🗄️ Database Scripts

### **`database/populate_database.py`**
Complete database initialization and population.
```bash
python scripts/database/populate_database.py
```

### **`database/init_db_simple.py`**
Simple database initialization with sample data.
```bash
python scripts/database/init_db_simple.py
```

## 🧹 Legacy Management

### **`cleanup_legacy.py`**
Clean up legacy files and archives.
```bash
python scripts/cleanup_legacy.py
```

### **`migrate_legacy.py`**
Migration utilities (now completed).
```bash
python scripts/migrate_legacy.py
```

## 📁 Subdirectories

- **`database/`** - Database initialization and management scripts
- **`maintenance/`** - System maintenance and monitoring tools

## 🎯 Quick Usage Guide

### First Time Setup
```bash
# 1. Initialize database
python scripts/database/populate_database.py

# 2. Validate system
python scripts/validate_deployment.py

# 3. Start bot
python main.py
```

### Regular Maintenance
```bash
# Check system status
python scripts/system_status.py

# Run tests
python scripts/test_system.py

# Create backup
python scripts/create_snapshot.py
```

## ⚠️ Important Notes

- **No pipeline scripts needed** - The Discord bot processes data automatically
- **Real-time processing** - Messages are handled as they arrive
- **Unified architecture** - All functionality integrated into main bot
- Environment variables (OPENAI_API_KEY, DISCORD_TOKEN, GUILD_ID)
- Python dependencies and package versions
- File structure and required directories
- Core system component importability
- Configuration validity

## Running Scripts

All scripts should be run from the project root directory to ensure proper path resolution:

```bash
# From project root
cd /Users/jose/Documents/apps/discord-bot-v2

# Run tests
python3 scripts/test_system.py

# Validate deployment readiness
python3 scripts/validate_deployment.py
```

## Integration with Main System

These scripts are designed to work with the agentic framework and will:
- Automatically create necessary data directories
- Use the same configuration as the main application
- Provide detailed logging and error reporting
- Maintain compatibility with the production environment
