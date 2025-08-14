# Quick Start Guide

Get the Discord bot agentic system running in minutes.

> **💡 To understand how the bot works, see [AGENTIC_ARCHITECTURE.md](../AGENTIC_ARCHITECTURE.md)**

## Prerequisites

- Python 3.12+
- Poetry for dependency management
- Discord Bot Token
- Ollama for local LLM models

## 1. Environment Setup

```bash
# Clone and enter directory
cd /Users/jose/Documents/apps/discord-bot-v2

# Install dependencies
poetry install

# Copy environment template
cp .env.example .env
# Edit .env with your tokens
```

## 2. Database Status Check

```bash
# Check current database status
python scripts/maintenance/check_vector_store.py
python scripts/maintenance/check_channels.py
```

**Current Status:**
- ✅ MCP SQLite Integration: Standardized database operations
- ✅ Conversation Memory: 138 conversations (older history summarized)
- ✅ Analytics Database: Functional
- ❌ SQLite Database: Missing (discord_messages.db)

## 3. Quick Launch Options

### Option A: Launch with Current Data
```bash
# Start bot with existing vector store
./launch.sh
# OR
python main.py
```

### Option B: Full Pipeline Setup
```bash
# Run complete pipeline (if you need SQLite database)
python scripts/run_pipeline.py

# Then start bot
python main.py
```

### Option C: Database Population Only
```bash
# Populate missing SQLite database
python scripts/database/populate_database.py

# Start bot
python main.py
```

## 4. Validation

```bash
# Test system functionality
python scripts/test_system.py

# Validate deployment
python scripts/validate_deployment.py
```

## 5. Monitoring

```bash
# Check logs
tail -f logs/agentic_bot.log

# Check reaction search status
python scripts/maintenance/reaction_search_status.py
```

## Quick Commands Reference

| Command | Purpose |
|---------|---------|
| `python main.py` | Start Discord bot |
| `python scripts/run_pipeline.py` | Run full data pipeline |
| `./launch.sh` | Quick start script |
| `python scripts/test_system.py` | System validation |
| `python scripts/maintenance/check_database.py` | Check database status |

## Troubleshooting

### Bot Won't Start
1. Check `.env` file has correct tokens
2. Verify vector store exists: `ls -la data/vectorstore/`
3. Check logs: `tail logs/agentic_bot.log`

### No Search Results
1. Check database population: `python scripts/maintenance/check_database.py`
2. Run pipeline: `python scripts/run_pipeline.py`

### Performance Issues
1. Check analytics: `ls -la data/analytics.db`
2. Monitor logs: `tail -f logs/agentic_bot.log`

## Architecture Overview

```
Discord Bot ← Discord Interface ← Agentic System → MCP SQLite Database
                                ↓
                           Conversation Memory
                                ↓  
                           Analytics Database
```

---
*For detailed documentation, see `docs/` directory*
