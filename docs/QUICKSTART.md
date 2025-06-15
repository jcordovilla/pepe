# Quick Start Guide

Get the Discord Bot v2 up and running quickly with this guide.

## Prerequisites

- Python 3.9+
- Discord Bot Token
- OpenAI API Key

## 1. Environment Setup

```bash
# Clone and enter directory
cd /Users/jose/Documents/apps/discord-bot-v2

# Install dependencies
pip install -r requirements.txt

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
- ✅ ChromaDB Vector Store: 4,943 records
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
| `python scripts/maintenance/check_vector_store.py` | Check vector store status |

## Troubleshooting

### Bot Won't Start
1. Check `.env` file has correct tokens
2. Verify vector store exists: `ls -la data/vectorstore/`
3. Check logs: `tail logs/agentic_bot.log`

### No Search Results
1. Check vector store population: `python scripts/maintenance/check_vector_store.py`
2. Run pipeline: `python scripts/run_pipeline.py`

### Performance Issues
1. Check analytics: `ls -la data/analytics.db`
2. Monitor logs: `tail -f logs/agentic_bot.log`

## Architecture Overview

```
Discord Bot ← Discord Interface ← Agentic System → Vector Store (ChromaDB)
                                ↓
                           Conversation Memory
                                ↓  
                           Analytics Database
```

---
*For detailed documentation, see `docs/` directory*
