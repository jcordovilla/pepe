# Deployment Checklist

Use this checklist to ensure proper deployment of the Discord Bot v2.

## Pre-Deployment Checks

### ✅ Environment Setup
- [ ] `.env` file created with all required variables
- [ ] `DISCORD_TOKEN` configured
- [ ] `OPENAI_API_KEY` configured
- [ ] Python 3.9+ installed
- [ ] Dependencies installed (`pip install -r requirements.txt`)

### ✅ Database Status
- [ ] ChromaDB vector store populated (check: `python scripts/system_status.py`)
- [ ] Conversation memory database exists
- [ ] Analytics database functional
- [ ] SQLite messages database (optional but recommended)

### ✅ System Validation
```bash
# Run comprehensive system check
python scripts/system_status.py

# Validate deployment readiness  
python scripts/validate_deployment.py

# Test core functionality
python scripts/test_system.py
```

## Deployment Steps

### 1. **System Status Check**
```bash
python scripts/system_status.py
```
**Expected:** All critical components marked ✅

### 2. **Database Population** (if needed)
```bash
# If SQLite database missing
python scripts/database/populate_database.py

# Or run full pipeline
python scripts/run_pipeline.py
```

### 3. **Test Launch**
```bash
# Test bot startup
python main.py
```
**Expected:** Bot connects to Discord successfully

### 4. **Functionality Test**
- [ ] Bot responds to mentions
- [ ] Slash commands work
- [ ] Search functionality operational
- [ ] Reaction search working

### 5. **Production Launch**
```bash
# Use launch script for production
./launch.sh

# Or start with logging
python main.py > logs/production.log 2>&1 &
```

## Post-Deployment Monitoring

### ✅ Health Checks
```bash
# Check logs
tail -f logs/agentic_bot.log

# Monitor system status
python scripts/system_status.py

# Check reaction search
python scripts/maintenance/reaction_search_status.py
```

### ✅ Performance Monitoring
- [ ] Response times < 5 seconds
- [ ] Memory usage stable
- [ ] No error spikes in logs
- [ ] Database queries efficient

## Troubleshooting Guide

### Bot Won't Start
1. **Check environment**: `python scripts/system_status.py`
2. **Verify tokens**: Ensure `.env` has valid tokens
3. **Check permissions**: Bot has necessary Discord permissions
4. **Review logs**: `tail logs/agentic_bot.log`

### No Search Results
1. **Check vector store**: `python scripts/maintenance/check_vector_store.py`
2. **Repopulate data**: `python scripts/run_pipeline.py`
3. **Verify embeddings**: Check ChromaDB has > 1000 records

### Performance Issues
1. **Monitor resources**: Check CPU/memory usage
2. **Database optimization**: Ensure indexes are built
3. **Cache performance**: Monitor hit rates
4. **Analytics review**: Check `data/analytics.db`

## Rollback Plan

### Emergency Rollback
```bash
# Stop current bot
pkill -f "python main.py"

# Restore from backup (if available)
cp .backup/main.py main.py

# Restart with minimal config
python main.py --safe-mode
```

### Data Recovery
```bash
# Restore vector store from backup
cp data/vectorstore_backup/* data/vectorstore/

# Restore databases
cp .backup/data/*.db data/
```

## Success Criteria

### ✅ Deployment Successful When:
- [ ] Bot online in Discord (green status)
- [ ] Responds to test queries within 5 seconds
- [ ] Reaction search returns relevant results
- [ ] No errors in logs for 10 minutes
- [ ] All agents operational (check with test query)
- [ ] Analytics tracking active
- [ ] Memory usage stable

### ✅ Production Ready When:
- [ ] All tests pass: `python scripts/test_system.py`
- [ ] Performance meets targets
- [ ] Monitoring systems active
- [ ] Documentation updated
- [ ] Team trained on operations

## Maintenance Schedule

### Daily
- [ ] Check logs for errors
- [ ] Monitor performance metrics
- [ ] Verify bot online status

### Weekly  
- [ ] Run system status check
- [ ] Review analytics data
- [ ] Update vector store if needed

### Monthly
- [ ] Full system validation
- [ ] Database optimization
- [ ] Performance review
- [ ] Security updates

---

**Last Updated:** June 4, 2025  
**Version:** 2.1.0  
**Status:** Production Ready ✅
