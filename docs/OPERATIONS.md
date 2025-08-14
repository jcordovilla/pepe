# 🚀 Pepe Discord Bot - Operations Guide

**Streamlined Operations for the Agentic Discord Bot System**

## 📋 **Quick Start**

### **1. Initial Setup**
```bash
# Clone and setup
git clone <repository>
cd discord-bot-agentic

# Environment setup
cp .env.example .env
# Edit .env with your tokens

# System initialization
./pepe-admin setup
```

### **2. Start the Bot**
```bash
# Start Discord bot (real-time processing)
python main.py
```

### **3. Admin Operations**
```bash
# Check system health
./pepe-admin status

# Monitor performance
./pepe-admin monitor

# View statistics
./pepe-admin stats
```

## 🏗️ **System Architecture Overview**

### **Simplified Architecture:**
```
Discord Users → /pepe command → Agentic System → Response
                     ↓
Admin Users → pepe-admin CLI → System Management
```

### **Core Components:**
1. **Discord Interface**: Single `/pepe` command for all user queries
2. **Agentic System**: Multi-agent processing (search, analysis, planning)
3. **Data Storage**: MCP SQLite integration + Analytics (SQLite)
4. **Admin CLI**: Unified `pepe-admin` tool for all operations

## 🎯 **Daily Operations**

### **System Health Monitoring**
```bash
# Daily health check
./pepe-admin status

# Performance monitoring
./pepe-admin monitor

# View usage statistics  
./pepe-admin stats
```

### **Maintenance Tasks**
```bash
# Weekly maintenance
./pepe-admin maintain

# Monthly backup
./pepe-admin backup

# Performance optimization (as needed)
./pepe-admin optimize
```

### **Data Synchronization**
```bash
# The Discord bot handles real-time sync automatically
# Manual sync only needed for historical data:
./pepe-admin sync --initial
```

## 🔧 **Administrative Commands**

### **Setup & Initialization**
```bash
./pepe-admin setup           # Complete system setup
./pepe-admin test            # Validate system functionality
./pepe-admin migrate         # Run resource migrations (if needed)
```

### **Monitoring & Status**
```bash
./pepe-admin status          # System health check
./pepe-admin monitor         # Performance metrics
./pepe-admin stats           # Database statistics
```

### **Maintenance & Optimization**
```bash
./pepe-admin maintain        # Cache cleanup, log rotation, DB optimization
./pepe-admin optimize        # Performance optimization
./pepe-admin backup          # Create system backup
```

### **Data Operations**
```bash
./pepe-admin sync            # Data synchronization
./pepe-admin sync --initial  # Initial historical data import
```

## 📊 **System Monitoring**

### **Key Metrics to Monitor:**
- **Response Time**: Should be < 2 seconds average
- **Success Rate**: Should be > 95%
- **Memory Usage**: Monitor for memory leaks
- **Database Size**: Track growth and performance
- **Cache Hit Rate**: Should be > 80%
- **Active Users**: Track engagement patterns

### **Health Check Indicators:**
```bash
./pepe-admin status
```
**Green Indicators:**
- ✅ All critical files present
- ✅ Database has messages
- ✅ All dependencies installed
- ✅ Environment variables set

**Red Flags:**
- ❌ Missing critical files
- ❌ Empty database
- ❌ Missing dependencies
- ❌ Environment variables not set

## 🚨 **Troubleshooting**

### **Common Issues:**

#### **Bot Not Responding**
```bash
# Check if bot is running
ps aux | grep "python main.py"

# Check system status
./pepe-admin status

# Restart bot
python main.py
```

#### **Poor Query Performance**
```bash
# Check performance metrics
./pepe-admin monitor

# Run optimization
./pepe-admin optimize

# Check database
./pepe-admin stats
```

#### **Data Sync Issues**
```bash
# Check system status
./pepe-admin status

# Re-sync data
./pepe-admin sync --initial

# Test system
./pepe-admin test
```

#### **Memory/Storage Issues**
```bash
# Run maintenance
./pepe-admin maintain

# Create backup before cleanup
./pepe-admin backup

# Check logs
tail -f logs/agentic_bot.log
```

## 📁 **File Structure**

### **Critical Files:**
```
├── main.py                 # Discord bot entry point
├── pepe-admin             # Admin CLI tool
├── agentic/               # Core agentic system
├── data/
│   ├── discord_messages.db # Main message database
│   ├── conversation_memory.db  # Chat history
│   └── analytics.db       # Analytics data
└── logs/                  # System logs
```

### **Configuration Files:**
```
├── .env                   # Environment variables
├── pyproject.toml        # Poetry dependencies
└── data/bot_config.json   # Bot configuration
```

## 🔐 **Security & Environment**

### **Required Environment Variables:**
```bash
DISCORD_TOKEN=your_discord_bot_token
GUILD_ID=your_discord_guild_id
LLM_MODEL=llama3.1:8b
```

### **Security Best Practices:**
1. **Keep tokens secure** - Never commit to version control
2. **Regular backups** - Use `./pepe-admin backup`
3. **Monitor logs** - Check for unusual activity
4. **Update dependencies** - Keep packages current
5. **Restrict admin access** - Limit who can run `pepe-admin`

## 🔄 **Backup & Recovery**

### **Regular Backups:**
```bash
# Create backup
./pepe-admin backup

# Backups stored in: ./backups/backup_YYYYMMDD_HHMMSS/
```

### **Backup Contents:**
- MCP SQLite database
- Conversation memory
- Analytics database
- Configuration files
- Backup manifest

### **Recovery Process:**
1. Stop the bot: `Ctrl+C`
2. Restore files from backup directory
3. Run system test: `./pepe-admin test`
4. Restart bot: `python main.py`

## 📈 **Performance Optimization**

### **Automatic Optimizations:**
- Real-time message indexing
- Smart caching with TTL
- Database connection pooling
- Query result caching

### **Manual Optimizations:**
```bash
# Run performance optimization
./pepe-admin optimize

# Database maintenance
./pepe-admin maintain

# Monitor improvements
./pepe-admin monitor
```

### **Performance Tuning:**
- **Cache Settings**: Adjust TTL in configuration
- **MCP SQLite**: Monitor database size vs. performance  
- **Memory Usage**: Regular maintenance prevents leaks
- **Query Patterns**: Monitor most common query types

## 🎯 **Operational Workflows**

### **Daily Routine:**
1. Check system status: `./pepe-admin status`
2. Review performance: `./pepe-admin monitor`
3. Check logs: `tail logs/agentic_bot.log`

### **Weekly Routine:**
1. Run maintenance: `./pepe-admin maintain`
2. Performance optimization: `./pepe-admin optimize`
3. Review statistics: `./pepe-admin stats`
4. Create backup: `./pepe-admin backup`

### **Monthly Routine:**
1. System tests: `./pepe-admin test`
2. Archive old logs
3. Review and rotate backups
4. Update dependencies if needed

## 📞 **Support & Escalation**

### **Self-Service Diagnostics:**
```bash
./pepe-admin test           # Comprehensive system test
./pepe-admin status         # System health check
./pepe-admin --help         # CLI documentation
```

### **Log Analysis:**
```bash
# Recent activity
tail -f logs/agentic_bot.log

# Error patterns
grep -i error logs/agentic_bot.log

# Performance metrics
grep "processing time" logs/agentic_bot.log
```

### **When to Escalate:**
- Persistent system failures after troubleshooting
- Security concerns or unusual access patterns
- Performance degradation that optimization doesn't fix
- Data corruption or significant data loss

---

## 💡 **Tips for Operators**

1. **Monitor regularly** - Use `pepe-admin status` daily
2. **Automate backups** - Set up cron jobs for `pepe-admin backup`
3. **Log rotation** - `pepe-admin maintain` handles this automatically
4. **Resource monitoring** - Watch memory and disk usage trends
5. **User feedback** - Monitor Discord for user complaints about performance

**Remember**: The system is designed to be self-managing. Most operations are automated through the Discord bot's real-time processing. Admin intervention should mainly be for monitoring, maintenance, and troubleshooting. 