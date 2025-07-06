# 🤖 Pepe Discord Bot - Streamlined Agentic System

**Intelligent Discord bot with agentic architecture for conversation analysis and insights.**

## ⚡ **Quick Start**

### **For Users (Discord):**
```
/pepe your question here
```
That's it! Single command for all queries.

### **For Admins (System Management):**
```bash
# Initial setup
./pepe-admin setup

# Start the bot
python main.py

# Monitor system
./pepe-admin status
```

## 🎯 **What Can It Do?**

- **🔍 Smart Search**: Find messages by content, user, channel, or timeframe
- **📊 Analytics**: Generate summaries, trends, and activity reports  
- **🤖 Capability Awareness**: Ask about the bot's features and get helpful responses
- **⚡ Real-time Processing**: Automatically indexes new messages as they arrive
- **📈 Performance Monitoring**: Built-in health checks and optimization

### **Example Queries:**
```
/pepe what kind of questions can you answer?
/pepe find messages about machine learning from last week
/pepe summarize the discussion in #general today
/pepe what are the most active channels?
/pepe show me resources shared about Python
```

## 🏗️ **Simple Architecture**

```
Discord Users → /pepe command → Agentic System → Intelligent Response
                     ↓
Admin Users → pepe-admin CLI → System Management
```

**Core Components:**
- **Discord Interface**: Single `/pepe` command handles everything
- **Agentic System**: Multi-agent processing (search, analysis, planning)  
- **Data Storage**: Vector store (ChromaDB) + Analytics (SQLite)
- **Admin CLI**: Unified management tool

## 🚀 **Installation**

### **Prerequisites:**
- Python 3.9+
- Discord Bot Token
- OpenAI API Key

### **Setup:**
```bash
# 1. Clone and install
git clone <repository>
cd discord-bot-agentic
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Edit .env with your tokens

# 3. Initialize system
./pepe-admin setup

# 4. Start bot
python main.py
```

### **Environment Variables:**
```bash
DISCORD_TOKEN=your_discord_bot_token
GUILD_ID=your_discord_guild_id  
OPENAI_API_KEY=your_openai_api_key
```

## 🛠️ **Administration**

All system management is handled through the unified `pepe-admin` CLI:

### **Daily Operations:**
```bash
./pepe-admin status     # System health check
./pepe-admin monitor    # Performance metrics
./pepe-admin stats      # Usage statistics
```

### **Maintenance:**
```bash
./pepe-admin maintain   # System cleanup
./pepe-admin backup     # Create backup
./pepe-admin optimize   # Performance tuning
```

### **Data Management:**
```bash
./pepe-admin sync       # Sync Discord data
./pepe-admin test       # System validation
./pepe-admin migrate    # Resource migration
```

### **Help:**
```bash
./pepe-admin --help
./pepe-admin <command> --help
```

## 📊 **System Status**

**Current Performance:**
- ✅ **Response Time**: ~0.7s average
- ✅ **Success Rate**: 98%+ 
- ✅ **Vector Store**: 13,491 embeddings
- ✅ **Health Score**: 6.4/10 (good and stable)

**System Requirements:**
- **RAM**: 2GB+ recommended
- **Storage**: 500MB+ for vector data
- **Network**: Stable internet for Discord/OpenAI APIs

## 🎛️ **Advanced Features**

### **Intelligent Processing:**
- Multi-agent orchestration with LangGraph
- Semantic search using vector embeddings
- Context-aware query understanding
- Automated content classification

### **Operational Excellence:**
- Real-time message indexing
- Smart caching for performance
- Comprehensive logging and monitoring
- Automatic error recovery

### **Analytics & Insights:**
- Query response tracking
- Performance metrics
- User activity patterns
- System health monitoring

## 🚨 **Troubleshooting**

### **Bot Not Responding:**
```bash
# Check system
./pepe-admin status

# Restart bot
python main.py
```

### **Performance Issues:**
```bash
# Check metrics
./pepe-admin monitor

# Run optimization
./pepe-admin optimize
```

### **Data Problems:**
```bash
# Re-sync data
./pepe-admin sync --initial

# Validate system
./pepe-admin test
```

## 📁 **Key Files**

```
├── main.py              # Bot entry point
├── pepe-admin           # Admin CLI tool
├── agentic/             # Core system
├── data/chromadb/       # Vector store
├── data/*.db            # Analytics & memory
├── logs/                # System logs
└── OPERATIONS.md        # Detailed ops guide
```

## 🔐 **Security**

- Environment variables for sensitive tokens
- No hardcoded credentials
- Regular automated backups
- Secure admin CLI access

## 📚 **Documentation**

- **[OPERATIONS.md](OPERATIONS.md)**: Comprehensive operations guide
- **[scripts/DEPRECATION_NOTICE.md](scripts/DEPRECATION_NOTICE.md)**: Legacy script migration
- **Inline help**: `./pepe-admin --help`

## 🎯 **Key Benefits**

✅ **Simple to use**: Single Discord command for everything
✅ **Easy to manage**: Unified CLI for all admin tasks  
✅ **Reliable**: Built-in monitoring and error handling
✅ **Intelligent**: Multi-agent processing for complex queries
✅ **Performant**: Sub-second response times with smart caching
✅ **Scalable**: Handles 10,000+ messages efficiently

---

## 💡 **Need Help?**

1. **Users**: Try `/pepe what can you do?` in Discord
2. **Admins**: Run `./pepe-admin status` for health check
3. **Issues**: Check logs with `tail -f logs/agentic_bot.log`
4. **Documentation**: See [OPERATIONS.md](OPERATIONS.md) for details

**System designed for simplicity and reliability.** Most operations are automated - admin intervention mainly needed for monitoring and maintenance. 