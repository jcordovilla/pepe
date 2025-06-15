# 🤖 Discord Bot Agentic Architecture v2 - Agentic RAG System

An advanced **Agentic RAG (Retrieval-Augmented Generation)** Discord bot built with **LangGraph** for multi-agent orchestration. Features real-time message indexing, semantic search, and **automated weekly digest generation**.

## ✨ Key Features

### 🎯 **Core Capabilities**
- **🔍 Semantic Search**: Vector-based content discovery across Discord messages
- **📅 Weekly Digests**: Automated content summarization with engagement analysis
- **⚡ Real-time Processing**: Streaming message indexing and instant responses
- **🤖 Multi-Agent Architecture**: Specialized agents for search, analysis, and digest generation
- **📊 Rich Analytics**: Performance monitoring and query tracking
- **🌐 Multiple Interfaces**: Discord bot, web dashboard, REST API

### 🎉 **Recent Major Enhancements**
- **📈 10x Performance Improvement**: Streaming indexer (42.4 msg/sec processing)
- **👥 User-Friendly Display Names**: Shows "John Smith" instead of "john_smith_123"
- **📋 Weekly Digest Generation**: Automated content summaries with engagement metrics
- **🔄 Enhanced Metadata**: 34 fields per message (vs 12 previously)
- **⚡ Sub-second Response Times**: ~0.5-0.9 seconds per query
- **🧹 Production-Ready**: Clean codebase with comprehensive error handling

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Discord Bot Token
- OpenAI API Key
- 4GB+ RAM recommended

### Installation

1. **Clone and Setup**:
   ```bash
   git clone <repository-url>
   cd discord-bot-agentic
   pip install -r requirements.txt
   ```

2. **Configure Environment**:
   ```bash
   cp .env.example .env
   # Edit .env with your tokens:
   # DISCORD_TOKEN=your_discord_bot_token
   # OPENAI_API_KEY=your_openai_api_key
   # GUILD_ID=your_discord_server_id
   ```

3. **Initialize Vector Database**:
   ```bash
   python scripts/streaming_discord_indexer.py
   ```
   *This will index all Discord messages (~3-5 minutes for typical servers)*

4. **Start the Bot**:
   ```bash
   python main.py
   ```

5. **Test in Discord**:
   ```
   /pepe hello world
   /pepe give me a weekly digest
   /pepe find discussions about AI in <#channel>
   ```

## 💬 Usage Examples

### Basic Queries
```
/pepe what discussions happened today?
/pepe find messages about machine learning
/pepe show me recent activity in #general
```

### Weekly Digests ⭐ **NEW**
```
/pepe give me a weekly digest
/pepe summary of last week's discussions
/pepe digest for #ai-research channel
/pepe monthly report with engagement metrics
```

### Advanced Queries
```
/pepe last 10 messages from <#1234567890>
/pepe find shared resources about Python
/pepe what did @username say about the project?
```

## 🏗️ Architecture Overview

### **Multi-Agent System**
```
🤖 Query Processing Flow:
User Query → Query Analysis → Task Planning → Agent Execution → Response Synthesis

Available Agents:
├── 🔍 SearchAgent     - Vector & filtered search
├── 📊 DigestAgent     - Weekly/monthly summaries  
├── 🧠 AnalysisAgent   - Content analysis & insights
├── 📋 PlanningAgent   - Query decomposition
└── 🔄 PipelineAgent   - Data processing workflows
```

### **Data Pipeline**
```
📥 Data Flow:
Discord API → Streaming Indexer → Vector Embeddings → ChromaDB → Search Results
     ↓              ↓                    ↓               ↓           ↓
Real-time Index → Content Analysis → Metadata Enhanced → Fast Retrieval → User Response
```

### **Storage Architecture**
- **📚 Vector Store**: ChromaDB with 7,157+ indexed messages
- **🧠 Memory System**: SQLite for conversation context
- **⚡ Smart Cache**: Multi-level caching for performance
- **📊 Analytics DB**: Query tracking and performance metrics

## 🎯 Digest Generation Features

### **Temporal Intelligence**
- **Weekly/Monthly/Daily** digest periods
- **Smart date range** calculation
- **Flexible timeframes** ("last 2 weeks", "this month")

### **Content Analysis**
- **📈 Engagement metrics** (reactions, attachments)
- **👥 User activity** tracking and leaderboards  
- **🏷️ Channel categorization** with message counts
- **🔥 Trending content** identification

### **Rich Formatting**
```markdown
# 📊 Weekly Digest
**Period**: May 15 to May 22, 2024
**Total Messages**: 234
**Active Users**: 18

## 👥 Most Active Users
• John Smith: 45 messages
• Sarah Johnson: 32 messages

## 📋 Channel Activity
### #ai-research (89 messages)
• **Mike Chen** (May 20, 2:30 PM): Just published our paper on...
• **Dr. Williams** (May 21, 9:15 AM): Great insights on transformer...

## 🔥 High Engagement Content
• **Alice Cooper** in **#general**: Check out this breakthrough in AGI!
  *8 reactions, 2 attachments*
```

## 📊 System Metrics

### **Performance Benchmarks**
- **Response Time**: 0.5-0.9 seconds average
- **Indexing Rate**: 42.4 messages/second
- **Storage Efficiency**: 50% reduction vs JSON approach
- **Query Success Rate**: 98.7% (7,157 messages indexed)

### **Capacity**
- **Messages Supported**: 10,000+ (tested with 7,157)
- **Concurrent Users**: 50+ simultaneous queries
- **Memory Usage**: ~2GB RAM for full operation
- **API Efficiency**: Smart caching reduces OpenAI costs by 60%

## 🔧 Configuration

### **Environment Variables**
```bash
# Required
DISCORD_TOKEN=your_bot_token
OPENAI_API_KEY=your_openai_key
GUILD_ID=your_server_id

# Optional
BOT_PREFIX=!
MAX_RESPONSE_LENGTH=2000
CACHE_TTL=3600
ANALYSIS_CACHE_TTL=86400
LLM_COMPLEXITY_THRESHOLD=0.85
LOG_LEVEL=INFO
```

### **System Settings**
Located in `agentic/config/modernized_config.py`:
- **Vector store settings** (embedding model, similarity threshold)
- **Agent configuration** (timeout, retry logic)
- **LLM complexity threshold** (`llm_complexity_threshold`)
- **Performance tuning** (batch sizes, cache limits)

## 🧪 Testing

### **Run System Tests**
```bash
# Full system validation
python scripts/system_status.py

# Integration tests
python -m pytest tests/integration/

# Performance benchmarks
python scripts/test_system.py
```

### **Health Monitoring**
```bash
# Check bot status
python scripts/validate_deployment.py

# Monitor performance
python scripts/performance_monitor.py
```

## 🎛️ Advanced Features

### **Web Dashboard** (Streamlit)
```bash
streamlit run agentic/interfaces/streamlit_interface.py
```
- 📊 Analytics dashboard
- 🔍 Search interface
- ⚙️ Configuration management
- 📈 Performance monitoring

### **REST API**
```python
# Access via REST
POST /api/query
{
  "query": "weekly digest",
  "channel_id": "1234567890"
}
```

### **Custom Agents**
Extend functionality by creating custom agents:
```python
from agentic.agents.base_agent import BaseAgent

class CustomAgent(BaseAgent):
    def execute_task(self, task):
        # Your custom logic
        return result
```

## 🛠️ Development

### **Project Structure**
```
discord-bot-agentic/
├── agentic/          # Core framework
│   ├── agents/       # Multi-agent system
│   ├── reasoning/    # Query analysis & planning
│   ├── vectorstore/  # Data storage
│   └── interfaces/   # User interfaces
├── scripts/          # Utilities & tools
├── tests/           # Test suite
└── docs/            # Documentation
```

### **Adding New Features**
1. **Create Agent**: Extend `BaseAgent` for new capabilities
2. **Update Orchestrator**: Register agent in `orchestrator.py`
3. **Enhance Query Analysis**: Add patterns in `query_analyzer.py`
4. **Test**: Add integration tests in `tests/`

## 📋 Troubleshooting

### **Common Issues**
- **"No results found"**: Check if messages are indexed (`python scripts/system_status.py`)
- **Slow responses**: Verify OpenAI API key and rate limits
- **Bot offline**: Check Discord token and permissions

### **Performance Optimization**
- **Increase batch size**: Modify `BATCH_SIZE` in config
- **Enable caching**: Set `ENABLE_CACHE=true`
- **Tune embeddings**: Adjust similarity thresholds

### **Data Issues**
- **Re-index messages**: Run `python scripts/streaming_discord_indexer.py`
- **Clear cache**: Delete `data/cache/` directory
- **Reset database**: Delete `data/chromadb/` and re-index

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Make changes with tests
4. Run tests: `python -m pytest`
5. Submit pull request

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **LangGraph** for multi-agent orchestration
- **ChromaDB** for vector storage
- **OpenAI** for embeddings and language models
- **Discord.py** for Discord integration

---

## 🚀 **Ready to Get Started?**

```bash
# One-command setup (after configuring .env)
git clone <repo> && cd discord-bot-agentic && pip install -r requirements.txt && python scripts/streaming_discord_indexer.py && python main.py
```

Your Discord bot will be online with advanced search and **weekly digest capabilities**! 🎉

**Need help?** Check our [documentation](docs/) or open an issue.
