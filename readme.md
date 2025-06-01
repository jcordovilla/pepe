# 🤖 Agentic Discord Bot
## Advanced Multi-Agent RAG System for Discord Intelligence
### Version: 2.0.0 - Production Ready

An intelligent Discord bot powered by a sophisticated **multi-agent architecture** using LangGraph, ChromaDB vector storage, and specialized AI agents for enhanced conversational intelligence.

---

## 🎯 **Key Features**

### 🤖 **Multi-Agent Architecture**
- **Planning Agent**: Query analysis and task decomposition
- **Search Agent**: Vector similarity search and message retrieval
- **Analysis Agent**: Content analysis and response synthesis
- **Orchestrator**: LangGraph-powered workflow coordination

### 💾 **Advanced Data Management**
- **Persistent Vector Store**: ChromaDB with OpenAI embeddings
- **Conversation Memory**: SQLite-backed history tracking
- **Smart Caching**: Redis-powered multi-level caching
- **Real-time Processing**: Async I/O operations throughout

### 🔧 **Production Features**
- **Error Handling**: Comprehensive error tracking and recovery
- **Performance Monitoring**: Built-in metrics and analytics
- **Scalable Design**: Modular architecture for easy extension
- **Environment Management**: Full Docker and environment support

---

## 🚀 **Quick Start**

### 1. **Setup Environment**
```bash
# Clone and setup
git clone <repository>
cd discord-bot-v2

# Setup project
./launch.sh setup
```

### 2. **Configure Environment Variables**
Create a `.env` file:
```env
DISCORD_TOKEN=your_discord_bot_token
OPENAI_API_KEY=your_openai_api_key
GUILD_ID=your_discord_guild_id
```

### 3. **Start the Bot**
```bash
# Start Discord bot
./launch.sh bot

# Or start web interface
./launch.sh streamlit
```

---

## 🏗️ **Architecture Overview**

```
🤖 Agentic Framework
├── 🎛️ Agent Orchestrator (LangGraph)
│   ├── 🎯 Planning Agent
│   ├── 🔍 Search Agent
│   └── 📈 Analysis Agent
├── 💾 Data Layer
│   ├── 🔍 Vector Store (ChromaDB)
│   ├── 🧠 Conversation Memory (SQLite)
│   └── ⚡ Smart Cache (Redis)
├── 🌐 Interfaces
│   ├── 💬 Discord Interface
│   ├── 🌊 Streamlit Web UI
│   └── 🔌 Agent API
└── 🔧 Core Systems
    ├── 🧪 Query Analysis
    ├── 📋 Task Planning
    └── 📊 Performance Monitoring
```

---

## 📁 **Project Structure**

```
discord-bot-v2/
├── 🤖 agentic/              # Core agentic framework
│   ├── agents/              # Specialized AI agents
│   ├── memory/              # Conversation memory system
│   ├── vectorstore/         # Vector storage with ChromaDB
│   ├── cache/               # Smart caching system
│   ├── interfaces/          # Discord, Streamlit, API interfaces
│   └── reasoning/           # Query analysis and task planning
├── 📊 data/                 # Persistent data storage
├── 📚 docs/                 # Documentation
├── 🏗️ architecture/         # System architecture docs
├── 🚀 main.py              # Main entry point
├── 🛠️ launch.sh            # Launch script
└── 📋 requirements.txt     # Dependencies
```
│   ├── Planning Agent (Query Analysis)
│   ├── Search Agent (Vector Retrieval)
│   ├── Analysis Agent (Content Synthesis)
│   └── Orchestrator (Workflow Management)
├── 💾 Persistent Storage
│   ├── ChromaDB Vector Store
│   ├── SQLite Conversation Memory
│   └── User Context Management
├── ⚡ Performance Layer
│   ├── Smart Cache (L1/L2/L3)
│   ├── Background Processing
│   └── Connection Pooling
└── 🔌 Interfaces
    ├── Discord Interface
    ├── Streamlit Web App
    └── REST API
```

---

## ✨ Key Features

### 🎯 Intelligent Query Processing
- **Multi-Agent Reasoning**: Specialized agents collaborate for optimal results
- **Context-Aware Search**: Understands user intent and conversation history
- **Semantic Understanding**: Goes beyond keyword matching to find meaning
- **Execution Planning**: Breaks down complex queries into manageable tasks

### 💬 Advanced Discord Integration
- **Real-time Responses**: Fast, intelligent answers to user questions
- **Message History Analysis**: Deep insights into server conversations
- **User Context Tracking**: Personalized responses based on interaction history
- **Performance Monitoring**: Built-in analytics and health monitoring

### 🌐 Modern Web Interface
- **Interactive Chat**: Streamlit-powered web interface
- **Real-time Analytics**: System performance and usage metrics
- **Export Capabilities**: Download conversations and insights
- **Admin Controls**: System optimization and health management

### 🔧 Enterprise Features
- **Persistent Storage**: No data loss with ChromaDB vector store
- **Smart Caching**: Multi-level caching for optimal performance
- **Scalable Architecture**: Designed for high-volume Discord servers
- **Monitoring & Observability**: Comprehensive system health tracking

---

## 📁 Project Structure

```
🤖 Agentic Discord Bot v2.0
├── 📋 launch.sh              # Easy launcher script
├── 🔄 migrate_to_agentic.py  # Migration from v1.x
├── 📄 requirements.txt       # Dependencies (LangGraph, ChromaDB, etc.)
├── 📚 readme.md              # This file

🧠 agentic/                   # Core agentic framework
├── 🤖 agents/               # Specialized AI agents
│   ├── base_agent.py        # Base agent class and registry
│   ├── orchestrator.py      # LangGraph workflow orchestrator
│   ├── planning_agent.py    # Query planning specialist
│   ├── search_agent.py      # Vector search specialist
│   └── analysis_agent.py    # Content analysis specialist
├── 💾 memory/              # Conversation memory system
│   └── conversation_memory.py # SQLite-backed memory
├── 🧩 reasoning/           # Reasoning components
│   ├── query_analyzer.py   # Intent detection & entity extraction
│   └── task_planner.py     # Execution plan generation
├── 📊 vectorstore/         # Persistent vector storage
│   └── persistent_store.py # ChromaDB implementation
├── ⚡ cache/               # Smart caching system
│   └── smart_cache.py      # Multi-level cache
└── 🔌 interfaces/          # Platform interfaces
    ├── agent_api.py         # High-level API
    ├── discord_interface.py # Discord integration
    └── streamlit_interface.py # Web interface

🎮 core/                     # Application entry points
├── agentic_bot.py          # New Discord bot
├── agentic_app.py          # New Streamlit app
├── bot.py                  # Legacy Discord bot
└── app.py                  # Legacy Streamlit app

🛠️ tools/                   # Utilities and tools
├── tools.py                # Tool functions
├── time_parser.py          # Natural language time parsing
└── full_pipeline.py        # Data processing pipeline

💾 data/                    # Data storage
├── discord_messages.db     # SQLite database
├── chroma_db/             # ChromaDB vector store
└── cache/                 # Cache storage

📊 logs/                    # Application logs
└── monitoring/            # System metrics
```


db/                       # Database models and migrations
    __init__.py
    db.py                 # Database session management, engine, and models
    models.py             # Data models
    alembic.ini           # Alembic config
    alembic/              # Alembic migrations


data/                     # Data files and vector indexes
    discord_messages.db   # Main SQLite database
    resources/            # Resource logs and exports
    *.json, *.jsonl       # Message and chat history exports
    index_faiss/          # FAISS vector index files

utils/                    # Utility functions and helpers
    __init__.py
    helpers.py            # Helper functions (jump URLs, validation, etc.)
    logger.py             # Logging setup
    embed_store.py        # Embedding helpers

tests/                    # Unit and integration tests
    test_*.py             # Test modules (run with pytest)
    conftest.py           # Pytest fixtures
    query_test_results.json# Test results

docs/                     # Project documentation (Markdown, resources)
    index.md
    resources/
        resources.json    # Exported/curated resources

jc_logs/                  # Performance and architecture logs (gitignored)
```

---

## How to Run

1. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
2. **Prepare the database and data files:**
   - Fetch Discord messages: `python core/fetch_messages.py`
   - (Optional) Migrate or clean data: see `tools/migrate_messages.py`, `tools/clean_resources_db.py`
3. **Configure environment variables:**
   - Copy `.env` and fill in your `DISCORD_TOKEN`, `OPENAI_API_KEY`, etc.
4. **Run the Streamlit app:**
   ```sh
   streamlit run core/app.py
   ```
5. **(Optional) Run the Discord bot:**
   ```sh
   python core/bot.py
   ```
6. **(Optional) Run the full pipeline:**
   ```sh
   python tools/full_pipeline.py
   ```

---

## Requirements

- Python 3.9+
- Discord API token (`DISCORD_TOKEN`)
- OpenAI API key (`OPENAI_API_KEY`)
- (Optional) FAISS, Streamlit, SQLAlchemy, LangChain, TQDM, Prometheus, etc. (see `requirements.txt`)

---

## Notes

- The `jc_logs/` directory and `.DS_Store` files are ignored by git (see `.gitignore`).
- The main database is located at `data/discord_messages.db`.
- For advanced documentation, see the `docs/` folder or build with MkDocs (`mkdocs serve`).
- Test coverage: run `pytest` in the `tests/` directory.
- For troubleshooting, see logs in `jc_logs/` and `tools/full_pipeline.log`.

---

**Author:**  
Jose Cordovilla
GenAI Global Network Architect