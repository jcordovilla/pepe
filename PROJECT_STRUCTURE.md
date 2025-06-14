# Discord Bot Agentic Architecture v2 - Project Structure

## Overview
This is an agentic RAG (Retrieval-Augmented Generation) application with Discord bot capabilities, built using LangGraph for multi-agent orchestration. The system now includes advanced digest generation capabilities for weekly/monthly summaries.

## Root Directory Structure

```
discord-bot-agentic/
├── .env                    # Environment variables (Discord token, OpenAI API key)
├── .gitignore             # Git ignore patterns
├── main.py                # Main entry point for the application
├── requirements.txt       # Python dependencies
├── readme.md             # Project documentation
├── PROJECT_STRUCTURE.md  # This file - project structure documentation
│
├── agentic/              # Core agentic framework
│   ├── agents/           # Multi-agent system
│   ├── analytics/        # Performance monitoring and analytics
│   ├── cache/           # Smart caching system
│   ├── config/          # Configuration management
│   ├── core/            # Core business logic
│   ├── interfaces/      # Discord, Streamlit, and API interfaces
│   ├── memory/          # Conversation memory management
│   ├── reasoning/       # Query analysis and task planning
│   ├── services/        # Business services layer
│   ├── utils/           # Utility functions and helpers
│   └── vectorstore/     # Vector database operations
│
├── data/                # Data storage
│   ├── chromadb/        # Vector database files
│   ├── messages/        # Discord message archives (JSON)
│   └── analytics/       # Analytics and performance data
│
├── docs/                # Documentation
│   ├── cleanup_complete_*.md  # Cleanup completion records
│   └── guides/          # Usage and deployment guides
│
├── scripts/             # Utility and management scripts
│   ├── streaming_discord_indexer.py  # Optimized message indexing
│   ├── system_status.py              # System health monitoring
│   └── validation_*.py               # System validation tools
│
└── tests/               # Test suite
    ├── integration/     # Integration tests
    └── unit/           # Unit tests
```

## Core Architecture Components

### 🤖 Multi-Agent System (`agentic/agents/`)

```
agents/
├── __init__.py              # Agent registry and initialization
├── base_agent.py           # Base agent class with common functionality
├── orchestrator.py         # LangGraph workflow coordinator
├── search_agent.py         # Vector and filtered search operations
├── analysis_agent.py       # Content analysis and insights
├── digest_agent.py         # Weekly/monthly digest generation
├── planning_agent.py       # Query analysis and execution planning
└── pipeline_agent.py       # Data processing workflows
```

**Key Features:**
- ✅ **Stateful workflow orchestration** with LangGraph
- ✅ **Specialized agent roles** for different query types
- ✅ **Task decomposition** with dependency tracking
- ✅ **Comprehensive error handling** and recovery
- ✅ **Digest generation** for weekly/monthly summaries

### 🧠 Reasoning System (`agentic/reasoning/`)

```
reasoning/
├── __init__.py
├── query_analyzer.py       # Intent detection, entity extraction, temporal parsing
└── task_planner.py        # Execution plan generation and task orchestration
```

**Capabilities:**
- Intent classification (search, digest, analyze, summarize)
- Entity extraction (channels, users, dates, time periods)
- Complex query decomposition with digest support
- Temporal pattern recognition (weekly, monthly, daily)
- Dependency resolution and task scheduling

### 💾 Data Management

#### Vector Store (`agentic/vectorstore/`)
```
vectorstore/
├── __init__.py
└── persistent_store.py     # ChromaDB operations with enhanced metadata
```

**Features:**
- ✅ **7,157+ indexed messages** with 34 metadata fields per message
- ✅ **Semantic search** with OpenAI embeddings
- ✅ **Temporal filtering** and chronological sorting
- ✅ **Enhanced metadata** (display names, attachments, reactions)

#### Memory System (`agentic/memory/`)
```
memory/
├── __init__.py
└── conversation_memory.py  # SQLite conversation tracking and context
```

#### Smart Caching (`agentic/cache/`)
```
cache/
├── __init__.py
└── smart_cache.py         # Multi-level caching with TTL and size limits
```

### 🔗 Interfaces (`agentic/interfaces/`)

```
interfaces/
├── __init__.py
├── discord_interface.py    # Discord bot integration (slash commands)
├── agent_api.py           # REST API for agent operations
└── streamlit_interface.py # Web dashboard and admin interface
```

**Discord Integration:**
- ✅ **Real-time slash commands** (`/pepe`)
- ✅ **Weekly digest commands** (`/digest weekly`)
- ✅ **Message indexing** with enhanced metadata
- ✅ **Channel mention resolution** (`<#channelID>`)
- ✅ **User-friendly display names**

### 📊 Analytics System (`agentic/analytics/`)

```
analytics/
├── __init__.py
├── query_answer_repository.py  # Query/answer tracking
├── performance_monitor.py      # System metrics and monitoring
├── validation_system.py        # Answer quality validation
└── analytics_dashboard.py      # Metrics visualization
```

### 🛠 Services Layer (`agentic/services/`)

```
services/
├── __init__.py
├── unified_data_manager.py    # Centralized data operations
├── discord_service.py         # Discord API operations
├── content_processor.py       # AI-powered content analysis
├── sync_service.py           # Data synchronization
└── channel_resolver.py       # Channel ID/name resolution
```

### 🔧 Utilities (`agentic/utils/`)

```
utils/
├── __init__.py
├── date_utils.py             # Date range calculation for digests
├── logging_utils.py          # Structured logging configuration
└── validation_utils.py       # Data validation helpers
```

## Data Flow Architecture

### 1. Message Processing Pipeline
```
Discord API → Streaming Indexer → Vector Embeddings → ChromaDB
     ↓              ↓                    ↓              ↓
Analytics Tracking → Content Analysis → Metadata Enhancement → Search Index
```

### 2. Query Processing Workflow (LangGraph)
```
User Query → Query Analysis → Task Planning → Agent Selection → Result Synthesis
     ↓           ↓              ↓              ↓              ↓
Intent Detection → Entity Extract → Task Creation → Agent Execution → Response Format
```

### 3. Digest Generation Workflow
```
Digest Request → Date Range Calc → Message Retrieval → Content Analysis → Digest Format
      ↓              ↓                   ↓                ↓               ↓
Temporal Parse → Filter Creation → Vector Search → Aggregation → Structured Output
```

## Key Capabilities

### ✅ Current Working Features
- **Semantic Search**: Vector-based content discovery
- **Temporal Queries**: "last X messages" with proper chronological sorting
- **Channel Filtering**: Discord channel mention support
- **User Display Names**: Friendly names instead of usernames
- **Real-time Indexing**: Streaming Discord message processing
- **Performance Analytics**: Query tracking and system monitoring
- **Weekly Digests**: Automated content summarization
- **Multi-interface Support**: Discord bot, web app, REST API

### 📊 System Metrics
- **Messages Indexed**: 7,157+ with enhanced metadata
- **Metadata Fields**: 34 fields per message
- **Processing Speed**: 42.4 messages/sec
- **Response Time**: ~0.5-0.9 seconds per query
- **Storage Efficiency**: 50% reduction vs JSON approach

### 🎯 Digest Generation Features
- **Temporal Patterns**: Weekly, monthly, daily digests
- **Content Aggregation**: By channels, users, engagement
- **Smart Summarization**: High-engagement content highlighting
- **Flexible Filtering**: Channel-specific or server-wide digests
- **Rich Formatting**: Structured output with metadata

## Recent Enhancements (Latest Update)

### 🚀 Major Optimizations Applied
1. **Streaming Indexer**: Direct Discord API → ChromaDB (eliminated JSON bottleneck)
2. **Enhanced Metadata**: 34 fields per message (vs 12 previously)
3. **Display Names**: User-friendly names in all responses
4. **Weekly Digests**: Full digest generation capability
5. **Temporal Analysis**: Advanced date range processing
6. **Performance Monitoring**: Real-time system metrics
7. **Codebase Cleanup**: 115+ temporary files archived, organized structure

### 📋 Production Readiness
- ✅ **Clean Architecture**: 80 essential Python files, no bloat
- ✅ **Comprehensive Testing**: Integration and unit test coverage
- ✅ **Error Handling**: Robust error recovery and logging
- ✅ **Documentation**: Complete setup and usage guides
- ✅ **Monitoring**: Health checks and performance tracking

## Configuration

### Environment Variables Required
```bash
DISCORD_TOKEN=your_discord_bot_token
OPENAI_API_KEY=your_openai_api_key
GUILD_ID=your_discord_server_id
```

### System Requirements
- Python 3.9+
- ChromaDB for vector storage
- OpenAI API access for embeddings
- Discord bot permissions for message reading

## Quick Start

1. **Setup Environment**:
   ```bash
   pip install -r requirements.txt
   cp .env.example .env  # Add your tokens
   ```

2. **Initialize Data**:
   ```bash
   python scripts/streaming_discord_indexer.py
   ```

3. **Start Bot**:
   ```bash
   python main.py
   ```

4. **Test Digest Feature**:
   ```
   /pepe give me a weekly digest
   /pepe summary of last week
   /pepe digest for #general channel
   ```

The system is now production-ready with advanced digest capabilities and optimized performance!
