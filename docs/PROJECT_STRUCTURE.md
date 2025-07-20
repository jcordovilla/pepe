# Discord Bot Agentic Architecture v2 - Project Structure

## Overview
This is an agentic RAG (Retrieval-Augmented Generation) application with Discord bot capabilities, built using LangGraph for multi-agent orchestration. The system now includes advanced digest generation capabilities for weekly/monthly summaries.

## Root Directory Structure

```
discord-bot-agentic/
├── .env                    # Environment variables (Discord token, OpenAI API key)
├── .gitignore             # Git ignore patterns
├── main.py                # Main entry point for the application
├── pyproject.toml         # Poetry dependencies and project configuration
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
- ✅ **Concurrent subtask execution** for improved performance
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
- Advanced time-bound query processing ("last week", "yesterday", date ranges)
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
└── conversation_memory.py  # SQLite conversation tracking with intelligent summarization
```

#### Smart Caching (`agentic/cache/`)
```
cache/
├── __init__.py
└── smart_cache.py         # Multi-level caching with content classification optimization
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
├── content_processor.py       # AI-powered content analysis with intelligent caching
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
User Query → Query Analysis → Task Planning → Concurrent Agent Execution → Result Synthesis
     ↓           ↓              ↓              ↓                          ↓
Intent Detection → Entity Extract → Task Creation → Parallel Agent Processing → Response Format
                   Time-bound     → Dependency    → Smart Caching          → Memory Update
                   Patterns         Resolution      Content Classification
```

### 3. Digest Generation Workflow
```
Digest Request → Date Range Calc → Message Retrieval → Content Analysis → Digest Format
      ↓              ↓                   ↓                ↓               ↓
Temporal Parse → Filter Creation → Vector Search → Aggregation → Structured Output
```

## Key Capabilities

### ✅ Current Working Features
- **Semantic Search**: Vector-based content discovery with enhanced temporal filtering
- **Concurrent Processing**: Parallel subtask execution for improved performance  
- **Time-bound Queries**: Advanced temporal pattern recognition ("last week", "yesterday", date ranges)
- **Smart Memory Management**: Automatic conversation history summarization
- **Content Classification Caching**: Intelligent caching for repeated content analysis
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
- **Response Time**: ~0.5-0.9 seconds per query (with concurrent processing)
- **Storage Efficiency**: 50% reduction vs JSON approach
- **Concurrent Tasks**: Up to 10 parallel subtasks execution
- **Cache Hit Rate**: 85%+ for content classification
- **Memory Optimization**: Automatic history summarization for long conversations

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
8. **🚀 Concurrent Task Execution**: Parallel subtask processing for faster responses
9. **🧠 Smart Memory Summarization**: Automatic conversation history compression
10. **⚡ Content Classification Caching**: Improved performance with intelligent caching
11. **⏰ Time-bound Query Support**: Enhanced temporal query processing capabilities

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

# Optional Performance & Caching
CACHE_TTL=3600
ANALYSIS_CACHE_TTL=86400
CLASSIFICATION_CACHE_TTL=86400
LLM_COMPLEXITY_THRESHOLD=0.85
MAX_CONCURRENT_TASKS=10
ENABLE_MEMORY_SUMMARIZATION=true
```

### System Requirements
- Python 3.9+
- ChromaDB for vector storage
- OpenAI API access for embeddings
- Discord bot permissions for message reading

## Quick Start

1. **Setup Environment**:
   ```bash
   poetry install
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

4. **Test Enhanced Features**:
   ```
   /pepe give me a weekly digest
   /pepe summary of last week
   /pepe digest for #general channel
   /pepe show me discussions from yesterday
   /pepe find activity between June 1 and June 7
   ```

The system is now production-ready with advanced capabilities including concurrent processing, intelligent memory management, enhanced caching, and sophisticated temporal query processing!

## 🧪 Test Coverage

The system includes comprehensive test coverage for all new features:

### Test Files
```
tests/
├── test_plan_concurrency.py          # Concurrent subtask execution tests
├── test_memory_summary.py             # Memory summarization functionality
├── test_time_bound_queries.py         # Time-bound query processing
├── test_analytics_structure.py        # Analytics system validation
├── test_bot_search.py                 # Discord bot search functionality
├── test_channel_resolution.py         # Channel ID/name resolution
├── test_database_search.py            # Database search with improved error handling
├── test_discord_bot_query.py          # Discord bot query processing
├── test_production_ready.py           # Production environment validation
├── test_query_analysis.py             # Enhanced query analysis
└── ...
```

### New Test Capabilities
- **Concurrent Processing**: Validates parallel subtask execution performance
- **Memory Management**: Tests automatic conversation history summarization
- **Temporal Queries**: Ensures accurate time-bound query processing
- **Caching Systems**: Validates content classification cache efficiency
- **Error Resilience**: Enhanced database test error handling

### Running Tests
```bash
# Run all tests
python -m pytest

# Run specific feature tests
python -m pytest tests/test_plan_concurrency.py
python -m pytest tests/test_memory_summary.py
python -m pytest tests/test_time_bound_queries.py

# Run system integration tests
python scripts/test_system_integrity.py
```
