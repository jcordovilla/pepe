# Discord Bot v2 - Project Structure

## Overview
This is an agentic RAG (Retrieval-Augmented Generation) application with Discord bot capabilities, built using LangGraph for multi-agent orchestration.

## Root Directory Structure

```
discord-bot-v2/
├── .env                    # Environment variables (Discord token, OpenAI API key)
├── .gitignore             # Git ignore patterns
├── main.py                # Main entry point for the application
├── requirements.txt       # Python dependencies
├── launch.sh             # Shell script for launching the application
├── readme.md             # Project documentation
├── PROJECT_STRUCTURE.md  # This file - project structure documentation
│
├── agentic/              # Core agentic framework
│   ├── config/           # Configuration management
│   ├── core/             # Core business logic
│   ├── interfaces/       # Discord, Streamlit, and API interfaces
│   ├── pipeline/         # Data processing pipeline
│   └── utils/            # Utility functions
│
├── data/                 # Data storage
│   ├── chromadb/         # Vector database (ChromaDB)
│   ├── conversations/    # Conversation history
│   ├── fetched_messages/ # Discord messages cache
│   └── analytics/        # Analytics database
│
├── scripts/              # Utility scripts
│   ├── system_status.py  # System health check
│   └── run_standalone_pipeline.py  # Pipeline runner
│
├── tests/                # Test suite
│   ├── test_*.py         # Unit tests
│   └── test_progress_bars.py  # Progress bar functionality tests
│
├── logs/                 # Application logs
│   ├── agentic_bot.log   # Main application log
│   └── pipeline.log      # Pipeline processing log
│
├── docs/                 # Documentation
├── debug/                # Debug scripts for development
├── migration/            # Migration reports and documentation
├── legacy_archive/       # Archived legacy code
└── .snapshots/           # Code snapshots
```

## Key Components

### 1. **Agentic Framework** (`agentic/`)
- **Multi-agent orchestration** using LangGraph
- **Three main interfaces**: Discord bot, Streamlit web app, REST API
- **Modular architecture** with clear separation of concerns

### 2. **Data Pipeline** (`agentic/pipeline/`)
- **Message fetching** from Discord channels
- **Embedding generation** using OpenAI
- **Vector storage** in ChromaDB
- **AI-powered classification** and analysis

### 3. **Interfaces** (`agentic/interfaces/`)
- **Discord Bot**: Real-time chat interface
- **Streamlit Web App**: Interactive web interface
- **REST API**: Programmatic access

### 4. **Configuration** (`agentic/config/`)
- **Modernized config management**
- **Environment-based settings**
- **Centralized configuration access**

## System Status

- ✅ **Environment**: Properly configured with Discord and OpenAI tokens
- ✅ **Database**: ChromaDB ready for vector storage
- ✅ **Interfaces**: All three interfaces deployment-ready
- ✅ **Pipeline**: Command-line pipeline functional
- ✅ **Tests**: Comprehensive test suite with 100% pass rate

## Usage

### Run Full Pipeline
```bash
python3 scripts/run_standalone_pipeline.py --mode fetch_and_process
```

### Launch Discord Bot
```bash
python3 main.py
```

### Launch Web Interface
```bash
python3 -m agentic.interfaces.streamlit_interface
```

### Check System Status
```bash
python3 scripts/system_status.py
```

## Development

- **Debug scripts** available in `debug/` directory
- **Migration history** preserved in `migration/` and `legacy_archive/`
- **Comprehensive logging** to `logs/` directory
- **Full test coverage** in `tests/` directory

## Next Steps

1. Run fresh pipeline to populate ChromaDB with Discord message data
2. Deploy Discord bot interface
3. Launch web interface for interactive access
4. Monitor system performance and logs
