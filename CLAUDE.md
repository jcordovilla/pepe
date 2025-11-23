# CLAUDE.md - Pepe Discord Bot

## Project Overview

Pepe is an intelligent Discord bot with agentic RAG (Retrieval-Augmented Generation) architecture. It transforms Discord conversations into actionable insights through semantic search, automated analysis, and multi-agent orchestration.

## Build & Run Commands

```bash
# Install dependencies (uses Poetry)
poetry install

# Run the Discord bot
python main.py

# Run tests
pytest                           # All tests
pytest tests/test_integration.py # Integration tests
pytest -v                        # Verbose output

# Code formatting
black .                          # Format code
isort .                          # Sort imports
flake8                           # Linting

# Admin CLI (standalone script)
./pepe-admin info               # Check system status
./pepe-admin setup              # Initial setup
./pepe-admin sync               # Sync Discord messages
./pepe-admin sync --full        # Full sync
./pepe-admin resources          # Process shared resources
./pepe-admin maintain           # System maintenance

# Resource detection (standalone scripts)
python scripts/resource_detector.py              # Run resource detection
python scripts/resource_detector.py --reset-cache # Full re-detection
python scripts/generate_resources_html.py        # Generate HTML from resources

# Database scripts
python scripts/discord_message_fetcher.py        # Fetch Discord messages
python scripts/index_database_messages.py        # Index to vector store
```

## Architecture

### Directory Structure

```
agentic/                  # Main package
├── agents/               # Multi-agent system
│   ├── orchestrator.py   # Main agent coordinator
│   ├── search_agent.py   # Semantic search
│   ├── analysis_agent.py # Analytics and insights
│   ├── digest_agent.py   # Weekly digest generation
│   ├── query_interpreter_agent.py # NL query parsing
│   └── planning_agent.py # Complex query planning
├── services/             # Core services
│   ├── resource_enrichment.py # Web scraping + LLM enrichment
│   ├── pdf_analyzer.py   # Two-tier PDF analysis
│   ├── gpt5_service.py   # OpenAI API wrapper
│   └── web_scraper.py    # Async web content fetching
├── interfaces/           # External interfaces
│   └── discord_interface.py # Discord.py integration
├── vectorstore/          # ChromaDB vector store
├── database/             # SQLite message storage
├── analytics/            # Performance metrics
├── config/               # Configuration management
└── utils/                # Shared utilities

scripts/                  # Standalone scripts
├── resource_detector.py  # Main resource detection
├── discord_message_fetcher.py # Message sync
└── generate_resources_html.py # HTML generation

docs/                     # Documentation and data
├── resources-data.json   # Detected resources (output)
└── resources.html        # Resource library website

data/                     # Runtime data (gitignored)
├── vectorstore/          # ChromaDB data
├── chromadb/             # Alternative vector storage
├── fetched_messages/     # Cached Discord messages
└── processed_resources.json # Processing checkpoint
```

### Key Components

- **Multi-Agent System**: LangGraph-based agents for search, analysis, digest generation
- **Vector Store**: ChromaDB for semantic search with OpenAI embeddings
- **Resource Detection**: Two-tier system (fast extraction + LLM enrichment)
- **PDF Analysis**: PyPDF2 extraction + LLM content analysis

### Technology Stack

- Python 3.11+
- Discord.py for bot interface
- LangGraph/LangChain for agent orchestration
- ChromaDB for vector storage
- OpenAI GPT-4 for LLM calls
- Ollama (llama3.2:3b, deepseek-r1:8b) for local LLM
- SQLite for message persistence

## Code Style

- Use `black` for formatting (line-length 88)
- Use `isort` for import sorting (profile: black)
- Type hints required (mypy strict mode)
- Docstrings for public functions

## Environment Variables

Required in `.env`:
```
DISCORD_TOKEN=          # Discord bot token
OPENAI_API_KEY=         # OpenAI API key
GUILD_ID=               # Discord server ID
```

Optional:
```
LLM_ENDPOINT=http://localhost:11434/api/generate  # Local Ollama
LLM_MODEL=llama3.2:3b   # Local model name
ENABLE_CHARTS=true      # Chart generation
MAX_MESSAGES=10000      # Message limit
```

## Testing

```bash
# Run all tests with coverage
pytest --cov=agentic --cov-report=term-missing

# Run specific test file
pytest tests/test_resource_enrichment.py -v

# Run performance tests
python tests/run_all_tests.py
```

## Common Tasks

### Adding a New Agent
1. Create class in `agentic/agents/` extending `BaseAgent`
2. Register in `agentic/agents/__init__.py`
3. Add to orchestrator workflow in `orchestrator.py`

### Modifying Resource Detection
1. Main logic in `scripts/resource_detector.py`
2. Web scraping in `agentic/services/web_scraper.py`
3. LLM enrichment in `agentic/services/resource_enrichment.py`
4. PDF handling in `agentic/services/pdf_analyzer.py`

### Running Resource Detection
```bash
# Incremental (skips already processed)
python scripts/resource_detector.py

# Full re-run (clears cache)
python scripts/resource_detector.py --reset-cache
```

Output goes to `docs/resources-data.json`.
