# Architecture Documentation

## Project Structure

```
discord-bot-agentic/
├── main.py                          # Entry point
├── pyproject.toml                   # Poetry configuration
├── poetry.lock                      # Dependency lock file
├── pytest.ini                      # Test configuration
├── pepe-admin                       # Admin script
├── agentic/                         # Main package
│   ├── __init__.py
│   ├── agents/                      # Agent system
│   │   ├── __init__.py
│   │   ├── base_agent.py           # Abstract base agent
│   │   ├── orchestrator.py         # Agent coordination
│   │   ├── search_agent.py         # Search functionality
│   │   ├── analysis_agent.py       # Content analysis
│   │   ├── digest_agent.py         # Content summarization
│   │   ├── planning_agent.py       # Task planning
│   │   ├── query_interpreter_agent.py # Query interpretation
│   │   ├── pipeline_agent.py       # Data pipeline management
│   │   ├── error_recovery_agent.py # Error handling
│   │   ├── result_aggregator.py    # Result combination
│   │   └── shared_state.py         # Shared state management
│   ├── analytics/                   # Analytics system
│   │   ├── __init__.py
│   │   ├── analytics_dashboard.py  # Dashboard interface
│   │   ├── performance_monitor.py  # Performance tracking
│   │   ├── query_answer_repository.py # Query storage
│   │   └── validation_system.py    # Answer validation
│   ├── cache/                       # Caching system
│   │   ├── __init__.py
│   │   └── smart_cache.py          # Intelligent caching
│   ├── config/                      # Configuration
│   │   ├── __init__.py
│   │   └── modernized_config.py    # System configuration
│   ├── database/                    # Database layer
│   │   └── __init__.py
│   ├── interfaces/                  # Interface layer
│   │   ├── __init__.py
│   │   ├── agent_api.py            # Agent API interface
│   │   ├── discord_interface.py    # Discord bot interface
│   │   └── streamlit_interface.py  # Web interface
│   ├── memory/                      # Memory system
│   │   ├── __init__.py
│   │   ├── conversation_memory.py  # Conversation history
│   │   └── enhanced_memory.py      # Advanced memory features
│   ├── reasoning/                   # Reasoning system
│   │   ├── __init__.py
│   │   └── task_planner.py         # Task planning logic
│   ├── services/                    # Service layer
│   │   ├── __init__.py
│   │   ├── channel_resolver.py     # Channel resolution
│   │   ├── discord_fetcher.py      # Discord data fetching
│   │   ├── discord_service.py      # Discord operations
│   │   ├── fetch_state_manager.py  # Fetch state management
│   │   ├── llm_client.py           # LLM interface
│   │   ├── service_container.py    # Dependency injection
│   │   ├── sync_service.py         # Data synchronization
│   │   └── unified_data_manager.py # Data management
│   ├── utils/                       # Utilities
│   │   ├── __init__.py
│   │   └── error_handling.py       # Error management
│   └── vectorstore/                 # Vector storage
│       ├── __init__.py
│       └── persistent_store.py     # ChromaDB implementation
├── data/                            # Data storage
│   ├── cache/                       # Cache files
│   ├── chromadb/                    # Vector database
│   ├── logging_config.json         # Logging configuration
│   ├── resource_checkpoint.json    # Resource state
│   └── resources_export.json       # Exported resources
├── logs/                            # Log files
├── scripts/                         # Utility scripts
│   ├── README.md
│   ├── discord_message_fetcher.py  # Message fetching
│   ├── index_database_messages.py  # Message indexing
│   └── resource_detector.py        # Resource detection
└── tests/                           # Test suite
    ├── __init__.py
    ├── README.md
    ├── run_all_tests.py            # Test runner
    ├── test_agent_registry.py      # Agent registry tests
    ├── test_discord_bot_core.py    # Core bot tests
    ├── test_integration.py         # Integration tests
    ├── test_orchestrator.py        # Orchestrator tests
    ├── test_workflow.py            # Workflow tests
    └── performance_test_suite/     # Performance tests
        ├── __init__.py
        ├── README.md
        ├── bot_runner.py           # Bot test runner
        ├── content_analyzer.py     # Content analysis tests
        ├── data/                   # Test data
        ├── evaluator.py            # Test evaluation
        ├── main_orchestrator.py    # Test orchestration
        ├── query_generator.py      # Query generation
        ├── report_generator.py     # Report generation
        ├── run_tests.py            # Test execution
        └── sample_config.json      # Test configuration
```

## Core Components

### Entry Point
- `main.py`: Initializes Discord bot, sets up logging, handles `/pepe` command

### Configuration
- `agentic/config/modernized_config.py`: Centralized configuration with Discord, LLM, data, processing, and interface settings

### Interface Layer
- `agentic/interfaces/discord_interface.py`: Discord bot interface with message handling, formatting, and analytics
- `agentic/interfaces/agent_api.py`: High-level API for agentic system with query processing and management
- `agentic/interfaces/streamlit_interface.py`: Web-based administration interface

### Agent System
- `agentic/agents/base_agent.py`: Abstract base class with AgentRole, TaskStatus, SubTask, ExecutionPlan, AgentState
- `agentic/agents/orchestrator.py`: LangGraph-based agent coordination and workflow management
- `agentic/agents/search_agent.py`: Semantic and keyword search functionality
- `agentic/agents/analysis_agent.py`: Content analysis and processing
- `agentic/agents/digest_agent.py`: Content summarization and synthesis
- `agentic/agents/planning_agent.py`: Task planning and execution strategy
- `agentic/agents/query_interpreter_agent.py`: Query analysis and categorization
- `agentic/agents/pipeline_agent.py`: Data pipeline management
- `agentic/agents/error_recovery_agent.py`: Error handling and recovery strategies
- `agentic/agents/result_aggregator.py`: Result combination and formatting
- `agentic/agents/shared_state.py`: Shared state management between agents

### Service Layer
- `agentic/services/service_container.py`: Dependency injection and shared service management
- `agentic/services/discord_service.py`: Discord-specific operations
- `agentic/services/unified_data_manager.py`: Centralized data access and management
- `agentic/services/sync_service.py`: Data synchronization and consistency
- `agentic/services/llm_client.py`: Unified language model interface
- `agentic/services/channel_resolver.py`: Channel and guild resolution
- `agentic/services/discord_fetcher.py`: Discord data fetching
- `agentic/services/fetch_state_manager.py`: Fetch state management

### Data Layer
- `agentic/vectorstore/persistent_store.py`: ChromaDB-based vector storage with similarity, keyword, and filtered search
- `agentic/memory/conversation_memory.py`: Conversation history management
- `agentic/memory/enhanced_memory.py`: Advanced memory features
- `agentic/cache/smart_cache.py`: Intelligent caching with TTL and memory management

### Analytics System
- `agentic/analytics/query_answer_repository.py`: Query-answer pair storage and retrieval
- `agentic/analytics/performance_monitor.py`: Performance metrics tracking
- `agentic/analytics/validation_system.py`: Answer quality validation
- `agentic/analytics/analytics_dashboard.py`: Analytics visualization and insights

### Utilities
- `agentic/utils/error_handling.py`: Comprehensive error management and recovery

### Scripts
- `scripts/discord_message_fetcher.py`: Discord message fetching utility
- `scripts/index_database_messages.py`: Message indexing for vector store
- `scripts/resource_detector.py`: Resource detection and processing
- `pepe-admin`: Administrative script

### Test Suite
- `tests/run_all_tests.py`: Main test runner
- `tests/test_agent_registry.py`: Agent registry testing
- `tests/test_discord_bot_core.py`: Core bot functionality testing
- `tests/test_integration.py`: Integration testing
- `tests/test_orchestrator.py`: Orchestrator testing
- `tests/test_workflow.py`: Workflow testing
- `tests/performance_test_suite/`: Performance testing framework

## Data Flow

1. User query enters through Discord interface
2. Query processed by Agent API
3. Orchestrator coordinates agent workflow
4. Specialized agents perform tasks (search, analysis, digest)
5. Results aggregated and formatted
6. Response returned through Discord interface

## Configuration Structure

```python
{
    "discord": {
        "token": "DISCORD_TOKEN",
        "page_size": 100,
        "rate_limit_delay": 1.0,
        "max_retries": 3
    },
    "llm": {
        "endpoint": "http://localhost:11434/api/generate",
        "model": "llama3.1:8b",
        "max_tokens": 2048,
        "temperature": 0.1,
        "timeout": 30,
        "retry_attempts": 3,
        "fallback_model": "llama2:latest"
    },
    "data": {
        "vector_config": {
            "persist_directory": "./data/chromadb",
            "collection_name": "discord_messages",
            "embedding_model": "msmarco-distilbert-base-v4",
            "embedding_type": "sentence_transformers",
            "batch_size": 100
        },
        "memory_config": {
            "database_url": "sqlite:///data/conversation_memory.db",
            "max_history": 50
        },
        "cache_config": {
            "cache_dir": "./data/cache",
            "default_ttl": 3600,
            "max_size_mb": 1000
        },
        "analytics_config": {
            "database_url": "sqlite:///data/analytics.db",
            "track_queries": True,
            "track_performance": True
        }
    },
    "processing": {
        "batch_size": 100,
        "max_retries": 3,
        "rate_limit_delay": 1.0,
        "enable_ai_classification": True,
        "preserve_legacy_patterns": True
    },
    "interfaces": {
        "discord_enabled": True,
        "api_enabled": True,
        "streamlit_enabled": True,
        "api_port": 8000,
        "streamlit_port": 8501
    }
}
```

## Dependencies

### Core
- Python 3.11
- Poetry (dependency management)
- discord.py (Discord bot framework)
- ChromaDB (vector database)
- LangGraph (workflow orchestration)
- SQLAlchemy (database ORM)

### AI/ML
- sentence-transformers (embeddings)
- OpenAI (optional cloud embeddings)
- LangChain (AI framework)
- FAISS (vector similarity)

### Development
- pytest (testing)
- Black (formatting)
- MyPy (type checking)
- Flake8 (linting)

### Utilities
- streamlit (web interface)
- plotly (visualization)
- pandas (data processing)
- tqdm (progress bars)
- python-dotenv (environment variables) 