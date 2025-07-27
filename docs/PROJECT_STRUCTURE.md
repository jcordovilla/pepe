# Project Structure

This document describes the architecture and organization of the agentic Discord bot system.

## Overview

The project follows a modular, agentic architecture with clear separation of concerns. The system uses Python 3.12+ and integrates MCP SQLite for standardized database operations.

## Core Architecture

```
discord-bot-agentic/
├── agentic/                    # Main application package
│   ├── agents/                 # Agent implementations
│   │   ├── v2/                # Latest agent versions
│   │   │   ├── qa_agent.py    # Question-answering agent
│   │   │   ├── digest_agent.py # Digest generation agent
│   │   │   ├── router_agent.py # Request routing agent
│   │   │   └── ...
│   │   └── base_agent.py      # Base agent class
│   ├── mcp/                   # Model Context Protocol
│   │   ├── mcp_server.py      # Legacy MCP server
│   │   ├── mcp_sqlite_server.py # MCP SQLite server
│   │   └── sqlite_query_service.py # SQLite service
│   ├── services/              # Core services
│   │   ├── service_container.py # Dependency injection
│   │   ├── llm_client.py      # LLM client
│   │   └── ...
│   ├── config/                # Configuration
│   │   └── modernized_config.py # Unified configuration
│   └── ...
├── data/                      # Data storage
├── docs/                      # Documentation
├── scripts/                   # Utility scripts
├── tests/                     # Test suite
└── pyproject.toml            # Poetry configuration
```

## Key Components

### Agents (`agentic/agents/`)

The system uses a multi-agent architecture where each agent has a specific role:

- **QA Agent**: Handles question-answering using RAG
- **Digest Agent**: Creates summaries and digests
- **Router Agent**: Routes requests to appropriate agents
- **Self-Check Agent**: Validates response quality
- **Trend Agent**: Analyzes trends and patterns

### MCP Integration (`agentic/mcp/`)

Model Context Protocol integration for standardized database operations:

- **MCPSQLiteServer**: Runs mcp-sqlite as subprocess
- **MCPServer**: Legacy MCP server implementation
- **SQLiteQueryService**: Direct SQLite operations

### Services (`agentic/services/`)

Core services providing shared functionality:

- **ServiceContainer**: Dependency injection and lifecycle management
- **UnifiedLLMClient**: LLM client with fallback support
- **DiscordService**: Discord API integration
- **SyncService**: Data synchronization

### Configuration (`agentic/config/`)

Unified configuration system:

- **modernized_config.py**: Centralized configuration with environment variable support
- Supports Python 3.12+ requirements
- MCP SQLite configuration
- LLM model configuration

## Technology Stack

### Core Technologies

- **Python**: 3.12+ (required for MCP SQLite)
- **Poetry**: Dependency management
- **SQLite**: Primary database
- **Ollama**: Local LLM models

### Key Dependencies

- **pydantic**: 2.11.5+ (MCP SQLite requirement)
- **aiosqlite**: 0.21.0+ (MCP SQLite requirement)
- **mcp-sqlite**: 0.1.0+ (standardized SQLite operations)
- **discord.py**: Discord API integration
- **langchain**: LLM framework

### LLM Models

- **Primary**: llama3.1:8b (complex tasks)
- **Fast**: phi3:mini (simple tasks)
- **Fallback**: llama2:latest (reliability)

## Data Flow

### Request Processing

1. **Input**: User query via Discord or API
2. **Routing**: Router agent determines appropriate handler
3. **Processing**: Specialized agent processes request
4. **Database**: MCP SQLite server handles data queries
5. **LLM**: Local models generate responses
6. **Output**: Formatted response returned to user

### Data Storage

- **SQLite Database**: `data/discord_messages.db`
- **MCP SQLite**: Standardized query interface
- **Cache**: Smart caching for performance
- **Memory**: Conversation context storage

## Configuration Management

### Environment Variables

Key configuration via `.env` file:

```bash
# Discord
DISCORD_TOKEN=your_token

# LLM Models
LLM_MODEL=llama3.1:8b
LLM_FAST_MODEL=phi3:mini
LLM_FALLBACK_MODEL=llama2:latest

# MCP SQLite
MCP_SQLITE_ENABLED=true
```

### Configuration Hierarchy

1. **Environment Variables**: Highest priority
2. **Configuration Files**: Default values
3. **Code Defaults**: Fallback values

## Development Workflow

### Setup

1. **Python 3.12+**: Required for MCP SQLite
2. **Poetry**: Install dependencies
3. **Ollama**: Install and configure models
4. **Environment**: Set up `.env` file

### Testing

- **Unit Tests**: Individual component testing
- **Integration Tests**: End-to-end workflow testing
- **Performance Tests**: Load and stress testing

### Deployment

- **Local Development**: Poetry environment
- **Production**: Containerized deployment
- **Monitoring**: Health checks and logging

## Migration Notes

### Python 3.12 Upgrade

The system has been upgraded to Python 3.12 to support MCP SQLite:

- ✅ All dependencies updated
- ✅ Compatibility verified
- ✅ Performance maintained

### MCP SQLite Integration

Standardized SQLite operations via MCP protocol:

- ✅ Subprocess architecture
- ✅ Natural language queries
- ✅ Schema introspection
- ✅ Backward compatibility

## Future Architecture

### Planned Enhancements

1. **Microservices**: Service decomposition
2. **Event Streaming**: Real-time processing
3. **Advanced Caching**: Distributed caching
4. **Monitoring**: Comprehensive observability

### Scalability Considerations

- **Horizontal Scaling**: Multiple bot instances
- **Database Sharding**: Distributed data storage
- **Load Balancing**: Request distribution
- **Caching Strategy**: Multi-level caching
