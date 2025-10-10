# MCP SQLite Integration

This document describes the integration of the `mcp-sqlite` library into the agentic Discord bot system.

## Overview

The MCP SQLite integration provides a standardized approach to SQLite database queries using the Model Context Protocol (MCP) framework. This replaces the custom SQLite implementation with a more robust and feature-rich solution.

## Implementation Approach

The MCP SQLite server runs as a separate subprocess using the `mcp-sqlite` library's CLI interface. This approach provides:

- **Process Isolation**: The MCP server runs independently from the main application
- **Standardized Interface**: Uses the official MCP protocol for communication
- **Natural Language Queries**: Built-in natural language to SQL translation
- **Schema Introspection**: Automatic discovery of database structure
- **Query Optimization**: Built-in query optimization and caching

## Configuration

The MCP SQLite server is configured in `agentic/config/modernized_config.py`:

```python
"mcp_sqlite": {
    "enabled": True,
    "database_path": "data/discord_messages.db",
    "enable_write": False,
    "metadata_path": None,
    "verbose": False
}
```

### Configuration Options

- **enabled**: Enable/disable MCP SQLite server
- **database_path**: Path to the SQLite database file
- **enable_write**: Allow write operations (default: False for safety)
- **metadata_path**: Path to Datasette-compatible metadata file (optional)
- **verbose**: Enable verbose logging

## Usage

### In Agents

The MCP SQLite server is automatically used by agents when enabled:

```python
# In QA Agent or Digest Agent
await self._ensure_mcp_server_ready()
results = await self.mcp_server.query_messages("show me recent messages")
```

### Direct Usage

```python
from agentic.mcp import MCPSQLiteServer

# Initialize server
mcp_server = MCPSQLiteServer(config)

# Start the server
await mcp_server.start()

# Query messages
results = await mcp_server.query_messages("find messages about Python")

# Search with filters
results = await mcp_server.search_messages(
    query="discord bot",
    filters={"author_bot": False},
    limit=10
)

# Get schema information
schema = await mcp_server.get_schema_info()

# Health check
health = await mcp_server.health_check()

# Stop the server
await mcp_server.stop()
```

## Features

### Natural Language Queries

The MCP SQLite server can translate natural language queries to SQL:

```python
# These queries are automatically translated to SQL
await mcp_server.query_messages("show me 5 recent messages")
await mcp_server.query_messages("count all messages from today")
await mcp_server.query_messages("find messages about AI and machine learning")
```

### Text Search

Direct text search with filtering:

```python
results = await mcp_server.search_messages(
    query="discord bot",
    filters={"author_bot": False, "channel_id": 123456789},
    limit=20
)
```

### Schema Introspection

Get database structure information:

```python
schema = await mcp_server.get_schema_info()
# Returns table names and their SQL definitions
```

### Health Monitoring

Monitor server health and status:

```python
health = await mcp_server.health_check()
# Returns status: "healthy", "unhealthy", "stopped", or "error"
```

## Migration from Custom Implementation

The MCP SQLite integration maintains backward compatibility with the existing API:

- `query_messages()` - Natural language queries
- `search_messages()` - Text search with filters
- `get_schema_info()` - Database schema information
- `health_check()` - Server health status

## Benefits

1. **Standardized Protocol**: Uses the official MCP protocol
2. **Natural Language Support**: Built-in NL-to-SQL translation
3. **Process Isolation**: Runs as separate process for stability
4. **Schema Awareness**: Automatic schema discovery and optimization
5. **Query Optimization**: Built-in query optimization
6. **Error Handling**: Robust error handling and recovery

## Limitations

1. **Subprocess Overhead**: Requires starting/stopping subprocess
2. **Limited Customization**: Less control over query translation
3. **Python 3.12+ Required**: Requires Python 3.12 or higher
4. **Dependency Conflicts**: May conflict with existing pydantic/aiosqlite versions

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure Python 3.12+ and correct dependencies
2. **Process Start Failures**: Check database path and permissions
3. **Query Translation Issues**: Use simple, clear natural language
4. **Performance Issues**: Monitor subprocess resource usage

### Debug Mode

Enable verbose logging for debugging:

```python
"mcp_sqlite": {
    "enabled": True,
    "verbose": True
}
```

## Future Enhancements

1. **LLM Integration**: Use LLM for better natural language translation
2. **Connection Pooling**: Implement connection pooling for better performance
3. **Caching**: Add query result caching
4. **Advanced Filtering**: Support more complex filter operations 