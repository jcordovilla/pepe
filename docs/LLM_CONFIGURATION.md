# LLM Configuration

This document describes the LLM (Large Language Model) configuration for the agentic Discord bot system.

## Overview

The system uses local LLM models via Ollama for all AI operations, including:
- Natural language query processing
- Message summarization and analysis
- Content generation and formatting
- Quality validation and self-checking

## Python 3.12 Requirements

The system now requires **Python 3.12** or higher to support the MCP SQLite integration and latest dependencies.

### Key Dependencies
- **Python**: 3.12+
- **pydantic**: 2.11.5+ (required for MCP SQLite)
- **aiosqlite**: 0.21.0+ (required for MCP SQLite)
- **mcp-sqlite**: 0.1.0+ (for standardized SQLite operations)

## Model Configuration

### Primary Models

The system supports multiple local models with automatic fallback:

```python
"llm": {
    "endpoint": "http://localhost:11434/api/generate",
    "model": "llama3.1:8b",           # Primary model for complex tasks
    "fast_model": "phi3:mini",        # Fast model for simple tasks
    "fallback_model": "llama2:latest" # Fallback if primary fails
}
```

### Recommended Models

1. **llama3.1:8b** - Best balance of quality and performance
2. **phi3:mini** - Fast inference for simple tasks
3. **llama2:latest** - Reliable fallback option

### Model Parameters

```python
"llm": {
    "max_tokens": 2048,        # Maximum response length
    "temperature": 0.1,        # Low temperature for consistent results
    "timeout": 120,            # Request timeout in seconds
    "retry_attempts": 3        # Number of retry attempts
}
```

## Environment Variables

Configure models via environment variables:

```bash
# Model selection
LLM_MODEL=llama3.1:8b
LLM_FAST_MODEL=phi3:mini
LLM_FALLBACK_MODEL=llama2:latest

# Ollama endpoint
LLM_ENDPOINT=http://localhost:11434/api/generate

# Model parameters
LLM_MAX_TOKENS=2048
LLM_TEMPERATURE=0.1
LLM_TIMEOUT=120
LLM_RETRY_ATTEMPTS=3
```

## MCP SQLite Integration

The MCP SQLite server uses the same LLM configuration for natural language to SQL translation:

```python
"mcp_sqlite": {
    "enabled": True,
    "database_path": "data/discord_messages.db",
    "enable_write": False,
    "metadata_path": None,
    "verbose": False
}
```

## Agent-Specific Configuration

### QA Agent
- Uses primary model for complex query processing
- Falls back to fast model for simple searches
- Implements retry logic with fallback models

### Digest Agent
- Uses primary model for summarization
- Employs fast model for content classification
- Implements batch processing with model selection

### Self-Check Agent
- Uses primary model for quality validation
- Implements factuality and relevance scoring
- Provides confidence metrics for responses

## Performance Optimization

### Model Selection Strategy
1. **Simple tasks** → Fast model (phi3:mini)
2. **Complex tasks** → Primary model (llama3.1:8b)
3. **Fallback scenarios** → Fallback model (llama2:latest)

### Caching
- Query results cached to reduce model calls
- Context information cached for reuse
- Model responses cached for similar queries

### Batch Processing
- Multiple queries processed in batches
- Model context shared across related operations
- Efficient token usage through batching

## Troubleshooting

### Common Issues

1. **Model Not Found**
   ```
   Error: Model 'llama3.1:8b' not found
   ```
   Solution: Install model via `ollama pull llama3.1:8b`

2. **Timeout Errors**
   ```
   Error: Request timeout after 120 seconds
   ```
   Solution: Increase `LLM_TIMEOUT` or use faster model

3. **Memory Issues**
   ```
   Error: CUDA out of memory
   ```
   Solution: Use smaller model or reduce batch size

### Debug Mode

Enable verbose logging for troubleshooting:

```python
"llm": {
    "verbose": True,
    "debug_mode": True
}
```

## Migration from OpenAI

The system has been completely migrated from OpenAI to local models:

- ✅ All OpenAI dependencies removed
- ✅ Local model integration complete
- ✅ MCP SQLite integration working
- ✅ Python 3.12 compatibility verified

## Future Enhancements

1. **Model Fine-tuning**: Custom models for Discord content
2. **Multi-model Ensemble**: Combine multiple model outputs
3. **Dynamic Model Selection**: AI-driven model choice
4. **Performance Monitoring**: Real-time model performance tracking 