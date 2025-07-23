# LLM Configuration Guide

This guide explains how to configure and change the AI models used by the Discord bot agentic system.

## 🎯 Overview

The system uses a unified LLM configuration that ensures all agents and components use the same AI models specified in your `.env` file. This makes it easy to change models across the entire system by updating a few environment variables.

## 🔧 Environment Variables

### Core LLM Configuration

Add these variables to your `.env` file:

```bash
# Primary model for most operations
LLM_MODEL=llama3.1:8b

# Fast model for resource detection (lighter/faster)
LLM_FAST_MODEL=phi3:mini

# Fallback model if primary fails
LLM_FALLBACK_MODEL=llama2:latest

# Ollama endpoint
LLM_ENDPOINT=http://localhost:11434/api/generate

# Model parameters
LLM_MAX_TOKENS=2048
LLM_TEMPERATURE=0.1
LLM_TIMEOUT=30
LLM_RETRY_ATTEMPTS=3
```

### Default Values

If not specified in `.env`, the system uses these defaults:

- **Primary Model**: `llama3.1:8b` (recommended for most operations)
- **Fast Model**: `phi3:mini` (for resource detection)
- **Fallback Model**: `llama2:latest` (backup if primary fails)
- **Endpoint**: `http://localhost:11434/api/generate` (local Ollama server)

## 🚀 Changing AI Models

### Quick Model Change

To change the AI model across the entire system:

1. **Update your `.env` file**:
   ```bash
   LLM_MODEL=your_new_model_name
   ```

2. **Restart the bot** - all agents will automatically use the new model

### Example Model Changes

#### Switch to a Different Llama Model
```bash
LLM_MODEL=llama3.1:70b
```

#### Use a Specialized Model
```bash
LLM_MODEL=codellama:13b
```

#### Use a Fast Model for All Operations
```bash
LLM_MODEL=phi3:mini
LLM_FAST_MODEL=phi3:mini
```

#### Use a Cloud Model (if supported)
```bash
LLM_ENDPOINT=https://your-cloud-endpoint.com/api/generate
LLM_MODEL=your-cloud-model
```

## 🔍 Which Components Use Which Models

### Primary Model (`LLM_MODEL`)
Used by:
- **QueryInterpreterAgent** - Query interpretation and analysis
- **SearchAgent** - Search result ranking and reranking
- **AnalysisAgent** - Content analysis and insights
- **QAAgent** - Question answering and context generation
- **DigestAgent** - Summary and digest generation
- **TrendAgent** - Trend analysis and pattern detection
- **UnifiedLLMClient** - All general LLM operations

### Fast Model (`LLM_FAST_MODEL`)
Used by:
- **ResourceDetector** - Fast resource analysis and classification
- **Performance-critical operations** - When speed is prioritized over quality

### Fallback Model (`LLM_FALLBACK_MODEL`)
Used by:
- **UnifiedLLMClient** - Automatic fallback when primary model fails
- **Error recovery** - When primary model is unavailable

## 🛠️ Configuration Architecture

### Unified Configuration System

The system uses a centralized configuration approach:

```python
# All components get config from the same source
from agentic.config.modernized_config import get_modernized_config

config = get_modernized_config()
llm_config = config.get("llm", {})

# Components use the unified config
model = llm_config.get("model", "llama3.1:8b")
```

### Component Integration

Each component automatically uses the configuration:

1. **UnifiedLLMClient** - Main LLM interface
2. **Agents** - All agentic components
3. **Scripts** - Resource detection and utilities
4. **Tests** - Performance evaluation and testing

## 📊 Model Performance Considerations

### Model Selection Guidelines

| Use Case | Recommended Model | Reason |
|----------|------------------|---------|
| **General Operations** | `llama3.1:8b` | Good balance of quality and speed |
| **Resource Detection** | `phi3:mini` | Fast processing of many resources |
| **Complex Analysis** | `llama3.1:70b` | Higher quality for complex tasks |
| **Code Analysis** | `codellama:13b` | Specialized for code understanding |
| **Fallback** | `llama2:latest` | Reliable backup option |

### Performance Trade-offs

- **Larger models** (70b) = Better quality, slower response
- **Smaller models** (mini) = Faster response, lower quality
- **Specialized models** = Better for specific tasks

## 🔧 Advanced Configuration

### Custom Model Parameters

You can override model parameters for specific operations:

```python
# In your agent code
response = await self.llm_client.generate(
    prompt=prompt,
    max_tokens=4096,  # Override default
    temperature=0.2   # Override default
)
```

### Model-Specific Configuration

For advanced users, you can add model-specific settings:

```bash
# In your .env file
LLM_MODEL_SPECIFIC_PARAMS={"context_length": 8192, "top_p": 0.9}
```

## 🧪 Testing Configuration

### Verify Configuration

Run this command to verify all components use the same configuration:

```bash
poetry run python -c "
from agentic.config.modernized_config import get_modernized_config
config = get_modernized_config()
print('Model:', config['llm']['model'])
print('Fast Model:', config['llm']['fast_model'])
print('Fallback Model:', config['llm']['fallback_model'])
"
```

### Test Model Availability

Check if your configured model is available:

```bash
# Test Ollama models
ollama list

# Test specific model
ollama show llama3.1:8b
```

## 🚨 Troubleshooting

### Common Issues

1. **Model Not Found**
   ```bash
   # Install the model
   ollama pull llama3.1:8b
   ```

2. **Inconsistent Models**
   - Check that all components use the unified config
   - Verify `.env` file is loaded correctly

3. **Performance Issues**
   - Try a smaller/faster model
   - Adjust `LLM_TIMEOUT` and `LLM_RETRY_ATTEMPTS`

### Debug Configuration

Enable debug logging to see which models are being used:

```bash
# In your .env file
LOG_LEVEL=DEBUG
```

## 📝 Best Practices

1. **Use Environment Variables** - Never hardcode model names
2. **Test Before Deploying** - Verify new models work in your environment
3. **Monitor Performance** - Track response times and quality
4. **Have Fallbacks** - Always configure a fallback model
5. **Document Changes** - Keep track of model changes and their impact

## 🔄 Migration Guide

### From Hardcoded Models

If you have hardcoded model references:

1. **Replace hardcoded values** with config lookups
2. **Update `.env` file** with your preferred models
3. **Test all components** to ensure consistency
4. **Remove hardcoded fallbacks** - use unified config

### Example Migration

```python
# Before (hardcoded)
model = "llama3.1:8b"

# After (config-based)
from agentic.config.modernized_config import get_modernized_config
config = get_modernized_config()
model = config.get("llm", {}).get("model", "llama3.1:8b")
```

---

**Need Help?** Check the main documentation or open an issue for configuration problems. 