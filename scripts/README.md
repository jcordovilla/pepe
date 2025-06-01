# Scripts Directory

This directory contains utility scripts for testing, validation, and deployment of the Agentic Discord Bot.

## Available Scripts

### 🧪 `test_system.py`
Comprehensive test suite for the agentic Discord bot system.

**Usage:**
```bash
cd /path/to/discord-bot-v2
python3 scripts/test_system.py
```

**Features:**
- Tests all core components (memory, cache, agents, interfaces)
- Validates system integration and end-to-end functionality
- Provides detailed test results and success/failure reporting
- 100% test coverage of agentic framework

### ✅ `validate_deployment.py`
Pre-deployment validation script to ensure system readiness.

**Usage:**
```bash
cd /path/to/discord-bot-v2
python3 scripts/validate_deployment.py
```

**Checks:**
- Environment variables (OPENAI_API_KEY, DISCORD_TOKEN, GUILD_ID)
- Python dependencies and package versions
- File structure and required directories
- Core system component importability
- Configuration validity

## Running Scripts

All scripts should be run from the project root directory to ensure proper path resolution:

```bash
# From project root
cd /Users/jose/Documents/apps/discord-bot-v2

# Run tests
python3 scripts/test_system.py

# Validate deployment readiness
python3 scripts/validate_deployment.py
```

## Integration with Main System

These scripts are designed to work with the agentic framework and will:
- Automatically create necessary data directories
- Use the same configuration as the main application
- Provide detailed logging and error reporting
- Maintain compatibility with the production environment
