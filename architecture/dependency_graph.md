# Dependency Graph

## Direct Dependencies (Import-based)

### `core/agent.py`
- Depends on:
  - `tools.tools` (direct)
  - `langchain` packages (direct)
  - `os` and `dotenv` (direct)

### `core/bot.py`
- Depends on:
  - `core.agent` (direct)
  - `discord` packages (direct)
  - `flask` (direct)
  - `os` and `dotenv` (direct)

### `core/rag_engine.py`
- Depends on:
  - `openai` (direct)
  - `core.agent` (direct)
  - Various utility functions (direct)

### `core/resource_detector.py`
- Depends on:
  - `openai` (direct)
  - URL processing utilities (direct)
  - Message processing utilities (direct)

### `core/classifier.py`
- Depends on:
  - `openai` (direct)
  - Regular expression utilities (direct)

### `core/app.py`
- Depends on:
  - `streamlit` (direct)
  - `core.agent` (direct)
  - Formatting utilities (direct)

### `core/batch_detect.py`
- Depends on:
  - Database models (direct)
  - Resource detection utilities (direct)
  - Message processing utilities (direct)

## Implicit Dependencies

### Environment Variables
- `OPENAI_API_KEY`: Used by multiple modules
- `DISCORD_TOKEN`: Used by `bot.py`
- `GPT_MODEL`: Used by multiple modules for OpenAI calls

### Database
- SQLAlchemy models referenced across multiple modules
- Alembic migrations for schema changes

### File System
- `data/resources/`: Used for logging and resource storage
- `index_faiss/`: Used for vector store operations
- `bot.log`: Used for application logging

## Circular Dependencies
- `core/agent.py` ↔ `tools/tools.py`: Potential circular import risk
- `core/rag_engine.py` ↔ `core/agent.py`: Potential circular import risk

## External Service Dependencies
- OpenAI API (used by multiple modules)
- Discord API (used by `bot.py`)
- Vector store (FAISS)
- SQL database 