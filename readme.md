# Discord Bot with RAG and Vector Search
# Version: beta-03

This project is a Discord bot that leverages Retrieval-Augmented Generation (RAG), vector search (using FAISS), advanced message storage, resource detection, and classification for enhanced chat interactions and AI-powered features.

---

## Key Features

- **Search & Summarize:** Query and summarize Discord messages using LLMs and vector search.
- **Resource Search:** Find and display links, files, and other resources shared in messages.
- **Classify:** Automatically classify messages or resources by type, topic, or intent.
- **Streamlit UI:** User-friendly web interface for searching, filtering, and copying results.
- **Discord Bot:** Interact with users in Discord channels, answer queries, detect resources, and classify content.
- **Batch Tools:** Scripts for fetching, migrating, and batch-processing messages and resources.

---

## Project Structure
```
mkdocs.yml                # MkDocs documentation config
readme.md                 # Project documentation (this file)
requirements.txt          # Python dependencies

core/                     # Core logic and orchestration
    __init__.py
    agent.py              # AI agent orchestration and Discord bot logic
    app.py                # Streamlit UI / bot runner
    classifier.py         # Message/resource classification logic
    resource_detector.py  # Resource detection utilities
    rag_engine.py         # Retrieval-Augmented Generation engine
    repo_sync.py          # Repo sync logic

tools/                    # Custom tools and scripts
    __init__.py
    tools.py              # Tool registry and main tool functions
    tools_metadata.py     # Tool metadata
    fetch_messages.py     # Fetch and store Discord messages
    migrate_messages.py   # Migrate message data
    batch_detect.py       # Batch resource detection
    time_parser.py        # Natural language time parsing

db/                       # Database models and migrations
    __init__.py
    db.py                 # Database session management, engine, and models
    models.py             # Data models
    alembic.ini           # Alembic config
    alembic/              # Alembic migrations

data/                     # Data files and vector indexes
    discord_messages.db   # Main SQLite database
    *.json, *.jsonl       # Message and chat history exports
    index_faiss/          # FAISS vector index files

utils/                    # Utility functions and helpers

tests/                    # Unit and integration tests

docs/                     # Project documentation

jc_logs/                  # Performance and architecture logs (gitignored)
```

---

## How to Run

1. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
2. **Prepare the database and data files** (see `tools/fetch_messages.py` and `tools/migrate_messages.py`).
3. **Configure environment variables:**
   - Copy `.env` and fill in your `DISCORD_TOKEN`, `OPENAI_API_KEY`, etc.
4. **Run the Streamlit app:**
   ```sh
   streamlit run core/app.py
   ```

---

## Scripts & Tools

- `tools/fetch_messages.py` — Fetches and stores Discord messages
- `tools/batch_detect.py` — **Batch resource detection** in messages
- `tools/time_parser.py` — Natural language time parsing
- `core/classifier.py` — **Classifies** messages/resources by type, topic, or intent (isolated run from batch_detect if needed. Otherwise, run batch_detect)
- `core/resource_detector.py` — **Detects resources** (links, files, etc.) in messages (isolated run from batch_detect if needed. Otherwise, run batch_detect)
- `tests/test_*.py` — Run tests with `pytest`

---

## Requirements

- Python 3.9+
- Discord API token
- OpenAI API key

---

## Notes

- The `jc_logs/` directory and `.DS_Store` files are ignored by git (see `.gitignore`).
- The main database is located at `data/discord_messages.db`.
- For advanced documentation, see the `docs/` folder or build with MkDocs.

---

**Author:**  
Jose Cordovilla