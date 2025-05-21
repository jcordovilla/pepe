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
    resource_detector.py  # Resource detection, enrichment, normalization, deduplication
    rag_engine.py         # Retrieval-Augmented Generation engine (FAISS, OpenAI)
    repo_sync.py          # Export resources to JSON/Markdown
    batch_detect.py       # Batch resource detection and enrichment
    fetch_messages.py     # Fetch and store Discord messages
    bot.py                # Discord bot entrypoint
    embed_store.py        # Embedding and vector store logic


tools/                    # Custom tools and scripts
    __init__.py
    tools.py              # Tool registry and main tool functions
    tools_metadata.py     # Tool metadata for agent/LLM tool-calling
    fetch_messages.py     # (Legacy/alt) Fetch Discord messages
    migrate_messages.py   # Migrate message data
    batch_detect.py       # (Legacy/alt) Batch resource detection
    time_parser.py        # Natural language time parsing
    clean_resources_db.py # Clean, deduplicate, and re-enrich resources in DB
    dedup_resources.py    # Deduplicate JSON resources by URL/title (CLI)
    fix_resource_titles.py# AI-based title/description enrichment for resources
    full_pipeline.py      # Run full pipeline (fetch, embed, detect, export)


db/                       # Database models and migrations
    __init__.py
    db.py                 # Database session management, engine, and models
    models.py             # Data models
    alembic.ini           # Alembic config
    alembic/              # Alembic migrations


data/                     # Data files and vector indexes
    discord_messages.db   # Main SQLite database
    resources/            # Resource logs and exports
    *.json, *.jsonl       # Message and chat history exports
    index_faiss/          # FAISS vector index files

utils/                    # Utility functions and helpers
    __init__.py
    helpers.py            # Helper functions (jump URLs, validation, etc.)
    logger.py             # Logging setup
    embed_store.py        # Embedding helpers

tests/                    # Unit and integration tests
    test_*.py             # Test modules (run with pytest)
    conftest.py           # Pytest fixtures
    query_test_results.json# Test results

docs/                     # Project documentation (Markdown, resources)
    index.md
    resources/
        resources.json    # Exported/curated resources

jc_logs/                  # Performance and architecture logs (gitignored)
```

---

## How to Run

1. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
2. **Prepare the database and data files:**
   - Fetch Discord messages: `python core/fetch_messages.py`
   - (Optional) Migrate or clean data: see `tools/migrate_messages.py`, `tools/clean_resources_db.py`
3. **Configure environment variables:**
   - Copy `.env` and fill in your `DISCORD_TOKEN`, `OPENAI_API_KEY`, etc.
4. **Run the Streamlit app:**
   ```sh
   streamlit run core/app.py
   ```
5. **(Optional) Run the Discord bot:**
   ```sh
   python core/bot.py
   ```
6. **(Optional) Run the full pipeline:**
   ```sh
   python tools/full_pipeline.py
   ```

---

## Requirements

- Python 3.9+
- Discord API token (`DISCORD_TOKEN`)
- OpenAI API key (`OPENAI_API_KEY`)
- (Optional) FAISS, Streamlit, SQLAlchemy, LangChain, TQDM, Prometheus, etc. (see `requirements.txt`)

---

## Notes

- The `jc_logs/` directory and `.DS_Store` files are ignored by git (see `.gitignore`).
- The main database is located at `data/discord_messages.db`.
- For advanced documentation, see the `docs/` folder or build with MkDocs (`mkdocs serve`).
- Test coverage: run `pytest` in the `tests/` directory.
- For troubleshooting, see logs in `jc_logs/` and `tools/full_pipeline.log`.

---

**Author:**  
Jose Cordovilla
GenAI Global Network Architect