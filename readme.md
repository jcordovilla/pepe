# Discord Bot with RAG and Vector Search
# Version: beta-03

This project is a Discord bot that leverages Retrieval-Augmented Generation (RAG), vector search (using FAISS), and advanced message storage for enhanced chat interactions and AI-powered features.

---

## Project Structure
```
mkdocs.yml                # MkDocs documentation config
readme.md                 # Project documentation (this file)
requirements.txt          # Python dependencies

core/                     # Core logic and orchestration
    __init__.py           # Makes core a package
    agent.py              # AI agent orchestration
    app.py                # Streamlit UI / bot runner
    classifier.py         # Message classification logic
    rag_engine.py         # RAG (Retrieval-Augmented Generation) core logic
    repo_sync.py          # Repo synchronization logic
    resource_detector.py  # Resource detection utilities

data/                     # Data and logs
    chat_history.jsonl    # Query and chat history log
    discord_messages.db   # SQLite database file (used by the app)
    discord_messages.json # Exported Discord messages
    discord_messages_v2.json # Exported Discord messages (v2)

index_faiss/              # FAISS vector index files
    index.faiss
    index.pkl

db/                       # Database models and migrations
    __init__.py           # Makes db a package
    db.py                 # Database session management, engine, and models
    models.py             # Data models
    alembic.ini           # Alembic config
    alembic/              # Alembic migrations
        env.py
        README
        script.py.mako
        versions/
            5fcb4fdc70c5_create_resources_table.py

docs/                     # Project documentation
    index.md
    resources/

jc_logs/                  # Performance and architecture logs
    2025-05-10-Perf-Simplify-Tools.md
    2025-05-11-evaluation-after-test2.md
    2025-05-11-Perf-after-eval2-&UI.md
    2025-05-11-project-architecture.md

tests/                    # Unit and integration tests, test runners, and results
    test_migration.py
    test_queries.py
    test_summarizer.py
    test_time_parser.py
    test_utils.py
    query_test_results.json

tools/                    # Tooling and scripts for bot features
    __init__.py           # Makes tools a package
    batch_detect.py
    fetch_messages.py
    migrate_messages.py
    time_parser.py
    tools_metadata.py
    tools.py

utils/                    # Helper functions and logging
    __init__.py           # Makes utils a package
    embed_store.py
    helpers.py
    logger.py
```

---

## App Functionality

### Overview
This Discord bot enhances server interactions by combining message storage, semantic search, and AI-powered responses. It is designed for:
- **Storing and indexing Discord messages** in a local SQLite database (`data/discord_messages.db`).
- **Vectorizing messages** using OpenAI embeddings and storing them with FAISS for fast similarity search.
- **Retrieval-Augmented Generation (RAG):** When a user asks a question, the bot retrieves relevant messages using vector search, then sends both the user’s query and the retrieved context to an OpenAI GPT model to generate a context-aware response.
- **Summarization:** The bot can summarize conversations or message windows using advanced summarization tools.
- **Utility Tools:** Includes scripts for message migration, time parsing, resource detection, and more.
- **Modern Streamlit UI:** The app provides a user-friendly interface for searching, summarizing, and copying results, with a real copy-to-clipboard button.

### Key Features
- **Message Storage:** Listens to Discord messages and stores them with metadata (author, channel, timestamp, mentions, reactions).
- **Embedding & Vector Search:** Converts messages to vector embeddings for semantic search and retrieval.
- **RAG Pipeline:** Combines retrieved context with user queries for more accurate, context-aware answers.
- **Summarization:** Summarizes conversations or time windows using LLM-powered tools.
- **Migration & Utilities:** Includes scripts for migrating data, parsing time expressions, and detecting resources.
- **Streamlit UI:** Modern, responsive interface with copy-to-clipboard for results.
- **Modular Architecture:** Easy to extend, test, and maintain.
- **Resource Detector:** Identifies and extracts resources (such as links, files, or references) from Discord messages, tagging them with type, context, and metadata for downstream use or analytics. See `core/resource_detector.py`.
- **Classifier:** Uses custom or ML-based logic to classify messages or resources (e.g., by type, topic, or intent). This enables advanced filtering, analytics, and automation. See `core/classifier.py`.

### Main Frameworks and Models Used
- **SQLAlchemy:** ORM for database modeling and interaction with SQLite.
- **OpenAI GPT Models:** For generating responses, summarizing conversations, and creating embeddings.
- **FAISS:** Local vector database for fast similarity search and retrieval.
- **discord.py (implied):** For Discord bot integration and event handling.
- **Streamlit:** For the web UI.
- **Standard Python Libraries:** For logging, context management, and utilities.

---

## Setup
1. **Clone the repository:**
   ```sh
   git clone <your-repo-url>
   cd discord-bot
   ```
2. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
3. **Configure environment variables:**
   - Copy `.env` and fill in your `DISCORD_TOKEN`, `OPENAI_API_KEY`, etc.
4. **Run the app:**
   ```sh
   streamlit run core/app.py
   ```

---

## Scripts & Tools
- `tools/fetch_messages.py` — Fetches and stores Discord messages
- `tools/migrate_messages.py` — Migrates message data
- `tools/time_parser.py` — Natural language time parsing
- `tools/batch_detect.py` — Batch resource detection
- `tests/test_*.py` — Run tests with `pytest`

---

## Requirements
- Python 3.8+
- Discord bot token
- OpenAI API key

---

## Notes
- All data is stored locally in `data/discord_messages.db` (SQLite)
- Vector indices are stored in `index_faiss/`
- See `docs/` and `jc_logs/` for architecture and performance notes
- `.DS_Store` and other system files are ignored via `.gitignore`

---

## License
MIT License

## Author
Jose Cordovilla