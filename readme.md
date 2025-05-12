# Discord Bot with RAG and Vector Search
# Version: beta-02

This project is a Discord bot that leverages Retrieval-Augmented Generation (RAG), vector search (using FAISS), and advanced message storage for enhanced chat interactions and AI-powered features.

## Project Tree
```
agent.py                # AI agent orchestration
app.py                  # Main entry point (Streamlit UI / bot runner)
chat_history.jsonl      # Query and chat history log
db.py                   # Database models and session management
discord_messages_v2.json# Exported Discord messages (v2)
discord_messages.db     # SQLite database file
discord_messages.json   # Exported Discord messages
embed_store.py          # Embedding and vector store utilities
fetch_messages.py       # Script: fetch and store Discord messages
Leeme_JC.txt            # (Documentation or notes)
migrate_messages.py     # Script: migrate message data
models.py               # (Additional data models)
rag_engine.py           # RAG (Retrieval-Augmented Generation) core logic
readme.md               # Project documentation (this file)
requirements.txt        # Python dependencies
time_parser.py          # Natural language time parsing
tools_metadata.py       # Tool definitions and metadata
tools.py                # Core tools for message handling

docs/                   # Project documentation
  ├── 2025-05-10-Perf-Simplify-Tools.md
  ├── 2025-05-11-evaluation-after-test2.md
  ├── 2025-05-11-Perf-after-eval2-&UI.md
  └── 2025-05-11-project-architecture.md

index_faiss/            # FAISS vector index files
  ├── index.faiss
  └── index.pkl

tests/                  # Unit and integration tests, test runners, and results
  ├── test_migration.py
  ├── test_summarizer.py
  ├── test_time_parser.py
  ├── test_utils.py
  ├── test_queries.py         # Test runner for queries
  └── query_test_results.json # Test results for queries

utils/                  # Helper functions and logging
  ├── __init__.py
  ├── helpers.py
  └── logger.py
```

## How It Works (Plain Language)
1. **Message Storage:** The bot listens to Discord messages and stores them in a local SQLite database, saving details like author, channel, timestamp, mentions, and reactions.
2. **Embedding & Vector Search:** Messages are converted into vector embeddings using OpenAI models. These embeddings are stored locally with FAISS, allowing the bot to quickly find similar or relevant messages.
3. **Retrieval-Augmented Generation (RAG):** When a user asks a question, the bot retrieves relevant past messages using vector search, then sends both the user’s query and the retrieved context to an OpenAI GPT model to generate a context-aware response.
4. **Summarization & Tools:** The bot can summarize conversations, answer questions, and perform other tasks using a set of tools and utility functions.

## Tools Available for the AI Agent
- **Message Fetching:** Retrieve messages from the database by time, channel, or similarity.
- **Summarization:** Summarize a set of messages or a conversation thread.
- **Vector Search:** Find messages similar to a query using FAISS.
- **Migration Scripts:** Move or update message data as the schema evolves.
- **Utility Functions:** Helpers for formatting, logging, and parsing time.

## Main Frameworks and Models Used
- **SQLAlchemy:** ORM for database modeling and interaction with SQLite.
- **OpenAI GPT Models:** Used for generating responses, summarizing conversations, and creating embeddings for vector search.
- **FAISS:** Local vector database for fast similarity search and retrieval.
- **discord.py (implied):** For Discord bot integration and event handling.
- **Standard Python Libraries:** For logging, context management, and utilities.

## Features
- Stores Discord messages in a local SQLite database using SQLAlchemy ORM
- Supports message retrieval, summarization, and search
- Integrates with OpenAI GPT models for RAG and chat
- Uses FAISS for vector search and embedding storage
- Modular architecture for easy extension and testing

## Project Structure
- `app.py` — Main entry point for running the bot
- `db.py` — Database models and session management
- `agent.py`, `rag_engine.py` — Core logic for RAG and agent orchestration
- `embed_store.py` — Embedding and vector store utilities
- `fetch_messages.py`, `migrate_messages.py` — Scripts for data migration and message fetching
- `tools.py`, `tools_metadata.py` — Tooling and metadata for bot features
- `utils/` — Helper functions and logging
- `tests/` — Unit and integration tests

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

4. **Run the bot:**
   ```sh
   python app.py
   ```

## Scripts
- `fetch_messages.py` — Fetches and stores Discord messages
- `migrate_messages.py` — Migrates message data
- `test_*.py` (in `tests/`) — Run tests with `pytest`

## Requirements
- Python 3.8+
- Discord bot token
- OpenAI API key

## Notes
- All data is stored locally in `discord_messages.db` (SQLite)
- Vector indices are stored in `index_faiss/`
- See `docs/` for architecture and performance notes

## License
MIT License
