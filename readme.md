# Discord Bot with RAG and Vector Search
# Version: beta-02

This project is a Discord bot that leverages Retrieval-Augmented Generation (RAG), vector search (using FAISS), and advanced message storage for enhanced chat interactions and AI-powered features.

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
- `docs/` — Project documentation and architecture notes

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
