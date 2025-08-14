# Pepe - Advanced Discord Bot with Agentic RAG Architecture

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Poetry](https://img.shields.io/badge/dependency%20management-poetry-60A5FA.svg)](https://python-poetry.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![MCP SQLite](https://img.shields.io/badge/MCP-SQLite-green.svg)](https://github.com/microsoft/mcp)

**Pepe** is an intelligent Discord bot powered by advanced agentic RAG (Retrieval-Augmented Generation) architecture. It transforms Discord conversations into actionable insights through natural language queries, automated analysis, and multi-agent orchestration. Whether you're managing a community, analyzing discussions, or extracting knowledge from conversations, Pepe provides deep intelligence about user behavior, conversation patterns, and community dynamics.

> **🚀 New to Pepe?** Jump straight to the Discord Bot Quick Start Guide for immediate hands-on testing!

## 🎯 What Pepe Does

### **Core Purpose**

Pepe analyzes Discord server activity to help you:

* **Understand Your Community**: Who are your most active members? When is your server busiest?
* **Extract Knowledge**: Find specific discussions, resources, and insights from message history
* **Generate Automated Digests**: Get weekly summaries with engagement analysis
* **Natural Language Queries**: Ask questions in plain English
* **Make Data-Driven Decisions**: Use real insights instead of gut feelings

### **Advanced Features**

* **🤖 Multi-Agent Architecture**: Specialized agents for search, analysis, and digest generation
* **🧠 AI-Powered Resource Detection**: Automatically extract and categorize links and resources
* **📊 Real-time Analytics**: Performance monitoring and query tracking
* **⚡ MCP SQLite Integration**: Standardized database operations with natural language queries
* **🌐 Multiple Interfaces**: Discord bot, web dashboard, REST API
* **🐍 Python 3.12**: Latest language features and performance improvements

## 🤝 Community & Contributing

Hey there! 👋 I'm José, and I built Pepe to help Discord communities get more insights from their conversations. This project has evolved from a simple bot to a full agentic RAG system, and I'd love your help making it even better!

### **Want to Contribute?**

**Simple stuff first:**
- Found a bug? Open an issue with clear steps to reproduce
- Have an idea? Let's discuss it in GitHub Discussions
- Want to help with docs? That's always appreciated!

**For code contributions:**
1. Fork the repo and create a feature branch
2. Keep it simple - one feature per PR
3. Add tests if you're adding new functionality
4. Use black for formatting (just run `black .` before committing)

**What I'm looking for:**
- Bug fixes and performance improvements
- Better error handling and user experience
- Documentation improvements
- Ideas for new features that would help Discord communities

**Questions?** Just ask! I'm pretty responsive and love discussing ideas. The goal is to make Discord communities more insightful and engaging.

---

## 🚀 Discord Bot Quick Start Guide

**For colleagues testing the bot for the first time** 🤖

This is a simple guide to try out the Discord bot commands. No setup needed - just type and see what happens!

### 🚀 Getting Started

1. **Find the bot** in your Discord server (look for "Pepe" in the member list)
2. **Type `/pepe`** in any channel to see available commands
3. **Try the commands below** - they all start with `/pepe`

---

### 📋 Essential Commands to Try

#### 🏠 **Server Overview**

```
/pepe what discussions happened today?
/pepe show me recent activity in #general
```

**What it shows**: Overall server stats, recent discussions, activity trends
**Good for**: Getting a bird's eye view of your Discord server

**Example output**: Recent messages, active users, discussion topics, engagement metrics

---

#### 🔍 **Semantic Search**

```
/pepe find messages about machine learning
/pepe what did people say about the project?
```

**What it shows**: Messages matching your query by meaning, not just keywords
**Good for**: Finding specific discussions and insights

**Example output**: Relevant messages with context, user attribution, timestamps

---

#### 📊 **Weekly Digests**

```
/pepe give me a weekly digest
/pepe summary of last week's discussions
```

**What it shows**: Automated summaries of recent activity with engagement analysis
**Good for**: Understanding trends and community health

**Example output**: Top discussions, most active users, engagement metrics, key topics

---

#### 👤 **User Analysis**

```
/pepe what did @username say about AI?
/pepe find shared resources from @username
```

**What it shows**: Deep dive into a specific user's activity and contributions
**Good for**: Understanding individual member behavior and expertise

**Example output**: User's messages, shared resources, discussion topics, activity patterns

---

#### 📈 **Channel Analysis**

```
/pepe analyze activity patterns in #general
/pepe what topics are discussed in #ai-research?
```

**What it shows**: Complete health check of specific channels
**Good for**: Understanding how well channels are performing

**Example output**: Message volume, top contributors, response times, engagement metrics

---

#### 🎯 **Resource Discovery**

```
/pepe find shared resources about Python
/pepe what links were shared in #resources?
```

**What it shows**: Automatically extracted and categorized links from conversations
**Good for**: Discovering valuable resources shared by the community

**Example output**: Categorized links with descriptions, sharing context, user attribution

---

### 🎨 What to Expect

#### **📊 Rich Analytics**

Some commands include detailed analytics showing:

* Daily activity patterns
* User engagement metrics
* Channel growth trends
* Discussion topic evolution

#### **🤖 Smart Analysis**

The bot automatically chooses the best analysis method:

* **Natural Language Queries** for finding specific content
* **MCP SQLite** for optimized database operations
* **AI-Powered Summarization** for digests

#### **⏱️ Response Times**

* Simple queries: Instant
* Search commands: 0.5-0.9 seconds
* Complex analysis: Up to 30 seconds
* Weekly digests: 1-2 minutes

---

### 💡 Pro Tips

#### **🔍 Use Natural Language**

* Ask questions naturally: "What did people say about the new feature?"
* Use time references: "last week", "yesterday", "this month"
* Mention users: "@username said something about..."

#### **📅 Try Different Time Periods**

```
/pepe what happened yesterday?
/pepe discussions from last week
/pepe activity this month
```

#### **🎯 Start Simple**

1. Try `/pepe what discussions happened today?` first
2. Then search for topics: `/pepe find messages about AI`
3. Then try user analysis: `/pepe what did @username say?`

#### **📱 Works Everywhere**

* Any channel (the bot will analyze the right data)
* Desktop and mobile Discord
* DMs with the bot

---

### 🤔 Common Questions

**Q: "No data found" - what's wrong?**
A: The bot needs to sync data first. Ask the admin to run a sync, or try a different channel/user.

**Q: Search is taking forever?**
A: Large servers take longer. Try being more specific or use time limits like "last week".

**Q: Can I break anything?**
A: Nope! These are read-only commands. You're just viewing data, not changing anything.

**Q: Commands not showing up?**
A: Make sure you type `/pepe` and the bot has proper permissions.

---

### 🎯 Quick Command Cheat Sheet

| What you want to know                | Command to use                                    |
| ------------------------------------ | ------------------------------------------------- |
| "What happened today?"               | /pepe what discussions happened today?            |
| "Find AI discussions"                | /pepe find messages about AI                      |
| "Weekly summary"                     | /pepe give me a weekly digest                     |
| "What did Alice say?"                | /pepe what did @alice say about the project?      |
| "Channel activity"                   | /pepe analyze activity patterns in #general       |
| "Shared resources"                   | /pepe find shared resources about Python          |

---

### 🚀 Have Fun Exploring!

The bot is designed to be intuitive - just ask questions naturally and see what insights you discover about your Discord community!

**Questions?** Ask in the channel or DM the person who set up the bot.

---

_This bot analyzes your Discord conversations to provide insights about community activity, user behavior, and discussion topics. All analysis is based on message history the bot can access._

---

## 🏗️ Architecture Overview

### **Multi-Agent System**

Pepe uses a sophisticated multi-agent architecture:

* **🔍 Search Agent**: Handles semantic search and content discovery
* **📊 Analysis Agent**: Processes analytics and generates insights
* **🤖 Planning Agent**: Orchestrates complex multi-step queries
* **📝 Digest Agent**: Creates automated weekly summaries
* **🎯 Query Interpreter**: Understands natural language requests

### **Technology Stack**

* **LangGraph**: Multi-agent orchestration and workflow management
* **MCP SQLite**: Standardized database operations with natural language queries
* **Ollama (Llama)**: Local LLM for analysis and generation
* **Python 3.12**: Latest language features and performance improvements
* **SQLite**: Message storage and metadata
* **Discord.py**: Bot interface and message handling

## 🚀 Installation & Setup

### **Prerequisites**

* Python 3.12+
* Poetry (for dependency management)
* Discord Bot Token
* Ollama (for local Llama models)
* 4GB+ RAM recommended

### **Install Poetry**

If you don't have Poetry installed:

```bash
# Install Poetry
curl -sSL https://install.python-poetry.org | python3 -

# Or with pip
pip install poetry
```

### **Quick Setup**

1. **Clone and Setup**:
   ```bash
   git clone https://github.com/jcordovilla/pepe
   cd pepe
   poetry install
   ```

2. **Configure Environment**:
   ```bash
   cp .env.example .env
   # Edit .env with your tokens:
   # DISCORD_TOKEN=your_discord_bot_token
   # LLM_MODEL=llama3.1:8b
   # LLM_FAST_MODEL=phi3:mini
   # GUILD_ID=your_discord_server_id
   ```

3. **Initialize System**:
   ```bash
   poetry run ./pepe-admin setup
   poetry run ./pepe-admin sync --full  # First time: full sync
   ```

4. **Start the Bot**:
   ```bash
   poetry run python main.py
   ```

### **Admin Commands**

```bash
# System management (always use Poetry)
poetry run ./pepe-admin info              # Check system status
poetry run ./pepe-admin setup             # Initial setup
poetry run ./pepe-admin sync              # Incremental sync (new messages only)
poetry run ./pepe-admin sync --full       # Full sync (all messages)
poetry run ./pepe-admin sync --fetch-only # Only fetch messages
poetry run ./pepe-admin sync --index-only # Only index messages
poetry run ./pepe-admin resources         # Process shared resources
poetry run ./pepe-admin maintain          # System maintenance
poetry run ./pepe-admin test              # Run tests

# Alternative: Activate Poetry shell first
poetry shell
./pepe-admin info
./pepe-admin setup
# ... then run commands directly
```

## 🔧 Configuration

### **Environment Variables**

```bash
# Required
DISCORD_TOKEN=your_discord_bot_token
LLM_MODEL=llama3.1:8b
LLM_FAST_MODEL=phi3:mini
GUILD_ID=your_discord_server_id

# Optional
ENABLE_CHARTS=true              # Enable chart generation
MAX_MESSAGES=10000              # Limit message processing
ANALYSIS_TIMEOUT=300            # Analysis timeout in seconds
ENABLE_RESOURCE_DETECTION=true  # Enable AI resource detection
MCP_SQLITE_ENABLED=true         # Enable MCP SQLite integration
```

### **Performance Tuning**

For large servers:

```bash
# Optimize database
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

# Limit analysis scope
MAX_MESSAGES=10000
ANALYSIS_TIMEOUT=300

# Use incremental sync (default behavior)
./pepe-admin sync
```

## 📚 Documentation

### **Guides**

* **[Poetry Workflow Guide](docs/setup/POETRY_WORKFLOW.md)** - Complete Poetry setup and usage
* **[Incremental Sync Guide](docs/setup/INCREMENTAL_SYNC_GUIDE.md)** - Efficient message syncing
* **[Model Comparison Guide](docs/setup/MODEL_COMPARISON_GUIDE.md)** - Fast vs Standard AI models
* **[Logging and Query Tracking Guide](docs/setup/LOGGING_AND_QUERY_TRACKING.md)** - User query logging and analytics
* **[Operations Guide](docs/OPERATIONS.md)** - Complete setup and configuration
* **[Architecture Guide](docs/AGENTIC_ARCHITECTURE.md)** - Technical implementation details
* **[Testing Guide](tests/README.md)** - Development and testing procedures

### **API Reference**

* **[Agent API](docs/AGENTIC_ARCHITECTURE.md)** - Multi-agent system documentation
* **[Discord Interface](docs/OPERATIONS.md)** - Bot command reference
* **[Analytics API](docs/AGENTIC_ARCHITECTURE.md)** - Data analysis endpoints

## 🎉 What's New

### **Latest Features**

* **🔄 Incremental Fetching**: Only sync new messages (1-5 min vs 10-30 min)
* **🧠 Multi-Agent Architecture**: Specialized agents for different tasks
* **📊 Automated Weekly Digests**: AI-generated summaries with engagement metrics
* **⚡ MCP SQLite Integration**: Standardized database operations with natural language queries
* **🎯 AI-Powered Resource Detection**: Automatic link extraction and categorization
* **🏗️ Production-Ready**: Clean codebase with comprehensive error handling

### **Recent Improvements**

* **Performance**: 10x faster processing with MCP SQLite integration
* **Sync Speed**: Incremental fetching reduces sync time by 80-90%
* **Accuracy**: Better semantic search with enhanced database operations
* **Usability**: Natural language queries and intelligent responses
* **Reliability**: Comprehensive test suite and error handling
* **Documentation**: Complete guides for all user types

---

**Ready to unlock insights from your Discord community?** Start with the quick setup and explore what Pepe can reveal about your server! 🚀

---

Created by Jose Cordovilla with vibecoding tools like Github Copilot and Cursor and a lot of dedication. Jose is Volunteer Network Architect at MIT Professional Education's GenAI Global Community
