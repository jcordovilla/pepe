# Pepe - Advanced Discord Bot with Agentic RAG Architecture

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Poetry](https://img.shields.io/badge/dependency-poetry-blue.svg)](https://python-poetry.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**Pepe** is an intelligent Discord bot powered by advanced agentic RAG (Retrieval-Augmented Generation) architecture. It transforms Discord conversations into actionable insights through semantic search, automated analysis, and multi-agent orchestration. Whether you're managing a community, analyzing discussions, or extracting knowledge from conversations, Pepe provides deep intelligence about user behavior, conversation patterns, and community dynamics.

> **🚀 New to Pepe?** Jump straight to the Discord Bot Quick Start Guide for immediate hands-on testing!

## 🎯 What Pepe Does

### **Core Purpose**

Pepe analyzes Discord server activity to help you:

* **Understand Your Community**: Who are your most active members? When is your server busiest?
* **Extract Knowledge**: Find specific discussions, resources, and insights from message history
* **Generate Automated Digests**: Get weekly summaries with engagement analysis
* **Semantic Search**: Find content by meaning, not just keywords
* **Make Data-Driven Decisions**: Use real insights instead of gut feelings

### **Advanced Features**

* **🤖 Multi-Agent Architecture**: Specialized agents for search, analysis, and digest generation
* **🧠 AI-Powered Resource Detection**: Automatically extract and categorize links and resources
* **📊 Real-time Analytics**: Performance monitoring and query tracking
* **⚡ Streaming Processing**: 42.4 messages/second indexing with sub-second response times
* **🌐 Multiple Interfaces**: Discord bot, web dashboard, REST API

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

* **Semantic Search** for finding specific content
* **Pattern Recognition** for activity analysis
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
* **ChromaDB**: Vector database for semantic search
* **OpenAI GPT-4**: Primary LLM for analysis and generation
* **Ollama (Llama)**: Local model for resource detection
* **SQLite**: Message storage and metadata
* **Discord.py**: Bot interface and message handling

## 🚀 Installation & Setup

### **Prerequisites**

* Python 3.11+
* Discord Bot Token
* OpenAI API Key
* Ollama (for local Llama model)
* 4GB+ RAM recommended

### **Quick Setup**

1. **Clone and Setup**:
   ```bash
   git clone https://github.com/jcordovilla/pepe
   cd pepe
   pip install -r requirements.txt
   ```

2. **Configure Environment**:
   ```bash
   cp .env.example .env
   # Edit .env with your tokens:
   # DISCORD_TOKEN=your_discord_bot_token
   # OPENAI_API_KEY=your_openai_api_key
   # GUILD_ID=your_discord_server_id
   ```

3. **Initialize System**:
   ```bash
   ./pepe-admin setup
   ./pepe-admin sync --full
   ```

4. **Start the Bot**:
   ```bash
   python main.py
   ```

### **Admin Commands**

```bash
# System management
./pepe-admin info              # Check system status
./pepe-admin setup             # Initial setup
./pepe-admin sync              # Sync Discord messages
./pepe-admin resources         # Process shared resources
./pepe-admin maintain          # System maintenance
./pepe-admin test              # Run tests
```



## 🔧 Configuration

### **Environment Variables**

```bash
# Required
DISCORD_TOKEN=your_discord_bot_token
OPENAI_API_KEY=your_openai_api_key
GUILD_ID=your_discord_server_id

# Optional
ENABLE_CHARTS=true              # Enable chart generation
MAX_MESSAGES=10000              # Limit message processing
ANALYSIS_TIMEOUT=300            # Analysis timeout in seconds
ENABLE_RESOURCE_DETECTION=true  # Enable AI resource detection
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

# Use faster sync
./pepe-admin sync --incremental
```

## 📚 Documentation

### **Guides**

* **Operations Guide** - Complete setup and configuration
* **Bot Operations** - Discord bot management
* **Architecture** - Technical implementation details
* **Testing Guide** - Development and testing procedures

### **API Reference**

* **Agent API** - Multi-agent system documentation
* **Discord Interface** - Bot command reference
* **Analytics API** - Data analysis endpoints

## 🎉 What's New

### **Latest Features**

* **🧠 Multi-Agent Architecture**: Specialized agents for different tasks
* **📊 Automated Weekly Digests**: AI-generated summaries with engagement metrics
* **⚡ Streaming Processing**: 42.4 messages/second indexing
* **🎯 AI-Powered Resource Detection**: Automatic link extraction and categorization
* **🏗️ Production-Ready**: Clean codebase with comprehensive error handling

### **Recent Improvements**

* **Performance**: 10x faster processing with streaming indexer
* **Accuracy**: Better semantic search with enhanced embeddings
* **Usability**: Natural language queries and intelligent responses
* **Reliability**: Comprehensive test suite and error handling
* **Documentation**: Complete guides for all user types

---

**Ready to unlock insights from your Discord community?** Start with the quick setup and explore what Pepe can reveal about your server! 🚀

---

Made with vibecoding tools like Github Copilot and Cursor by Jose Cordovilla, Volunteer Network Architect at MIT Professional Education's GenAI Global Community
