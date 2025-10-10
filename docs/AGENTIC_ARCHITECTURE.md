# Agentic Architecture - Simple Guide

This document explains how our Discord bot works in simple terms. Think of it as a team of specialized AI assistants working together to help you.

## 🏗️ How It Works (The Big Picture)

Imagine you have a team of experts, each with a specific job:

1. **The Receptionist** (Router Agent) - Figures out what you need and sends you to the right expert
2. **The Researcher** (Search Agent) - Finds relevant information from your Discord history
3. **The Analyst** (Analysis Agent) - Digs deep into topics and creates detailed reports
4. **The Summarizer** (Digest Agent) - Creates weekly summaries of channel activity
5. **The Planner** (Planning Agent) - Breaks down complex tasks into steps
6. **The Coordinator** (Orchestrator) - Makes sure everyone works together smoothly

### System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    DISCORD BOT AGENTIC SYSTEM                   │
│                        (Python 3.12 + MCP SQLite)              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │   Discord   │    │   REST API  │    │             │         │
│  │  Interface  │    │  Interface  │    │             │         │
│  └─────┬───────┘    └─────┬───────┘    └─────┬───────┘         │
│        │                  │                  │                 │
│        └──────────────────┼──────────────────┘                 │
│                           │                                    │
│                    ┌──────▼───────┐                            │
│                    │ ORCHESTRATOR │                            │
│                    │ (Coordinator)│                            │
│                    └──────┬───────┘                            │
│                           │                                    │
│        ┌──────────────────┼──────────────────┐                 │
│        │                  │                  │                 │
│  ┌─────▼──────┐    ┌──────▼──────┐    ┌─────▼──────┐           │
│  │   ROUTER   │    │   SEARCH    │    │  ANALYSIS  │           │
│  │  AGENT     │    │   AGENT     │    │   AGENT    │           │
│  │(Receptionist)│  │(Researcher) │    │(Analyst)   │           │
│  └─────┬──────┘    └──────┬──────┘    └─────┬──────┘           │
│        │                  │                  │                 │
│  ┌─────▼──────┐    ┌──────▼──────┐    ┌─────▼──────┐           │
│  │   DIGEST   │    │   PLANNING  │    │  PIPELINE  │           │
│  │   AGENT    │    │   AGENT     │    │   AGENT    │           │
│  │(Summarizer)│    │(Planner)    │    │(Processor) │           │
│  └────────────┘    └─────────────┘    └────────────┘           │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                        DATA LAYER                              │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │   MCP       │    │   SQLite    │    │   Memory    │         │
│  │ SQLite      │    │  Database   │    │   System    │         │
│  │(Smart Query)│    │(Metadata)   │    │(Conversation)│        │
│  └─────────────┘    └─────────────┘    └─────────────┘         │
└─────────────────────────────────────────────────────────────────┘
```

## 🔄 Typical Workflows

### Workflow 1: "Give me a weekly digest of #general-forum"

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│    USER     │───▶│   ROUTER    │───▶│   DIGEST    │───▶│   MCP       │
│   Request   │    │   AGENT     │    │   AGENT     │    │  SQLite     │
│             │    │(Receptionist)│    │(Summarizer) │    │(Smart Query)│
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                                                              │
                                                              ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│    USER     │◀───│ ORCHESTRATOR│◀───│   DIGEST    │◀───│  SQLite     │
│  Response   │    │(Coordinator)│    │   AGENT     │    │  Database   │
│             │    │             │    │(Summarizer) │    │             │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

**What happens behind the scenes:**
1. You type: "Give me a weekly digest of #general-forum"
2. The Router Agent recognizes this is a digest request
3. It calls the Digest Agent with the channel information
4. The Digest Agent uses MCP SQLite to query all messages from that channel in the past week
5. It groups messages by topic and creates a summary using local LLM models
6. You get a nicely formatted digest with all the important discussions

### Workflow 2: "What was discussed about AI safety last month?"

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│    USER     │───▶│   ROUTER    │───▶│  ANALYSIS   │───▶│   MCP       │
│   Request   │    │   AGENT     │    │   AGENT     │    │  SQLite     │
│             │    │(Receptionist)│    │  (Analyst)  │    │(Smart Query)│
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                                                              │
                                                              ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│    USER     │◀───│ ORCHESTRATOR│◀───│  ANALYSIS   │◀───│  SQLite     │
│  Response   │    │(Coordinator)│    │   AGENT     │    │  Database   │
│             │    │             │    │  (Analyst)  │    │             │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

**What happens behind the scenes:**
1. You type: "What was discussed about AI safety last month?"
2. The Router Agent recognizes this is an analysis request
3. It calls the Analysis Agent with your question
4. The Analysis Agent searches for messages about AI safety
5. It analyzes the content and creates a comprehensive report
6. You get a detailed analysis with insights and trends

### Workflow 3: "Help me understand the project structure"

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│    USER     │───▶│   ROUTER    │───▶│  PLANNING   │───▶│   MCP       │
│   Request   │    │   AGENT     │    │   AGENT     │    │  SQLite     │
│             │    │(Receptionist)│    │ (Planner)   │    │(Smart Query)│
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                                                              │
                                                              ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│    USER     │◀───│ ORCHESTRATOR│◀───│  PLANNING   │◀───│  SQLite     │
│  Response   │    │(Coordinator)│    │   AGENT     │    │  Database   │
│             │    │             │    │ (Planner)   │    │             │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

## 🧠 The Memory System

The bot remembers your conversations and preferences:

- **Short-term memory**: What you just talked about
- **Long-term memory**: Your preferences and past interactions
- **Context awareness**: It knows what you're working on

### Memory System Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    MEMORY SYSTEM                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────┐    ┌─────────────────┐                │
│  │  SHORT-TERM     │    │   LONG-TERM     │                │
│  │    MEMORY       │    │    MEMORY       │                │
│  │                 │    │                 │                │
│  │ • Current       │    │ • User          │                │
│  │   conversation  │    │   preferences   │                │
│  │ • Recent        │    │ • Past          │                │
│  │   context       │    │   interactions  │                │
│  │ • Active        │    │ • Summarized    │                │
│  │   topics        │    │   history       │                │
│  └─────────────────┘    └─────────────────┘                │
│           │                       │                        │
│           └───────────┬───────────┘                        │
│                       │                                    │
│              ┌────────▼────────┐                           │
│              │   CONTEXT       │                           │
│              │   AWARENESS     │                           │
│              │                 │                           │
│              │ • Knows what    │                           │
│              │   you're        │                           │
│              │   working on    │                           │
│              │ • Maintains     │                           │
│              │   conversation  │                           │
│              │   flow          │                           │
│              └─────────────────┘                           │
└─────────────────────────────────────────────────────────────┘
```

## 🔍 The Search System

The bot can find information in your Discord history using:

- **Natural language queries**: Ask questions in plain English
- **MCP SQLite**: Standardized database queries with schema introspection
- **Metadata filtering**: Searches by channel, date, user, etc.
- **Context understanding**: Knows what's relevant to your question

### Search System Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    SEARCH SYSTEM                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────┐    ┌─────────────────┐                │
│  │   NATURAL       │    │   MCP SQLITE    │                │
│  │   LANGUAGE      │    │   QUERIES       │                │
│  │   QUERIES       │    │                 │                │
│  │                 │    │ • Schema        │                │
│  │ • Ask questions │    │   introspection │                │
│  │   in plain      │    │ • Optimized     │                │
│  │   English       │    │   SQL queries   │                │
│  │ • Automatic     │    │ • Process       │                │
│  │   translation   │    │   isolation     │                │
│  │ • Context       │    │ • Error         │                │
│  │   awareness     │    │   recovery      │                │
│  └─────────────────┘    └─────────────────┘                │
│           │                       │                        │
│           └───────────┬───────────┘                        │
│                       │                                    │
│              ┌────────▼────────┐                           │
│              │   CONTEXT       │                           │
│              │ UNDERSTANDING   │                           │
│              │                 │                           │
│              │ • Knows what's  │                           │
│              │   relevant      │                           │
│              │ • Ranks results │                           │
│              │   by relevance  │                           │
│              │ • Combines      │                           │
│              │   multiple      │                           │
│              │   search types  │                           │
│              └─────────────────┘                           │
└─────────────────────────────────────────────────────────────┘
```

## 🎯 Key Concepts

### Agents
Each agent is like a specialist with a specific skill:
- **Router**: Traffic controller - decides which agent should handle your request
- **Search**: Librarian - finds the information you need
- **Analysis**: Researcher - digs deep and creates reports
- **Digest**: Journalist - creates summaries and digests
- **Planning**: Project manager - breaks down complex tasks
- **Orchestrator**: Team leader - coordinates everyone

### MCP SQLite Database
Think of this as a smart database:
- Stores all your Discord messages with full context
- Uses natural language queries to find relevant content
- Organized by standard database principles for fast access

### MCP SQLite Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    MCP SQLITE DATABASE                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │                 SQLITE DATABASE                          │ │
│  │                                                         │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │ │
│  │  │   MESSAGE   │  │   MESSAGE   │  │   MESSAGE   │     │ │
│  │  │   VECTOR    │  │   VECTOR    │  │   VECTOR    │     │ │
│  │  │             │  │             │  │             │     │ │
│  │  │ • Content   │  │ • Content   │  │ • Content   │     │ │
│  │  │ • Metadata  │  │ • Metadata  │  │ • Metadata  │     │ │
│  │  │ • Embedding │  │ • Embedding │  │ • Embedding │     │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘     │ │
│  │                                                         │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │ │
│  │  │   MESSAGE   │  │   MESSAGE   │  │   MESSAGE   │     │ │
│  │  │   VECTOR    │  │   VECTOR    │  │   VECTOR    │     │ │
│  │  │             │  │             │  │             │     │ │
│  │  │ • Content   │  │ • Content   │  │ • Content   │     │ │
│  │  │ • Metadata  │  │ • Metadata  │  │ • Metadata  │     │ │
│  │  │ • Embedding │  │ • Embedding │  │ • Embedding │     │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘     │ │
│  │                                                         │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │ │
│  │  │   MESSAGE   │  │   MESSAGE   │  │   MESSAGE   │     │ │
│  │  │   VECTOR    │  │   VECTOR    │  │   VECTOR    │     │ │
│  │  │             │  │             │  │             │     │ │
│  │  │ • Content   │  │ • Content   │  │ • Content   │     │ │
│  │  │ • Metadata  │  │ • Metadata  │  │ • Metadata  │     │ │
│  │  │ • Embedding │  │ • Embedding │  │ • Embedding │     │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘     │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                             │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │              EMBEDDING MODEL                            │ │
│  │              (msmarco-distilbert-base-v4)               │ │
│  │                                                         │ │
│  │  • Converts text to numerical vectors                   │ │
│  │  • Optimized for Discord content types                  │ │
│  │  • Supports technical discussions, Q&A, tutorials       │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### Memory
The bot's brain:
- Remembers what you've talked about
- Learns your preferences
- Provides context for better responses

## 🚀 What Makes This Special

1. **Teamwork**: Multiple specialized agents work together
2. **Intelligence**: Each agent is optimized for its specific task
3. **Memory**: The system remembers and learns from interactions
4. **Flexibility**: Can handle many different types of requests
5. **Context**: Understands the full context of your Discord server

### System Benefits Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    SYSTEM BENEFITS                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │  TEAMWORK   │  │INTELLIGENCE │  │   MEMORY    │         │
│  │             │  │             │  │             │         │
│  │ • Multiple  │  │ • Each agent│  │ • Remembers │         │
│  │   agents    │  │   optimized │  │   past      │         │
│  │   working   │  │   for task  │  │   chats     │         │
│  │   together  │  │ • Specialized│  │ • Learns    │         │
│  │ • Coordinated│  │   expertise │  │   preferences│        │
│  │   workflow  │  │ • Better    │  │ • Context   │         │
│  └─────────────┘  │   results   │  │   awareness │         │
│                   └─────────────┘  └─────────────┘         │
│                                                             │
│  ┌─────────────┐  ┌─────────────┐                          │
│  │ FLEXIBILITY │  │   CONTEXT   │                          │
│  │             │  │             │                          │
│  │ • Handles   │  │ • Understands│                          │
│  │   many      │  │   full      │                          │
│  │   request   │  │   Discord   │                          │
│  │   types     │  │   context   │                          │
│  │ • Scalable  │  │ • Knows     │                          │
│  │   system    │  │   community │                          │
│  │ • Adaptable │  │   dynamics  │                          │
│  └─────────────┘  └─────────────┘                          │
└─────────────────────────────────────────────────────────────┘
```

## 📝 Example Interactions

**User**: "Give me a weekly digest of #ai-discussions"
**Bot**: Creates a comprehensive summary of all AI-related discussions from the past week, grouped by topics like "AI Safety", "New Developments", "Community Questions", etc.

**User**: "What are the main concerns about AI regulation?"
**Bot**: Searches through all discussions, finds relevant messages, and creates a detailed analysis of the main concerns, supporting arguments, and community sentiment.

**User**: "Help me plan a community event about AI ethics"
**Bot**: Analyzes past events, community interests, and creates a step-by-step plan with recommendations based on what the community has discussed.

## 🔧 Technical Details (Simplified)

- **LangGraph**: The framework that coordinates all the agents
- **MCP SQLite**: The standardized database that stores and searches messages
- **SQLite**: The database that stores conversation history and metadata
- **Discord API**: How the bot connects to Discord
- **MCP Protocol**: How the bot communicates with the database

## 🎯 Why This Architecture?

This multi-agent approach allows the bot to:
- Handle complex requests that require multiple steps
- Provide specialized expertise for different types of tasks
- Scale and improve individual components independently
- Maintain context and memory across interactions
- Deliver more accurate and relevant responses

The result is a bot that feels intelligent and helpful, capable of understanding complex requests and providing valuable insights from your Discord community's discussions.

## 🔄 Complete Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           COMPLETE DATA FLOW                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐                                                           │
│  │   DISCORD   │                                                           │
│  │   SERVER    │                                                           │
│  └─────┬───────┘                                                           │
│        │                                                                   │
│        ▼                                                                   │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                    │
│  │   DISCORD   │───▶│   FETCHER   │───▶│   INDEXER   │                    │
│  │   FETCHER   │    │   SERVICE   │    │   SERVICE   │                    │
│  └─────────────┘    └─────────────┘    └─────────────┘                    │
│                              │                    │                        │
│                              ▼                    ▼                        │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                    │
│  │   SQLITE    │◀───│   UNIFIED   │───▶│   MCP       │                    │
│  │  DATABASE   │    │   DATA      │    │  SQLITE     │                    │
│  │             │    │  MANAGER    │    │             │                    │
│  └─────────────┘    └─────────────┘    └─────────────┘                    │
│        │                    │                    │                        │
│        └────────────────────┼────────────────────┘                        │
│                             │                                             │
│                             ▼                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    AGENTIC SYSTEM                                  │   │
│  │                                                                     │   │
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐            │   │
│  │  │   DISCORD   │    │   REST API  │    │             │            │   │
│  │  │ INTERFACE   │    │ INTERFACE   │    │             │            │   │
│  │  └─────┬───────┘    └─────┬───────┘    └─────┬───────┘            │   │
│  │        │                  │                  │                     │   │
│  │        └──────────────────┼──────────────────┘                     │   │
│  │                           │                                        │   │
│  │                    ┌──────▼───────┐                                │   │
│  │                    │ ORCHESTRATOR │                                │   │
│  │                    │              │                                │   │
│  │                    └──────┬───────┘                                │   │
│  │                           │                                        │   │
│  │        ┌──────────────────┼──────────────────┐                     │   │
│  │        │                  │                  │                     │   │
│  │  ┌─────▼──────┐    ┌──────▼──────┐    ┌─────▼──────┐              │   │
│  │  │   ROUTER   │    │   SEARCH    │    │  ANALYSIS  │              │   │
│  │  │   AGENT    │    │   AGENT     │    │   AGENT    │              │   │
│  │  └─────┬──────┘    └──────┬──────┘    └─────┬──────┘              │   │
│  │        │                  │                  │                     │   │
│  │  ┌─────▼──────┐    ┌──────▼──────┐    ┌─────▼──────┐              │   │
│  │  │   DIGEST   │    │   PLANNING  │    │  PIPELINE  │              │   │
│  │  │   AGENT    │    │   AGENT     │    │   AGENT    │              │   │
│  │  └────────────┘    └─────────────┘    └────────────┘              │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    MEMORY & CACHE                                   │   │
│  │                                                                     │   │
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐            │   │
│  │  │ CONVERSATION│    │   SMART     │    │   ANALYTICS │            │   │
│  │  │   MEMORY    │    │   CACHE     │    │   DASHBOARD │            │   │
│  │  └─────────────┘    └─────────────┘    └─────────────┘            │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Key Data Flow Steps:

1. **Data Ingestion**: Discord messages are fetched and indexed
2. **Storage**: Messages stored in SQLite with MCP SQLite integration for standardized operations
3. **Processing**: User requests flow through the agentic system
4. **Retrieval**: Agents query the MCP SQLite database
5. **Response**: Results are processed and returned to the user
6. **Memory**: Conversations and preferences are stored for context 