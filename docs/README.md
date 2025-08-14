# Documentation

This directory contains documentation for the Discord RAG Bot with Agentic Architecture.

## Structure

- **[AGENTIC_ARCHITECTURE.md](AGENTIC_ARCHITECTURE.md)** - Simple guide to how the agentic system works
- **[OPERATIONS.md](OPERATIONS.md)** - Complete operations and maintenance guide
- **[ORGANIZATION.md](ORGANIZATION.md)** - Project organization and structure
- **[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)** - Detailed project architecture
- **[guides/](guides/)** - Usage guides and examples
- **[setup/](setup/)** - Setup and deployment instructions

## Quick Start

For deployment and setup instructions, see:
- **[setup/DEPLOYMENT.md](setup/DEPLOYMENT.md)** - Complete deployment guide
- **[setup/QUICKSTART.md](setup/QUICKSTART.md)** - Quick start guide
- **[setup/DEPLOYMENT_CHECKLIST.md](setup/DEPLOYMENT_CHECKLIST.md)** - Deployment checklist

## Usage Examples

- **[guides/example_queries.md](guides/example_queries.md)** - Examples of supported queries and interactions

## System Architecture

The agentic system provides:
- Discord slash commands with intelligent responses
- RESTful API endpoints (see `agentic/interfaces/agent_api.py`)
- Multi-agent orchestration with LangGraph
- MCP SQLite integration for standardized database operations
- Conversation memory with SQLite storage
