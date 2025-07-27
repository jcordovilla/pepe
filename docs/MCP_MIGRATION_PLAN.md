# MCP Server Migration Plan

## Overview
This document outlines the migration from ChromaDB + SQLite metadata to an MCP (Model Context Protocol) server architecture with SQLite-based vector storage.

## Current Architecture Analysis

### ChromaDB Usage Patterns
The current system uses ChromaDB for:
1. **Semantic Search**: `similarity_search()` with OpenAI embeddings
2. **Metadata Storage**: 34+ metadata fields per message stored in ChromaDB
3. **Filtering**: Complex where clauses for temporal, author, channel filtering
4. **Batch Operations**: Upsert operations for message indexing
5. **Caching**: Smart cache integration for search results

### SQLite Database Structure
Current SQLite schema includes:
- `messages` table with comprehensive Discord message data
- `conversation_memory` for user context
- `query_answers` for analytics tracking
- Various metadata fields for filtering and analysis

## MCP Server Architecture Design

### 1. MCP Server Interface
The MCP server will provide these core operations:

```python
# Core MCP Server Methods
class MCPServer:
    async def generate_embedding(self, text: str) -> List[float]
    async def similarity_search(self, query: str, k: int, filters: Dict) -> List[Dict]
    async def batch_embed(self, texts: List[str]) -> List[List[float]]
    async def health_check(self) -> Dict[str, Any]
```

### 2. SQLite Vector Store Schema
New tables to support MCP server:

```sql
-- Message embeddings table
CREATE TABLE message_embeddings (
    message_id TEXT PRIMARY KEY,
    embedding_vector BLOB,  -- Store as binary blob
    embedding_model TEXT NOT NULL,
    content_hash TEXT,      -- For deduplication
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Search cache table
CREATE TABLE search_cache (
    query_hash TEXT PRIMARY KEY,
    results TEXT,           -- JSON results
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP,
    cache_hits INTEGER DEFAULT 0
);

-- Embedding model registry
CREATE TABLE embedding_models (
    model_name TEXT PRIMARY KEY,
    model_type TEXT NOT NULL,  -- 'openai', 'sentence_transformers', etc.
    dimensions INTEGER NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### 3. Migration Strategy

#### Phase 1: Foundation (Current)
- [x] Create MCP server architecture design
- [ ] Implement MCP server foundation
- [ ] Create SQLite-based vector store
- [ ] Implement embedding service

#### Phase 2: Core Implementation
- [ ] Replace ChromaDB operations with MCP server calls
- [ ] Update data manager to use new vector store
- [ ] Implement batch processing for embeddings
- [ ] Add caching layer for search results

#### Phase 3: Migration & Testing
- [ ] Create data migration script
- [ ] Test with subset of data
- [ ] Validate search accuracy and performance
- [ ] Update all dependent components

#### Phase 4: Cleanup
- [ ] Remove ChromaDB dependencies
- [ ] Update documentation
- [ ] Performance optimization
- [ ] Final validation

## Implementation Plan

### Step 1: MCP Server Foundation
Create the basic MCP server structure in `agentic/mcp/`:
- `mcp_server.py` - Main server implementation
- `mcp_client.py` - Client for communicating with server
- `embedding_service.py` - Handle embedding generation
- `search_service.py` - Implement semantic search

### Step 2: SQLite Vector Store
Create `agentic/vectorstore/sqlite_store.py` to replace `persistent_store.py`:
- Maintain same interface as ChromaDB implementation
- Use SQLite for metadata storage
- Use MCP server for embedding operations
- Implement all current search methods

### Step 3: Data Migration
Create migration scripts to:
- Export current ChromaDB data
- Generate embeddings for all messages
- Import into new SQLite schema
- Validate data integrity

### Step 4: Integration
Update all components to use new architecture:
- `unified_data_manager.py`
- `agent_api.py`
- Indexing scripts
- Test suites

## Benefits of MCP Architecture

1. **Simplified Dependencies**: Remove ChromaDB dependency
2. **Better Control**: Direct control over embedding generation
3. **Flexibility**: Easy to switch embedding models
4. **Performance**: Optimized SQLite queries for metadata
5. **Scalability**: MCP server can be scaled independently
6. **Cost Control**: Better control over API usage

## Risk Mitigation

1. **Backward Compatibility**: Maintain same API interfaces
2. **Gradual Migration**: Migrate one component at a time
3. **Extensive Testing**: Test with real data before full deployment
4. **Performance Monitoring**: Track response times and accuracy
5. **Rollback Plan**: Keep ChromaDB code available until validated

## Success Criteria

1. **Functional Parity**: All current features work identically
2. **Performance**: Response times within 10% of current system
3. **Accuracy**: Search results maintain same quality
4. **Reliability**: System stability maintained
5. **Cost Efficiency**: Reduced API costs through better caching 