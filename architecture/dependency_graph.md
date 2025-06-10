# 🔗 **Corrected System Architecture & Dependencies**
*Updated: June 10, 2025 - Based on Actual Code Analysis*

## 🎯 **Core System Flow (Production)**

```mermaid
graph TD
    %% Data Ingestion
    A[Discord Server] --> B[core/fetch_messages.py]
    B --> C[data/discord_messages.db]
    
    %% User Interfaces
    D[User Query] --> E{Interface}
    E -->|Web UI| F[core/app.py - Streamlit]
    E -->|Discord| G[core/bot.py - Discord Bot]
    
    %% Core Processing
    F --> H[core/agent.py - AI Agent]
    G --> H
    H --> I[core/rag_engine.py - RAG Engine]
    
    %% RAG Engine Dependencies
    I --> J[core/ai_client.py - Ollama]
    I --> K[tools/tools.py - Message Utils]
    I --> L[tools/time_parser.py - Time Parsing]
    I --> M[core/enhanced_fallback_system.py]
    I --> N[core/enhanced_k_determination.py]
    
    %% Data Sources
    K --> C
    I --> O[data/indices/ - FAISS Indices]
    
    %% Configuration
    J --> P[core/config.py]
    B --> P
```

## 🏗️ **Data Processing Pipeline**

```mermaid
graph LR
    %% Current Production Pipeline
    A[Raw Discord Messages] --> B[scripts/pipeline.py]
    B --> C[scripts/content_preprocessor.py]
    C --> D[Preprocessing & Filtering]
    D --> E[scripts/enhanced_faiss_index.py]
    E --> F[Vector Embeddings]
    F --> G[FAISS Index Creation]
    G --> H[data/indices/]
    
    %% Resource Processing
    A --> I[core/resource_detector.py]
    I --> J[Resource Extraction]
    J --> K[core/classifier.py]
    K --> L[Resource Classification]
    
    %% Enhanced Pipeline
    M[scripts/enhanced_pipeline_with_resources.py] --> C
    M --> I
```

## 🔍 **Actual Dependencies (Import Analysis)**

### **⭐ TIER 1: ESSENTIAL CORE**

#### **Main Applications**
- **`core/app.py`** (Streamlit Web UI)
  - → `tools.tools` (get_channels)
  - → `core.agent` (get_agent_answer, analyze_query_type)
  - → `db.query_logs` (logging functions)

- **`core/bot.py`** (Discord Bot)
  - → `core.agent` (get_agent_answer)
  - → `core.config` (get_config)
  - → `db.query_logs` (logging functions)

- **`core/fetch_messages.py`** (Data Ingestion) ⚠️ **CORRECTED**
  - → `db` (SessionLocal, Message)
  - → `discord` library
  - → `utils.logger`

#### **Core Intelligence**
- **`core/agent.py`** (AI Orchestrator)
  - → `core.ai_client` (get_ai_client)
  - → `core.rag_engine` (get_answer, get_agent_answer, etc.)
  - → `core.enhanced_fallback_system` (conditional import)
  - → `tools.tools` (search_messages, etc.)

- **`core/rag_engine.py`** (RAG Engine)
  - → `core.ai_client` (get_ai_client)
  - → `core.config` (get_config)
  - → `core.enhanced_fallback_system` (EnhancedFallbackSystem)
  - → `tools.tools` (resolve_channel_name, summarize_messages, etc.)
  - → `tools.time_parser` (parse_timeframe, extract_time_reference, etc.)

- **`core/ai_client.py`** (LLM Client)
  - → `core.config` (get_config)

- **`core/config.py`** (Configuration)
  - → Environment variables and settings

#### **Data Processing**
- **`scripts/pipeline.py`** (Main Pipeline)
  - → `scripts.content_preprocessor` (ContentPreprocessor, PreprocessingConfig)
  - → `scripts.enhanced_faiss_index` (EnhancedFAISSIndex, IndexConfig, SearchResult)
  - → `utils.logger`

- **`scripts/content_preprocessor.py`** (Message Preprocessing)
  - → `db` (SessionLocal, Message)
  - → `utils.logger`

- **`scripts/enhanced_faiss_index.py`** (Vector Index Builder)
  - → `sentence_transformers`
  - → `faiss`
  - → `db`

#### **Database Layer**
- **`db/db.py`** - Database connection
- **`db/models.py`** - ORM models  
- **`db/query_logs.py`** - Query logging

### **🟡 TIER 2: PRODUCTION FEATURES**

#### **Enhanced Systems**
- **`core/enhanced_fallback_system.py`**
  - → `core.ai_client`
  - → `core.config`

- **`core/enhanced_k_determination.py`**
  - → `db` (database queries)

- **`core/resource_detector.py`**
  - → `core.classifier` (classify_resource)

- **`core/classifier.py`**
  - → Regex patterns and classification logic

#### **Production Pipelines**
- **`scripts/enhanced_pipeline_with_resources.py`**
  - → `scripts.content_preprocessor`
  - → `scripts.enhanced_faiss_index`
  - → `core.resource_detector`

### **🔵 TIER 3: UTILITIES & TOOLS**

- **`tools/tools.py`** (Message Utilities)
  - → `db` (SessionLocal, Message)
  - → Various utility functions

- **`tools/time_parser.py`** (Time Parsing)
  - → Standalone natural language processing

### **🔴 TIER 5: ARCHIVED (Already Removed)**

**✅ Successfully Archived to `archive/legacy_pipeline_removed_20250611/`:**
- `core/embed_store.py` - Legacy Index Builder (superseded by enhanced_faiss_index.py)
- `core/batch_detect.py` - Legacy Resource Detection  
- `core/repo_sync.py` - Export Tool
- `core/preprocessing.py` - Unused Pipeline Orchestrator
- `tools/full_pipeline.py` - Legacy Pipeline Orchestrator
- `scripts/fix_embedding_model.py` - One-time migration (complete)
- `tools/fix_resource_titles.py` - Obsolete utility tool

## 🎯 **Environment Dependencies**

### **Required Environment Variables**
- `DISCORD_TOKEN` - Used by `core/fetch_messages.py`, `core/bot.py`
- `OLLAMA_URL` - Used by `core/ai_client.py` (default: http://localhost:11434)
- Various config settings in `core/config.py`

### **External Libraries**
- **Discord.py** - Discord API integration
- **Streamlit** - Web interface
- **FAISS** - Vector similarity search
- **Sentence Transformers** - Text embeddings
- **SQLAlchemy** - Database ORM
- **Ollama** - Local LLM integration
- `GPT_MODEL`: Used by multiple modules for OpenAI calls

### Database
- SQLAlchemy models referenced across multiple modules
- Alembic migrations for schema changes

### File System
- `data/resources/`: Used for logging and resource storage
- `index_faiss/`: Used for vector store operations
- `bot.log`: Used for application logging

## Circular Dependencies
- `core/agent.py` ↔ `tools/tools.py`: Potential circular import risk
- `core/rag_engine.py` ↔ `core/agent.py`: Potential circular import risk

## External Service Dependencies
- OpenAI API (used by multiple modules)
- Discord API (used by `bot.py`)
- Vector store (FAISS)
- SQL database

## 📋 **Data Flow Analysis**

### **1. Message Ingestion Flow**
```
Discord Server → core/fetch_messages.py → data/discord_messages.db
```

### **2. Index Building Flow**
```
data/discord_messages.db → scripts/content_preprocessor.py → scripts/enhanced_faiss_index.py → data/indices/
```

### **3. Query Processing Flow**
```
User Query → core/app.py|core/bot.py → core/agent.py → core/rag_engine.py → tools/tools.py → Database + FAISS → Response
```

### **4. Resource Detection Flow**
```
Messages → core/resource_detector.py → core/classifier.py → Resource Classification
```

## ⚠️ **Critical Dependencies**

### **System Cannot Function Without:**
1. **`core/fetch_messages.py`** - No data without this
2. **`core/agent.py`** - No AI processing without this
3. **`core/rag_engine.py`** - No search without this
4. **`core/ai_client.py`** - No LLM integration without this
5. **Database layer** (`db/`) - No data persistence without this

### **System Degraded Without:**
- **`core/enhanced_fallback_system.py`** - Errors become generic
- **`tools/tools.py`** - Limited search capabilities
- **`tools/time_parser.py`** - No time-based queries

### **Features Missing Without:**
- **`core/resource_detector.py`** - No resource detection
- **Index builders** - No semantic search
- **Pipelines** - No data processing automation

## 🧹 **Cleanup Recommendations (Updated Post-Legacy Removal)**

### **✅ COMPLETED - Legacy Pipeline Removed**
Successfully archived to `archive/legacy_pipeline_removed_20250611/`:
- Complete legacy pipeline (9 files)
- Obsolete migration scripts  
- Legacy log files

### **NEXT - Archive Remaining Migrations (2 files)**
Move to `archive/migrations/`:
- `scripts/migrate_add_preprocessing_fields.py` - One-time database migration (complete)
- `scripts/populate_preprocessing_data.py` - One-time data population (complete)

### **ORGANIZE - Development Tools (7+ files)**
Move to `scripts/analysis/`:
- All `analyze_*.py` scripts (4 files)
- `evaluate_embedding_models.py`
- `enhanced_community_preprocessor.py`
- `cleanup_root.py`

## 🎯 **Final Architecture Summary**

**ESSENTIAL CORE (24 files):**
- User Interfaces: 3 files
- AI/RAG Engine: 4 files  
- Data Processing: 3 files
- Database: 3 files
- Enhanced Features: 5 files
- Production Pipelines: 3 files
- Index Builders: 3 files

**SUPPORT SYSTEM (8 files):**
- Tools & Utilities: 2 files
- Analytics: 3 files
- Testing: 3 files

**CLEANUP CANDIDATES (25+ files):**
- Legacy: 4 files
- Migrations: 3 files
- Development: 10+ files
- Analysis: 7+ files

**Result:** Clean, focused codebase with **32 core files** instead of **60+ mixed files**