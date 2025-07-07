# Codebase Organization

This document describes the organized structure of the Discord Bot v2 codebase after cleanup.

## Root Directory Structure

```
├── .env                      # Environment variables
├── .gitignore               # Git ignore rules
├── main.py                  # Main Discord bot entry point
├── scripts/                  # Utility and maintenance scripts
│   ├── run_pipeline.py       # Database pipeline runner
├── launch.sh                # Quick start script
├── requirements.txt         # Python dependencies
├── readme.md               # Main project documentation
├── README_NEW.md           # Updated documentation
├── ORGANIZATION.md         # This file
│
├── agentic/                # Core agentic system
├── core/                   # Core application systems
├── data/                   # Application data and databases
├── docs/                   # Project documentation
├── logs/                   # Application logs
├── venv/                   # Python virtual environment
│
├── tests/                  # All test files (35 files)
├── debug/                  # Debug and troubleshooting scripts (8 files)
└── scripts/                # Utility and maintenance scripts
    ├── maintenance/        # System maintenance scripts
    ├── database/          # Database setup and population scripts
    ├── test_system.py     # System validation
    └── validate_deployment.py # Deployment validation
```

## Directory Purposes

### Core Directories
- **`agentic/`** - Multi-agent system implementation
- **`core/`** - Core Discord bot functionality and data processing
- **`data/`** - All databases, cache, and persistent data

### Documentation
- **`docs/`** - Comprehensive project documentation
- **`logs/`** - Runtime logs and debugging information

### Development & Maintenance
- **`tests/`** - All test files (moved from root for cleanliness)
- **`debug/`** - Debug scripts for troubleshooting specific issues
- **`scripts/maintenance/`** - System health checks and maintenance
- **`scripts/database/`** - Database initialization and population tools

## Key Files

### Essential Runtime Files
- `main.py` - Discord bot entry point
- `scripts/run_pipeline.py` - Database processing pipeline
- `launch.sh` - Quick start script

### Configuration
- `.env` - Environment configuration
- `requirements.txt` - Python dependencies

### Documentation
- `readme.md` - Main project README
- `ORGANIZATION.md` - This organization guide

## Recent Cleanup Actions

1. **Moved 30 test files** from root to `tests/` directory
2. **Moved 8 debug files** from root to `debug/` directory  
3. **Organized maintenance scripts** into `scripts/maintenance/`
4. **Organized database scripts** into `scripts/database/`
5. **Achieved clean root directory** with only essential files

## Database Status

- ✅ **ChromaDB Vector Store**: 4,943 records (fully populated)
- ✅ **Conversation Memory**: 138 conversations 
- ✅ **Analytics Database**: Populated and functional
- ❌ **Traditional SQLite Database**: Missing (`discord_messages.db`)

## Next Steps

1. **Run database population** using scripts in `scripts/database/`
2. **Validate system functionality** using `scripts/test_system.py`
3. **Deploy using** `scripts/validate_deployment.py`

---
*Last updated: June 4, 2025*
