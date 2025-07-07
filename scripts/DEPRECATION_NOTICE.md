# 🚨 SCRIPTS DEPRECATION NOTICE

**Effective Date:** December 19, 2024

The majority of scripts in this directory have been **DEPRECATED** and replaced by the unified **`pepe-admin`** CLI tool located in the project root.

## ✅ **New Unified Interface**

Instead of running scattered scripts, use the single admin CLI:

```bash
# Old way (deprecated)
python scripts/system_status.py
python scripts/validate_deployment.py
python scripts/database/populate_database.py

# New way (recommended)
./pepe-admin status
./pepe-admin test
./pepe-admin setup
```

## 📋 **Script Migration Map**

### **Replaced by `pepe-admin` commands:**

| **Old Script** | **New Command** | **Status** |
|---|---|---|
| `system_status.py` | `./pepe-admin status` | ✅ **REPLACED** |
| `validate_deployment.py` | `./pepe-admin test` | ✅ **REPLACED** |
| `database/populate_database.py` | `./pepe-admin setup` | ✅ **REPLACED** |
| `database/init_db_simple.py` | `./pepe-admin setup` | ✅ **REPLACED** |
| `apply_performance_optimizations.py` | `./pepe-admin optimize` | ✅ **REPLACED** |
| `maintenance/check_vector_store.py` | `./pepe-admin stats` | ✅ **REPLACED** |
| `maintenance/check_channels.py` | `./pepe-admin status` | ✅ **REPLACED** |
| Various backup scripts | `./pepe-admin backup` | ✅ **REPLACED** |

### **Legacy/Deprecated scripts:**
- `run_pipeline.py` - **OBSOLETE** (real-time processing via Discord bot)
- `streaming_discord_indexer.py` - **INTEGRATED** into main bot
- `comprehensive_codebase_cleanup.py` - **ONE-TIME** use completed
- `fix_*.py` scripts - **ONE-TIME** fixes completed
- `reindex_*.py` scripts - **USE** `pepe-admin sync --initial`

### **Still Active (Specialized Use):**
- `migrate_to_enhanced_resources.py` - Use `./pepe-admin migrate`
- `test_enhanced_resource_detection.py` - Use `./pepe-admin test`

## 🎯 **Quick Reference**

### **Daily Operations:**
```bash
./pepe-admin status          # Check system health
./pepe-admin monitor         # Performance monitoring  
./pepe-admin maintain        # Run maintenance tasks
```

### **Initial Setup:**
```bash
./pepe-admin setup           # Complete system setup
./pepe-admin sync --initial  # Import existing data
./pepe-admin test            # Validate everything works
```

### **Backup & Recovery:**
```bash
./pepe-admin backup          # Create system backup
./pepe-admin stats           # System statistics
```

## ⚠️ **Important Notes**

1. **Real-time processing** is now handled automatically by the Discord bot (`python main.py`)
2. **No manual pipeline runs** needed - everything is automated
3. **Single point of control** through `pepe-admin` for all operations
4. **Old scripts will be removed** in future cleanup

## 🚀 **Migration Steps**

1. Replace any script usage with equivalent `pepe-admin` commands
2. Update documentation/automation to use the new CLI
3. Test the new commands to ensure functionality
4. Remove references to old scripts from processes

---

**For questions or issues with the new CLI, check:**
```bash
./pepe-admin --help
./pepe-admin <command> --help
``` 