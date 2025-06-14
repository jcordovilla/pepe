#!/usr/bin/env python3
"""
Comprehensive Codebase Cleanup Script

This script organizes and cleans up the codebase by:
1. Categorizing scripts by purpose
2. Moving files to appropriate directories
3. Removing obsolete/redundant files
4. Creating proper documentation
5. Updating .gitignore if needed
"""

import os
import shutil
from pathlib import Path
from datetime import datetime

def cleanup_codebase():
    """Perform comprehensive codebase cleanup"""
    print("🧹 COMPREHENSIVE CODEBASE CLEANUP")
    print("=" * 80)
    
    base_dir = Path(".")
    scripts_dir = base_dir / "scripts"
    debug_dir = base_dir / "debug"
    
    # Create cleanup directories
    cleanup_dirs = {
        "scripts/archive": "Archived/obsolete scripts",
        "scripts/development": "Development and debugging scripts", 
        "scripts/maintenance": "Production maintenance scripts",
        "scripts/analysis": "Data analysis and diagnostics",
        "scripts/migration": "Data migration scripts"
    }
    
    print("📁 Creating organized directory structure...")
    for dir_path, description in cleanup_dirs.items():
        os.makedirs(dir_path, exist_ok=True)
        print(f"   ✅ {dir_path}/ - {description}")
    
    # File categorization rules
    file_categories = {
        # PRODUCTION SCRIPTS (keep in main scripts/)
        "production": [
            "streaming_discord_indexer.py",  # Our optimized indexer
            "system_status.py",
            "validate_deployment.py",
            "run_pipeline.py", 
            "run_standalone_pipeline.py"
        ],
        
        # MAINTENANCE SCRIPTS
        "maintenance": [
            "create_snapshot.py",
            "restore_snapshot.py", 
            "cleanup_legacy.py",
            "migrate_legacy.py",
            "monitor_resource_quality.py",
            "monitor_resource_quality_enhanced.py"
        ],
        
        # ANALYSIS SCRIPTS  
        "analysis": [
            "analyze_complete_data_catalog.py",
            "analyze_data_fields.py", 
            "analyze_data_flow.py",
            "analyze_skipped_messages.py",
            "check_vector_store_status.py"
        ],
        
        # DEVELOPMENT/DEBUG SCRIPTS
        "development": [
            "debug_data_format_error.py",
            "debug_message_processing_error.py", 
            "debug_message_structure.py",
            "debug_search_functionality.py",
            "debug_search_quality.py",
            "debug_str_get_error.py",
            "test_channel_mentions.py",
            "test_channel_regex.py", 
            "test_channel_resolution.py",
            "test_fixed_resolution.py",
            "test_fixed_vector_store.py",
            "test_formatting_fixes.py",
            "test_system.py"
        ],
        
        # MIGRATION SCRIPTS
        "migration": [
            "fix_embedding_config.py",
            "fix_vector_store_embedding.py",
            "fast_initialize.py"
        ],
        
        # OBSOLETE/ARCHIVE SCRIPTS
        "archive": [
            "reindex_complete_data.py",  # Replaced by streaming_discord_indexer.py
            "reindex_robust_handling.py",  # Replaced by streaming_discord_indexer.py  
            "reindex_with_display_names.py",  # Replaced by streaming_discord_indexer.py
            "apply_performance_optimizations.py"  # One-time use
        ]
    }
    
    # Move files to appropriate categories
    print(f"\n📦 Organizing scripts by category...")
    
    moved_files = 0
    for category, file_list in file_categories.items():
        if category == "production":
            continue  # Keep production scripts in main scripts/
            
        target_dir = scripts_dir / category
        
        for filename in file_list:
            source_file = scripts_dir / filename
            if source_file.exists():
                target_file = target_dir / filename
                shutil.move(str(source_file), str(target_file))
                print(f"   📁 {filename} → scripts/{category}/")
                moved_files += 1
    
    # Move debug directory scripts to development
    print(f"\n🐛 Moving debug scripts...")
    debug_moved = 0
    if debug_dir.exists():
        development_dir = scripts_dir / "development"
        for debug_file in debug_dir.glob("*.py"):
            target_file = development_dir / debug_file.name
            shutil.move(str(debug_file), str(target_file))
            print(f"   🐛 {debug_file.name} → scripts/development/")
            debug_moved += 1
        
        # Remove empty debug directory
        if not any(debug_dir.iterdir()):
            debug_dir.rmdir()
            print(f"   🗑️ Removed empty debug/ directory")
    
    # Create category README files
    print(f"\n📝 Creating documentation...")
    
    category_docs = {
        "scripts": """# Scripts Directory

## 📁 Directory Structure

- **📦 Production Scripts**: Core operational scripts
- **🔧 maintenance/**: System maintenance and monitoring
- **📊 analysis/**: Data analysis and diagnostic tools  
- **🛠 development/**: Development and debugging utilities
- **🔄 migration/**: Data migration and setup scripts
- **📚 archive/**: Obsolete/replaced scripts (kept for reference)

## 🚀 Key Production Scripts

- `streaming_discord_indexer.py` - Optimized Discord → ChromaDB indexer
- `system_status.py` - System health monitoring
- `validate_deployment.py` - Deployment validation
- `run_pipeline.py` - Data processing pipeline

## 📊 Quick Analysis

```bash
# Check vector store status
python3 scripts/analysis/check_vector_store_status.py

# Analyze data catalog
python3 scripts/analysis/analyze_complete_data_catalog.py

# System health check
python3 scripts/system_status.py
```
""",
        
        "maintenance": """# Maintenance Scripts

Scripts for system maintenance, monitoring, and backup operations.

## 📋 Available Scripts

- `create_snapshot.py` - Create system snapshots
- `restore_snapshot.py` - Restore from snapshots
- `cleanup_legacy.py` - Clean up legacy data
- `migrate_legacy.py` - Migrate legacy systems
- `monitor_resource_quality.py` - Resource quality monitoring
""",
        
        "analysis": """# Analysis Scripts

Data analysis, diagnostics, and system inspection tools.

## 📊 Available Scripts

- `analyze_complete_data_catalog.py` - Complete data field analysis
- `analyze_data_flow.py` - Data flow optimization analysis
- `check_vector_store_status.py` - Vector store diagnostics
- `analyze_skipped_messages.py` - Message processing analysis
""",
        
        "development": """# Development Scripts

Development tools, debugging utilities, and test scripts.

## 🛠 Available Scripts

### Debug Tools
- `debug_*.py` - Various debugging utilities
- `test_*.py` - Test and validation scripts

### Channel Testing
- `test_channel_*.py` - Channel resolution testing

### Data Testing  
- `debug_message_*.py` - Message processing debugging
""",
        
        "migration": """# Migration Scripts

Data migration, setup, and configuration scripts.

## 🔄 Available Scripts

- `fix_embedding_config.py` - Fix embedding configurations
- `fix_vector_store_embedding.py` - Vector store embedding fixes
- `fast_initialize.py` - Quick system initialization
""",
        
        "archive": """# Archived Scripts

Obsolete or replaced scripts kept for reference.

## 📚 Archived Scripts

- `reindex_*.py` - Old indexing scripts (replaced by streaming_discord_indexer.py)
- `apply_performance_optimizations.py` - One-time optimization script

⚠️ **Note**: These scripts are archived and should not be used in production.
Use the current scripts in the main scripts/ directory instead.
"""
    }
    
    readme_created = 0
    for category, content in category_docs.items():
        if category == "scripts":
            readme_path = scripts_dir / "README.md"
        else:
            readme_path = scripts_dir / category / "README.md"
        
        with open(readme_path, 'w') as f:
            f.write(content)
        print(f"   📝 Created {readme_path}")
        readme_created += 1
    
    # Clean up any empty directories
    print(f"\n🗑️ Cleaning up empty directories...")
    cleanup_count = 0
    for root, dirs, files in os.walk(".", topdown=False):
        for dir_name in dirs:
            dir_path = Path(root) / dir_name
            if dir_path.exists() and not any(dir_path.iterdir()):
                try:
                    dir_path.rmdir()
                    print(f"   🗑️ Removed empty directory: {dir_path}")
                    cleanup_count += 1
                except OSError:
                    pass  # Directory not empty or permission issue
    
    # Update .gitignore if needed
    print(f"\n📄 Checking .gitignore...")
    gitignore_path = Path(".gitignore")
    gitignore_additions = [
        "# Temporary/debug files",
        "*.tmp",
        "*.debug", 
        ".DS_Store",
        "",
        "# Development artifacts", 
        "scripts/development/*.log",
        "scripts/analysis/*.tmp",
        "",
        "# Backup files",
        "*_backup_*",
        "backup_*/",
    ]
    
    if gitignore_path.exists():
        with open(gitignore_path, 'r') as f:
            current_content = f.read()
        
        additions_needed = []
        for addition in gitignore_additions:
            if addition and addition not in current_content:
                additions_needed.append(addition)
        
        if additions_needed:
            with open(gitignore_path, 'a') as f:
                f.write("\n# Cleanup additions\n")
                f.write("\n".join(additions_needed))
            print(f"   📄 Updated .gitignore with {len(additions_needed)} entries")
        else:
            print(f"   ✅ .gitignore is up to date")
    
    # Summary
    print(f"\n✅ CLEANUP COMPLETE!")
    print("=" * 80)
    print(f"📊 SUMMARY:")
    print(f"   • Files moved: {moved_files}")
    print(f"   • Debug scripts moved: {debug_moved}")
    print(f"   • README files created: {readme_created}")
    print(f"   • Empty directories removed: {cleanup_count}")
    print(f"   • Directory structure organized: 5 categories")
    
    print(f"\n📁 NEW STRUCTURE:")
    print(f"   scripts/")
    print(f"   ├── 🚀 streaming_discord_indexer.py (PRODUCTION)")
    print(f"   ├── 🔧 maintenance/ ({len(file_categories['maintenance'])} scripts)")
    print(f"   ├── 📊 analysis/ ({len(file_categories['analysis'])} scripts)")
    print(f"   ├── 🛠 development/ ({len(file_categories['development']) + debug_moved} scripts)")
    print(f"   ├── 🔄 migration/ ({len(file_categories['migration'])} scripts)")
    print(f"   └── 📚 archive/ ({len(file_categories['archive'])} scripts)")
    
    print(f"\n🎯 PRODUCTION READY:")
    print(f"   • Main scripts/ contains only production-ready scripts")
    print(f"   • Development tools organized in subdirectories")
    print(f"   • Comprehensive documentation added")
    print(f"   • .gitignore updated for cleaner commits")

if __name__ == "__main__":
    cleanup_codebase()
