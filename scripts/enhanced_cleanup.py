#!/usr/bin/env python3
"""
Enhanced cleanup script for remaining temporary files and backups.
Handles ChromaDB backups, root-level temporary files, and final organization.
"""

import os
import shutil
from pathlib import Path
from datetime import datetime
import json

def enhanced_cleanup():
    """Execute enhanced cleanup for remaining temporary files"""
    print("🧹 ENHANCED CLEANUP - PHASE 2")
    print("=" * 80)
    
    # Create archive directories
    archive_base = Path("archive")
    cleanup_date = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Define additional cleanup categories
    additional_cleanup = {
        "chromadb_backups": {
            "archive_dir": archive_base / f"chromadb_backups_{cleanup_date}",
            "description": "ChromaDB backup directories",
            "patterns": [
                "./data/chromadb_backup_*",
                "./data/chromadb_conflict_backup_*",
            ],
            "action": "archive"
        },
        
        "config_backups": {
            "archive_dir": archive_base / f"config_backups_{cleanup_date}",
            "description": "Configuration backup directories",
            "patterns": [
                "./backup_db_config_*",
            ],
            "action": "archive"
        },
        
        "root_temporary_files": {
            "archive_dir": archive_base / f"root_temp_files_{cleanup_date}",
            "description": "Root-level temporary and utility files",
            "patterns": [
                "./cleanup_codebase.py",
                "./start_optimized.py",
                "./launch.sh",
            ],
            "action": "archive"
        },
        
        "redundant_configs": {
            "archive_dir": archive_base / f"redundant_configs_{cleanup_date}",
            "description": "Redundant configuration files",
            "patterns": [
                "./data/optimized_analytics_config.json",
                "./data/optimized_discord_config.json", 
                "./data/optimized_memory_config.json",
                "./data/optimized_vector_config.json",
                "./data/performance_config.json",
                "./data/performance_monitoring_config.json",
                "./data/unified_performance_config.json",
            ],
            "action": "archive"
        }
    }
    
    stats = {
        "archived_files": 0,
        "removed_files": 0,
        "archived_dirs": 0,
        "removed_dirs": 0
    }
    
    # Execute additional cleanup
    for category, config in additional_cleanup.items():
        print(f"\n📂 {category.upper()}: {config['description']}")
        print("-" * 60)
        
        if config["action"] == "archive":
            config["archive_dir"].mkdir(exist_ok=True, parents=True)
        
        for pattern in config["patterns"]:
            # Handle both files and directories
            if "*" in pattern:
                # Handle wildcard patterns
                base_path = Path(pattern).parent
                file_pattern = Path(pattern).name
                
                if base_path.exists():
                    for item_path in base_path.glob(file_pattern):
                        if config["action"] == "archive":
                            dest = config["archive_dir"] / item_path.name
                            print(f"   📁 Archiving: {item_path} → {dest}")
                            shutil.move(str(item_path), str(dest))
                            if item_path.is_dir():
                                stats["archived_dirs"] += 1
                            else:
                                stats["archived_files"] += 1
            else:
                # Handle exact paths
                item_path = Path(pattern)
                if item_path.exists():
                    if config["action"] == "archive":
                        dest = config["archive_dir"] / item_path.name
                        print(f"   📄 Archiving: {item_path} → {dest}")
                        shutil.move(str(item_path), str(dest))
                        if item_path.is_dir():
                            stats["archived_dirs"] += 1
                        else:
                            stats["archived_files"] += 1
    
    # Check for and clean any remaining empty export directories
    print(f"\n🧽 CLEANING REMAINING EMPTY DIRECTORIES")
    print("-" * 60)
    
    exports_dir = Path("data/exports")
    if exports_dir.exists():
        remaining_files = list(exports_dir.rglob("*"))
        if not remaining_files:
            print(f"   🗑️ Removing empty exports directory")
            exports_dir.rmdir()
            stats["removed_dirs"] += 1
        else:
            print(f"   ✅ Exports directory has {len(remaining_files)} files, keeping")
    
    # Clean up .pytest_cache if it exists
    pytest_cache = Path(".pytest_cache")
    if pytest_cache.exists():
        print(f"   🗑️ Removing pytest cache directory")
        shutil.rmtree(pytest_cache)
        stats["removed_dirs"] += 1
    
    # Create enhanced cleanup summary
    summary = {
        "cleanup_date": cleanup_date,
        "phase": "Enhanced Cleanup Phase 2",
        "description": "Final cleanup of remaining temporary files and backups",
        "categories_processed": list(additional_cleanup.keys()),
        "statistics": stats,
        "remaining_items_check": {
            "chromadb_backups_cleaned": not any(Path("data").glob("chromadb_backup_*")),
            "config_backups_cleaned": not any(Path(".").glob("backup_db_config_*")),
            "root_temp_files_cleaned": not Path("cleanup_codebase.py").exists(),
            "redundant_configs_cleaned": not Path("data/optimized_analytics_config.json").exists()
        }
    }
    
    # Save enhanced cleanup summary
    summary_file = archive_base / f"enhanced_cleanup_summary_{cleanup_date}.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✅ ENHANCED CLEANUP COMPLETED!")
    print("=" * 80)
    print(f"📊 ADDITIONAL STATISTICS:")
    print(f"   • Additional files archived: {stats['archived_files']}")
    print(f"   • Additional directories archived: {stats['archived_dirs']}")
    print(f"   • Additional directories removed: {stats['removed_dirs']}")
    print(f"   • Enhanced cleanup summary: {summary_file}")
    
    print(f"\n🧹 COMPLETE CLEANUP STATUS:")
    all_clean = all(summary["remaining_items_check"].values())
    if all_clean:
        print(f"   ✅ All temporary files and backups cleaned!")
        print(f"   ✅ ChromaDB backups archived")
        print(f"   ✅ Configuration backups archived") 
        print(f"   ✅ Root temporary files archived")
        print(f"   ✅ Redundant configs archived")
    else:
        print(f"   ⚠️ Some items may still need attention:")
        for item, status in summary["remaining_items_check"].items():
            status_icon = "✅" if status else "❌"
            print(f"     {status_icon} {item}")
    
    # Final directory structure overview
    print(f"\n📁 FINAL CLEAN STRUCTURE:")
    essential_dirs = ["agentic", "scripts", "tests", "docs", "data", "archive"]
    for dir_name in essential_dirs:
        dir_path = Path(dir_name)
        if dir_path.exists():
            if dir_name == "data":
                # Count important data subdirectories
                chromadb_size = len(list(Path("data/chromadb").rglob("*"))) if Path("data/chromadb").exists() else 0
                fetched_size = len(list(Path("data/fetched_messages").glob("*.json"))) if Path("data/fetched_messages").exists() else 0
                print(f"   📂 {dir_name}/ - ChromaDB: {chromadb_size} files, Messages: {fetched_size} files")
            elif dir_name == "archive":
                # Count archive items
                archive_items = len(list(dir_path.rglob("*"))) 
                print(f"   📦 {dir_name}/ - {archive_items} archived items")
            else:
                # Count Python files
                py_files = len(list(dir_path.rglob("*.py"))) if dir_path.exists() else 0
                print(f"   🔥 {dir_name}/ - {py_files} Python files")
    
    return summary

if __name__ == "__main__":
    try:
        summary = enhanced_cleanup()
        print("\n🎉 Enhanced cleanup completed successfully!")
        print("\n🎯 FINAL STATUS: Codebase is now fully optimized and ready for commit!")
        
    except Exception as e:
        print(f"\n❌ Error during enhanced cleanup: {e}")
        import traceback
        traceback.print_exc()
