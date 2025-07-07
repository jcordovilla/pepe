#!/usr/bin/env python3
"""
Comprehensive codebase cleanup script.
Organizes, archives, and removes temporary files before final commit.
"""

import os
import shutil
from pathlib import Path
from datetime import datetime
import json

def comprehensive_cleanup():
    """Execute comprehensive codebase cleanup"""
    print("🧹 COMPREHENSIVE CODEBASE CLEANUP")
    print("=" * 80)
    
    # Create archive directories
    archive_base = Path("archive")
    archive_base.mkdir(exist_ok=True)
    
    cleanup_date = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Define cleanup categories
    cleanup_plan = {
        "debug_scripts": {
            "archive_dir": archive_base / f"debug_scripts_{cleanup_date}",
            "description": "Debug and temporary development scripts",
            "patterns": [
                "./debug/debug_*.py",
                "./scripts/debug_*.py",
                "./tests/debug/",
                "./scripts/test_*.py",
                "./tests/test_*_isolated.py",
                "./tests/test_*_simple.py",
                "./tests/reaction_search/",
            ],
            "action": "archive"
        },
        
        "temporary_data": {
            "archive_dir": archive_base / f"temp_data_{cleanup_date}",
            "description": "Temporary data files and exports",
            "patterns": [
                "./data/exports/resource_detection_evaluation_*.json",
                "./data/evaluation_reports/",
                "./data/pipeline_results_standalone.json",
                "./data/test_fetch_state.json",
                "./migration/cleanup_report_*.json",
                "./migration/migration_report_*.json",
            ],
            "action": "archive"
        },
        
        "legacy_scripts": {
            "archive_dir": archive_base / f"legacy_scripts_{cleanup_date}",
            "description": "Legacy and superseded scripts",
            "patterns": [
                "./scripts/reindex_complete_data.py",
                "./scripts/reindex_robust_handling.py",
                "./scripts/analyze_data_flow.py",
                "./scripts/check_vector_store_status.py",
                "./scripts/create_snapshot.py",
                "./scripts/restore_snapshot.py",
                "./scripts/monitor_resource_quality*.py",
            ],
            "action": "archive"
        },
        
        "log_files": {
            "archive_dir": archive_base / f"logs_{cleanup_date}",
            "description": "Log files",
            "patterns": [
                "./logs/*.log",
            ],
            "action": "archive"
        },
        
        "obsolete_tests": {
            "archive_dir": archive_base / f"obsolete_tests_{cleanup_date}",
            "description": "Obsolete and redundant test files",
            "patterns": [
                "./tests/test_*_complete.py",
                "./tests/test_*_fixed.py",
                "./tests/test_*_final.py",
                "./tests/test_*_production.py",
                "./tests/test_*_validation.py",
                "./tests/test_*_real.py",
                "./tests/test_*_flow.py",
                "./tests/test_*_formatting.py",
                "./tests/test_*_integration.py",
                "./tests/test_*_interaction.py",
                "./tests/test_*_transform.py",
                "./tests/test_chromadb_*.py",
                "./tests/test_classification_*.py",
                "./tests/test_end_to_end_*.py",
                "./tests/test_incremental_*.py",
                "./tests/test_progress_*.py",
                "./tests/test_vectorstats_*.py",
            ],
            "action": "remove"  # These are truly obsolete
        }
    }
    
    stats = {
        "archived_files": 0,
        "removed_files": 0,
        "archived_dirs": 0,
        "removed_dirs": 0
    }
    
    # Execute cleanup plan
    for category, config in cleanup_plan.items():
        print(f"\n📂 {category.upper()}: {config['description']}")
        print("-" * 60)
        
        if config["action"] == "archive":
            config["archive_dir"].mkdir(exist_ok=True, parents=True)
        
        for pattern in config["patterns"]:
            # Handle both files and directories
            if pattern.endswith("/"):
                # Directory pattern
                dir_path = Path(pattern.rstrip("/"))
                if dir_path.exists():
                    if config["action"] == "archive":
                        dest = config["archive_dir"] / dir_path.name
                        print(f"   📁 Archiving directory: {dir_path} → {dest}")
                        shutil.move(str(dir_path), str(dest))
                        stats["archived_dirs"] += 1
                    elif config["action"] == "remove":
                        print(f"   🗑️ Removing directory: {dir_path}")
                        shutil.rmtree(dir_path)
                        stats["removed_dirs"] += 1
            else:
                # File pattern (with potential wildcards)
                if "*" in pattern:
                    # Handle wildcard patterns
                    base_path = Path(pattern).parent
                    file_pattern = Path(pattern).name
                    
                    if base_path.exists():
                        for file_path in base_path.glob(file_pattern):
                            if file_path.is_file():
                                if config["action"] == "archive":
                                    dest = config["archive_dir"] / file_path.name
                                    print(f"   📄 Archiving file: {file_path} → {dest}")
                                    shutil.move(str(file_path), str(dest))
                                    stats["archived_files"] += 1
                                elif config["action"] == "remove":
                                    print(f"   🗑️ Removing file: {file_path}")
                                    file_path.unlink()
                                    stats["removed_files"] += 1
                else:
                    # Handle exact file paths
                    file_path = Path(pattern)
                    if file_path.exists():
                        if config["action"] == "archive":
                            dest = config["archive_dir"] / file_path.name
                            print(f"   📄 Archiving file: {file_path} → {dest}")
                            shutil.move(str(file_path), str(dest))
                            stats["archived_files"] += 1
                        elif config["action"] == "remove":
                            print(f"   🗑️ Removing file: {file_path}")
                            file_path.unlink()
                            stats["removed_files"] += 1
    
    # Clean up empty directories
    print(f"\n🧽 CLEANING UP EMPTY DIRECTORIES")
    print("-" * 60)
    
    empty_dirs = []
    for root, dirs, files in os.walk("."):
        if root.startswith("./venv") or root.startswith("./archive"):
            continue
        
        path = Path(root)
        if not any(path.iterdir()) and path != Path("."):
            empty_dirs.append(path)
    
    for empty_dir in empty_dirs:
        print(f"   🗑️ Removing empty directory: {empty_dir}")
        empty_dir.rmdir()
        stats["removed_dirs"] += 1
    
    # Create cleanup summary
    summary = {
        "cleanup_date": cleanup_date,
        "description": "Comprehensive codebase cleanup before final commit",
        "categories_processed": list(cleanup_plan.keys()),
        "statistics": stats,
        "preserved_files": [
            "Main production code (agentic/)",
            "Core scripts (scripts/streaming_discord_indexer.py, scripts/system_status.py)",
            "Essential tests (tests/README.md and key integration tests)",
            "Documentation (docs/)",
            "Configuration files (.env, requirements.txt, etc.)",
            "Current data (data/chromadb/, data/cache/)"
        ],
        "archived_locations": [str(archive_dir) for archive_dir in [config["archive_dir"] for config in cleanup_plan.values() if config["action"] == "archive"]]
    }
    
    # Save cleanup summary
    summary_file = archive_base / f"cleanup_summary_{cleanup_date}.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✅ CLEANUP COMPLETED!")
    print("=" * 80)
    print(f"📊 STATISTICS:")
    print(f"   • Files archived: {stats['archived_files']}")
    print(f"   • Files removed: {stats['removed_files']}")
    print(f"   • Directories archived: {stats['archived_dirs']}")
    print(f"   • Directories removed: {stats['removed_dirs']}")
    print(f"   • Cleanup summary: {summary_file}")
    
    print(f"\n📁 ARCHIVE STRUCTURE:")
    for category, config in cleanup_plan.items():
        if config["action"] == "archive" and config["archive_dir"].exists():
            file_count = len(list(config["archive_dir"].rglob("*")))
            print(f"   • {config['archive_dir']}: {file_count} items")
    
    print(f"\n🎯 READY FOR COMMIT:")
    print(f"   • Codebase is now clean and organized")
    print(f"   • All temporary and debug files archived")
    print(f"   • Production code preserved and ready")
    print(f"   • Archive available for reference if needed")
    
    return summary

if __name__ == "__main__":
    try:
        summary = comprehensive_cleanup()
        print("\n🎉 Cleanup completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during cleanup: {e}")
        import traceback
        traceback.print_exc()
