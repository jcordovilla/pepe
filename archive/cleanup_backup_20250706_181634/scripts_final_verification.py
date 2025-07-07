#!/usr/bin/env python3
"""
Final verification script to confirm codebase is ready for commit.
"""

import os
from pathlib import Path
import json

def final_verification():
    """Perform final verification checks"""
    print("🔍 FINAL VERIFICATION CHECKLIST")
    print("=" * 80)
    
    checks = {
        "core_structure": False,
        "no_temp_files": False,
        "bot_data_intact": False,
        "archive_organized": False,
        "documentation_complete": False
    }
    
    # Check 1: Core structure intact
    print("\n✅ CHECKING CORE STRUCTURE")
    print("-" * 40)
    essential_dirs = ["agentic", "scripts", "tests", "docs", "data"]
    essential_files = ["main.py", "requirements.txt", ".env", "readme.md"]
    
    all_dirs_exist = all(Path(d).exists() for d in essential_dirs)
    all_files_exist = all(Path(f).exists() for f in essential_files)
    
    if all_dirs_exist and all_files_exist:
        checks["core_structure"] = True
        print("   ✅ All essential directories and files present")
        
        # Count Python files
        agentic_py = len(list(Path("agentic").rglob("*.py")))
        scripts_py = len(list(Path("scripts").rglob("*.py")))
        tests_py = len(list(Path("tests").rglob("*.py")))
        print(f"   📊 Python files: agentic/{agentic_py}, scripts/{scripts_py}, tests/{tests_py}")
    else:
        print("   ❌ Missing essential directories or files")
    
    # Check 2: No temporary files in main directories
    print("\n✅ CHECKING FOR TEMPORARY FILES")
    print("-" * 40)
    
    temp_patterns = ["*backup*", "*temp*", "*debug*", "*test_*", "*tmp*"]
    temp_files_found = []
    
    for pattern in temp_patterns:
        temp_files_found.extend(list(Path(".").glob(pattern)))
        temp_files_found.extend(list(Path("data").glob(pattern)))
        temp_files_found.extend(list(Path("scripts").glob(pattern)))
    
    # Filter out archive directory
    temp_files_found = [f for f in temp_files_found if not str(f).startswith("archive")]
    
    if not temp_files_found:
        checks["no_temp_files"] = True
        print("   ✅ No temporary files in main directories")
    else:
        print(f"   ⚠️ Found {len(temp_files_found)} potential temporary files:")
        for f in temp_files_found[:5]:  # Show first 5
            print(f"      - {f}")
    
    # Check 3: Bot data intact
    print("\n✅ CHECKING BOT DATA INTEGRITY")
    print("-" * 40)
    
    chromadb_exists = Path("data/chromadb").exists()
    messages_exist = Path("data/fetched_messages").exists()
    config_exists = Path("data/bot_config.json").exists()
    
    if chromadb_exists and messages_exist and config_exists:
        checks["bot_data_intact"] = True
        
        # Count data files
        chromadb_files = len(list(Path("data/chromadb").rglob("*")))
        message_files = len(list(Path("data/fetched_messages").glob("*.json")))
        
        print("   ✅ All bot data directories present")
        print(f"   📊 ChromaDB files: {chromadb_files}")
        print(f"   📊 Message files: {message_files}")
    else:
        print("   ❌ Missing bot data directories")
    
    # Check 4: Archive organized
    print("\n✅ CHECKING ARCHIVE ORGANIZATION")
    print("-" * 40)
    
    archive_dir = Path("archive")
    if archive_dir.exists():
        archive_subdirs = [d for d in archive_dir.iterdir() if d.is_dir()]
        archive_files = len(list(archive_dir.rglob("*")))
        
        if len(archive_subdirs) >= 6:  # Should have multiple cleanup phases
            checks["archive_organized"] = True
            print(f"   ✅ Archive organized with {len(archive_subdirs)} categories")
            print(f"   📦 Total archived items: {archive_files}")
            
            for subdir in sorted(archive_subdirs):
                items = len(list(subdir.rglob("*")))
                print(f"      - {subdir.name}: {items} items")
        else:
            print(f"   ⚠️ Archive exists but may be incomplete ({len(archive_subdirs)} categories)")
    else:
        print("   ❌ Archive directory not found")
    
    # Check 5: Documentation complete
    print("\n✅ CHECKING DOCUMENTATION")
    print("-" * 40)
    
    doc_files = [
        "readme.md",
        "PROJECT_STRUCTURE.md", 
        "CLEANUP_COMPLETE.md",
        "docs/QUICKSTART.md",
        "docs/DEPLOYMENT.md"
    ]
    
    existing_docs = [doc for doc in doc_files if Path(doc).exists()]
    
    if len(existing_docs) >= 4:
        checks["documentation_complete"] = True
        print(f"   ✅ Documentation complete ({len(existing_docs)}/{len(doc_files)} files)")
        for doc in existing_docs:
            size = Path(doc).stat().st_size
            print(f"      - {doc}: {size:,} bytes")
    else:
        print(f"   ⚠️ Documentation incomplete ({len(existing_docs)}/{len(doc_files)} files)")
    
    # Final summary
    print(f"\n🎯 FINAL VERIFICATION SUMMARY")
    print("=" * 80)
    
    passed_checks = sum(checks.values())
    total_checks = len(checks)
    
    print(f"📊 VERIFICATION RESULTS: {passed_checks}/{total_checks} checks passed")
    print()
    
    for check_name, passed in checks.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {status} {check_name.replace('_', ' ').title()}")
    
    if passed_checks == total_checks:
        print(f"\n🎉 ALL CHECKS PASSED!")
        print(f"🚀 Codebase is ready for commit!")
        return True
    else:
        print(f"\n⚠️ {total_checks - passed_checks} checks failed")
        print(f"📋 Please review failed items before committing")
        return False

if __name__ == "__main__":
    success = final_verification()
    exit(0 if success else 1)
