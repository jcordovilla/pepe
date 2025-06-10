#!/usr/bin/env python3
"""
Root Directory Cleanup Script

This script helps maintain a clean root directory by moving files to appropriate locations.
Run this periodically to keep the project organized.

Usage: python scripts/cleanup_root.py
"""

import os
import shutil
from pathlib import Path
from datetime import datetime

# Define cleanup rules
CLEANUP_RULES = {
    # File patterns and their destination directories
    '*.md': 'docs/',                    # Documentation files
    '*_test*.py': 'tests/',             # Test files  
    'test_*.py': 'tests/',              # Test files
    '*_report_*.json': 'data/reports/', # Report files
    '*.index': 'data/indices/',         # FAISS index files
    '*_metadata.json': 'data/indices/', # FAISS metadata files
    'discord_index_*': 'data/indices/', # Discord index directories
    'enhanced_faiss_*': 'data/indices/', # Enhanced FAISS files
    '*_example.py': 'scripts/examples/', # Example scripts
    'example_*.py': 'scripts/examples/', # Example scripts
}

# Files to keep in root (whitelist)
KEEP_IN_ROOT = {
    'readme.md', 'README.md', 'requirements.txt', 'setup.py', 
    '.env', '.gitignore', 'pytest.ini', 'mkdocs.yml', 
    'render.yaml', '.flake8', 'pyproject.toml'
}

def cleanup_root_directory():
    """Clean up the root directory by moving files to appropriate locations."""
    
    root_dir = Path(__file__).parent.parent
    print(f"🧹 Cleaning up root directory: {root_dir}")
    
    moved_files = []
    created_dirs = []
    
    # Get all files in root directory
    root_files = [f for f in root_dir.iterdir() if f.is_file()]
    
    for file_path in root_files:
        filename = file_path.name.lower()
        
        # Skip files that should stay in root
        if filename in KEEP_IN_ROOT:
            continue
            
        # Check against cleanup rules
        moved = False
        for pattern, dest_dir in CLEANUP_RULES.items():
            # Simple pattern matching (you could use fnmatch for more complex patterns)
            if pattern.startswith('*') and filename.endswith(pattern[1:]):
                # Create destination directory if it doesn't exist
                dest_path = root_dir / dest_dir
                if not dest_path.exists():
                    dest_path.mkdir(parents=True, exist_ok=True)
                    created_dirs.append(str(dest_path))
                
                # Move the file
                new_path = dest_path / file_path.name
                if not new_path.exists():
                    shutil.move(str(file_path), str(new_path))
                    moved_files.append(f"{file_path.name} → {dest_dir}")
                    moved = True
                    break
            elif pattern == filename:
                # Exact match
                dest_path = root_dir / dest_dir
                if not dest_path.exists():
                    dest_path.mkdir(parents=True, exist_ok=True)
                    created_dirs.append(str(dest_path))
                
                new_path = dest_path / file_path.name
                if not new_path.exists():
                    shutil.move(str(file_path), str(new_path))
                    moved_files.append(f"{file_path.name} → {dest_dir}")
                    moved = True
                    break
    
    # Handle directories
    root_dirs = [d for d in root_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
    for dir_path in root_dirs:
        dirname = dir_path.name
        
        # Move index directories
        if dirname.startswith(('discord_index_', 'enhanced_faiss_', 'faiss_index_')):
            dest_path = root_dir / 'data' / 'indices'
            if not dest_path.exists():
                dest_path.mkdir(parents=True, exist_ok=True)
                created_dirs.append(str(dest_path))
            
            new_path = dest_path / dirname
            if not new_path.exists():
                shutil.move(str(dir_path), str(new_path))
                moved_files.append(f"{dirname}/ → data/indices/")
    
    # Print results
    print(f"\n📊 Cleanup Results:")
    if created_dirs:
        print(f"✅ Created directories: {len(created_dirs)}")
        for dir_path in created_dirs:
            print(f"   📁 {dir_path}")
    
    if moved_files:
        print(f"✅ Moved files: {len(moved_files)}")
        for move in moved_files:
            print(f"   📄 {move}")
    else:
        print("✅ Root directory is already clean!")
    
    # Show current root structure
    print(f"\n📁 Current root structure:")
    root_items = sorted([item.name for item in root_dir.iterdir() 
                        if not item.name.startswith('.') and item.name != '__pycache__'])
    for item in root_items:
        item_path = root_dir / item
        icon = "📁" if item_path.is_dir() else "📄"
        print(f"   {icon} {item}")

if __name__ == "__main__":
    cleanup_root_directory()
