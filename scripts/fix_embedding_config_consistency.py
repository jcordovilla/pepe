#!/usr/bin/env python3
"""
Fix Embedding Configuration Consistency

This script updates all hardcoded embedding model references to use
environment variables or the modernized configuration system.
"""

import os
import re
import glob
from pathlib import Path

def update_file_embedding_config(file_path: str) -> bool:
    """Update embedding model configuration in a single file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Pattern 1: Direct hardcoded embedding_model in config dicts
        content = re.sub(
            r'"embedding_model":\s*"text-embedding-3-small"',
            '"embedding_model": os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")',
            content
        )
        
        # Pattern 2: Hardcoded embedding_model in variable assignments
        content = re.sub(
            r'embedding_model\s*=\s*["\']text-embedding-3-small["\']',
            'embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")',
            content
        )
        
        # Pattern 3: Hardcoded embedding_model in get() calls
        content = re.sub(
            r'\.get\(["\']embedding_model["\'],\s*["\']text-embedding-3-small["\']\)',
            '.get("embedding_model", os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"))',
            content
        )
        
        # Only write if content changed
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        
        return False
        
    except Exception as e:
        print(f"Error updating {file_path}: {e}")
        return False

def main():
    """Main function to update all embedding configurations."""
    print("🔧 Fixing embedding configuration consistency...")
    
    # Get the project root
    project_root = Path(__file__).parent.parent
    
    # Files to update (excluding archives and tests)
    patterns = [
        "agentic/**/*.py",
        "scripts/**/*.py", 
        "main.py",
        "tests/**/*.py"
    ]
    
    # Exclude archive directories
    exclude_dirs = ["archive", "venv", "__pycache__", ".git"]
    
    updated_files = []
    total_files = 0
    
    for pattern in patterns:
        for file_path in glob.glob(str(project_root / pattern), recursive=True):
            # Skip excluded directories
            if any(exclude_dir in file_path for exclude_dir in exclude_dirs):
                continue
                
            total_files += 1
            
            # Check if file contains embedding model references
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                if 'text-embedding-3-small' in content:
                    if update_file_embedding_config(file_path):
                        updated_files.append(file_path)
                        print(f"✅ Updated: {file_path}")
                        
            except Exception as e:
                print(f"❌ Error reading {file_path}: {e}")
    
    print(f"\n📊 Summary:")
    print(f"   Total files checked: {total_files}")
    print(f"   Files updated: {len(updated_files)}")
    
    if updated_files:
        print(f"\n📝 Updated files:")
        for file_path in updated_files:
            print(f"   - {file_path}")
    
    print(f"\n✅ Embedding configuration consistency fix complete!")
    print(f"💡 All embedding model references now use OPENAI_EMBEDDING_MODEL environment variable")

if __name__ == "__main__":
    main() 