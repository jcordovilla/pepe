#!/usr/bin/env python3
"""
Create comprehensive project snapshot before migration
"""
import os
import subprocess
import shutil
import json
import time
from datetime import datetime
from pathlib import Path

def print_progress_bar(iteration, total, prefix='', suffix='', length=40, fill='█'):
    """Print a progress bar to the console"""
    percent = f"{100 * (iteration / float(total)):.1f}"
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='', flush=True)

def print_snapshot_header(title):
    """Print formatted snapshot header"""
    print(f"\n{'=' * 60}")
    print(f"📸 {title}")
    print('=' * 60)

def get_current_branch():
    """Get the current branch name"""
    try:
        result = subprocess.run(["git", "branch", "--show-current"], 
                              capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return None

def create_snapshot():
    """Create complete project snapshot"""
    
    # Get current branch
    current_branch = get_current_branch()
    if not current_branch:
        print("❌ Could not determine current branch")
        return False
    
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    snapshot_name = f"snapshot-pre-migration-{timestamp}"
    
    print(f"📍 Current branch: {current_branch}")
    print(f"🔄 Creating snapshot: {snapshot_name}")
    
    try:
        # 1. Create Git snapshot branch
        subprocess.run(["git", "checkout", "-b", snapshot_name], check=True)
        
        # Add snapshot commit message
        commit_message = f"""📸 SNAPSHOT: Pre-migration state - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Source branch: {current_branch}
Purpose: Safe snapshot before architectural cleanup/migration
Contains:
- Complete legacy core/ system
- Modern agentic/ framework  
- All data files and configurations
- Test suites and documentation

This snapshot allows safe rollback if migration issues occur."""

        subprocess.run(["git", "commit", "--allow-empty", "-m", commit_message], check=True)
        
        print(f"✅ Git snapshot created: {snapshot_name}")
        
        # Return to original branch
        subprocess.run(["git", "checkout", current_branch], check=True)
        print(f"✅ Returned to original branch: {current_branch}")
        
        # 2. Create file system backup
        snapshot_dir = Path(f".snapshots/{snapshot_name}")
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        
        # Create complete project archive (excluding git and temp files)
        archive_path = snapshot_dir / "complete-project.tar.gz"
        subprocess.run([
            "tar", "-czf", str(archive_path),
            "--exclude=.git",
            "--exclude=__pycache__",
            "--exclude=venv",
            "--exclude=env",
            "--exclude=.snapshots",
            "--exclude=*.pyc",
            "--exclude=.DS_Store",
            "."
        ], check=True)
        
        # Create data-only backup
        if Path("data").exists():
            data_archive = snapshot_dir / "data-backup.tar.gz"
            subprocess.run(["tar", "-czf", str(data_archive), "data/"], check=True)
        
        # Create core-only backup (legacy system)
        if Path("core").exists():
            core_archive = snapshot_dir / "core-legacy-backup.tar.gz"
            subprocess.run(["tar", "-czf", str(core_archive), "core/"], check=True)
        
        # Copy critical configuration files
        critical_files = [".env", ".gitignore", "main.py", "requirements.txt", "launch.sh"]
        for file in critical_files:
            if Path(file).exists():
                shutil.copy2(file, snapshot_dir / f"{file}.backup")
        
        # Create snapshot manifest
        manifest = {
            "timestamp": datetime.now().isoformat(),
            "source_branch": current_branch,
            "snapshot_branch": snapshot_name,
            "files_count": len(list(Path(".").rglob("*"))),
            "data_size_mb": round(get_directory_size("data") / (1024*1024), 2) if Path("data").exists() else 0,
            "core_preserved": Path("core").exists(),
            "agentic_present": Path("agentic").exists(),
            "git_commit": get_git_commit_hash(),
            "critical_files": [f for f in critical_files if Path(f).exists()],
            "backup_files": {
                "complete_project": str(archive_path),
                "data_only": str(snapshot_dir / "data-backup.tar.gz") if Path("data").exists() else None,
                "core_legacy": str(snapshot_dir / "core-legacy-backup.tar.gz") if Path("core").exists() else None
            }
        }
        
        with open(snapshot_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
        
        # Create restoration instructions
        create_restoration_guide(snapshot_dir, snapshot_name, current_branch)
        
        print(f"✅ File system backup created: {snapshot_dir}")
        print(f"📊 Project size: {manifest['files_count']} files, {manifest['data_size_mb']} MB data")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Snapshot creation failed: {e}")
        # Try to return to original branch
        try:
            subprocess.run(["git", "checkout", current_branch], check=False)
        except:
            pass
        return False

def get_directory_size(path):
    """Get directory size in bytes"""
    total = 0
    try:
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if os.path.exists(filepath):
                    total += os.path.getsize(filepath)
    except:
        pass
    return total

def get_git_commit_hash():
    """Get current git commit hash"""
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], 
                              capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except:
        return "unknown"

def create_restoration_guide(snapshot_dir, snapshot_name, original_branch):
    """Create detailed restoration instructions"""
    guide = f"""# Snapshot Restoration Guide

## Snapshot Information
- **Created**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Source Branch**: {original_branch}
- **Snapshot Branch**: {snapshot_name}
- **Purpose**: Pre-migration safety snapshot

## Quick Restoration Options

### Option 1: Git Branch Restoration (Fastest)
```bash
# Switch to snapshot branch
git checkout {snapshot_name}

# Create working branch from snapshot
git checkout -b restore-from-snapshot-$(date +%Y%m%d-%H%M%S)

# Or merge snapshot into current branch
git checkout {original_branch}
git merge {snapshot_name}
```

### Option 2: File System Restoration
```bash
# Full project restoration
tar -xzf complete-project.tar.gz

# Data-only restoration
tar -xzf data-backup.tar.gz

# Legacy core system only
tar -xzf core-legacy-backup.tar.gz
```

### Option 3: Selective File Restoration
```bash
# Restore specific config files
cp .env.backup .env
cp .gitignore.backup .gitignore
cp main.py.backup main.py
```

## Emergency Commands
```bash
# If migration goes wrong, immediately run:
git checkout {snapshot_name}
git checkout -b emergency-restore-$(date +%Y%m%d-%H%M%S)

# Then assess and restore as needed
```

## Validation After Restoration
```bash
# Check git status
git status
git log --oneline -5

# Verify critical files exist
ls -la main.py requirements.txt .env

# Test basic functionality
python main.py --help
```

## Notes
- This snapshot preserves both legacy core/ and modern agentic/ systems
- All data files are backed up separately for safety
- Git history is preserved in the snapshot branch
- Multiple restoration paths available for different scenarios
"""
    
    with open(snapshot_dir / "RESTORATION_GUIDE.md", "w") as f:
        f.write(guide)

def list_snapshots():
    """List all available snapshots"""
    print("📋 Available Snapshots:")
    print("=" * 50)
    
    # Git snapshot branches
    try:
        result = subprocess.run(["git", "branch", "--list", "snapshot-*"], 
                              capture_output=True, text=True)
        branches = [b.strip().replace("* ", "") for b in result.stdout.split("\n") if b.strip()]
        if branches:
            print("🌿 Git Branches:")
            for branch in branches:
                print(f"   {branch}")
    except:
        pass
    
    # File system snapshots
    snapshots_dir = Path(".snapshots")
    if snapshots_dir.exists():
        print("📁 File System Snapshots:")
        for snapshot in sorted(snapshots_dir.iterdir()):
            if snapshot.is_dir():
                manifest_path = snapshot / "manifest.json"
                if manifest_path.exists():
                    try:
                        with open(manifest_path) as f:
                            manifest = json.load(f)
                        print(f"   {snapshot.name}")
                        print(f"      Source: {manifest.get('source_branch', 'unknown')}")
                        print(f"      Created: {manifest.get('timestamp', 'unknown')}")
                        print(f"      Files: {manifest.get('files_count', 0)}")
                    except:
                        print(f"   {snapshot.name} (no manifest)")
                else:
                    print(f"   {snapshot.name}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "list":
        list_snapshots()
    else:
        print("🔄 Creating comprehensive snapshot...")
        if create_snapshot():
            print("\n🎉 Snapshot created successfully!")
            print("💡 You can now safely proceed with migration")
            print("🔗 Use 'python scripts/create_snapshot.py list' to see all snapshots")
        else:
            print("\n❌ Snapshot creation failed!")
