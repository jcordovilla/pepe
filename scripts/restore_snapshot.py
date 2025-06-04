#!/usr/bin/env python3
"""
Quick restoration from snapshot - Emergency use
"""
import subprocess
import sys
from pathlib import Path

def restore_from_git_snapshot(snapshot_name):
    """Restore from Git snapshot branch"""
    try:
        print(f"🔄 Restoring from Git snapshot: {snapshot_name}")
        
        # Stash current changes if any
        result = subprocess.run(["git", "status", "--porcelain"], 
                              capture_output=True, text=True)
        if result.stdout.strip():
            print("📦 Stashing current changes...")
            subprocess.run(["git", "stash", "push", "-m", "Pre-restore stash"], check=False)
        
        # Create restoration branch from snapshot
        restore_branch = f"restore-from-{snapshot_name}"
        subprocess.run(["git", "checkout", snapshot_name], check=True)
        subprocess.run(["git", "checkout", "-b", restore_branch], check=True)
        
        print(f"✅ Restored to branch: {restore_branch}")
        print("🔄 You are now on the restored state")
        print("💡 Use 'git checkout agentic-architecture-v2' to return if needed")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Git restoration failed: {e}")
        return False

def list_available_snapshots():
    """List available snapshots for restoration"""
    print("📋 Available Snapshots for Restoration:")
    print("=" * 50)
    
    # Git snapshot branches
    try:
        result = subprocess.run(["git", "branch", "--list", "snapshot-*"], 
                              capture_output=True, text=True)
        branches = [b.strip().replace("* ", "").replace("  ", "") for b in result.stdout.split("\n") if b.strip() and "snapshot-" in b]
        if branches:
            print("🌿 Git Branch Snapshots:")
            for i, branch in enumerate(branches, 1):
                print(f"   {i}. {branch}")
            return branches
    except:
        pass
    
    return []

def interactive_restore():
    """Interactive restoration interface"""
    snapshots = list_available_snapshots()
    
    if not snapshots:
        print("❌ No snapshots available for restoration")
        return False
    
    print("\n🚨 WARNING: This will change your current working state!")
    print("💡 Make sure you've committed or stashed any important changes")
    
    choice = input(f"\nSelect snapshot to restore (1-{len(snapshots)}) or 'q' to quit: ").strip()
    
    if choice.lower() == 'q':
        print("Restoration cancelled")
        return False
    
    try:
        index = int(choice) - 1
        if 0 <= index < len(snapshots):
            snapshot_name = snapshots[index]
            confirm = input(f"\n⚠️  Restore from {snapshot_name}? (y/N): ")
            if confirm.lower() == 'y':
                return restore_from_git_snapshot(snapshot_name)
            else:
                print("Restoration cancelled")
                return False
        else:
            print("❌ Invalid selection")
            return False
    except ValueError:
        print("❌ Invalid input")
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1:
        snapshot_name = sys.argv[1]
        restore_from_git_snapshot(snapshot_name)
    else:
        interactive_restore()
