#!/usr/bin/env python3
"""
System Status Check - Discord Bot v2

Comprehensive status check for all system components.
"""

import os
import sys
import sqlite3
import json
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def print_header(title):
    """Print formatted section header"""
    print(f"\n{'=' * 60}")
    print(f"🔍 {title}")
    print('=' * 60)

def print_status(component, status, details=""):
    """Print component status"""
    icon = "✅" if status else "❌"
    print(f"{icon} {component}: {'OK' if status else 'MISSING'}")
    if details:
        print(f"   └─ {details}")

def check_environment():
    """Check environment variables"""
    print_header("Environment Configuration")
    
    env_file = Path(".env")
    if env_file.exists():
        print_status("Environment file", True, f"{env_file.stat().st_size} bytes")
        
        # Check for key variables (without exposing them)
        from dotenv import load_dotenv
        load_dotenv()
        
        discord_token = bool(os.getenv("DISCORD_TOKEN"))
        openai_key = bool(os.getenv("OPENAI_API_KEY"))
        
        print_status("Discord Token", discord_token)
        print_status("OpenAI API Key", openai_key)
    else:
        print_status("Environment file", False, "Create .env file")

def check_databases():
    """Check database status"""
    print_header("Database Status")
    
    databases = {
        "ChromaDB Vector Store": "data/chromadb/chroma.sqlite3",
        "Conversation Memory": "data/conversation_memory.db", 
        "Analytics Database": "data/analytics.db",
        "SQLite Messages": "data/discord_messages.db"
    }
    
    for name, path in databases.items():
        db_path = Path(path)
        if db_path.exists():
            size_mb = db_path.stat().st_size / (1024 * 1024)
            
            # Try to get record count for SQLite databases
            if path.endswith('.db') or path.endswith('.sqlite3'):
                try:
                    conn = sqlite3.connect(db_path)
                    cursor = conn.cursor()
                    
                    if 'chroma.sqlite3' in path:
                        cursor.execute("SELECT COUNT(*) FROM embeddings")
                        count = cursor.fetchone()[0]
                        details = f"{size_mb:.1f}MB, {count} embeddings"
                    elif 'conversation_memory.db' in path:
                        cursor.execute("SELECT COUNT(*) FROM conversations")
                        count = cursor.fetchone()[0]
                        details = f"{size_mb:.1f}MB, {count} conversations"
                    elif 'discord_messages.db' in path:
                        cursor.execute("SELECT COUNT(*) FROM messages")
                        count = cursor.fetchone()[0]
                        details = f"{size_mb:.1f}MB, {count} messages"
                    else:
                        details = f"{size_mb:.1f}MB"
                    
                    conn.close()
                    print_status(name, True, details)
                except Exception as e:
                    print_status(name, True, f"{size_mb:.1f}MB (query failed: {e})")
            else:
                print_status(name, True, f"{size_mb:.1f}MB")
        else:
            print_status(name, False, "File not found")

def check_data_directories():
    """Check data directory structure"""
    print_header("Data Directory Structure")
    
    directories = {
        "Fetched Messages": "data/fetched_messages",
        "Vector Store": "data/chromadb", 
        "Cache": "data/cache",
        "Exports": "data/exports",
        "Processing Markers": "data/processing_markers"
    }
    
    for name, path in directories.items():
        dir_path = Path(path)
        if dir_path.exists():
            if dir_path.is_dir():
                file_count = len(list(dir_path.iterdir()))
                print_status(name, True, f"{file_count} files/dirs")
            else:
                print_status(name, False, "Path exists but not a directory")
        else:
            print_status(name, False, "Directory not found")

def check_core_files():
    """Check core application files"""
    print_header("Core Application Files")
    
    core_files = {
        "Main Bot": "main.py",
        "Pipeline Runner": "run_pipeline.py",
        "Launch Script": "launch.sh",
        "Requirements": "requirements.txt",
        "Agentic Core": "agentic/__init__.py",
        "Discord Interface": "agentic/interfaces/discord_interface.py",
        "Vector Store": "agentic/vectorstore/persistent_store.py"
    }
    
    for name, path in core_files.items():
        file_path = Path(path)
        if file_path.exists():
            size_kb = file_path.stat().st_size / 1024
            print_status(name, True, f"{size_kb:.1f}KB")
        else:
            print_status(name, False, "File not found")

def check_logs():
    """Check log files"""
    print_header("Log Files")
    
    logs_dir = Path("logs")
    if logs_dir.exists():
        log_files = list(logs_dir.glob("*.log"))
        if log_files:
            for log_file in sorted(log_files)[-3:]:  # Show last 3 log files
                size_kb = log_file.stat().st_size / 1024
                modified = datetime.fromtimestamp(log_file.stat().st_mtime)
                print_status(log_file.name, True, 
                           f"{size_kb:.1f}KB, modified {modified.strftime('%Y-%m-%d %H:%M')}")
        else:
            print_status("Log files", False, "No log files found")
    else:
        print_status("Logs directory", False, "Directory not found")

def get_system_summary():
    """Get overall system summary"""
    print_header("System Summary")
    
    # Check critical components
    critical_files = [
        Path("main.py"),
        Path("data/chromadb/chroma.sqlite3"),
        Path("agentic/interfaces/discord_interface.py")
    ]
    
    missing_critical = [f for f in critical_files if not f.exists()]
    
    if not missing_critical:
        print("🎉 All critical components are present!")
        print("✅ System appears ready to run")
        
        # Check if vector store has data
        chroma_db = Path("data/chromadb/chroma.sqlite3")
        if chroma_db.exists():
            try:
                conn = sqlite3.connect(chroma_db)
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM embeddings")
                count = cursor.fetchone()[0]
                conn.close()
                
                if count > 0:
                    print(f"📊 Vector store has {count} embeddings")
                    print("🚀 Ready to start: python main.py")
                else:
                    print("⚠️  Vector store is empty")
                    print("📥 Run pipeline: python run_pipeline.py")
            except Exception:
                print("⚠️  Could not check vector store content")
    else:
        print("❌ Missing critical components:")
        for missing in missing_critical:
            print(f"   • {missing}")
        print("🔧 Please set up missing components before running")

def main():
    """Main status check function"""
    print("🤖 Discord Bot v2 - System Status Check")
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Change to script directory
    script_dir = Path(__file__).parent
    if script_dir.name == 'scripts':
        os.chdir(script_dir.parent)
    
    try:
        check_environment()
        check_databases()
        check_data_directories()
        check_core_files()
        check_logs()
        get_system_summary()
        
        print(f"\n{'=' * 60}")
        print("✅ Status check complete!")
        print("📖 See QUICKSTART.md for next steps")
        print(f"{'=' * 60}")
        
    except Exception as e:
        print(f"\n❌ Error during status check: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
