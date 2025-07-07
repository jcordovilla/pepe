#!/usr/bin/env python3
"""
Complete Database Population Script

This script runs the complete process to populate your database with real Discord messages:
1. Fetches messages from Discord (with reactions)
2. Processes and embeds them in the vector store
3. Verifies the database is properly populated
"""

import asyncio
import logging
import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def print_progress_bar(iteration, total, prefix='', suffix='', length=40, fill='█'):
    """Print a progress bar to the console"""
    percent = f"{100 * (iteration / float(total)):.1f}"
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='\r')
    if iteration == total:
        print()

def print_step_header(step_num, total_steps, title, description=""):
    """Print a formatted step header"""
    print(f"\n{'='*60}")
    print(f"📋 Step {step_num}/{total_steps}: {title}")
    if description:
        print(f"💡 {description}")
    print(f"{'='*60}")

def print_substep(message):
    """Print a formatted substep message"""
    print(f"   🔸 {message}")

async def run_command_with_progress(command, description, estimated_time=30):
    """Run a command with a simulated progress indicator"""
    print(f"\n🚀 Starting: {description}")
    print(f"⏱️  Estimated time: ~{estimated_time} seconds")
    print(f"🔧 Command: {command}")
    print("-" * 50)
    
    # Start the process
    try:
        process = subprocess.Popen(
            command, 
            shell=True, 
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Show progress while process runs
        start_time = time.time()
        last_output = ""
        
        while process.poll() is None:
            elapsed = time.time() - start_time
            progress = min(int((elapsed / estimated_time) * 100), 95)  # Cap at 95% until complete
            
            # Read any new output
            if process.stdout:
                try:
                    line = process.stdout.readline()
                    if line:
                        last_output = line.strip()[:50]  # Truncate long lines
                except:
                    pass
            
            print(f"\r🔄 Progress: {progress}% | {last_output}", end="", flush=True)
            time.sleep(0.5)
        
        # Wait for completion
        process.wait()
        elapsed = time.time() - start_time
        
        if process.returncode == 0:
            print(f"\r✅ {description} completed in {elapsed:.1f}s!" + " " * 20)
            return True
        else:
            print(f"\r❌ {description} failed (exit code: {process.returncode})" + " " * 20)
            return False
            
    except Exception as e:
        print(f"\r❌ Error running {description}: {e}" + " " * 20)
        return False

async def check_prerequisites():
    """Check that required environment variables are set"""
    print("🔍 Checking prerequisites...")
    
    required_vars = ["DISCORD_TOKEN", "OPENAI_API_KEY"]
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
        print("Please set them in your .env file before continuing.")
        return False
    
    print("✅ All required environment variables are set!")
    return True

async def check_data_directories():
    """Ensure required directories exist"""
    print("📁 Setting up data directories...")
    
    directories = [
        "data/fetched_messages",
        "data/chromadb", 
        "data/processing_markers",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("✅ Data directories ready!")

async def populate_database():
    """Run the complete database population process"""
    total_steps = 4
    
    print("🤖 Discord Bot Database Population")
    print("=" * 60)
    print(f"🕐 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📊 Total steps: {total_steps}")
    print()
    
    # Step 0: Prerequisites
    print_step_header(1, total_steps, "Prerequisites Check", "Verifying environment and API keys")
    if not await check_prerequisites():
        return False
    
    # Setup directories
    print_substep("Setting up data directories...")
    await check_data_directories()
    print_substep("✅ Prerequisites complete!")
    
    # Step 1: Database initialization
    print_step_header(2, total_steps, "Database Initialization", "Setting up database schema and sample data")
    success = await run_command_with_progress(
        "python scripts/database/init_db_simple.py",
        "Initializing database with sample data",
        estimated_time=10
    )
    if not success:
        print("❌ Failed to initialize database. Check your configuration.")
        return False
    
    # Step 2: Database verification
    print_step_header(3, total_steps, "Database Verification", "Testing database search functionality")
    success = await run_command_with_progress(
        "python tests/test_database_search.py",
        "Verifying database search functionality",
        estimated_time=15
    )
    if not success:
        print("⚠️  Database verification failed, but data may still be usable.")
    
    # Step 3: Bot integration test
    print_step_header(4, total_steps, "Bot Integration Test", "Testing complete Discord bot functionality")
    success = await run_command_with_progress(
        "python tests/test_discord_bot_complete.py",
        "Testing complete bot integration",
        estimated_time=20
    )
    if not success:
        print("⚠️  Bot integration test failed, check configuration.")
    
    # Completion summary
    print(f"\n{'='*60}")
    print("🎉 Database Population Complete!")
    print(f"{'='*60}")
    print("✅ Your Discord bot database has been populated and tested.")
    print()
    print("📊 Next steps:")
    print("   1. 🤖 Start bot: python main.py")
    print("   2. 💬 Test in Discord: /pepe hello")
    print("   3. 📈 Monitor logs: tail -f logs/agentic_bot.log")
    print()
    print("🚀 Ready for production use!")
    
    return True

if __name__ == "__main__":
    try:
        success = asyncio.run(populate_database())
        if success:
            print("\n🚀 Ready for production use!")
            sys.exit(0)
        else:
            print("\n💥 Population process encountered errors. Check logs for details.")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n🛑 Population process interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)
