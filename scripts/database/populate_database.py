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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def run_command(command, description):
    """Run a command and return success status"""
    print(f"\n🔄 {description}")
    print(f"Running: {command}")
    print("-" * 50)
    
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            check=True, 
            capture_output=False,  # Show output in real-time
            text=True
        )
        print(f"✅ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with error code {e.returncode}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error in {description}: {e}")
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
    print("🤖 Discord Bot Database Population")
    print("=" * 50)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Check prerequisites
    if not await check_prerequisites():
        return False
    
    # Setup directories
    await check_data_directories()
    
    # Step 1: Fetch messages from Discord
    success = await run_command(
        "python core/fetch_messages.py",
        "Step 1: Fetching messages from Discord"
    )
    if not success:
        print("❌ Failed to fetch messages. Check your Discord token and bot permissions.")
        return False
    
    # Step 2: Embed and store messages in vector database
    success = await run_command(
        "python core/embed_store.py", 
        "Step 2: Processing and embedding messages"
    )
    if not success:
        print("❌ Failed to embed messages. Check your OpenAI API key.")
        return False
    
    # Step 3: Verify database population
    success = await run_command(
        "python test_database_search.py",
        "Step 3: Verifying database population"
    )
    if not success:
        print("⚠️  Database verification failed, but data may still be usable.")
    
    # Step 4: Test complete bot integration  
    success = await run_command(
        "python test_discord_bot_complete.py",
        "Step 4: Testing complete bot integration"
    )
    if not success:
        print("⚠️  Bot integration test failed, check configuration.")
    
    print("\n🎉 Database Population Complete!")
    print("=" * 50)
    print("Your Discord bot database has been populated with real message data.")
    print()
    print("📊 Next steps:")
    print("1. Run 'python main.py' to start the Discord bot")
    print("2. Test queries in Discord with /pepe commands")
    print("3. Monitor logs in logs/agentic_bot.log")
    print()
    print("✨ The bot now has access to your actual Discord message history!")
    
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
