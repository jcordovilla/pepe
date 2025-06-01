#!/usr/bin/env python3
"""
Agentic Discord Bot - Main Entry Point

This is the main entry point for the new agentic Discord bot that replaces
the old core/ implementation with a sophisticated multi-agent architecture.
"""

import os
import asyncio
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/agentic_bot.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


async def main():
    """Main function to start the agentic Discord bot."""
    
    # Ensure logs directory exists
    os.makedirs('logs', exist_ok=True)
    
    logger.info("🚀 Starting Agentic Discord Bot...")
    
    try:
        # Import and initialize the Discord interface
        from agentic.interfaces.discord_interface import DiscordInterface
        
        # Configuration for the agentic system
        config = {
            'orchestrator': {
                'memory_config': {
                    'db_path': 'data/conversation_memory.db',
                    'max_history_length': 50,
                    'context_window_hours': 24
                }
            },
            'vectorstore': {
                'persist_directory': 'data/vectorstore',
                'collection_name': 'discord_messages',
                'embedding_model': 'text-embedding-3-small'
            },
            'cache': {
                'redis_url': 'redis://localhost:6379',
                'default_ttl': 3600
            },
            'discord': {
                'token': os.getenv('DISCORD_TOKEN'),
                'guild_id': os.getenv('GUILD_ID'),
                'command_prefix': '!'
            }
        }
        
        # Initialize Discord interface
        discord_bot = DiscordInterface(config)
        
        # Start the bot
        await discord_bot.start()
        
    except KeyboardInterrupt:
        logger.info("🛑 Bot stopped by user")
    except Exception as e:
        logger.error(f"❌ Error starting bot: {e}")
        raise
    finally:
        logger.info("🔄 Shutting down agentic bot...")


if __name__ == "__main__":
    """Entry point when running the script directly."""
    asyncio.run(main())
