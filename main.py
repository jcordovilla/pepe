#!/usr/bin/env python3
"""
Agentic Discord Bot - Main Entry Point

This is the main entry point for the modernized Discord bot with unified
agentic architecture integrating battle-tested legacy patterns.
"""

import os
import asyncio
import logging
from dotenv import load_dotenv
import discord
from discord import app_commands

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
        
        # Import modernized configuration
        from agentic.config.modernized_config import get_modernized_config
        
        # Configuration for the agentic system
        config = get_modernized_config()
        
        # Initialize Discord interface
        discord_bot = DiscordInterface(config)
        bot = await discord_bot.setup_bot()
        
        # Define commands
        @bot.tree.command(name="pepe", description="Ask the AI assistant something")
        @app_commands.describe(query="Your question or search query")
        async def pepe_command(interaction: discord.Interaction, query: str):
            try:
                logger.info(f"🎯 Received pepe command: {query[:50]}...")
                
                # Defer the interaction IMMEDIATELY
                # The Discord interface will know the interaction was already deferred
                try:
                    await asyncio.wait_for(
                        interaction.response.defer(ephemeral=False, thinking=True),
                        timeout=1.5
                    )
                    logger.info("✅ Interaction deferred successfully by main handler")
                except asyncio.TimeoutError:
                    logger.error("❌ Initial defer timed out, continuing anyway")
                except Exception as defer_error:
                    logger.error(f"❌ Failed to defer in main handler: {defer_error}")
                
                # Handle the command - notify that the interaction is already deferred
                await discord_bot.handle_slash_command(interaction, query)
                logger.info("✅ Slash command handled successfully")
                
            except discord.errors.NotFound as nf_error:
                logger.error(f"❌ Discord interaction not found (likely timed out): {nf_error}")
                # Don't try to respond if the interaction is invalid
            except Exception as e:
                logger.error(f"❌ Error in pepe command: {e}", exc_info=True)
                try:
                    if interaction.response.is_done():
                        await interaction.followup.send(f"❌ An error occurred: {str(e)}")
                    else:
                        await interaction.response.send_message(f"❌ An error occurred: {str(e)}")
                except Exception as send_error:
                    logger.error(f"❌ Failed to send error message: {send_error}")



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
