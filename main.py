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
        
        # Configuration for the agentic system
        config = {
            'orchestrator': {
                'memory_config': {
                    'db_path': 'data/conversation_memory.db',
                    'max_history_length': 50,
                    'context_window_hours': 24
                }
            },
            'vector_store': {
                'persist_directory': './data/chromadb',
                'collection_name': 'discord_messages',
                'embedding_model': 'text-embedding-3-small',
                'batch_size': 100
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

        @bot.tree.command(name="vectorstats", description="Show vector store statistics")
        async def vectorstats_command(interaction: discord.Interaction):
            """Show vector store statistics"""
            try:
                logger.info("🎯 Received vectorstats command")
                
                # Defer the interaction to prevent timeout
                await interaction.response.defer(ephemeral=False)
                logger.info("✅ Vectorstats interaction deferred successfully")
                
                # Get vector store stats using the correct method
                stats = await discord_bot.agent_api.get_system_stats()
                
                if not stats.get("success"):
                    await interaction.followup.send(f"Error getting vector store statistics: {stats.get('error')}")
                    return
                    
                vector_stats = stats.get("vector_store", {})
                
                # Format the response
                response = "📊 **Vector Store Statistics**\n\n"
                response += f"**Collection:** {vector_stats.get('collection_name', 'N/A')}\n"
                response += f"**Total Documents:** {vector_stats.get('total_documents', 0)}\n"
                response += f"**Embedding Model:** {vector_stats.get('embedding_model', 'N/A')}\n"
                response += f"**Last Updated:** {vector_stats.get('last_updated', 'N/A')}\n\n"
                
                # Add content statistics if available
                if "content_stats" in vector_stats:
                    content_stats = vector_stats["content_stats"]
                    response += "**Content Statistics:**\n"
                    response += f"• Total Tokens: {content_stats.get('total_tokens', 0)}\n"
                    response += f"• Average Length: {content_stats.get('avg_length', 0):.1f} tokens\n"
                    response += f"• Longest Document: {content_stats.get('max_length', 0)} tokens\n\n"
                
                # Add top channels if available
                if "top_channels" in vector_stats:
                    response += "**Top Channels:**\n"
                    for channel in vector_stats["top_channels"][:5]:
                        response += f"• #{channel['name']}: {channel['count']} messages\n"
                    response += "\n"
                
                # Add top authors if available
                if "top_authors" in vector_stats:
                    response += "**Top Authors:**\n"
                    for author in vector_stats["top_authors"][:5]:
                        response += f"• {author['username']}: {author['count']} messages\n"
                
                await interaction.followup.send(response)
                logger.info("✅ Vectorstats response sent successfully")
                
            except discord.errors.NotFound as nf_error:
                logger.error(f"❌ Discord interaction not found (likely timed out): {nf_error}")
                # Don't try to respond if the interaction is invalid
            except Exception as e:
                logger.error(f"❌ Error in vectorstats command: {e}", exc_info=True)
                try:
                    if interaction.response.is_done():
                        await interaction.followup.send(f"Error getting vector store statistics: {str(e)}")
                    else:
                        await interaction.response.send_message(f"Error getting vector store statistics: {str(e)}")
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
