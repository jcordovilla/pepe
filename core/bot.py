import os
import discord
from discord.ext import commands
from discord import app_commands
from dotenv import load_dotenv
from core.agent import get_agent_answer
from flask import Flask
import threading
import asyncio
import logging
import json
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bot.log', mode='w', encoding='utf-8'),
        logging.StreamHandler()
    ],
    force=True  # Force reconfiguration of the root logger
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
if not DISCORD_TOKEN:
    raise ValueError("DISCORD_TOKEN environment variable is not set")

# Set up Flask app
app = Flask(__name__)

@app.route('/health')
def health_check():
    return 'OK', 200

# Set up bot with command prefix '!' (kept for potential future use)
intents = discord.Intents.default()
intents.message_content = True
intents.guilds = True
intents.messages = True
bot = commands.Bot(command_prefix='!', intents=intents)

@bot.event
async def on_ready():
    logger.info(f'{bot.user} has connected to Discord!')
    try:
        synced = await bot.tree.sync()  # Register slash commands globally
        logger.info(f"Synced {len(synced)} command(s).")
    except Exception as e:
        logger.error(f"Failed to sync commands: {e}")

@bot.tree.command(name="pepe", description="Ask Pepe the AI assistant something")
@app_commands.describe(query="Your question for Pepe")
async def pepe(interaction: discord.Interaction, query: str):
    try:
        # Log the incoming query
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'user_id': interaction.user.id,
            'username': str(interaction.user),
            'query': query,
            'channel_id': interaction.channel_id,
            'guild_id': interaction.guild_id if interaction.guild else None
        }
        logger.info(f"Query received: {json.dumps(log_data)}")
        
        # Acknowledge the interaction first
        await interaction.response.defer(ephemeral=False)
        
        # Get the response from the agent
        response = get_agent_answer(query)

        # Log the agent's response
        response_log = {
            'timestamp': datetime.utcnow().isoformat(),
            'query': query,
            'response_type': type(response).__name__,
            'response_length': len(str(response)) if isinstance(response, (str, list, dict)) else 0,
            'response_content': str(response) if isinstance(response, (str, dict)) else [str(msg) for msg in response] if isinstance(response, list) else None
        }
        logger.info(f"Agent response: {json.dumps(response_log, ensure_ascii=False)}")

        # Add header with user's question
        header = f"**Question:** {query}\n\n"

        if isinstance(response, list):
            if not response:
                logger.info(f"Empty response for query: {query}")
                await interaction.followup.send(header + "No messages found matching your query.")
                return

            current_chunk = header
            for msg in response:
                # Ensure we have all required fields
                author = msg.get('author', {})
                author_name = author.get('username', 'Unknown')
                timestamp = msg.get('timestamp', '')
                content = msg.get('content', '')
                jump_url = msg.get('jump_url', '')
                channel_name = msg.get('channel_name', 'Unknown Channel')
                
                # Format the message with proper Discord markdown
                msg_str = f"**{author_name}** ({timestamp}) in **#{channel_name}**\n{content}\n"
                if jump_url:
                    msg_str += f"[View message]({jump_url})\n"
                msg_str += "---\n"

                if len(current_chunk) + len(msg_str) > 1900:
                    try:
                        await interaction.followup.send(current_chunk)
                    except discord.NotFound:
                        logger.warning("Interaction not found when sending chunk")
                        return
                    except Exception as e:
                        logger.error(f"Error sending chunk: {str(e)}", exc_info=True)
                        return
                    current_chunk = msg_str
                else:
                    current_chunk += msg_str

            if current_chunk:
                try:
                    await interaction.followup.send(current_chunk)
                except discord.NotFound:
                    logger.warning("Interaction not found when sending final chunk")
                except Exception as e:
                    logger.error(f"Error sending final chunk: {str(e)}", exc_info=True)

        elif isinstance(response, dict):
            # Format dictionary response with header
            formatted_response = header
            if "timeframe" in response:
                formatted_response += f"**Timeframe:** {response['timeframe']}\n"
            if "channel" in response:
                formatted_response += f"**Channel:** {response['channel']}\n"
            if "summary" in response:
                formatted_response += f"\n{response['summary']}\n"
            if "messages" in response and response["messages"]:
                formatted_response += "\n**Messages:**\n"
                for msg in response["messages"]:
                    author = msg.get("author", {}).get("username", "Unknown")
                    timestamp = msg.get("timestamp", "")
                    content = msg.get("content", "")
                    jump_url = msg.get("jump_url", "")
                    formatted_response += f"\n**{author}** ({timestamp}):\n{content}\n"
                    if jump_url:
                        formatted_response += f"[View message]({jump_url})\n"
                    formatted_response += "---\n"
            try:
                await interaction.followup.send(formatted_response)
            except discord.NotFound:
                logger.warning("Interaction not found when sending dictionary response")
            except Exception as e:
                logger.error(f"Error sending dictionary response: {str(e)}", exc_info=True)
        else:
            # Format string response with header
            formatted_response = header + str(response)
            try:
                await interaction.followup.send(formatted_response)
            except discord.NotFound:
                logger.warning("Interaction not found when sending string response")
            except Exception as e:
                logger.error(f"Error sending string response: {str(e)}", exc_info=True)

    except discord.NotFound:
        logger.warning("Interaction not found - it may have timed out")
    except Exception as e:
        logger.error(f"Error in pepe command: {str(e)}", exc_info=True)
        try:
            await interaction.followup.send(f"An error occurred: {str(e)}")
        except:
            pass

def run_flask():
    app.run(host='0.0.0.0', port=8080)

# Run the bot
if __name__ == "__main__":
    # Start Flask in a separate thread
    flask_thread = threading.Thread(target=run_flask)
    flask_thread.daemon = True  # Make thread daemon so it exits when main thread exits
    flask_thread.start()
    
    # Run the Discord bot
    try:
        bot.run(DISCORD_TOKEN)
    except Exception as e:
        logger.error(f"Error running bot: {str(e)}")
        raise 