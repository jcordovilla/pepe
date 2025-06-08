import os
import discord
from discord.ext import commands
from discord import app_commands
from core.agent import get_agent_answer
from core.config import get_config
from flask import Flask
import threading
import asyncio
import logging
import json
import time
from datetime import datetime

# Import query logging system
from db.query_logs import log_query_start, update_query_analysis, log_query_completion, log_simple_query

# Load configuration
config = get_config()

# Create logs directory if it doesn't exist
logs_dir = 'logs'
if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)

# Generate log filename with timestamp
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_filename = os.path.join(logs_dir, f'bot_{timestamp}.log')

# Set up logging
logging.basicConfig(
    level=getattr(logging, config.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename, mode='a', encoding='utf-8'),
        logging.StreamHandler()
    ],
    force=True
)
logger = logging.getLogger(__name__)

# Use Discord token from config
DISCORD_TOKEN = config.discord_token

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
    query_log_id = -1
    start_time = time.time()
    
    try:
        # Start query logging
        channel_name = interaction.channel.name if hasattr(interaction.channel, 'name') else None
        query_log_id = log_query_start(
            user_id=str(interaction.user.id),
            username=str(interaction.user),
            query_text=query,
            guild_id=str(interaction.guild_id) if interaction.guild else None,
            channel_id=str(interaction.channel_id),
            channel_name=channel_name
        )
        
        # Log the incoming query (legacy logging)
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'user_id': interaction.user.id,
            'username': str(interaction.user),
            'query': query,
            'channel_id': interaction.channel_id,
            'guild_id': interaction.guild_id if interaction.guild else None,
            'query_log_id': query_log_id
        }
        logger.info(f"Query received: {json.dumps(log_data)}")
        
        # Acknowledge the interaction first
        await interaction.response.defer(ephemeral=False)
        
        # Import analyze_query_type for debugging capabilities
        from core.agent import analyze_query_type
        
        # Analyze query for logging and debugging
        query_analysis = analyze_query_type(query)
        logger.info(f"Query analysis: {json.dumps(query_analysis)}")
        
        # Update query log with analysis
        if query_log_id > 0:
            update_query_analysis(query_log_id, query_analysis)
        
        # Get the response from the enhanced agent
        response_start_time = time.time()
        response = get_agent_answer(query)
        response_time = int((time.time() - response_start_time) * 1000)
        
        total_processing_time = int((time.time() - start_time) * 1000)

        # Log the agent's response with analysis info (legacy logging)
        response_log = {
            'timestamp': datetime.utcnow().isoformat(),
            'query': query,
            'query_type': query_analysis.get('query_type', 'unknown'),
            'strategy': query_analysis.get('strategy', 'unknown'),
            'confidence': query_analysis.get('confidence', 0.0),
            'response_type': type(response).__name__,
            'response_length': len(str(response)) if isinstance(response, (str, list, dict)) else 0,
            'processing_time_ms': total_processing_time,
            'query_log_id': query_log_id
        }
        logger.info(f"Agent response: {json.dumps(response_log, ensure_ascii=False)}")

        # Complete query logging
        if query_log_id > 0:
            response_text = str(response) if response else None
            search_results_count = len(response) if isinstance(response, list) else (len(response.get('messages', [])) if isinstance(response, dict) else None)
            
            log_query_completion(
                query_log_id=query_log_id,
                response_text=response_text,
                response_type=type(response).__name__,
                response_status='success',
                processing_time_ms=total_processing_time,
                search_results_count=search_results_count,
                performance_metrics={
                    'llm_generation_time_ms': response_time,
                    'k_parameter': query_analysis.get('k_parameter'),
                    'strategy': query_analysis.get('strategy')
                }
            )

            # Also log to simple text file for easy searching
            log_simple_query(
                user_id=str(interaction.user.id),
                username=str(interaction.user),
                query_text=query,
                response_text=str(response),
                interface="discord",
                guild_id=str(interaction.guild_id) if interaction.guild else None,
                channel_name=interaction.channel.name if hasattr(interaction.channel, 'name') else None
            )

        # Add enhanced header with query analysis for transparency
        strategy_emojis = {
            'data_status': '📊',
            'channel_list': '📋', 
            'resources_only': '📚',
            'hybrid_search': '🔍',
            'agent_summary': '📝',
            'messages_only': '💬'
        }
        
        strategy = query_analysis.get('strategy', 'messages_only')
        emoji = strategy_emojis.get(strategy, '🤖')
        
        header = f"{emoji} **Question:** {query}\n"
        if query_analysis.get('confidence', 0) > 0.8:
            header += f"*Using {strategy.replace('_', ' ').title()} Strategy*\n\n"
        else:
            header += "\n"

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
                    msg_str = f"\n**{author}** ({timestamp}):\n{content}\n"
                    if jump_url:
                        msg_str += f"[View message]({jump_url})\n"
                    msg_str += "---\n"
                    # Chunking for Discord 2000 char limit
                    if len(formatted_response) + len(msg_str) > 1900:
                        try:
                            await interaction.followup.send(formatted_response)
                        except discord.NotFound:
                            logger.warning("Interaction not found when sending dictionary chunk")
                            return
                        except Exception as e:
                            logger.error(f"Error sending dictionary chunk: {str(e)}", exc_info=True)
                            return
                        formatted_response = msg_str
                    else:
                        formatted_response += msg_str
                if formatted_response:
                    try:
                        await interaction.followup.send(formatted_response)
                    except discord.NotFound:
                        logger.warning("Interaction not found when sending final dictionary chunk")
                    except Exception as e:
                        logger.error(f"Error sending final dictionary chunk: {str(e)}", exc_info=True)
        else:
            # Format string response with header
            formatted_response = header + str(response)
            # Chunk if too long
            while formatted_response:
                chunk = formatted_response[:1900]
                formatted_response = formatted_response[1900:]
                try:
                    await interaction.followup.send(chunk)
                except discord.NotFound:
                    logger.warning("Interaction not found when sending string response")
                    break
                except Exception as e:
                    logger.error(f"Error sending string response: {str(e)}", exc_info=True)
                    break

    except discord.NotFound:
        logger.warning("Interaction not found - it may have timed out")
        # Log the failure
        if query_log_id > 0:
            log_query_completion(
                query_log_id=query_log_id,
                response_text=None,
                response_type='None',
                response_status='timeout',
                processing_time_ms=int((time.time() - start_time) * 1000),
                error_message="Interaction timed out"
            )
        
        # Log simple error entry for timeout
        log_simple_query(
            user_id=str(interaction.user.id) if hasattr(interaction, 'user') else 'unknown',
            username=str(interaction.user) if hasattr(interaction, 'user') else 'Unknown User',
            query_text=query if 'query' in locals() else 'Unknown Query',
            response_text="ERROR: Interaction timed out",
            interface="discord"
        )

    except Exception as e:
        logger.error(f"Error in pepe command: {str(e)}", exc_info=True)
        # Log the failure
        if query_log_id > 0:
            log_query_completion(
                query_log_id=query_log_id,
                response_text=None,
                response_type='None',
                response_status='error',
                processing_time_ms=int((time.time() - start_time) * 1000),
                error_message=str(e)
            )
        
        # Log simple error entry
        log_simple_query(
            user_id=str(interaction.user.id) if hasattr(interaction, 'user') else 'unknown',
            username=str(interaction.user) if hasattr(interaction, 'user') else 'Unknown User',
            query_text=query if 'query' in locals() else 'Unknown Query',
            response_text=f"ERROR: {str(e)}",
            interface="discord"
        )
        
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