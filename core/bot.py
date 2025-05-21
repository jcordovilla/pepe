import os
import discord
from discord.ext import commands
from dotenv import load_dotenv
from core.agent import get_agent_answer

# Load environment variables
load_dotenv()
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
if not DISCORD_TOKEN:
    raise ValueError("DISCORD_TOKEN environment variable is not set")

# Set up bot with command prefix '!'
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents)

@bot.event
async def on_ready():
    print(f'{bot.user} has connected to Discord!')

@bot.command(name='pepe')
async def pepe(ctx, *, query: str):
    """
    Command to interact with the AI agent.
    Usage: !pepe <your question>
    Example: !pepe what was discussed about python in the last 2 days?
    """
    try:
        # Show typing indicator while processing
        async with ctx.typing():
            # Get response from agent
            response = get_agent_answer(query)
            
            # Handle different response types
            if isinstance(response, list):
                # If response is a list of messages
                if not response:
                    await ctx.send("No messages found matching your query.")
                    return
                
                # Send messages in chunks to avoid Discord's message length limit
                current_chunk = ""
                for msg in response:
                    msg_str = f"**{msg.get('author', {}).get('username', 'Unknown')}** ({msg.get('timestamp', '')})\n{msg.get('content', '')}\n[🔗 Jump to message]({msg.get('jump_url', '')})\n\n"
                    
                    if len(current_chunk) + len(msg_str) > 1900:  # Discord's limit is 2000
                        await ctx.send(current_chunk)
                        current_chunk = msg_str
                    else:
                        current_chunk += msg_str
                
                if current_chunk:
                    await ctx.send(current_chunk)
            
            elif isinstance(response, dict):
                # If response is a dictionary (e.g., data availability info)
                await ctx.send(str(response))
            
            else:
                # If response is a string or other type
                await ctx.send(str(response))
    
    except Exception as e:
        await ctx.send(f"Error: {str(e)}")

# Run the bot
if __name__ == "__main__":
    bot.run(DISCORD_TOKEN) 