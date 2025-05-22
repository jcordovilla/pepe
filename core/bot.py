import os
import discord
from discord.ext import commands
from discord import app_commands
from dotenv import load_dotenv
from core.agent import get_agent_answer
from flask import Flask
import threading

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
bot = commands.Bot(command_prefix='!', intents=intents)

@bot.event
async def on_ready():
    print(f'{bot.user} has connected to Discord!')
    try:
        synced = await bot.tree.sync()  # Register slash commands globally
        print(f"Synced {len(synced)} command(s).")
    except Exception as e:
        print(f"Failed to sync commands: {e}")

@bot.tree.command(name="pepe", description="Ask Pepe the AI assistant something")
@app_commands.describe(query="Your question for Pepe")
async def pepe(interaction: discord.Interaction, query: str):
    await interaction.response.defer()  # Indicate bot is processing
    try:
        response = get_agent_answer(query)

        if isinstance(response, list):
            if not response:
                await interaction.followup.send("No messages found matching your query.")
                return

            current_chunk = ""
            for msg in response:
                msg_str = f"**{msg.get('author', {}).get('username', 'Unknown')}** ({msg.get('timestamp', '')})\n{msg.get('content', '')}\n[🔗 Jump to message]({msg.get('jump_url', '')})\n\n"
                if len(current_chunk) + len(msg_str) > 1900:
                    await interaction.followup.send(current_chunk)
                    current_chunk = msg_str
                else:
                    current_chunk += msg_str

            if current_chunk:
                await interaction.followup.send(current_chunk)

        elif isinstance(response, dict):
            await interaction.followup.send(str(response))
        else:
            await interaction.followup.send(str(response))

    except Exception as e:
        await interaction.followup.send(f"Error: {str(e)}")

def run_flask():
    app.run(host='0.0.0.0', port=8080)

# Run the bot
if __name__ == "__main__":
    # Start Flask in a separate thread
    flask_thread = threading.Thread(target=run_flask)
    flask_thread.start()
    
    # Run the Discord bot
    bot.run(DISCORD_TOKEN) 