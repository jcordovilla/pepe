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



        # Analytics commands
        @bot.tree.command(name="user-report", description="Generate an activity report for a user")
        @app_commands.describe(
            user="The user to analyze (mention or username)",
            days="Number of days to analyze (default: 7)"
        )
        async def user_report_command(
            interaction: discord.Interaction,
            user: str,
            days: int = 7
        ):
            try:
                await interaction.response.defer(ephemeral=False, thinking=True)

                from agentic.services.analytics_service import get_analytics_service
                analytics = get_analytics_service()

                # Extract username from mention if needed
                username = user.strip()
                if username.startswith("<@") and username.endswith(">"):
                    # Extract user ID from mention
                    user_id = username.replace("<@", "").replace(">", "").replace("!", "")
                    result = await analytics.get_user_activity_report(user_id=user_id, days=days)
                else:
                    # Use as username
                    username = username.lstrip("@")
                    result = await analytics.get_user_activity_report(username=username, days=days)

                if result.get("success"):
                    # Format response
                    embed = discord.Embed(
                        title=f"Activity Report: {result['user']}",
                        description=result.get("summary", "")[:4000],
                        color=discord.Color.blue()
                    )
                    metrics = result.get("metrics", {})
                    embed.add_field(
                        name="Stats",
                        value=f"Messages: {metrics.get('total_messages', 0)}\n"
                              f"Channels: {metrics.get('unique_channels', 0)}\n"
                              f"Reactions: {metrics.get('total_reactions_received', 0)}",
                        inline=True
                    )
                    embed.set_footer(text=f"Last {days} days")
                    await interaction.followup.send(embed=embed)
                else:
                    await interaction.followup.send(f"Could not generate report: {result.get('error')}")

            except Exception as e:
                logger.error(f"Error in user-report command: {e}", exc_info=True)
                await interaction.followup.send(f"Error generating report: {str(e)}")

        @bot.tree.command(name="channel-summary", description="Generate a summary of channel activity")
        @app_commands.describe(
            channel="The channel to analyze",
            days="Number of days to analyze (default: 7)"
        )
        async def channel_summary_command(
            interaction: discord.Interaction,
            channel: str,
            days: int = 7
        ):
            try:
                await interaction.response.defer(ephemeral=False, thinking=True)

                from agentic.services.analytics_service import get_analytics_service
                analytics = get_analytics_service()

                # Clean channel name
                channel_name = channel.lstrip("#")
                result = await analytics.get_channel_summary(channel_name=channel_name, days=days)

                if result.get("success"):
                    embed = discord.Embed(
                        title=f"Channel Summary: #{result['channel']}",
                        description=result.get("summary", "")[:4000],
                        color=discord.Color.green()
                    )
                    metrics = result.get("metrics", {})
                    embed.add_field(
                        name="Activity",
                        value=f"Messages: {metrics.get('total_messages', 0)}\n"
                              f"Contributors: {metrics.get('unique_contributors', 0)}\n"
                              f"Msgs/day: {metrics.get('avg_messages_per_day', 0)}",
                        inline=True
                    )
                    embed.set_footer(text=f"Last {days} days")
                    await interaction.followup.send(embed=embed)
                else:
                    await interaction.followup.send(f"Could not generate summary: {result.get('error')}")

            except Exception as e:
                logger.error(f"Error in channel-summary command: {e}", exc_info=True)
                await interaction.followup.send(f"Error generating summary: {str(e)}")

        @bot.tree.command(name="digest", description="Generate a community digest")
        @app_commands.describe(
            period="Time period: daily, weekly, or monthly (default: weekly)"
        )
        @app_commands.choices(period=[
            app_commands.Choice(name="Daily", value="daily"),
            app_commands.Choice(name="Weekly", value="weekly"),
            app_commands.Choice(name="Monthly", value="monthly")
        ])
        async def digest_command(
            interaction: discord.Interaction,
            period: str = "weekly"
        ):
            try:
                await interaction.response.defer(ephemeral=False, thinking=True)

                from agentic.services.analytics_service import get_analytics_service
                analytics = get_analytics_service()

                result = await analytics.get_community_digest(period=period)

                if result.get("success"):
                    embed = discord.Embed(
                        title=f"Community Digest ({period.capitalize()})",
                        description=result.get("digest", "")[:4000],
                        color=discord.Color.gold()
                    )
                    metrics = result.get("metrics", {})
                    embed.add_field(
                        name="Overview",
                        value=f"Messages: {metrics.get('total_messages', 0)}\n"
                              f"Channels: {metrics.get('unique_channels', 0)}\n"
                              f"Members: {metrics.get('unique_contributors', 0)}",
                        inline=True
                    )
                    await interaction.followup.send(embed=embed)
                else:
                    await interaction.followup.send(f"Could not generate digest: {result.get('error')}")

            except Exception as e:
                logger.error(f"Error in digest command: {e}", exc_info=True)
                await interaction.followup.send(f"Error generating digest: {str(e)}")

        @bot.tree.command(name="trends", description="Analyze trending topics")
        @app_commands.describe(days="Number of days to analyze (default: 7)")
        async def trends_command(interaction: discord.Interaction, days: int = 7):
            try:
                await interaction.response.defer(ephemeral=False, thinking=True)

                from agentic.services.analytics_service import get_analytics_service
                analytics = get_analytics_service()

                result = await analytics.get_trends(days=days)

                if result.get("success"):
                    embed = discord.Embed(
                        title=f"Trending Topics (Last {days} days)",
                        description=result.get("analysis", "")[:4000],
                        color=discord.Color.purple()
                    )
                    metrics = result.get("metrics", {})
                    topics = list(metrics.get("trending_topics", {}).keys())[:10]
                    if topics:
                        embed.add_field(
                            name="Top Keywords",
                            value=", ".join(topics),
                            inline=False
                        )
                    await interaction.followup.send(embed=embed)
                else:
                    await interaction.followup.send(f"Could not analyze trends: {result.get('error')}")

            except Exception as e:
                logger.error(f"Error in trends command: {e}", exc_info=True)
                await interaction.followup.send(f"Error analyzing trends: {str(e)}")

        @bot.tree.command(name="stats", description="Quick server statistics")
        async def stats_command(interaction: discord.Interaction):
            try:
                await interaction.response.defer(ephemeral=False, thinking=True)

                from agentic.services.analytics_service import get_analytics_service
                analytics = get_analytics_service()

                result = await analytics.get_quick_stats()

                if result.get("success"):
                    stats = result.get("stats", {})
                    embed = discord.Embed(
                        title="Server Statistics",
                        color=discord.Color.teal()
                    )
                    embed.add_field(
                        name="All Time",
                        value=f"Messages: {stats.get('total_messages', 0):,}\n"
                              f"Users: {stats.get('unique_users', 0):,}\n"
                              f"Channels: {stats.get('unique_channels', 0)}",
                        inline=True
                    )
                    embed.add_field(
                        name="This Week",
                        value=f"Messages: {stats.get('messages_this_week', 0):,}\n"
                              f"Today: {stats.get('messages_today', 0):,}",
                        inline=True
                    )
                    embed.add_field(
                        name="Top This Week",
                        value=f"Channel: #{stats.get('top_channel_this_week', 'N/A')}\n"
                              f"User: {stats.get('top_user_this_week', 'N/A')}",
                        inline=False
                    )
                    await interaction.followup.send(embed=embed)
                else:
                    await interaction.followup.send(f"Could not get stats: {result.get('error')}")

            except Exception as e:
                logger.error(f"Error in stats command: {e}", exc_info=True)
                await interaction.followup.send(f"Error getting stats: {str(e)}")

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
