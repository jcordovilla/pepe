"""Discord Interface for Agentic RAG System

This module provides the Discord-specific interface for the agentic RAG framework,
replacing the existing LangChain-based agent with a sophisticated multi-agent system.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass

import discord
from discord.ext import commands

from ..interfaces.agent_api import AgentAPI
from ..agents.base_agent import AgentRole, TaskStatus
from ..memory.conversation_memory import ConversationMemory
from ..cache.smart_cache import SmartCache

logger = logging.getLogger(__name__)


@dataclass
class DiscordContext:
    """Discord-specific context information"""
    user_id: int
    username: str
    channel_id: int
    guild_id: Optional[int]
    channel_name: str
    guild_name: Optional[str]
    timestamp: datetime


class DiscordInterface:
    """
    Discord interface for the agentic RAG system.
    
    This class provides Discord-specific functionality including:
    - Message formatting and chunking for Discord's limits
    - User context management
    - Conversation history tracking
    - Performance monitoring and analytics
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        agent_api: Optional[AgentAPI] = None,
        cache_enabled: bool = True,
        max_message_length: int = 1900,
        enable_analytics: bool = True
    ):
        """
        Initialize Discord interface.
        
        Args:
            config: Configuration dictionary for the agentic system
            agent_api: Agent API instance (created if None)
            cache_enabled: Whether to enable caching
            max_message_length: Maximum Discord message length
            enable_analytics: Whether to track analytics
        """
        self.config = config
        self.agent_api = agent_api or AgentAPI(config)
        self.cache_enabled = cache_enabled
        self.max_message_length = max_message_length
        self.enable_analytics = enable_analytics
        
        # Initialize components
        memory_config = config.get('orchestrator', {}).get('memory_config', {})
        self.memory = ConversationMemory(memory_config)
        cache_config = config.get('cache', {})
        self.cache = SmartCache(cache_config) if cache_enabled else None
        
        # Performance tracking
        self.query_count = 0
        self.error_count = 0
        self.average_response_time = 0.0
        
        # Initialize Discord bot
        self.bot: Optional[commands.Bot] = None
        self.discord_config = config.get('discord', {})
        
        logger.info("Discord interface initialized")
    
    async def setup_bot(self) -> commands.Bot:
        """Set up the Discord bot instance"""
        if self.bot is not None:
            return self.bot
            
        # Set up Discord intents
        intents = discord.Intents.default()
        intents.message_content = True
        intents.guilds = True
        intents.messages = True
        
        # Create bot instance
        command_prefix = self.discord_config.get('command_prefix', '!')
        self.bot = commands.Bot(command_prefix=command_prefix, intents=intents)
        
        # Set up event handlers
        @self.bot.event
        async def on_ready():
            if self.bot:
                logger.info(f'{self.bot.user} has connected to Discord!')
                try:
                    synced = await self.bot.tree.sync()
                    logger.info(f"Synced {len(synced)} command(s)")
                except Exception as e:
                    logger.error(f"Failed to sync commands: {e}")
        
        # Set up slash command
        @self.bot.tree.command(name="pepe", description="Ask the AI assistant something")
        async def pepe_command(interaction: discord.Interaction, query: str):
            await self.handle_slash_command(interaction, query)
            
        return self.bot
    
    async def start(self):
        """Start the Discord bot"""
        token = self.discord_config.get('token')
        if not token:
            raise ValueError("Discord token not found in configuration")
            
        # Set up bot first
        bot = await self.setup_bot()
        
        # Start the bot
        await bot.start(token)
    
    async def process_query(
        self,
        query: str,
        discord_context: DiscordContext,
        interaction: Optional[discord.Interaction] = None
    ) -> List[str]:
        """
        Process a user query through the agentic system.
        
        Args:
            query: User's question
            discord_context: Discord-specific context
            interaction: Discord interaction for progress updates
            
        Returns:
            List of formatted message chunks ready for Discord
        """
        start_time = datetime.now()
        session_id = f"discord_{discord_context.user_id}_{int(start_time.timestamp())}"
        
        try:
            # Log query
            await self._log_query(query, discord_context)
            
            # Check cache first
            if self.cache_enabled:
                cached_response = await self._check_cache(query, discord_context.user_id)
                if cached_response:
                    logger.info(f"Cache hit for query: {query[:50]}...")
                    return await self._format_response(cached_response, query, discord_context)
            
            # Update user context
            await self._update_user_context(discord_context)
            
            # Send typing indicator if interaction provided
            if interaction and not interaction.response.is_done():
                await interaction.response.defer(ephemeral=False)
            
            # Process through agentic system
            result = await self.agent_api.query(
                query=query,
                user_id=str(discord_context.user_id),
                context={
                    "platform": "discord",
                    "channel_id": discord_context.channel_id,
                    "guild_id": discord_context.guild_id,
                    "timestamp": discord_context.timestamp.isoformat()
                }
            )
            
            # Cache successful results
            if self.cache_enabled and result.get("status") == "success":
                await self._cache_response(query, discord_context.user_id, result)
            
            # Format for Discord
            formatted_messages = await self._format_response(result, query, discord_context)
            
            # Update analytics
            await self._update_analytics(start_time, success=True)
            
            # Store conversation
            await self._store_conversation(query, result, discord_context, session_id)
            
            return formatted_messages
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}", exc_info=True)
            await self._update_analytics(start_time, success=False)
            
            error_message = self._format_error_message(str(e))
            return [error_message]
    
    async def _format_response(
        self,
        result: Dict[str, Any],
        query: str,
        discord_context: DiscordContext
    ) -> List[str]:
        """Format agent response for Discord display"""
        messages = []
        
        # Add header with user's question
        header = f"**Question:** {query}\n\n"
        
        if result.get("status") == "error":
            return [header + f"**Error:** {result.get('message', 'Unknown error occurred')}"]
        
        response_data = result.get("response", {})
        
        # Handle different response types
        if "messages" in response_data:
            messages.extend(await self._format_message_list(
                response_data["messages"], header, response_data
            ))
        elif "summary" in response_data:
            messages.extend(await self._format_summary_response(
                response_data, header
            ))
        elif "answer" in response_data:
            messages.extend(await self._format_text_response(
                response_data["answer"], header
            ))
        else:
            # Fallback for unknown response format
            messages.extend(await self._format_text_response(
                str(response_data), header
            ))
        
        # Add performance info if enabled
        if self.enable_analytics and "execution_time" in result:
            perf_info = f"\n*Processed in {result['execution_time']:.2f}s*"
            if messages:
                messages[-1] += perf_info
        
        return messages if messages else [header + "No results found."]
    
    async def _format_message_list(
        self,
        messages: List[Dict[str, Any]],
        header: str,
        metadata: Dict[str, Any]
    ) -> List[str]:
        """Format a list of Discord messages"""
        if not messages:
            return [header + "No messages found matching your query."]
        
        chunks = []
        current_chunk = header
        
        # Add metadata if available
        if "timeframe" in metadata:
            current_chunk += f"**Timeframe:** {metadata['timeframe']}\n"
        if "channel" in metadata:
            current_chunk += f"**Channel:** {metadata['channel']}\n"
        if "total_count" in metadata:
            current_chunk += f"**Total Messages:** {metadata['total_count']}\n"
        
        current_chunk += "\n**Messages:**\n\n"
        
        for msg in messages:
            author = msg.get('author', {})
            author_name = author.get('username', 'Unknown')
            timestamp = msg.get('timestamp', '')
            content = msg.get('content', '')
            jump_url = msg.get('jump_url', '')
            channel_name = msg.get('channel_name', 'Unknown Channel')
            
            # Format message
            msg_str = f"**{author_name}** ({timestamp}) in **#{channel_name}**\n{content}\n"
            if jump_url:
                msg_str += f"[View message]({jump_url})\n"
            msg_str += "───\n"
            
            # Check if we need to start a new chunk
            if len(current_chunk) + len(msg_str) > self.max_message_length:
                chunks.append(current_chunk)
                current_chunk = msg_str
            else:
                current_chunk += msg_str
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    async def _format_summary_response(
        self,
        response_data: Dict[str, Any],
        header: str
    ) -> List[str]:
        """Format a summary response"""
        content = header
        
        if "timeframe" in response_data:
            content += f"**Timeframe:** {response_data['timeframe']}\n"
        if "channel" in response_data:
            content += f"**Channel:** {response_data['channel']}\n"
        if "message_count" in response_data:
            content += f"**Messages Analyzed:** {response_data['message_count']}\n"
        
        content += f"\n**Summary:**\n{response_data['summary']}\n"
        
        # Add key topics if available
        if "topics" in response_data:
            content += f"\n**Key Topics:**\n"
            for topic in response_data["topics"]:
                content += f"• {topic}\n"
        
        # Add insights if available
        if "insights" in response_data:
            content += f"\n**Insights:**\n{response_data['insights']}\n"
        
        return await self._chunk_text(content)
    
    async def _format_text_response(
        self,
        text: str,
        header: str
    ) -> List[str]:
        """Format a simple text response"""
        content = header + text
        return await self._chunk_text(content)
    
    async def _chunk_text(self, text: str) -> List[str]:
        """Split text into Discord-compatible chunks"""
        if len(text) <= self.max_message_length:
            return [text]
        
        chunks = []
        remaining = text
        
        while remaining:
            if len(remaining) <= self.max_message_length:
                chunks.append(remaining)
                break
            
            # Find a good break point (prefer newlines)
            chunk = remaining[:self.max_message_length]
            last_newline = chunk.rfind('\n')
            
            if last_newline > self.max_message_length * 0.5:
                # Use newline break
                chunks.append(remaining[:last_newline])
                remaining = remaining[last_newline + 1:]
            else:
                # Use space break or force break
                last_space = chunk.rfind(' ')
                if last_space > self.max_message_length * 0.5:
                    chunks.append(remaining[:last_space])
                    remaining = remaining[last_space + 1:]
                else:
                    # Force break
                    chunks.append(chunk)
                    remaining = remaining[self.max_message_length:]
        
        return chunks
    
    def _format_error_message(self, error: str) -> str:
        """Format error message for Discord"""
        return f"**Error:** An error occurred while processing your request.\n\n```\n{error}\n```\n\nPlease try again or contact support if the issue persists."
    
    async def _log_query(self, query: str, discord_context: DiscordContext):
        """Log incoming query with context"""
        log_data = {
            'timestamp': discord_context.timestamp.isoformat(),
            'user_id': discord_context.user_id,
            'username': discord_context.username,
            'query': query,
            'channel_id': discord_context.channel_id,
            'guild_id': discord_context.guild_id
        }
        logger.info(f"Query received: {json.dumps(log_data)}")
    
    async def _update_user_context(self, discord_context: DiscordContext):
        """Update user context in the system"""
        context_data = {
            "platform": "discord",
            "username": discord_context.username,
            "channel_id": discord_context.channel_id,
            "guild_id": discord_context.guild_id,
            "last_activity": discord_context.timestamp.isoformat()
        }
        
        await self.agent_api.update_user_context(
            str(discord_context.user_id),
            context_data
        )
    
    async def _check_cache(self, query: str, user_id: int) -> Optional[Dict[str, Any]]:
        """Check cache for previous response"""
        if not self.cache:
            return None
        
        cache_key = f"discord_query_{user_id}_{hash(query)}"
        return await self.cache.get(cache_key)
    
    async def _cache_response(self, query: str, user_id: int, result: Dict[str, Any]):
        """Cache successful response"""
        if not self.cache:
            return
        
        cache_key = f"discord_query_{user_id}_{hash(query)}"
        await self.cache.set(cache_key, result, ttl=3600)  # 1 hour cache
    
    async def _store_conversation(
        self,
        query: str,
        result: Dict[str, Any],
        discord_context: DiscordContext,
        session_id: str
    ):
        """Store conversation in memory"""
        try:
            response_text = self._extract_response_text(result)
            await self.memory.add_interaction(
                str(discord_context.user_id),
                query,
                response_text,
                context={
                    "platform": "discord",
                    "channel_id": discord_context.channel_id,
                    "guild_id": discord_context.guild_id,
                    "session_id": session_id
                },
                metadata={
                    "execution_time": result.get("execution_time"),
                    "agents_used": result.get("agents_used", []),
                    "status": result.get("status")
                }
            )
        except Exception as e:
            logger.warning(f"Failed to store conversation: {e}")
    
    def _extract_response_text(self, result: Dict[str, Any]) -> str:
        """Extract text response from result for storage"""
        response_data = result.get("response", {})
        
        if "answer" in response_data:
            return response_data["answer"]
        elif "summary" in response_data:
            return response_data["summary"]
        elif "messages" in response_data:
            return f"Found {len(response_data['messages'])} messages"
        else:
            return str(response_data)
    
    async def _update_analytics(self, start_time: datetime, success: bool):
        """Update performance analytics"""
        if not self.enable_analytics:
            return
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        # Simple local tracking for legacy compatibility
        self.query_count += 1
        if not success:
            self.error_count += 1
        
        # Update rolling average
        self.average_response_time = (
            (self.average_response_time * (self.query_count - 1) + execution_time) 
            / self.query_count
        )
        
        # Note: Detailed analytics are automatically recorded by AgentAPI.query()
        # This method now mainly handles legacy local counters
        
        if self.query_count % 10 == 0:  # Log every 10 queries
            logger.info(
                f"Analytics: {self.query_count} queries, "
                f"{self.error_count} errors, "
                f"avg response time: {self.average_response_time:.2f}s"
            )
    
    async def get_analytics_dashboard(self) -> Dict[str, Any]:
        """Get analytics dashboard data from the comprehensive analytics system"""
        try:
            if hasattr(self.agent_api, 'analytics_dashboard'):
                # Get comprehensive analytics from the AgentAPI analytics system
                return await self.agent_api.analytics_dashboard.generate_overview_dashboard(
                    hours_back=24,
                    platform="discord"
                )
            else:
                # Fallback to basic analytics
                return await self.get_user_stats(0)  # Global stats
        except Exception as e:
            logger.error(f"Error getting analytics dashboard: {e}")
            return {"error": str(e)}
    
    async def get_performance_analytics(self, hours_back: int = 24) -> Dict[str, Any]:
        """Get performance analytics for Discord platform"""
        try:
            if hasattr(self.agent_api, 'query_repository'):
                return await self.agent_api.query_repository.get_performance_analytics(
                    hours_back=hours_back,
                    platform="discord"
                )
            else:
                # Fallback to basic stats
                return await self.get_user_stats(0)
        except Exception as e:
            logger.error(f"Error getting performance analytics: {e}")
            return {"error": str(e)}
    
    async def get_user_stats(self, user_id: int) -> Dict[str, Any]:
        """Get user interaction statistics"""
        try:
            history = await self.memory.get_history(str(user_id), limit=100)
            total_messages = len(history)
            
            stats = await self.memory.get_conversation_stats(str(user_id))
            
            return {
                "total_conversations": total_messages,
                "first_conversation": stats.get("first_conversation"),
                "last_conversation": stats.get("last_conversation"),
                "average_response_time": self.average_response_time,
                "total_queries": self.query_count,
                "error_rate": (self.error_count / max(self.query_count, 1)) * 100
            }
        except Exception as e:
            logger.error(f"Error getting user stats: {e}")
            return {"status": "error", "message": str(e)}
    
    async def get_system_health(self) -> Dict[str, Any]:
        """Get system health information"""
        try:
            health = await self.agent_api.health_check()
            
            # Add Discord-specific metrics
            health.update({
                "discord_interface": {
                    "query_count": self.query_count,
                    "error_count": self.error_count,
                    "error_rate": self.error_count / max(self.query_count, 1),
                    "average_response_time": self.average_response_time,
                    "cache_enabled": self.cache_enabled
                }
            })
            
            return health
        except Exception as e:
            logger.error(f"Error getting system health: {e}")
            return {"status": "error", "message": str(e)}
    
    async def optimize_system(self) -> Dict[str, Any]:
        """Trigger system optimization"""
        try:
            # Basic cleanup operations
            optimizations = 0
            
            # Cache cleanup if available
            if self.cache:
                await self.cache.cleanup_expired()
                optimizations += 1
            
            # Note: Memory cleanup would need to be implemented in ConversationMemory
            # For now, we'll just report the optimizations we can do
            
            return {
                "status": "success",
                "optimizations_performed": optimizations,
                "average_response_time": self.average_response_time,
                "total_queries": self.query_count,
                "error_count": self.error_count
            }
        except Exception as e:
            logger.error(f"Error optimizing system: {e}")
            return {"status": "error", "message": str(e)}
    
    async def shutdown(self):
        """Gracefully shutdown the interface"""
        logger.info("Shutting down Discord interface...")
        
        try:
            # Note: Add proper shutdown methods to AgentAPI and ConversationMemory if needed
            # For now, we'll just log the shutdown
            logger.info("Discord interface shutdown complete")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
    
    async def handle_slash_command(self, interaction: discord.Interaction, query: str):
        """Handle Discord slash command interactions"""
        try:
            # Create Discord context
            discord_context = DiscordContext(
                user_id=interaction.user.id,
                username=interaction.user.display_name,
                channel_id=interaction.channel_id or 0,
                guild_id=interaction.guild_id,
                channel_name=getattr(interaction.channel, 'name', 'DM'),
                guild_name=interaction.guild.name if interaction.guild else None,
                timestamp=datetime.now()
            )
            
            # Process the query
            response_messages = await self.process_query(query, discord_context, interaction)
            
            # Send response(s)
            for i, message in enumerate(response_messages):
                if i == 0:
                    # First message - use the interaction response
                    if interaction.response.is_done():
                        await interaction.followup.send(message)
                    else:
                        await interaction.response.send_message(message)
                else:
                    # Subsequent messages - use followup
                    await interaction.followup.send(message)
                    
        except Exception as e:
            logger.error(f"Error handling slash command: {e}", exc_info=True)
            error_message = f"Sorry, I encountered an error: {str(e)}"
            
            try:
                if interaction.response.is_done():
                    await interaction.followup.send(error_message, ephemeral=True)
                else:
                    await interaction.response.send_message(error_message, ephemeral=True)
            except Exception as send_error:
                logger.error(f"Failed to send error message: {send_error}")
