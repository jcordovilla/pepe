"""Discord Interface for Agentic RAG System

This module provides the Discord-specific interface for the agentic RAG framework,
replacing the existing LangChain-based agent with a sophisticated multi-agent system.
"""

import asyncio
import json
import logging
import random
from datetime import datetime
from typing import Dict, List, Optional, Union, Any, cast
from dataclasses import dataclass
import re

import discord
from discord import Webhook, WebhookMessage
from discord.ext import commands

from .agent_api import AgentAPI

from ..services.discord_service import DiscordMessageService
from ..services.unified_data_manager import UnifiedDataManager
from ..services.sync_service import DataSynchronizationService
from ..memory.conversation_memory import ConversationMemory
from ..cache.smart_cache import SmartCache
from ..analytics.query_answer_repository import QueryAnswerRepository
from ..config.modernized_config import get_modernized_config

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
        
        # Initialize modernized services (migrated from legacy core/)
        self._initialize_modernized_services()
        
        # Performance tracking
        self.query_count = 0
        self.error_count = 0
        self.average_response_time = 0.0
        
        # Initialize Discord bot
        self.bot: Optional[commands.Bot] = None
        self.discord_config = config.get('discord', {})
        
        logger.info("Discord interface initialized")
    
    def _initialize_modernized_services(self):
        """Initialize modernized services with unified architecture"""
        try:
            # Initialize unified data manager
            self.data_manager = UnifiedDataManager(self.config)
            
            # Initialize Discord message service
            discord_token = self.config.get('discord', {}).get('token')
            if discord_token and self.cache and self.memory:
                self.discord_service = DiscordMessageService(
                    token=discord_token,
                    cache=self.cache,
                    memory=self.memory
                )
            else:
                self.discord_service = None
                logger.warning("Discord service not initialized - missing dependencies")
            
            # Initialize sync service
            self.sync_service = DataSynchronizationService(
                self.config.get('sync', {})
            )
            
            logger.info("🔄 Modernized services initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize modernized services: {e}")
            # Set services to None if initialization fails
            self.data_manager = None
            self.discord_service = None
            self.sync_service = None
    
    async def setup_bot(self) -> commands.Bot:
        """Set up the Discord bot instance"""
        if self.bot is not None:
            return self.bot
        
        # Initialize AgentAPI if not already done
        if not hasattr(self.agent_api, 'memory'):
            await self.agent_api.initialize()
            
        # Initialize async services if needed
        if hasattr(self, 'data_manager') and self.data_manager:
            await self.data_manager.initialize()
            
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

        @self.bot.event
        async def on_message(message):
            # Ignore messages from the bot itself
            if self.bot and message.author == self.bot.user:
                return

            try:
                # Note: Messages are automatically stored in SQLite database
                # No need for explicit vector store indexing with MCP server
                logger.debug(f"Message {message.id} from {message.author.display_name} will be processed by MCP server")

            except Exception as e:
                logger.error(f"Error processing message: {e}")

            # Process commands
            if self.bot:
                await self.bot.process_commands(message)
        
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
            
            # Extract channel mention (e.g., <#1234567890>) from the query
            channel_mention_match = re.search(r'<#(\d+)>', query)
            extracted_channel_id = None
            if channel_mention_match:
                extracted_channel_id = channel_mention_match.group(1)
                logger.info(f"[DiscordInterface] Extracted channel mention: {extracted_channel_id}")

            # Check cache first
            if self.cache_enabled:
                cached_response = await self._check_cache(query, discord_context.user_id)
                if cached_response:
                    logger.info(f"Cache hit for query: {query[:50]}...")
                    return await self._format_response(cached_response, query, discord_context)
            
            # Update user context
            await self._update_user_context(discord_context)
            
            # The interaction should already be deferred by the command handler
            # No need to defer again here
            
            # Process through agentic system
            logger.info(f"⏳ Starting agent API query processing at {datetime.now().isoformat()}")
            query_start_time = datetime.now()
            
            # Ensure AgentAPI is initialized
            if not hasattr(self.agent_api, 'memory'):
                await self.agent_api.initialize()
            
            # Build context for agent
            agent_context = {
                    "platform": "discord",
                "channel_id": extracted_channel_id if extracted_channel_id else discord_context.channel_id,
                    "guild_id": discord_context.guild_id,
                "timestamp": discord_context.timestamp.isoformat(),
                }
            if extracted_channel_id:
                agent_context["channel_id"] = extracted_channel_id
                # Pass as agent_args for router/orchestrator
                agent_context["agent_args"] = {"channel_id": extracted_channel_id}
            result = await self.agent_api.query(
                query=query,
                user_id=str(discord_context.user_id),
                context=agent_context
            )
            
            query_duration = (datetime.now() - query_start_time).total_seconds()
            logger.info(f"✅ Agent API query completed in {query_duration:.2f}s")
            
            # Cache successful results
            if self.cache_enabled and result.get("success"):
                await self._cache_response(query, discord_context.user_id, result)
            
            # Transform agent API response to expected format
            transform_start = datetime.now()
            transformed_result = self._transform_agent_response(result)
            logger.info(f"✅ Response transformation completed in {(datetime.now() - transform_start).total_seconds():.2f}s")
            
            # Note: Removed interim status updates as users shouldn't see processing steps
            # Users should only see the final result
            
            # Format for Discord
            format_start = datetime.now()
            formatted_messages = await self._format_response(transformed_result, query, discord_context)
            logger.info(f"✅ Response formatting completed in {(datetime.now() - format_start).total_seconds():.2f}s")
            
            # Update analytics
            await self._update_analytics(start_time, success=True)
            
            # Store conversation
            await self._store_conversation(query, result, discord_context, session_id)
            
            total_processing = (datetime.now() - start_time).total_seconds()
            logger.info(f"🏁 Total query processing completed in {total_processing:.2f}s")
            
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
            # Fix username extraction - prefer author_display_name, fallback to author_username
            author_name = msg.get('author_display_name') or msg.get('author_username') or msg.get('author', {}).get('display_name') or msg.get('author', {}).get('username') or 'Unknown'
            
            # Fix timestamp formatting - convert from ISO to readable format
            timestamp = msg.get('timestamp', '')
            if timestamp:
                try:
                    from datetime import datetime
                    # Parse the ISO format timestamp
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    # Format to readable format: "May 29, 11:31 PM"
                    formatted_timestamp = dt.strftime("%b %d, %I:%M %p")
                except Exception:
                    # Fallback to original if parsing fails
                    formatted_timestamp = timestamp[:16]  # Just the date and time part
            else:
                formatted_timestamp = 'Unknown time'
            
            content = msg.get('content', '')
            jump_url = msg.get('jump_url', '')
            channel_name = msg.get('channel_name', 'Unknown Channel')
            
            # Enhanced message formatting with additional metadata
            msg_str = f"**{author_name}** ({formatted_timestamp}) in **#{channel_name}**\n{content}\n"
            
            # Add attachment info if present
            attachment_count = msg.get('attachment_count', 0)
            if attachment_count > 0:
                attachment_filenames = msg.get('attachment_filenames', '')
                if attachment_filenames:
                    files = attachment_filenames.split(',')[:3]  # Show max 3 filenames
                    file_list = ', '.join(files)
                    if attachment_count > 3:
                        file_list += f" (+{attachment_count - 3} more)"
                    msg_str += f"📎 Attachments: {file_list}\n"
                else:
                    msg_str += f"📎 {attachment_count} attachment(s)\n"
            
            # Add reaction info if present
            total_reactions = msg.get('total_reactions', 0)
            if total_reactions > 0:
                reaction_emojis = msg.get('reaction_emojis', '')
                if reaction_emojis:
                    emoji_list = reaction_emojis.replace(',', ' ')
                    msg_str += f"😊 Reactions: {emoji_list} ({total_reactions})\n"
                else:
                    msg_str += f"😊 {total_reactions} reaction(s)\n"
            
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
        try:
            # Ensure AgentAPI is initialized
            if not hasattr(self.agent_api, 'memory'):
                await self.agent_api.initialize()
                
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
        except Exception as e:
            logger.warning(f"Failed to update user context: {e}")
    
    async def _check_cache(self, query: str, user_id: int) -> Optional[Dict[str, Any]]:
        """Check if query result is cached."""
        if not self.cache_enabled:
            return None
        
        # Don't cache digest queries to avoid stale responses
        if any(keyword in query.lower() for keyword in ["digest", "key topics", "discussed", "summar", "overview"]):
            logger.info(f"Cache bypassed for digest query: {query[:50]}...")
            return None
        
        cache_key = f"query:{user_id}:{hash(query)}"
        return await self.cache.get(cache_key)
    
    async def _cache_response(self, query: str, user_id: int, result: Dict[str, Any]):
        """Cache query result."""
        if not self.cache_enabled:
            return
        
        # Don't cache digest queries to avoid stale responses
        if any(keyword in query.lower() for keyword in ["digest", "key topics", "discussed", "summar", "overview"]):
            return
        
        cache_key = f"query:{user_id}:{hash(query)}"
        await self.cache.set(cache_key, result, ttl=3600)  # Cache for 1 hour
    
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
    
    async def handle_slash_command(self, interaction: discord.Interaction, query: str):
        """Handle slash command interactions."""
        import time
        start_time = time.time()
        
        try:
            logger.info(f"🎯 Starting slash command handling for query: {query[:50]}...")
            
            # CRITICAL FIX: Defer the response IMMEDIATELY to prevent interaction expiry
            # This must be the FIRST thing we do with the interaction - before any other operations
            try:
                # Check if the interaction is still valid
                if not interaction.response.is_done():
                    # Set a shorter timeout for the deferral to ensure it happens quickly
                    await asyncio.wait_for(
                        interaction.response.defer(ephemeral=False, thinking=True),
                        timeout=2.0
                    )
                    logger.info("✅ Interaction deferred successfully (immediate)")
                else:
                    logger.warning("⚠️ Interaction was already responded to - continuing with followup")
            except asyncio.TimeoutError:
                logger.error("❌ Deferring the interaction timed out")
                # Try again with a simple approach as last resort
                try:
                    await interaction.response.defer(ephemeral=False)
                    logger.info("✅ Interaction deferred successfully (retry)")
                except Exception as e:
                    logger.error(f"❌ Failed to defer interaction after retry: {e}")
            except Exception as defer_error:
                logger.error(f"❌ Failed to defer interaction: {defer_error}")
                # Continue anyway - we'll try to use followup
            
            # Validate that the interaction is still valid
            if not hasattr(interaction, 'followup'):
                logger.error("❌ Invalid interaction object - missing followup")
                return
            
            # Create Discord context from interaction
            discord_context = DiscordContext(
                user_id=interaction.user.id,
                username=interaction.user.display_name,
                channel_id=interaction.channel_id or 0,
                guild_id=interaction.guild_id,
                channel_name=getattr(interaction.channel, 'name', 'Unknown'),
                guild_name=interaction.guild.name if interaction.guild else None,
                timestamp=datetime.now()
            )
            
            logger.info(f"📝 Created Discord context for user {discord_context.username} in #{discord_context.channel_name}")
            
            # Note: Status updates removed - users should only see final results
            # Initial processing notification removed to keep the interface clean
            
            # CRITICAL FIX: Set shorter timeout to ensure we respond within Discord's limits
            logger.info("🔄 Processing query through agentic system with timeout...")
            
            # Create a future for the query processing
            query_task = asyncio.create_task(
                self.process_query(query, discord_context, interaction)
            )
            
            try:
                # Wait for the query with a timeout
                messages = await asyncio.wait_for(
                    query_task,
                    timeout=90.0  # Increased timeout to 90 seconds for digest operations
                )
            except asyncio.TimeoutError:
                logger.error("❌ Query processing timed out after timeout period")
                error_msg = "⏰ Your request is taking longer than expected (90s timeout). Please try a simpler query or a shorter time period."
                query_task.cancel()
                
                # Try to send a timeout message
                try:
                    await interaction.followup.send(error_msg)
                except Exception as e:
                    logger.error(f"Could not send timeout message: {e}")
                return
            except Exception as e:
                logger.error(f"❌ Error in query processing: {str(e)}")
                query_task.cancel()
                
                # Try to send an error message
                try:
                    await interaction.followup.send(f"❌ An error occurred: {str(e)}")
                except Exception as send_error:
                    logger.error(f"Could not send error message: {send_error}")
                return
            
            processing_time = time.time() - start_time
            logger.info(f"✅ Query processed in {processing_time:.2f}s, got {len(messages)} response messages")
            
            # Send response messages with improved handling
            message_sent = False  # Track if at least one message was sent
            
            async def send_message_with_retries(msg_content, msg_index):
                nonlocal message_sent
                max_retries = 4  # Increase retries
                base_delay = 0.5  # Start with a shorter delay
                
                logger.info(f"📤 Sending response message {msg_index+1}/{len(messages)} (length: {len(msg_content)})")
                
                for retry in range(max_retries):
                    try:
                        # For long messages, try to chunk them further if they might be close to Discord's limit
                        if len(msg_content) > 1800:  # Discord's limit is ~2000, be conservative
                            # Split on paragraph boundaries for more natural breaks
                            paragraphs = msg_content.split('\n\n')
                            current_chunk = ""
                            
                            for para in paragraphs:
                                if len(current_chunk) + len(para) + 2 < 1800:  # +2 for the newlines
                                    if current_chunk:
                                        current_chunk += "\n\n" + para
                                    else:
                                        current_chunk = para
                                else:
                                    # Send the current chunk before it gets too big
                                    if current_chunk:
                                        await interaction.followup.send(current_chunk)
                                        message_sent = True
                                        await asyncio.sleep(0.5)  # Brief pause between chunks
                                        current_chunk = para
                                    else:
                                        # Single paragraph is too long, send it anyway
                                        await interaction.followup.send(para[:1800])
                                        message_sent = True
                                        await asyncio.sleep(0.5)
                                        # If there's remaining text, add to current chunk
                                        current_chunk = para[1800:]
                            
                            # Send any remaining text
                            if current_chunk:
                                await interaction.followup.send(current_chunk)
                                message_sent = True
                        else:
                            # Regular message that's under the limit
                            await interaction.followup.send(msg_content)
                            message_sent = True
                            
                        logger.info(f"✅ Successfully sent message {msg_index+1}")
                        return True  # Success
                        
                    except discord.errors.NotFound:
                        logger.error(f"❌ Interaction expired while sending message {msg_index+1}")
                        return False  # Fatal error
                    except Exception as send_error:
                        # Check for fatal errors that indicate we should stop trying
                        if "Unknown interaction" in str(send_error) or "Invalid Webhook Token" in str(send_error):
                            logger.error(f"❌ Interaction has expired: {send_error}")
                            return False  # Fatal error
                            
                        if retry < max_retries - 1:
                            # Calculate delay with jitter to prevent thundering herd
                            delay = base_delay * (2 ** retry) * (0.8 + 0.4 * random.random())
                            logger.warning(f"⚠️ Retry {retry+1}/{max_retries} for message {msg_index+1} in {delay:.2f}s: {send_error}")
                            await asyncio.sleep(delay)  # Exponential backoff with jitter
                        else:
                            logger.error(f"❌ Failed to send message {msg_index+1} after {max_retries} retries: {send_error}")
                            return False
                
                return False  # All retries failed
            
            # Process all messages
            for i, message in enumerate(messages):
                success = await send_message_with_retries(message, i)
                if not success:
                    # If we can't send a message, try one last fallback with a simple message
                    if not message_sent:  # Only if no messages were sent yet
                        try:
                            simple_msg = "I processed your query but encountered an error sending the full response. Please try again."
                            await interaction.followup.send(simple_msg)
                            logger.info("✅ Sent simple fallback message")
                            message_sent = True
                        except Exception as e:
                            logger.error(f"❌ Failed to send fallback message: {e}")
                    break  # Stop processing remaining messages
                
            total_time = time.time() - start_time
            logger.info(f"🎉 All response messages sent successfully! Total time: {total_time:.2f}s")
                
        except discord.errors.NotFound as nf_error:
            logger.error(f"❌ Discord interaction not found: {nf_error}")
            # Don't try to respond if the interaction is invalid
        except Exception as e:
            logger.error(f"❌ Error handling slash command: {e}", exc_info=True)
            
            error_message = self._format_error_message(str(e))
            logger.info(f"📤 Attempting to send error message: {error_message[:100]}...")
            
            # Try to send an error message with retry logic
            max_retries = 2
            for retry in range(max_retries):
                try:
                    await interaction.followup.send(f"❌ An error occurred: {error_message}")
                    break
                except discord.errors.NotFound:
                    logger.error("❌ Cannot send error message - interaction expired")
                    break
                except Exception as error_send_error:
                    if retry < max_retries - 1:
                        await asyncio.sleep(1)
                        continue
                    logger.error(f"❌ Failed to send error message after retries: {error_send_error}")
    
    async def shutdown(self):
        """Gracefully shutdown the interface"""
        logger.info("Shutting down Discord interface...")
        
        try:
            # Note: Add proper shutdown methods to AgentAPI and ConversationMemory if needed
            # For now, we'll just log the shutdown
            logger.info("Discord interface shutdown complete")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
    
    def _transform_agent_response(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Transform agent API response to Discord interface expected format"""
        # Allow either `success` boolean or legacy `status` string
        success = result.get("success")
        if success is None:
            success = result.get("status") == "success"

        if not success:
            return {
                "status": "error",
                "message": result.get("error", "Unknown error occurred")
            }
        
        # Get the sources (search results)
        response_payload = result.get("response", {}) if isinstance(result.get("response"), dict) else {}

        sources = result.get("sources")
        if sources is None:
            sources = response_payload.get("messages") or response_payload.get("sources") or []

        answer = result.get("answer")
        if answer is None:
            answer = response_payload.get("answer", "")
        
        # Determine response type based on sources content
        if sources and isinstance(sources, list) and len(sources) > 0:
            # Check if sources contain message-like objects
            first_source = sources[0]
            if isinstance(first_source, dict) and ("content" in first_source or "author" in first_source):
                # Format as message list
                return {
                    "response": {
                        "messages": sources,
                        "total_count": len(sources)
                    }
                }
            else:
                # Format as text response with answer
                return {
                    "response": {
                        "answer": answer if answer else f"Found {len(sources)} results."
                    }
                }
        else:
            # No sources, use answer as text response
            return {
                "response": {
                    "answer": answer if answer else "No results found."
                }
            }
