"""Streamlit Interface for Agentic RAG System

This module provides the Streamlit-specific interface for the agentic RAG framework,
enabling a modern web-based chat interface with the sophisticated multi-agent system.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from ..interfaces.agent_api import AgentAPI
from ..agents.base_agent import AgentRole, TaskStatus
from ..memory.conversation_memory import ConversationMemory
from ..cache.smart_cache import SmartCache

logger = logging.getLogger(__name__)


@dataclass
class StreamlitContext:
    """Streamlit-specific context information"""
    session_id: str
    user_id: str
    timestamp: datetime
    page: str
    widget_states: Dict[str, Any]


class StreamlitInterface:
    """
    Streamlit interface for the agentic RAG system.
    
    This class provides Streamlit-specific functionality including:
    - Chat interface with message history
    - Real-time progress tracking
    - System analytics and monitoring
    - Interactive visualizations
    - Export capabilities
    """
    
    def __init__(
        self,
        agent_api: Optional[AgentAPI] = None,
        cache_enabled: bool = True,
        enable_analytics: bool = True,
        session_state_key: str = "agentic_rag"
    ):
        """
        Initialize Streamlit interface.
        
        Args:
            agent_api: Agent API instance (created if None)
            cache_enabled: Whether to enable caching
            enable_analytics: Whether to track analytics
            session_state_key: Key for Streamlit session state
        """
        self.agent_api = agent_api or AgentAPI()
        self.cache_enabled = cache_enabled
        self.enable_analytics = enable_analytics
        self.session_state_key = session_state_key
        
        # Initialize session state
        self._init_session_state()
        
        # Initialize components
        self.memory = ConversationMemory()
        self.cache = SmartCache() if cache_enabled else None
        
        logger.info("Streamlit interface initialized")
    
    def _init_session_state(self):
        """Initialize Streamlit session state"""
        if self.session_state_key not in st.session_state:
            st.session_state[self.session_state_key] = {
                "messages": [],
                "user_id": f"streamlit_{id(st.session_state)}",
                "session_id": f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "query_count": 0,
                "last_query_time": None,
                "system_stats": {}
            }
    
    def render_chat_interface(self):
        """Render the main chat interface"""
        st.title("🤖 Pepe - Agentic RAG Assistant")
        st.markdown("Ask me anything about your Discord server data!")
        
        # Sidebar with controls
        self._render_sidebar()
        
        # Main chat area
        self._render_chat_history()
        
        # Query input
        self._render_query_input()
        
        # Display system status if enabled
        if self.enable_analytics:
            self._render_system_status()
    
    def _render_sidebar(self):
        """Render sidebar with controls and information"""
        with st.sidebar:
            st.header("🎛️ Controls")
            
            # Clear conversation button
            if st.button("🗑️ Clear Conversation", type="secondary"):
                self._clear_conversation()
                st.rerun()
            
            # Export conversation button
            if st.button("💾 Export Conversation", type="secondary"):
                self._export_conversation()
            
            # System optimization
            if st.button("🔧 Optimize System", type="secondary"):
                self._optimize_system()
            
            st.header("📊 Session Info")
            session_data = st.session_state[self.session_state_key]
            
            st.metric("Messages", len(session_data["messages"]))
            st.metric("Queries", session_data["query_count"])
            
            if session_data["last_query_time"]:
                st.metric("Last Query", session_data["last_query_time"].strftime("%H:%M:%S"))
            
            # Advanced settings
            with st.expander("⚙️ Advanced Settings"):
                st.session_state.enable_streaming = st.checkbox(
                    "Enable Streaming Responses", 
                    value=True
                )
                st.session_state.show_agent_details = st.checkbox(
                    "Show Agent Details", 
                    value=False
                )
                st.session_state.max_history = st.slider(
                    "Max History Messages", 
                    min_value=10, 
                    max_value=100, 
                    value=50
                )
    
    def _render_chat_history(self):
        """Render chat message history"""
        session_data = st.session_state[self.session_state_key]
        
        # Create container for messages
        chat_container = st.container()
        
        with chat_container:
            for message in session_data["messages"]:
                self._render_message(message)
    
    def _render_message(self, message: Dict[str, Any]):
        """Render a single message"""
        if message["role"] == "user":
            with st.chat_message("user"):
                st.write(message["content"])
                if "timestamp" in message:
                    st.caption(f"Sent at {message['timestamp']}")
        
        elif message["role"] == "assistant":
            with st.chat_message("assistant"):
                # Main response
                st.write(message["content"])
                
                # Show agent details if enabled
                if (st.session_state.get("show_agent_details", False) and 
                    "metadata" in message):
                    self._render_agent_details(message["metadata"])
                
                # Show timestamp
                if "timestamp" in message:
                    st.caption(f"Response at {message['timestamp']}")
        
        elif message["role"] == "system":
            with st.chat_message("assistant"):
                st.info(message["content"])
    
    def _render_agent_details(self, metadata: Dict[str, Any]):
        """Render agent execution details"""
        with st.expander("🔍 Agent Details"):
            col1, col2 = st.columns(2)
            
            with col1:
                if "execution_time" in metadata:
                    st.metric("Execution Time", f"{metadata['execution_time']:.2f}s")
                
                if "agents_used" in metadata:
                    st.write("**Agents Used:**")
                    for agent in metadata["agents_used"]:
                        st.write(f"• {agent}")
            
            with col2:
                if "reasoning_steps" in metadata:
                    st.write("**Reasoning Steps:**")
                    for step in metadata["reasoning_steps"]:
                        st.write(f"• {step}")
    
    def _render_query_input(self):
        """Render query input area"""
        # Query input
        query = st.chat_input("Ask me anything...")
        
        if query:
            asyncio.run(self._process_user_query(query))
            st.rerun()
    
    async def _process_user_query(self, query: str):
        """Process user query through agentic system"""
        session_data = st.session_state[self.session_state_key]
        
        # Add user message
        user_message = {
            "role": "user",
            "content": query,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        session_data["messages"].append(user_message)
        
        # Update session stats
        session_data["query_count"] += 1
        session_data["last_query_time"] = datetime.now()
        
        # Create context
        streamlit_context = StreamlitContext(
            session_id=session_data["session_id"],
            user_id=session_data["user_id"],
            timestamp=datetime.now(),
            page="chat",
            widget_states={}
        )
        
        # Show processing indicator
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                try:
                    # Process through agentic system
                    result = await self.agent_api.query(
                        query=query,
                        user_id=streamlit_context.user_id,
                        session_id=streamlit_context.session_id,
                        context={
                            "platform": "streamlit",
                            "page": streamlit_context.page,
                            "timestamp": streamlit_context.timestamp.isoformat()
                        }
                    )
                    
                    # Format response
                    response_content = self._format_response(result)
                    
                    # Add assistant message
                    assistant_message = {
                        "role": "assistant",
                        "content": response_content,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "metadata": {
                            "execution_time": result.get("execution_time"),
                            "agents_used": result.get("agents_used", []),
                            "reasoning_steps": result.get("reasoning_steps", [])
                        }
                    }
                    session_data["messages"].append(assistant_message)
                    
                    # Store in memory
                    await self._store_conversation(query, result, streamlit_context)
                    
                except Exception as e:
                    logger.error(f"Error processing query: {e}", exc_info=True)
                    
                    error_message = {
                        "role": "system",
                        "content": f"❌ **Error:** {str(e)}",
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    session_data["messages"].append(error_message)
    
    def _format_response(self, result: Dict[str, Any]) -> str:
        """Format agent response for Streamlit display"""
        if result.get("status") == "error":
            return f"**Error:** {result.get('message', 'Unknown error occurred')}"
        
        response_data = result.get("response", {})
        
        # Handle different response types
        if "messages" in response_data:
            return self._format_message_list(response_data["messages"], response_data)
        elif "summary" in response_data:
            return self._format_summary_response(response_data)
        elif "answer" in response_data:
            return response_data["answer"]
        else:
            return str(response_data)
    
    def _format_message_list(self, messages: List[Dict[str, Any]], metadata: Dict[str, Any]) -> str:
        """Format a list of Discord messages"""
        if not messages:
            return "No messages found matching your query."
        
        content = ""
        
        # Add metadata
        if "timeframe" in metadata:
            content += f"**📅 Timeframe:** {metadata['timeframe']}\n"
        if "channel" in metadata:
            content += f"**📢 Channel:** {metadata['channel']}\n"
        if "total_count" in metadata:
            content += f"**📊 Total Messages:** {metadata['total_count']}\n"
        
        content += "\n**💬 Messages:**\n\n"
        
        for i, msg in enumerate(messages[:10]):  # Limit to first 10 for display
            author = msg.get('author', {})
            author_name = author.get('username', 'Unknown')
            timestamp = msg.get('timestamp', '')
            message_content = msg.get('content', '')
            channel_name = msg.get('channel_name', 'Unknown Channel')
            
            content += f"**{i+1}. {author_name}** ({timestamp}) in **#{channel_name}**\n"
            content += f"{message_content}\n"
            
            if msg.get('jump_url'):
                content += f"[🔗 View Message]({msg['jump_url']})\n"
            
            content += "\n---\n"
        
        if len(messages) > 10:
            content += f"\n*...and {len(messages) - 10} more messages*"
        
        return content
    
    def _format_summary_response(self, response_data: Dict[str, Any]) -> str:
        """Format a summary response"""
        content = ""
        
        if "timeframe" in response_data:
            content += f"**📅 Timeframe:** {response_data['timeframe']}\n"
        if "channel" in response_data:
            content += f"**📢 Channel:** {response_data['channel']}\n"
        if "message_count" in response_data:
            content += f"**📊 Messages Analyzed:** {response_data['message_count']}\n"
        
        content += f"\n**📝 Summary:**\n{response_data['summary']}\n"
        
        # Add key topics if available
        if "topics" in response_data:
            content += f"\n**🔍 Key Topics:**\n"
            for topic in response_data["topics"]:
                content += f"• {topic}\n"
        
        # Add insights if available
        if "insights" in response_data:
            content += f"\n**💡 Insights:**\n{response_data['insights']}\n"
        
        return content
    
    def _render_system_status(self):
        """Render system status indicators"""
        with st.expander("📊 System Status"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Cache Status", "Enabled" if self.cache_enabled else "Disabled")
            
            with col2:
                st.metric("Analytics", "Enabled" if self.enable_analytics else "Disabled")
            
            with col3:
                if hasattr(self, '_last_health_check'):
                    status = "Healthy" if self._last_health_check.get("status") == "healthy" else "Issues"
                    st.metric("System Health", status)
                else:
                    st.metric("System Health", "Unknown")
            
            # Health check button
            if st.button("🏥 Check System Health", key="health_check"):
                asyncio.run(self._check_system_health())
                st.rerun()
    
    async def _check_system_health(self):
        """Check system health"""
        try:
            health = await self.agent_api.health_check()
            self._last_health_check = health
            
            if health.get("status") == "healthy":
                st.success("✅ System is healthy!")
            else:
                st.warning("⚠️ System has issues. Check logs for details.")
                
        except Exception as e:
            st.error(f"❌ Health check failed: {e}")
    
    def _clear_conversation(self):
        """Clear conversation history"""
        session_data = st.session_state[self.session_state_key]
        session_data["messages"] = []
        session_data["query_count"] = 0
        session_data["last_query_time"] = None
        
        st.success("🗑️ Conversation cleared!")
    
    def _export_conversation(self):
        """Export conversation to downloadable format"""
        session_data = st.session_state[self.session_state_key]
        
        if not session_data["messages"]:
            st.warning("No conversation to export!")
            return
        
        # Create export data
        export_data = {
            "session_id": session_data["session_id"],
            "user_id": session_data["user_id"],
            "export_timestamp": datetime.now().isoformat(),
            "total_messages": len(session_data["messages"]),
            "messages": session_data["messages"]
        }
        
        # Convert to JSON
        json_str = json.dumps(export_data, indent=2, ensure_ascii=False)
        
        # Provide download
        st.download_button(
            label="📥 Download Conversation",
            data=json_str,
            file_name=f"conversation_{session_data['session_id']}.json",
            mime="application/json"
        )
        
        st.success("💾 Conversation ready for download!")
    
    def _optimize_system(self):
        """Trigger system optimization"""
        try:
            with st.spinner("🔧 Optimizing system..."):
                result = asyncio.run(self.agent_api.optimize())
                
                if result.get("status") == "success":
                    st.success("✅ System optimization complete!")
                else:
                    st.error(f"❌ Optimization failed: {result.get('message')}")
                    
        except Exception as e:
            st.error(f"❌ Optimization error: {e}")
    
    async def _store_conversation(
        self,
        query: str,
        result: Dict[str, Any],
        streamlit_context: StreamlitContext
    ):
        """Store conversation in memory"""
        try:
            await asyncio.to_thread(
                self.memory.add_user_message,
                streamlit_context.user_id,
                query,
                {
                    "platform": "streamlit",
                    "session_id": streamlit_context.session_id,
                    "page": streamlit_context.page
                }
            )
            
            response_text = self._extract_response_text(result)
            await asyncio.to_thread(
                self.memory.add_assistant_message,
                streamlit_context.user_id,
                response_text,
                {
                    "execution_time": result.get("execution_time"),
                    "agents_used": result.get("agents_used", []),
                    "session_id": streamlit_context.session_id
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
    
    def render_analytics_page(self):
        """Render analytics and monitoring page"""
        st.title("📊 System Analytics")
        
        # System health overview
        self._render_health_overview()
        
        # Usage statistics
        self._render_usage_stats()
        
        # Performance metrics
        self._render_performance_metrics()
        
        # Agent activity
        self._render_agent_activity()
    
    def _render_health_overview(self):
        """Render system health overview"""
        st.header("🏥 System Health")
        
        try:
            health = asyncio.run(self.agent_api.health_check())
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                status = "🟢 Healthy" if health.get("status") == "healthy" else "🔴 Issues"
                st.metric("Overall Status", status)
            
            with col2:
                components = health.get("components", {})
                healthy_count = sum(1 for c in components.values() if c.get("status") == "healthy")
                st.metric("Healthy Components", f"{healthy_count}/{len(components)}")
            
            with col3:
                st.metric("Uptime", health.get("uptime", "Unknown"))
            
            with col4:
                st.metric("Last Check", datetime.now().strftime("%H:%M:%S"))
            
            # Component details
            if components:
                st.subheader("Component Status")
                df = pd.DataFrame([
                    {
                        "Component": name.title(),
                        "Status": info.get("status", "unknown"),
                        "Details": info.get("message", "No details")
                    }
                    for name, info in components.items()
                ])
                st.dataframe(df, use_container_width=True)
                
        except Exception as e:
            st.error(f"Failed to load health data: {e}")
    
    def _render_usage_stats(self):
        """Render usage statistics"""
        st.header("📈 Usage Statistics")
        
        # Mock data - replace with actual analytics
        col1, col2 = st.columns(2)
        
        with col1:
            # Query frequency chart
            dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='D')
            queries = [10 + i % 20 for i in range(len(dates))]
            
            fig = px.line(
                x=dates, 
                y=queries,
                title="Daily Query Volume",
                labels={"x": "Date", "y": "Queries"}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Response time distribution
            response_times = [0.5, 1.2, 0.8, 2.1, 1.5, 0.9, 1.8, 1.1, 0.7, 2.3]
            
            fig = px.histogram(
                x=response_times,
                title="Response Time Distribution",
                labels={"x": "Response Time (s)", "y": "Frequency"}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def _render_performance_metrics(self):
        """Render performance metrics"""
        st.header("⚡ Performance Metrics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Avg Response Time", "1.2s", delta="-0.3s")
        
        with col2:
            st.metric("Success Rate", "98.5%", delta="0.5%")
        
        with col3:
            st.metric("Cache Hit Rate", "75%", delta="5%")
    
    def _render_agent_activity(self):
        """Render agent activity overview"""
        st.header("🤖 Agent Activity")
        
        # Mock agent usage data
        agent_data = {
            "Agent": ["Planning", "Search", "Analysis"],
            "Queries Handled": [150, 200, 120],
            "Avg Response Time": [0.8, 1.5, 2.1],
            "Success Rate": [99.2, 97.8, 98.5]
        }
        
        df = pd.DataFrame(agent_data)
        st.dataframe(df, use_container_width=True)
        
        # Agent usage pie chart
        fig = px.pie(
            values=agent_data["Queries Handled"],
            names=agent_data["Agent"],
            title="Agent Usage Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    async def shutdown(self):
        """Gracefully shutdown the interface"""
        logger.info("Shutting down Streamlit interface...")
        
        try:
            if self.agent_api:
                await self.agent_api.shutdown()
            
            if self.memory:
                await asyncio.to_thread(self.memory.close)
            
            if self.cache:
                await asyncio.to_thread(self.cache.close)
                
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
        
        logger.info("Streamlit interface shutdown complete")
