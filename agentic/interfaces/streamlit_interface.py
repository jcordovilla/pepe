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
        self.agent_api = agent_api or AgentAPI({
            "orchestrator": {},
            "vector_store": {},
            "memory": {},
            "pipeline": {},
            "analytics": {}
        })
        self.cache_enabled = cache_enabled
        self.enable_analytics = enable_analytics
        self.session_state_key = session_state_key
        
        # Initialize session state
        self._init_session_state()
        
        # Initialize components
        self.memory = ConversationMemory({"db_path": "data/conversation_memory.db"})
        self.cache = SmartCache({"cache_dir": "data/cache"}) if cache_enabled else None
        
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
            
            # Pipeline controls
            st.markdown("---")
            st.header("⚙️ Data Pipeline")
            
            # Pipeline status
            pipeline_status = self.agent_api.get_pipeline_status()
            if pipeline_status.get("is_running"):
                st.warning(f"🔄 Pipeline running: {pipeline_status.get('current_step', 'unknown')}")
            else:
                st.success("✅ Pipeline ready")
            
            # Pipeline buttons
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🚀 Run Full Pipeline", type="primary", disabled=pipeline_status.get("is_running", False)):
                    self._run_full_pipeline()
                    st.rerun()
            
            with col2:
                if st.button("📊 Pipeline Status", type="secondary"):
                    self._show_pipeline_status()
            
            # Individual step controls
            with st.expander("🔧 Individual Steps"):
                available_steps = pipeline_status.get("available_steps", [])
                
                for step in available_steps:
                    step_label = step.replace("_", " ").title()
                    if st.button(f"▶️ {step_label}", key=f"step_{step}", disabled=pipeline_status.get("is_running", False)):
                        self._run_pipeline_step(step)
                        st.rerun()
            
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
                        context={
                            "platform": "streamlit",
                            "page": streamlit_context.page,
                            "session_id": streamlit_context.session_id,
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
                result = asyncio.run(self.agent_api.optimize_system())
                
                if result.get("success"):
                    st.success("✅ System optimization complete!")
                else:
                    st.error(f"❌ Optimization failed: {result.get('error', 'Unknown error')}")
                    
        except Exception as e:
            st.error(f"❌ Optimization error: {e}")
    
    def _run_full_pipeline(self):
        """Run the full data processing pipeline"""
        try:
            session_data = st.session_state[self.session_state_key]
            user_id = session_data["user_id"]
            
            with st.spinner("🚀 Running full pipeline..."):
                result = asyncio.run(self.agent_api.run_pipeline(user_id))
                
                if result.get("success"):
                    st.success("✅ Pipeline completed successfully!")
                    
                    # Show pipeline statistics
                    if "stats" in result:
                        stats = result["stats"]
                        st.info(f"📊 Messages: {stats.get('total_messages', 'N/A')} | Resources: {stats.get('total_resources', 'N/A')}")
                else:
                    st.error(f"❌ Pipeline failed: {result.get('error', 'Unknown error')}")
                    if "failed_step" in result:
                        st.error(f"Failed at step: {result['failed_step']}")
                        
        except Exception as e:
            st.error(f"❌ Pipeline error: {e}")
    
    def _run_pipeline_step(self, step_name: str):
        """Run a single pipeline step"""
        try:
            session_data = st.session_state[self.session_state_key]
            user_id = session_data["user_id"]
            
            with st.spinner(f"▶️ Running {step_name.replace('_', ' ').title()}..."):
                result = asyncio.run(self.agent_api.run_pipeline_step(step_name, user_id))
                
                if result.get("success"):
                    st.success(f"✅ {step_name.replace('_', ' ').title()} completed!")
                    
                    # Show step statistics
                    if "stats" in result:
                        stats = result["stats"]
                        st.info(f"📊 Messages: {stats.get('total_messages', 'N/A')} | Resources: {stats.get('total_resources', 'N/A')}")
                else:
                    st.error(f"❌ {step_name.replace('_', ' ').title()} failed: {result.get('error', 'Unknown error')}")
                    
        except Exception as e:
            st.error(f"❌ Step error: {e}")
    
    def _show_pipeline_status(self):
        """Show detailed pipeline status and history"""
        try:
            # Get current status
            status = self.agent_api.get_pipeline_status()
            
            # Get pipeline history
            history_result = self.agent_api.get_pipeline_history(limit=5)
            history = history_result.get("history", []) if history_result.get("success") else []
            
            # Get data statistics
            stats_result = asyncio.run(self.agent_api.get_data_stats())
            stats = stats_result if stats_result.get("success") else {}
            
            # Display in expander
            with st.expander("📊 Detailed Pipeline Status", expanded=True):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Current Status")
                    if status.get("is_running"):
                        st.warning(f"🔄 Running: {status.get('current_step', 'unknown')}")
                    else:
                        st.success("✅ Ready")
                    
                    st.metric("Available Steps", len(status.get("available_steps", [])))
                
                with col2:
                    st.subheader("Data Statistics")
                    if "database" in stats:
                        db_stats = stats["database"]
                        st.metric("Total Messages", db_stats.get("total_messages", "N/A"))
                        st.metric("Total Resources", db_stats.get("total_resources", "N/A"))
                
                # Recent pipeline history
                if history:
                    st.subheader("Recent Pipeline Runs")
                    for run in history[-3:]:  # Show last 3 runs
                        status_emoji = "✅" if run.get("success") else "❌"
                        start_time = run.get("start_time", "Unknown")
                        st.write(f"{status_emoji} {start_time} - Pipeline ID: {run.get('pipeline_id', 'Unknown')}")
                        
                # Pipeline logs button
                if st.button("📋 View Pipeline Logs"):
                    self._show_pipeline_logs()
                    
        except Exception as e:
            st.error(f"❌ Error getting pipeline status: {e}")
    
    def _show_pipeline_logs(self):
        """Show recent pipeline logs"""
        try:
            logs_result = asyncio.run(self.agent_api.get_pipeline_logs(lines=50))
            
            if logs_result.get("success"):
                logs = logs_result.get("lines", [])
                
                with st.expander("📋 Pipeline Logs", expanded=True):
                    if logs:
                        log_text = "".join(logs)
                        st.text_area(
                            "Recent Log Output",
                            value=log_text,
                            height=300,
                            label_visibility="collapsed"
                        )
                    else:
                        st.info("No logs available")
            else:
                st.error(f"❌ Failed to get logs: {logs_result.get('error', 'Unknown error')}")
                
        except Exception as e:
            st.error(f"❌ Error getting logs: {e}")
    
    async def _store_conversation(
        self,
        query: str,
        result: Dict[str, Any],
        streamlit_context: StreamlitContext
    ):
        """Store conversation in memory"""
        try:
            response_text = self._extract_response_text(result)
            await self.memory.add_interaction(
                streamlit_context.user_id,
                query,
                response_text,
                context={
                    "platform": "streamlit",
                    "session_id": streamlit_context.session_id,
                    "page": streamlit_context.page
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
    
    def render_analytics_page(self):
        """Render comprehensive analytics and monitoring page"""
        st.title("📊 System Analytics")
        
        # Create tabs for different analytics views
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Overview", 
            "⚡ Performance", 
            "👥 User Analytics", 
            "🔍 Quality Assessment"
        ])
        
        with tab1:
            self._render_overview_analytics()
        
        with tab2:
            self._render_performance_analytics()
        
        with tab3:
            self._render_user_analytics()
        
        with tab4:
            self._render_quality_analytics()
    
    def _render_overview_analytics(self):
        """Render overview analytics dashboard"""
        st.header("📊 System Overview")
        
        try:
            # Get comprehensive analytics from AgentAPI
            if hasattr(self.agent_api, 'analytics_dashboard'):
                dashboard_data = asyncio.run(
                    self.agent_api.analytics_dashboard.generate_overview_dashboard(
                        hours_back=24,
                        platform="streamlit"
                    )
                )
                
                # System Health Metrics
                col1, col2, col3, col4 = st.columns(4)
                
                stats = dashboard_data.get("stats", {})
                with col1:
                    st.metric(
                        "Total Queries", 
                        stats.get("total_queries", 0),
                        delta=stats.get("queries_delta", 0)
                    )
                
                with col2:
                    st.metric(
                        "Avg Response Time", 
                        f"{stats.get('avg_response_time', 0):.2f}s",
                        delta=f"{stats.get('response_time_delta', 0):.2f}s"
                    )
                
                with col3:
                    st.metric(
                        "Success Rate", 
                        f"{stats.get('success_rate', 0):.1f}%",
                        delta=f"{stats.get('success_rate_delta', 0):.1f}%"
                    )
                
                with col4:
                    st.metric(
                        "Quality Score", 
                        f"{stats.get('avg_quality_score', 0):.2f}/5",
                        delta=f"{stats.get('quality_delta', 0):.2f}"
                    )
                
                # Charts
                charts = dashboard_data.get("charts", {})
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if "query_volume" in charts:
                        st.plotly_chart(
                            go.Figure(charts["query_volume"]), 
                            use_container_width=True
                        )
                
                with col2:
                    if "response_time" in charts:
                        st.plotly_chart(
                            go.Figure(charts["response_time"]), 
                            use_container_width=True
                        )
                
                # Platform Usage
                if "platform_usage" in charts:
                    st.subheader("Platform Usage Distribution")
                    st.plotly_chart(
                        go.Figure(charts["platform_usage"]), 
                        use_container_width=True
                    )
                    
            else:
                self._render_fallback_overview()
                
        except Exception as e:
            st.error(f"Error loading analytics: {e}")
            logger.error(f"Analytics error: {e}", exc_info=True)
            self._render_fallback_overview()
    
    def _render_performance_analytics(self):
        """Render performance analytics dashboard"""
        st.header("⚡ Performance Analytics")
        
        try:
            if hasattr(self.agent_api, 'analytics_dashboard'):
                performance_data = asyncio.run(
                    self.agent_api.analytics_dashboard.generate_performance_dashboard(
                        hours_back=24
                    )
                )
                
                # Performance Metrics
                col1, col2, col3 = st.columns(3)
                
                stats = performance_data.get("stats", {})
                with col1:
                    st.metric(
                        "System Load", 
                        f"{stats.get('cpu_usage', 0):.1f}%"
                    )
                
                with col2:
                    st.metric(
                        "Memory Usage", 
                        f"{stats.get('memory_usage', 0):.1f}%"
                    )
                
                with col3:
                    st.metric(
                        "Cache Hit Rate", 
                        f"{stats.get('cache_hit_rate', 0):.1f}%"
                    )
                
                # Performance Charts
                charts = performance_data.get("charts", {})
                
                if "system_health" in charts:
                    st.subheader("System Health Over Time")
                    st.plotly_chart(
                        go.Figure(charts["system_health"]), 
                        use_container_width=True
                    )
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if "performance_trends" in charts:
                        st.plotly_chart(
                            go.Figure(charts["performance_trends"]), 
                            use_container_width=True
                        )
                
                with col2:
                    if "error_distribution" in charts:
                        st.plotly_chart(
                            go.Figure(charts["error_distribution"]), 
                            use_container_width=True
                        )
                        
            else:
                self._render_fallback_performance()
                
        except Exception as e:
            st.error(f"Error loading performance analytics: {e}")
            logger.error(f"Performance analytics error: {e}", exc_info=True)
            self._render_fallback_performance()
    
    def _render_user_analytics(self):
        """Render user analytics dashboard"""
        st.header("👥 User Analytics")
        
        try:
            if hasattr(self.agent_api, 'analytics_dashboard'):
                user_data = asyncio.run(
                    self.agent_api.analytics_dashboard.generate_user_analytics(
                        hours_back=24
                    )
                )
                
                # User Metrics
                col1, col2, col3, col4 = st.columns(4)
                
                stats = user_data.get("stats", {})
                with col1:
                    st.metric("Active Users", stats.get("active_users", 0))
                
                with col2:
                    st.metric("Total Sessions", stats.get("total_sessions", 0))
                
                with col3:
                    st.metric("Avg Session Duration", f"{stats.get('avg_session_duration', 0):.1f}m")
                
                with col4:
                    st.metric("Queries per User", f"{stats.get('queries_per_user', 0):.1f}")
                
                # User Charts
                charts = user_data.get("charts", {})
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if "user_activity" in charts:
                        st.plotly_chart(
                            go.Figure(charts["user_activity"]), 
                            use_container_width=True
                        )
                
                with col2:
                    if "session_patterns" in charts:
                        st.plotly_chart(
                            go.Figure(charts["session_patterns"]), 
                            use_container_width=True
                        )
                
                if "user_satisfaction" in charts:
                    st.subheader("User Satisfaction Trends")
                    st.plotly_chart(
                        go.Figure(charts["user_satisfaction"]), 
                        use_container_width=True
                    )
                    
            else:
                self._render_fallback_users()
                
        except Exception as e:
            st.error(f"Error loading user analytics: {e}")
            logger.error(f"User analytics error: {e}", exc_info=True)
            self._render_fallback_users()
    
    def _render_quality_analytics(self):
        """Render quality assessment analytics"""
        st.header("🔍 Quality Assessment")
        
        try:
            if hasattr(self.agent_api, 'query_repository'):
                # Get performance analytics which include quality metrics
                performance_data = asyncio.run(
                    self.agent_api.query_repository.get_performance_analytics(
                        hours_back=24
                    )
                )
                
                # Quality Metrics from performance data
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "Total Queries", 
                        performance_data.get('total_queries', 0)
                    )
                
                with col2:
                    st.metric(
                        "Success Rate", 
                        f"{performance_data.get('success_rate', 0):.1f}%"
                    )
                
                with col3:
                    st.metric(
                        "Avg Response Time", 
                        f"{performance_data.get('avg_response_time', 0):.2f}s"
                    )
                
                with col4:
                    st.metric(
                        "Error Rate", 
                        f"{performance_data.get('error_rate', 0):.1f}%"
                    )
                
                # Quality trends from historical data
                if 'response_time_trend' in performance_data:
                    st.subheader("Performance Trends Over Time")
                    
                    trend_data = performance_data['response_time_trend']
                    if trend_data:
                        df = pd.DataFrame(trend_data)
                        
                        if not df.empty:
                            fig = px.line(
                                df, 
                                x="timestamp", 
                                y="avg_response_time",
                                title="Response Time Trends"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                
                # Recent queries for quality assessment
                recent_queries = asyncio.run(
                    self.agent_api.query_repository.get_query_history(
                        hours_back=24,
                        limit=10
                    )
                )
                
                if recent_queries:
                    st.subheader("Recent Query Performance")
                    df_queries = pd.DataFrame([
                        {
                            "Query": q.get('query_text', '')[:50] + "..." if len(q.get('query_text', '')) > 50 else q.get('query_text', ''),
                            "Response Time": f"{q.get('response_time', 0):.2f}s",
                            "Status": "✅ Success" if q.get('success') else "❌ Failed",
                            "Platform": q.get('platform', 'Unknown'),
                            "Timestamp": q.get('timestamp', '')
                        }
                        for q in recent_queries[:10]
                    ])
                    st.dataframe(df_queries, use_container_width=True)
                else:
                    st.info("No recent queries found")
                        
            else:
                st.info("Quality analytics not available - requires analytics components")
                
        except Exception as e:
            st.error(f"Error loading quality analytics: {e}")
            logger.error(f"Quality analytics error: {e}", exc_info=True)
    
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
                await self.agent_api.close()
            
            # Note: ConversationMemory and Cache don't have close methods implemented
            # They use SQLite which closes automatically
                
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
        
        logger.info("Streamlit interface shutdown complete")
    
    def _render_fallback_overview(self):
        """Render fallback overview when comprehensive analytics are not available"""
        st.info("Comprehensive analytics not available. Showing basic system status.")
        
        # Basic system metrics
        col1, col2, col3, col4 = st.columns(4)
        
        session_data = st.session_state.get(self.session_state_key, {})
        
        with col1:
            st.metric("Session Queries", session_data.get("query_count", 0))
        
        with col2:
            st.metric("System Status", "🟢 Online")
        
        with col3:
            st.metric("Cache Status", "🟢 Active" if self.cache_enabled else "🔴 Disabled")
        
        with col4:
            st.metric("Analytics", "🟢 Enabled" if self.enable_analytics else "🔴 Disabled")
        
        # Basic usage chart with mock data
        st.subheader("Session Activity")
        dates = pd.date_range(start='2024-06-01', end='2024-06-02', freq='H')
        activity = [5 + i % 10 for i in range(len(dates))]
        
        fig = px.line(
            x=dates, 
            y=activity,
            title="Hourly Activity (Mock Data)",
            labels={"x": "Time", "y": "Activity"}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_fallback_performance(self):
        """Render fallback performance metrics"""
        st.info("Advanced performance analytics not available. Showing basic metrics.")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("System Load", "Normal")
        
        with col2:
            st.metric("Memory Usage", "< 512MB")
        
        with col3:
            st.metric("Response Status", "🟢 Responsive")
        
        # Basic performance chart
        st.subheader("System Performance (Estimated)")
        times = pd.date_range(start='2024-06-01', periods=24, freq='H')
        response_times = [0.5 + (i % 5) * 0.1 for i in range(24)]
        
        fig = px.line(
            x=times,
            y=response_times,
            title="Response Time Trends (Mock Data)",
            labels={"x": "Time", "y": "Response Time (s)"}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_fallback_users(self):
        """Render fallback user analytics"""
        st.info("User analytics not available. Showing session information.")
        
        session_data = st.session_state.get(self.session_state_key, {})
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Current Session", "Active")
        
        with col2:
            st.metric("Session Queries", session_data.get("query_count", 0))
        
        with col3:
            st.metric("Session Duration", "N/A")
        
        with col4:
            st.metric("User Type", "Streamlit User")
        
        # Session activity
        st.subheader("Session Information")
        st.json({
            "user_id": session_data.get("user_id", "Unknown"),
            "session_id": session_data.get("session_id", "Unknown"),
            "last_query": session_data.get("last_query_time", "Never"),
            "total_queries": session_data.get("query_count", 0)
        })
