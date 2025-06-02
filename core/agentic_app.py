"""
Agentic Discord Bot - Streamlit Interface

Modern Streamlit interface for the agentic Discord bot system with pipeline management.
"""

import streamlit as st
import asyncio
import logging
import sys
import os
from datetime import datetime
from typing import Dict, Any

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Now we can import the agentic modules
from agentic.interfaces.agent_api import AgentAPI
from agentic.interfaces.streamlit_interface import StreamlitInterface

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Agentic Discord Bot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    /* Main container */
    .main {
        padding: 1rem;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #1E88E5;
        font-weight: 600;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background-color: #f8f9fa;
    }
    
    /* Success messages */
    .stSuccess {
        background-color: #d4edda;
        border-color: #c3e6cb;
        color: #155724;
    }
    
    /* Warning messages */
    .stWarning {
        background-color: #fff3cd;
        border-color: #ffeaa7;
        color: #856404;
    }
    
    /* Error messages */
    .stError {
        background-color: #f8d7da;
        border-color: #f5c6cb;
        color: #721c24;
    }
    
    /* Buttons */
    .stButton button {
        transition: all 0.3s ease;
        border-radius: 6px;
    }
    
    .stButton button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    /* Metrics */
    .metric-container {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)


def initialize_system():
    """Initialize the agentic system"""
    if "agentic_system" not in st.session_state:
        with st.spinner("🚀 Initializing agentic system..."):
            try:
                # Configuration for the system
                config = {
                    "orchestrator": {},
                    "vector_store": {
                        "collection_name": "discord_messages",
                        "persist_directory": "data/vectorstore"
                    },
                    "memory": {
                        "db_path": "data/conversation_memory.db"
                    },
                    "pipeline": {
                        "base_path": ".",
                        "db_path": "data/discord_messages.db"
                    }
                }
                
                # Initialize Agent API
                agent_api = AgentAPI(config)
                
                # Initialize Streamlit Interface
                streamlit_interface = StreamlitInterface(
                    agent_api=agent_api,
                    cache_enabled=True,
                    enable_analytics=True
                )
                
                st.session_state.agentic_system = {
                    "agent_api": agent_api,
                    "streamlit_interface": streamlit_interface,
                    "initialized": True,
                    "init_time": datetime.now()
                }
                
                st.success("✅ Agentic system initialized successfully!")
                
            except Exception as e:
                st.error(f"❌ Failed to initialize system: {e}")
                st.session_state.agentic_system = {"initialized": False, "error": str(e)}


def render_main_page():
    """Render the main application page"""
    st.title("🤖 Agentic Discord Bot")
    st.markdown("### Intelligent Discord Message Analysis & Pipeline Management")
    
    # System status
    system = st.session_state.get("agentic_system", {})
    
    if not system.get("initialized"):
        st.error("❌ System not properly initialized")
        if st.button("🔄 Retry Initialization"):
            if "agentic_system" in st.session_state:
                del st.session_state.agentic_system
            st.rerun()
        return
    
    streamlit_interface = system["streamlit_interface"]
    agent_api = system["agent_api"]
    
    # Navigation
    tab1, tab2, tab3, tab4 = st.tabs(["💬 Chat", "⚙️ Pipeline", "📊 Analytics", "🔧 System"])
    
    with tab1:
        render_chat_tab(streamlit_interface)
    
    with tab2:
        render_pipeline_tab(agent_api)
    
    with tab3:
        render_analytics_tab(streamlit_interface)
    
    with tab4:
        render_system_tab(agent_api)


def render_chat_tab(streamlit_interface):
    """Render the chat interface tab"""
    st.header("💬 Chat with your Discord Data")
    
    # Render the main chat interface
    streamlit_interface.render_chat_interface()


def render_pipeline_tab(agent_api):
    """Render the pipeline management tab"""
    st.header("⚙️ Data Processing Pipeline")
    st.markdown("Manage the Discord message processing pipeline")
    
    # Get pipeline status
    try:
        status = agent_api.get_pipeline_status()
        
        # Status overview
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if status.get("is_running"):
                st.error(f"🔄 **Running**: {status.get('current_step', 'Unknown step')}")
            else:
                st.success("✅ **Ready**")
        
        with col2:
            available_steps = status.get("available_steps", [])
            st.info(f"📋 **Available Steps**: {len(available_steps)}")
        
        with col3:
            st.info(f"🕒 **Status Check**: {datetime.now().strftime('%H:%M:%S')}")
        
        st.markdown("---")
        
        # Pipeline controls
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("🚀 Pipeline Controls")
            
            # Full pipeline button
            if st.button(
                "🚀 Run Full Pipeline",
                disabled=status.get("is_running", False),
                help="Run all pipeline steps sequentially: fetch, embed, detect, sync"
            ):
                run_full_pipeline(agent_api)
                st.rerun()
            
            # Individual step controls
            st.markdown("**Individual Steps:**")
            step_cols = st.columns(len(available_steps) if available_steps else 1)
            
            for i, step in enumerate(available_steps):
                with step_cols[i % len(step_cols)]:
                    step_label = step.replace("_", " ").title()
                    if st.button(
                        f"▶️ {step_label}",
                        key=f"step_{step}",
                        disabled=status.get("is_running", False),
                        help=f"Run only the {step_label} step"
                    ):
                        run_pipeline_step(agent_api, step)
                        st.rerun()
        
        with col2:
            st.subheader("📊 Quick Stats")
            
            # Get data statistics
            try:
                stats_result = asyncio.run(agent_api.get_data_stats())
                if stats_result.get("success"):
                    db_stats = stats_result.get("database", {})
                    vector_stats = stats_result.get("vector_store", {})
                    
                    st.metric("Messages", db_stats.get("total_messages", "N/A"))
                    st.metric("Resources", db_stats.get("total_resources", "N/A"))
                    
                    if "document_count" in vector_stats:
                        st.metric("Indexed Docs", vector_stats["document_count"])
                else:
                    st.warning("Unable to load stats")
            except Exception as e:
                st.error(f"Stats error: {e}")
        
        # Pipeline history and logs
        render_pipeline_details(agent_api)
        
    except Exception as e:
        st.error(f"❌ Error getting pipeline status: {e}")


def render_pipeline_details(agent_api):
    """Render detailed pipeline information"""
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📜 Recent Pipeline Runs")
        
        try:
            history_result = agent_api.get_pipeline_history(limit=5)
            if history_result.get("success"):
                history = history_result.get("history", [])
                
                if history:
                    for run in history[-3:]:  # Show last 3 runs
                        success = run.get("success", False)
                        status_emoji = "✅" if success else "❌"
                        start_time = run.get("start_time", "Unknown")
                        pipeline_id = run.get("pipeline_id", "Unknown")
                        
                        with st.expander(f"{status_emoji} {start_time} - {pipeline_id}"):
                            st.json(run)
                else:
                    st.info("No pipeline runs found")
            else:
                st.warning("Unable to load pipeline history")
        except Exception as e:
            st.error(f"Error loading history: {e}")
    
    with col2:
        st.subheader("📋 Pipeline Logs")
        
        if st.button("🔄 Refresh Logs"):
            st.rerun()
        
        try:
            logs_result = asyncio.run(agent_api.get_pipeline_logs(lines=20))
            if logs_result.get("success"):
                logs = logs_result.get("lines", [])
                if logs:
                    log_text = "".join(logs)
                    st.text_area(
                        "Recent Log Output",
                        value=log_text,
                        height=200,
                        label_visibility="collapsed"
                    )
                else:
                    st.info("No logs available")
            else:
                st.warning("Unable to load logs")
        except Exception as e:
            st.error(f"Error loading logs: {e}")


def run_full_pipeline(agent_api):
    """Run the full pipeline"""
    try:
        with st.spinner("🚀 Running full pipeline..."):
            result = asyncio.run(agent_api.run_pipeline("streamlit_user"))
            
            if result.get("success"):
                st.success("✅ Pipeline completed successfully!")
                
                # Show statistics
                if "stats" in result:
                    stats = result["stats"]
                    st.balloons()
                    st.info(f"📊 **Results**: {stats.get('total_messages', 'N/A')} messages, {stats.get('total_resources', 'N/A')} resources")
            else:
                st.error(f"❌ Pipeline failed: {result.get('error', 'Unknown error')}")
                if "failed_step" in result:
                    st.error(f"**Failed at step**: {result['failed_step']}")
                    
    except Exception as e:
        st.error(f"❌ Pipeline error: {e}")


def run_pipeline_step(agent_api, step_name):
    """Run a single pipeline step"""
    try:
        step_label = step_name.replace("_", " ").title()
        with st.spinner(f"▶️ Running {step_label}..."):
            result = asyncio.run(agent_api.run_pipeline_step(step_name, "streamlit_user"))
            
            if result.get("success"):
                st.success(f"✅ {step_label} completed successfully!")
                
                # Show statistics
                if "stats" in result:
                    stats = result["stats"]
                    st.info(f"📊 **Results**: {stats.get('total_messages', 'N/A')} messages, {stats.get('total_resources', 'N/A')} resources")
            else:
                st.error(f"❌ {step_label} failed: {result.get('error', 'Unknown error')}")
                
    except Exception as e:
        st.error(f"❌ Step error: {e}")


def render_analytics_tab(streamlit_interface):
    """Render the analytics tab"""
    st.header("📊 System Analytics")
    
    # Render the analytics page from the streamlit interface
    streamlit_interface.render_analytics_page()


def render_system_tab(agent_api):
    """Render the system management tab"""
    st.header("🔧 System Management")
    
    # Health check
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏥 System Health")
        
        if st.button("🔄 Check Health"):
            check_system_health(agent_api)
    
    with col2:
        st.subheader("🔧 Maintenance")
        
        if st.button("⚡ Optimize System"):
            optimize_system(agent_api)


def check_system_health(agent_api):
    """Check system health"""
    try:
        with st.spinner("🏥 Checking system health..."):
            health = asyncio.run(agent_api.health_check())
            
            if health.get("status") == "healthy":
                st.success("✅ System is healthy!")
            else:
                st.warning("⚠️ System has issues")
            
            # Show component details
            components = health.get("components", {})
            if components:
                for name, info in components.items():
                    status = info.get("status", "unknown")
                    if status == "healthy":
                        st.success(f"✅ {name.title()}: {status}")
                    else:
                        st.error(f"❌ {name.title()}: {status}")
                        
    except Exception as e:
        st.error(f"❌ Health check failed: {e}")


def optimize_system(agent_api):
    """Optimize system performance"""
    try:
        with st.spinner("⚡ Optimizing system..."):
            result = asyncio.run(agent_api.optimize_system())
            
            if result.get("success"):
                st.success("✅ System optimization completed!")
                
                # Show optimization details
                optimizations = result.get("optimizations", [])
                for opt in optimizations:
                    component = opt.get("component", "unknown")
                    success = opt.get("success", False)
                    status = "✅" if success else "❌"
                    st.info(f"{status} {component.title()} optimization")
            else:
                st.error(f"❌ Optimization failed: {result.get('error', 'Unknown error')}")
                
    except Exception as e:
        st.error(f"❌ Optimization error: {e}")


def main():
    """Main application entry point"""
    # Initialize system
    initialize_system()
    
    # Render main page
    render_main_page()


if __name__ == "__main__":
    main()
