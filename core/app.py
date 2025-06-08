import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import json
from typing import Dict, Any, Optional, List
import re
import time
from datetime import datetime, timedelta
import pandas as pd
from tools.tools import get_channels
from core.agent import get_agent_answer, analyze_query_type
from db.query_logs import (
    log_query_start, update_query_analysis, log_query_completion, 
    log_simple_query, get_recent_queries, get_query_analytics
)

# Enhanced page config
st.set_page_config(
    page_title="PEPE - Predictive Engine for Prompt Experimentation",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo/discord-bot',
        'Report a bug': 'https://github.com/your-repo/discord-bot/issues',
        'About': """
        # PEPE Discord AI Search
        
        An intelligent search interface for Discord messages using advanced RAG technology.
        
        **Features:**
        - Semantic search across Discord messages
        - AI-powered query understanding
        - Multiple search strategies (messages, resources, hybrid)
        - Real-time analytics and performance metrics
        
        Built with ❤️ using Streamlit, FAISS, and OpenAI.
        """
    }
)

# Dark Discord theme
st.markdown("""
<style>
/* Dark Discord-style theme */
.stApp {
    background: #36393f;
    color: #dcddde;
}

/* Main container */
.main .block-container {
    padding-top: 1rem;
    max-width: 1000px;
    background: #36393f;
}

/* Sidebar styling */
.css-1d391kg {
    background: #2f3136 !important;
    border-right: 1px solid #202225;
}

/* Text styling */
h1, h2, h3, h4, h5, h6 {
    color: #ffffff !important;
}

p, div, span {
    color: #dcddde !important;
}

/* Button styling */
.stButton button {
    background: #5865F2;
    color: white;
    border: none;
    border-radius: 6px;
    padding: 0.5rem 1rem;
    font-weight: 500;
    transition: background 0.2s ease;
}

.stButton button:hover {
    background: #4752C4;
}

/* Input styling */
.stTextInput input {
    background: #40444b !important;
    border: 1px solid #202225;
    border-radius: 6px;
    padding: 0.5rem;
    font-size: 1rem;
    color: #dcddde !important;
}

.stTextInput input:focus {
    border-color: #5865F2;
    outline: none;
    box-shadow: 0 0 0 2px rgba(88, 101, 242, 0.3);
}

.stTextInput label {
    color: #dcddde !important;
}

/* Selectbox styling */
.stSelectbox select {
    background: #40444b !important;
    border: 1px solid #202225;
    color: #dcddde !important;
    border-radius: 6px;
}

.stSelectbox label {
    color: #dcddde !important;
}

/* Slider styling */
.stSlider label {
    color: #dcddde !important;
}

/* Radio button styling */
.stRadio label {
    color: #dcddde !important;
}

/* Tabs styling */
.stTabs [data-baseweb="tab-list"] {
    background: #2f3136;
    border-radius: 8px;
    padding: 0.25rem;
}

.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #dcddde;
    border-radius: 6px;
    padding: 0.5rem 1rem;
}

.stTabs [aria-selected="true"] {
    background: #5865F2;
    color: white;
}

/* Message styling */
.message-box {
    background: #2f3136;
    border-radius: 8px;
    padding: 1rem;
    margin: 0.5rem 0;
    border: 1px solid #202225;
}

.message-header {
    display: flex;
    align-items: center;
    margin-bottom: 0.5rem;
    font-size: 0.9rem;
}

.author-name {
    font-weight: 600;
    color: #5865F2;
    margin-right: 0.5rem;
}

.message-time {
    color: #72767d;
    font-size: 0.8rem;
}

.channel-tag {
    background: #5865F2;
    color: white;
    padding: 0.2rem 0.5rem;
    border-radius: 10px;
    font-size: 0.7rem;
    margin-left: auto;
}

.jump-url {
    color: #00b0f4;
    text-decoration: none;
    font-size: 0.85rem;
}

.jump-url:hover {
    text-decoration: underline;
}

/* Info card styling */
.info-card {
    background: #2f3136;
    padding: 1rem;
    border-radius: 8px;
    border: 1px solid #202225;
    text-align: center;
    margin: 0.5rem 0;
    color: #dcddde;
}

/* Expander styling */
.streamlit-expander {
    background: #2f3136;
    border: 1px solid #202225;
    border-radius: 8px;
}

.streamlit-expander .streamlit-expanderHeader {
    background: #2f3136;
    color: #dcddde;
}

/* Markdown styling */
.stMarkdown {
    color: #dcddde;
}

/* Metric styling */
.metric-container {
    background: #2f3136;
    border: 1px solid #202225;
    border-radius: 8px;
    padding: 1rem;
}

/* Progress bar */
.stProgress .st-bo {
    background: #202225;
}

.stProgress .st-bp {
    background: #5865F2;
}

/* Success/Error messages */
.stSuccess {
    background: rgba(59, 165, 93, 0.2);
    border: 1px solid #3ba55d;
    color: #dcddde;
}

.stError {
    background: rgba(237, 66, 69, 0.2);
    border: 1px solid #ed4245;
    color: #dcddde;
}

.stInfo {
    background: rgba(88, 101, 242, 0.2);
    border: 1px solid #5865F2;
    color: #dcddde;
}

.stWarning {
    background: rgba(254, 231, 92, 0.2);
    border: 1px solid #fee75c;
    color: #dcddde;
}

/* Dataframe styling */
.dataframe {
    background: #2f3136 !important;
    color: #dcddde !important;
}

/* Custom status indicators */
.status-success {
    background: rgba(59, 165, 93, 0.2);
    color: #3ba55d;
    border: 1px solid #3ba55d;
    padding: 0.25rem 0.5rem;
    border-radius: 4px;
    font-size: 0.8rem;
}

.status-warning {
    background: rgba(254, 231, 92, 0.2);
    color: #fee75c;
    border: 1px solid #fee75c;
    padding: 0.25rem 0.5rem;
    border-radius: 4px;
    font-size: 0.8rem;
}

.status-error {
    background: rgba(237, 66, 69, 0.2);
    color: #ed4245;
    border: 1px solid #ed4245;
    padding: 0.25rem 0.5rem;
    border-radius: 4px;
    font-size: 0.8rem;
}
</style>
""", unsafe_allow_html=True)

def create_status_dashboard():
    """Create a minimal system status line."""
    try:
        # Get system status - simplified version
        channels = get_channels()
        
        # Get message count and user count from database
        from db.db import SessionLocal, Message
        from sqlalchemy import func
        
        session = SessionLocal()
        try:
            total_messages = session.query(Message).count()
            unique_users = session.query(func.count(func.distinct(func.json_extract(Message.author, '$.username')))).scalar()
        except Exception:
            total_messages = 0
            unique_users = 0
        finally:
            session.close()
        
        # Display as a single thin status line
        st.markdown(f"""
        <div style="background: #2f3136; padding: 0.5rem 1rem; border-radius: 6px; border-left: 3px solid #5865F2; margin: 0.5rem 0; font-size: 0.9rem; color: #dcddde;">
            📊 <strong style="color: #ffffff;">Status:</strong> {len(channels) if channels else 0} channels | {total_messages:,} messages | {unique_users} users | <span style="color: #3ba55d;">✅ Ready</span>
        </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.markdown(f"""
        <div style="background: #40444b; padding: 0.5rem 1rem; border-radius: 6px; border-left: 3px solid #fee75c; margin: 0.5rem 0; font-size: 0.9rem; color: #dcddde;">
            ⚠️ <strong style="color: #fee75c;">Status:</strong> Unable to load system status
        </div>
        """, unsafe_allow_html=True)

def create_query_examples():
    """Create query examples section."""
    st.markdown("#### 💡 Example Queries")
    
    examples = [
        {
            "title": "📅 Time-based Search",
            "queries": [
                "Show me messages from last week",
                "What was discussed yesterday?", 
                "Messages from the past 3 days about AI"
            ]
        },
        {
            "title": "🎯 Topic Search", 
            "queries": [
                "Find discussions about machine learning",
                "Show me troubleshooting conversations",
                "Messages about API integrations"
            ]
        },
        {
            "title": "👥 People Search",
            "queries": [
                "What did @john say about the project?",
                "Show me messages from team leads",
                "Conversations involving multiple people"
            ]
        },
        {
            "title": "📊 Analytical Queries",
            "queries": [
                "Summarize today's discussions",
                "Key topics from this week",
                "Most active conversations recently"
            ]
        }
    ]
    
    cols = st.columns(2)
    for i, category in enumerate(examples):
        with cols[i % 2]:
            with st.expander(f"{category['title']}", expanded=False):
                for query in category['queries']:
                    if st.button(f"📝 {query}", key=f"example_{i}_{query[:20]}", use_container_width=True):
                        st.session_state.example_query = query
                        st.rerun()

def create_recent_queries_sidebar():
    """Show recent queries in sidebar - simplified."""
    st.markdown("**Recent Queries:**")
    
    try:
        recent = get_recent_queries(hours=24, limit=3)
        if recent:
            for query in recent:
                query_text = query.get('query_text', '')
                if len(query_text) > 30:
                    query_text = query_text[:27] + "..."
                
                status = "✅" if query.get('is_successful') else "❌"
                st.markdown(f"- {status} {query_text}")
        else:
            st.markdown("*No recent queries*")
    except Exception:
        st.markdown("*Unable to load recent queries*")

def enhanced_format_message(message: Dict[str, Any]) -> str:
    """Enhanced message formatting with better visual hierarchy."""
    author_info = message.get("author", {})
    display_name = author_info.get("display_name", "")
    username = author_info.get("username", "Unknown")
    timestamp = message.get("timestamp", "")
    content = message.get("content", "")
    jump_url = message.get("jump_url", None)
    channel_name = message.get("channel_name", "")
    
    # Format timestamp for better readability
    formatted_timestamp = timestamp
    if timestamp:
        try:
            time_obj = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            formatted_timestamp = time_obj.strftime('%B %d, %Y at %I:%M %p')
        except:
            pass
    
    # Format author name
    author_display = f"{display_name} (@{username})" if display_name and display_name != username else username
    
    # Extract URLs and mentions
    external_links = []
    mentions = []
    if content:
        urls = re.findall(r'https?://[^\s<>"]+|www\.[^\s<>"]+', content)
        external_links = list(set(urls))
        
        mention_matches = re.findall(r'@(\w+)', content)
        mentions = list(set(mention_matches))
    
    # Build the formatted message
    formatted = f"""
<div style="border-left: 4px solid #5865F2; padding-left: 1rem; margin: 1rem 0; background: #2f3136; border-radius: 0 8px 8px 0; padding: 1rem; border: 1px solid #202225;">
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
        <strong style="color: #5865F2; font-size: 1.1rem;">{author_display}</strong>
        <span style="color: #72767d; font-size: 0.9rem;">{formatted_timestamp}</span>
    </div>
    
    {f'<div style="color: #72767d; font-size: 0.9rem; margin-bottom: 0.5rem;">#{channel_name}</div>' if channel_name else ''}
    
    <div style="margin: 0.75rem 0; line-height: 1.5; color: #dcddde;">
        {content}
    </div>
    
    <div style="margin-top: 1rem; display: flex; gap: 0.5rem; flex-wrap: wrap;">
"""
    
    # Add jump URL
    if jump_url:
        formatted += f'<a href="{jump_url}" target="_blank" style="background: #5865F2; color: white; padding: 0.25rem 0.5rem; border-radius: 4px; text-decoration: none; font-size: 0.8rem;">🔗 Jump to Message</a>'
    else:
        formatted += '<span style="background: #ed4245; color: white; padding: 0.25rem 0.5rem; border-radius: 4px; font-size: 0.8rem;">❌ No Jump URL</span>'
    
    # Add external links
    for url in external_links[:3]:  # Limit to 3 links
        formatted += f'<a href="{url}" target="_blank" style="background: #3ba55d; color: white; padding: 0.25rem 0.5rem; border-radius: 4px; text-decoration: none; font-size: 0.8rem;">🌐 External Link</a>'
    
    # Add mentions
    if mentions:
        formatted += f'<span style="background: #fee75c; color: #2f3136; padding: 0.25rem 0.5rem; border-radius: 4px; font-size: 0.8rem;">👥 Mentions: {", ".join(mentions[:3])}</span>'
    
    formatted += """
    </div>
</div>
"""
    
    return formatted

def format_message(message: Dict[str, Any]) -> str:
    """Format a message for display with enhanced link handling. Always includes jump_url."""
    author_info = message.get("author", {})
    display_name = author_info.get("display_name", "")
    username = author_info.get("username", "Unknown")
    timestamp = message.get("timestamp", "")
    content = message.get("content", "")
    jump_url = message.get("jump_url", None)
    
    # Format author name to show both display name and username if different
    author_display = f"{display_name} (@{username})" if display_name and display_name != username else username
    
    # Extract and format external links
    external_links = []
    if content:
        urls = re.findall(r'https?://[^\s<>"]+|www\.[^\s<>"]+', content)
        for url in urls:
            if url not in external_links:
                external_links.append(url)
    
    # Format the message
    formatted = f"""
**{author_display}** (_{timestamp}_)
{content}
"""
    # Always add jump URL, or a warning if missing
    if jump_url:
        formatted += f"[🔗 Jump to message]({jump_url})\n"
    else:
        formatted += f"<span style='color:red'>[No jump_url available]</span>\n"
    # Add external links section if any found
    if external_links:
        formatted += "\n**External Links:**\n"
        for url in external_links:
            formatted += f"- [🔗 {url}]({url})\n"
    return formatted

def format_summary(messages: List[Dict[str, Any]]) -> str:
    """Format a summary of messages with improved organization and link handling. Always includes jump_url for each post."""
    if not messages:
        return "No messages found."
    # Group messages by author
    author_messages = {}
    for msg in messages:
        author_info = msg.get("author", {})
        display_name = author_info.get("display_name", "")
        username = author_info.get("username", "Unknown")
        author_key = display_name if display_name else username
        if author_key not in author_messages:
            author_messages[author_key] = {
                "messages": [],
                "display_name": display_name,
                "username": username
            }
        author_messages[author_key]["messages"].append(msg)
    summary = "## Discord Message Summary\n\n"
    if messages:
        timestamps = [msg.get("timestamp") for msg in messages if msg.get("timestamp")]
        if timestamps:
            start_date = min(timestamps)
            end_date = max(timestamps)
            summary += f"**Timeframe:** {start_date} to {end_date}\n\n"
    for author_key, author_data in author_messages.items():
        display_name = author_data["display_name"]
        username = author_data["username"]
        author_display = f"{display_name} (@{username})" if display_name and display_name != username else username
        summary += f"### {author_display}\n\n"
        for msg in sorted(author_data["messages"], key=lambda x: x.get("timestamp", "")):
            summary += format_message(msg) + "\n---\n\n"
    return summary

def main():
    # Initialize session state
    if 'example_query' not in st.session_state:
        st.session_state.example_query = ""
    
    # Header with minimalist design
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0; margin-bottom: 1rem;">
        <h1 style="margin: 0; color: #ffffff; font-size: 2rem;">🤖 PEPE Discord AI Search</h1>
        <p style="color: #72767d; font-size: 1rem; margin: 0.25rem 0;">Intelligent semantic search across Discord messages</p>
    </div>
    """, unsafe_allow_html=True)
    
    # System status dashboard
    with st.container():
        create_status_dashboard()
    
    st.markdown("---")
    
    # Main layout with simplified sidebar
    with st.sidebar:
        st.markdown("### 🔧 Search Settings")
        
        # Channel filter
        st.markdown("**Channel:**")
        try:
            channels = get_channels()
            channel_names = [ch['name'] for ch in channels]
            total_channels = len(channel_names)
            
            selected_channel = st.selectbox(
                f"Select from {total_channels} channels",
                ["All Channels"] + channel_names,
                label_visibility="collapsed",
                key="channel_selector"
            )
                
        except Exception as e:
            st.error("Unable to load channels")
            selected_channel = "All Channels"
        
        # Advanced search options
        with st.expander("Advanced Options"):
            # Output format
            output_format = st.radio(
                "Output Format",
                ["Enhanced View", "Classic View", "JSON Debug"],
                help="Choose how you want to see the results"
            )
            
            # Number of results
            k_results = st.slider(
                "Max Results", 
                min_value=1, 
                max_value=50, 
                value=10,
                help="Number of results to return"
            )
            
            # Search strategy hint
            search_strategy = st.selectbox(
                "Search Strategy",
                ["Auto (Recommended)", "Messages Only", "Resources Only", "Hybrid Search"],
                help="Hint for the AI agent on search strategy"
            )
        
        # Recent queries section - simplified
        create_recent_queries_sidebar()
    
    # Main content area with tabs
    tab1, tab2, tab3 = st.tabs(["🔍 Search", "💡 Examples", "📊 Analytics"])
    
    with tab1:
        # Simplified search interface
        st.markdown("### 🔍 Search")
        
        # Use example query if selected
        default_query = st.session_state.example_query if st.session_state.example_query else ""
        if default_query:
            st.session_state.example_query = ""  # Clear after use
        
        col1, col2 = st.columns([4, 1])
        
        with col1:
            query = st.text_input(
                "Search query",
                value=default_query,
                placeholder="e.g., 'messages about AI from last week' or 'what did the team discuss?'",
                help="Use natural language - the AI understands context and time references."
            )
        
        with col2:
            search_clicked = st.button("Search", key="search_button", use_container_width=True, type="primary")
        
        # Quick search chips
        if not query:
            st.markdown("**Quick searches:**")
            quick_col1, quick_col2, quick_col3, quick_col4 = st.columns(4)
            
            quick_searches = [
                ("📅 Today", "messages from today"),
                ("🔥 Trending", "most active discussions"),
                ("❓ Questions", "questions and help requests"),
                ("🐛 Issues", "bugs and troubleshooting")
            ]
            
            for i, (label, search_query) in enumerate(quick_searches):
                with [quick_col1, quick_col2, quick_col3, quick_col4][i]:
                    if st.button(label, key=f"quick_{i}", use_container_width=True):
                        # Trigger search immediately with the quick query
                        process_search_query(
                            query=search_query,
                            selected_channel=selected_channel,
                            k_results=k_results,
                            search_strategy=search_strategy,
                            output_format=output_format
                        )
        
        # Process search
        if search_clicked and query:
            process_search_query(
                query=query,
                selected_channel=selected_channel,
                k_results=k_results,
                search_strategy=search_strategy,
                output_format=output_format
            )
    
    with tab2:
        create_query_examples()
    
    with tab3:
        display_analytics()

def process_search_query(query: str, selected_channel: str, k_results: int, search_strategy: str, output_format: str):
    """Process a search query with all the enhanced features."""
    query_log_id = -1
    start_time = time.time()
    
    try:
        # Build enhanced query with options
        user_query = query
        if selected_channel != "All Channels":
            user_query += f" in #{selected_channel}"
        if k_results != 10:
            user_query += f", top {k_results}"
        if search_strategy != "Auto (Recommended)":
            user_query += f", strategy: {search_strategy.lower()}"
        
        # Generate session identifier
        import uuid
        session_id = str(uuid.uuid4())[:8]
        streamlit_user_id = f"streamlit_{session_id}"
        
        # Log query start
        query_log_id = log_query_start(
            user_id=streamlit_user_id,
            username="Streamlit User",
            query_text=user_query,
            guild_id=None,
            channel_id=None,
            channel_name=selected_channel if selected_channel != "All Channels" else None,
            session_id=session_id
        )
        
        # Show progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        with st.spinner("🤖 AI is analyzing your query..."):
            progress_bar.progress(25)
            status_text.text("🔍 Analyzing query and determining strategy...")
            
            # Analyze query
            query_analysis = analyze_query_type(user_query)
            
            progress_bar.progress(50)
            status_text.text("📊 Searching through message database...")
            
            # Update query log with analysis
            if query_log_id > 0:
                update_query_analysis(query_log_id, query_analysis)
            
            # Get response
            response_start_time = time.time()
            results = get_agent_answer(user_query)
            response_time = int((time.time() - response_start_time) * 1000)
            
            progress_bar.progress(100)
            status_text.text("✅ Search completed!")
            
            time.sleep(0.5)  # Brief pause for UX
            progress_bar.empty()
            status_text.empty()
        
        total_processing_time = int((time.time() - start_time) * 1000)
        
        # Display results with enhanced formatting
        st.markdown("### 📝 Search Results")
        
        # Show query analysis info
        if query_analysis:
            analysis_col1, analysis_col2, analysis_col3 = st.columns(3)
            
            with analysis_col1:
                strategy = query_analysis.get('strategy', 'unknown')
                st.markdown(f"""
                <div class="info-card">
                    <strong>🎯 Strategy:</strong><br>
                    <span style="color: #5865F2;">{strategy.replace('_', ' ').title()}</span>
                </div>
                """, unsafe_allow_html=True)
            
            with analysis_col2:
                confidence = query_analysis.get('confidence', 0)
                st.markdown(f"""
                <div class="info-card">
                    <strong>🎲 Confidence:</strong><br>
                    <span style="color: #5865F2;">{confidence:.2f}</span>
                </div>
                """, unsafe_allow_html=True)
            
            with analysis_col3:
                st.markdown(f"""
                <div class="info-card">
                    <strong>⚡ Response Time:</strong><br>
                    <span style="color: #5865F2;">{total_processing_time}ms</span>
                </div>
                """, unsafe_allow_html=True)
        
        # Format and display results
        st.markdown('<div class="message-box fade-in">', unsafe_allow_html=True)
        
        if output_format == "JSON Debug":
            st.json(results)
        elif output_format == "Enhanced View":
            display_enhanced_results(results)
        else:  # Classic View
            display_classic_results(results)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Complete logging
        complete_query_logging(
            query_log_id, results, total_processing_time, response_time, 
            k_results, query_analysis, streamlit_user_id, user_query, selected_channel
        )
        
    except Exception as e:
        handle_search_error(e, query_log_id, start_time, query)

def display_enhanced_results(results):
    """Display results in enhanced format."""
    if isinstance(results, str):
        # Handle string responses from agent
        st.markdown(results)
    elif isinstance(results, dict) and 'messages' in results:
        messages = results['messages']
        if messages:
            st.markdown(f"**Found {len(messages)} relevant messages:**")
            for i, msg in enumerate(messages):
                st.markdown(enhanced_format_message(msg), unsafe_allow_html=True)
        else:
            st.info("No messages found matching your criteria.")
    elif isinstance(results, list):
        if results:
            st.markdown(f"**Found {len(results)} relevant messages:**")
            for i, msg in enumerate(results):
                st.markdown(enhanced_format_message(msg), unsafe_allow_html=True)
        else:
            st.info("No messages found matching your criteria.")
    else:
        st.markdown(str(results))

def display_classic_results(results):
    """Display results in classic format."""
    if isinstance(results, str):
        # Handle string responses from agent
        st.markdown(results)
    elif isinstance(results, dict) and 'messages' in results:
        formatted_results = format_summary(results['messages'])
        st.markdown(formatted_results, unsafe_allow_html=True)
    elif isinstance(results, list):
        formatted_results = format_summary(results)
        st.markdown(formatted_results, unsafe_allow_html=True)
    else:
        st.markdown(str(results))

def display_analytics():
    """Display analytics tab content."""
    st.markdown("### 📊 Usage Analytics")
    
    try:
        analytics = get_query_analytics(days=7)
        
        if analytics.get('total_queries', 0) > 0:
            # Analytics overview
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            
            with metric_col1:
                st.metric(
                    "Total Queries (7 days)",
                    analytics.get('total_queries', 0)
                )
            
            with metric_col2:
                success_rate = analytics.get('success_rate', 0) * 100
                st.metric(
                    "Success Rate",
                    f"{success_rate:.1f}%"
                )
            
            with metric_col3:
                avg_time = analytics.get('average_processing_time_ms', 0)
                st.metric(
                    "Avg Response Time",
                    f"{avg_time:.0f}ms"
                )
            
            # Strategy distribution
            strategy_dist = analytics.get('strategy_distribution', {})
            if strategy_dist:
                st.markdown("#### 🎯 Search Strategy Distribution")
                
                # Create a simple bar chart using Streamlit
                strategy_df = pd.DataFrame(list(strategy_dist.items()), columns=['Strategy', 'Count'])
                st.bar_chart(strategy_df.set_index('Strategy'))
            
            # Recent activity
            st.markdown("#### 🕒 Recent Query Activity")
            recent_queries = get_recent_queries(hours=168, limit=10)  # Last week
            
            if recent_queries:
                activity_df = pd.DataFrame(recent_queries)
                activity_df['timestamp'] = pd.to_datetime(activity_df['timestamp'])
                activity_df = activity_df.sort_values('timestamp', ascending=False)
                
                # Display recent queries table
                display_df = activity_df[['timestamp', 'username', 'query_text', 'is_successful', 'processing_time_ms']].copy()
                display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
                display_df['query_text'] = display_df['query_text'].str[:60] + '...'
                display_df = display_df.rename(columns={
                    'timestamp': 'Time',
                    'username': 'User', 
                    'query_text': 'Query',
                    'is_successful': 'Success',
                    'processing_time_ms': 'Time (ms)'
                })
                
                st.dataframe(display_df, use_container_width=True)
            else:
                st.info("No recent query activity to display.")
        else:
            st.info("No analytics data available. Start searching to see analytics!")
            
    except Exception as e:
        st.error(f"Unable to load analytics: {str(e)}")

def complete_query_logging(query_log_id, results, total_processing_time, response_time, 
                          k_results, query_analysis, streamlit_user_id, user_query, selected_channel):
    """Complete the query logging process."""
    if query_log_id > 0:
        response_text = str(results) if results else None
        search_results_count = len(results) if isinstance(results, list) else (len(results.get('messages', [])) if isinstance(results, dict) else None)
        
        log_query_completion(
            query_log_id=query_log_id,
            response_text=response_text,
            response_type=type(results).__name__,
            response_status='success',
            processing_time_ms=total_processing_time,
            search_results_count=search_results_count,
            performance_metrics={
                'llm_generation_time_ms': response_time,
                'k_parameter': k_results,
                'strategy': query_analysis.get('strategy') if query_analysis else None,
                'interface': 'streamlit'
            }
        )
    
    # Simple text logging
    log_simple_query(
        user_id=streamlit_user_id,
        username="Streamlit User",
        query_text=user_query,
        response_text=str(results),
        interface="streamlit",
        channel_name=selected_channel if selected_channel != "All Channels" else None
    )

def handle_search_error(error, query_log_id, start_time, query):
    """Handle search errors with proper logging."""
    st.error(f"❌ Error processing query: {str(error)}")
    st.error("💡 Try rephrasing your query or check if the selected channel contains messages.")
    
    # Log failure
    if query_log_id > 0:
        log_query_completion(
            query_log_id=query_log_id,
            response_text=None,
            response_type='None',
            response_status='error',
            processing_time_ms=int((time.time() - start_time) * 1000),
            error_message=str(error),
            performance_metrics={'interface': 'streamlit'}
        )
    
    # Simple error logging
    import uuid
    session_id = str(uuid.uuid4())[:8]
    log_simple_query(
        user_id=f"streamlit_{session_id}",
        username="Streamlit User",
        query_text=query,
        response_text=f"ERROR: {str(error)}",
        interface="streamlit"
    )

if __name__ == "__main__":
    main()
