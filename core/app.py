import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import json
from datetime import datetime, timedelta
from tools.tools import get_channels, search_messages, summarize_messages
from tools.time_parser import parse_timeframe
from typing import Dict, Any, Optional, List
import re

# Page config
st.set_page_config(
    page_title="Discord Message Search",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern look
st.markdown("""
<style>
    /* Main container */
    .main {
        padding: 2rem;
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
    
    /* Buttons */
    .stButton button {
        background-color: #1E88E5;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        transition: all 0.3s ease;
    }
    
    .stButton button:hover {
        background-color: #1565C0;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }
    
    /* Input fields */
    .stTextInput input {
        border-radius: 5px;
        border: 1px solid #e0e0e0;
        padding: 0.5rem;
    }
    
    .stTextInput input:focus {
        border-color: #1E88E5;
        box-shadow: 0 0 0 1px #1E88E5;
    }
    
    /* Message boxes */
    .message-box {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border: 1px solid #e0e0e0;
    }
    
    /* Success message */
    .stSuccess {
        background-color: #E8F5E9;
        color: #2E7D32;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    /* Error message */
    .stError {
        background-color: #FFEBEE;
        color: #C62828;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #f8f9fa;
        border-radius: 5px;
        padding: 0.5rem;
    }
    
    /* Radio buttons */
    .stRadio > div {
        padding: 0.5rem;
        border-radius: 5px;
    }
    
    /* Slider */
    .stSlider {
        padding: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

def format_message(message: Dict[str, Any]) -> str:
    """Format a message for display with enhanced link handling."""
    author_info = message.get("author", {})
    display_name = author_info.get("display_name", "")
    username = author_info.get("username", "Unknown")
    timestamp = message.get("timestamp", "")
    content = message.get("content", "")
    jump_url = message.get("jump_url", "")
    
    # Format author name to show both display name and username if different
    author_display = f"{display_name} (@{username})" if display_name and display_name != username else username
    
    # Extract and format external links
    external_links = []
    if content:
        # Look for URLs in the content
        urls = re.findall(r'https?://[^\s<>"]+|www\.[^\s<>"]+', content)
        for url in urls:
            # Add to external links list if not already included
            if url not in external_links:
                external_links.append(url)
    
    # Format the message
    formatted = f"""
**{author_display}** (_{timestamp}_)
{content}
"""
    
    # Add jump URL if it exists
    if jump_url:
        formatted += f"[🔗 Jump to message]({jump_url})\n"
    
    # Add external links section if any found
    if external_links:
        formatted += "\n**External Links:**\n"
        for url in external_links:
            formatted += f"- [🔗 {url}]({url})\n"
    
    return formatted

def format_summary(messages: List[Dict[str, Any]]) -> str:
    """Format a summary of messages with improved organization and link handling."""
    # Group messages by author
    author_messages = {}
    for msg in messages:
        author_info = msg.get("author", {})
        display_name = author_info.get("display_name", "")
        username = author_info.get("username", "Unknown")
        # Use display name if available, otherwise use username
        author_key = display_name if display_name else username
        if author_key not in author_messages:
            author_messages[author_key] = {
                "messages": [],
                "display_name": display_name,
                "username": username
            }
        author_messages[author_key]["messages"].append(msg)
    
    # Format the summary
    summary = "## Discord Message Summary\n\n"
    
    # Add timeframe if available
    if messages:
        timestamps = [msg.get("timestamp") for msg in messages if msg.get("timestamp")]
        if timestamps:
            start_date = min(timestamps)
            end_date = max(timestamps)
            summary += f"**Timeframe:** {start_date} to {end_date}\n\n"
    
    # Add messages by author
    for author_key, author_data in author_messages.items():
        # Format author name to show both display name and username if different
        display_name = author_data["display_name"]
        username = author_data["username"]
        author_display = f"{display_name} (@{username})" if display_name and display_name != username else username
        
        summary += f"### {author_display}\n\n"
        for msg in sorted(author_data["messages"], key=lambda x: x.get("timestamp", "")):
            timestamp = msg.get("timestamp", "")
            content = msg.get("content", "")
            jump_url = msg.get("jump_url", "")
            
            # Extract external links
            external_links = []
            if content:
                urls = re.findall(r'https?://[^\s<>"]+|www\.[^\s<>"]+', content)
                external_links = [url for url in urls if url not in external_links]
            
            # Format the message with better visual hierarchy
            summary += f"#### {timestamp}\n\n"
            summary += f"{content}\n\n"
            
            if jump_url:
                summary += f"[🔗 Jump to message]({jump_url})\n\n"
            
            if external_links:
                summary += "**External Links:**\n"
                for url in external_links:
                    summary += f"- [🔗 {url}]({url})\n"
                summary += "\n"
            
            summary += "---\n\n"
    
    return summary

def main():
    # Header with logo and title
    col1, col2 = st.columns([1, 4])
    with col1:
        st.image("https://discord.com/assets/3437c10597c1526c3dbd98c737c2bcae.svg", width=50)
    with col2:
        st.title("Discord Message Search")
    
    # Sidebar with a nice header
    with st.sidebar:
        st.markdown("### 🔧 Search Options")
        st.markdown("---")
        
        # Channel filter
        st.markdown("#### 📢 Channel")
        channels = get_channels()
        st.write("[DEBUG] Channels:", channels)
        channel_names = [ch['name'] for ch in channels]
        selected_channel = st.selectbox(
            "Select a channel to filter results",
            ["All Channels"] + channel_names,
            label_visibility="collapsed",
            key="channel_selector"
        )
        st.write("[DEBUG] Selected channel:", selected_channel)
        
        st.markdown("---")
        
        # Output format
        st.markdown("#### 📝 Output Format")
        output_format = st.radio(
            "Choose how you want to see the results",
            ["Formatted Text", "JSON"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        k_results = st.slider("Number of Results", 1, 20, 5)

    # Main search interface
    st.markdown("### 🔍 Search Messages")
    query = st.text_input(
        "Enter your search query",
        placeholder="e.g., 'Show me messages about AI from the last week'",
        label_visibility="collapsed"
    )
    st.write("[DEBUG] Query:", query)
    
    if st.button("🔍 Search", key="search_button") and query:
        try:
            channel_id = None
            if selected_channel != "All Channels":
                channel_id = next((ch['id'] for ch in channels if ch['name'] == selected_channel), None)
            st.write("[DEBUG] Channel ID:", channel_id)
            with st.spinner("🔍 Searching messages..."):
                try:
                    start_dt, end_dt = parse_timeframe(query)
                    st.markdown(f"**🕒 Timeframe:** {start_dt.date()} → {end_dt.date()}")
                    results = summarize_messages(
                        start_iso=start_dt.isoformat(),
                        end_iso=end_dt.isoformat(),
                        channel_id=channel_id,
                        as_json=(output_format == "JSON")
                    )
                except ValueError as e:
                    if "No timeframe specified" in str(e):
                        results = search_messages(
                            query=query,
                            channel_id=channel_id,
                            k=k_results
                        )
                    else:
                        raise
                st.write("[DEBUG] Raw results:", results)
                if isinstance(results, str):
                    formatted_results = results
                elif isinstance(results, dict):
                    formatted_results = json.dumps(results, indent=2) if output_format == "JSON" else format_summary(results.get("messages", []))
                elif isinstance(results, list):
                    formatted_results = format_summary(results)
                else:
                    formatted_results = str(results)
                st.markdown("### 📝 Results")
                st.markdown('<div class="message-box">', unsafe_allow_html=True)
                if output_format == "JSON":
                    st.json(results)
                else:
                    st.markdown(formatted_results)
                st.markdown('</div>', unsafe_allow_html=True)
        except Exception as e:
            import traceback
            st.error(f"❌ Error processing query: {str(e)}")
            st.error("Debug info: Please check if the channel exists and contains messages in the specified timeframe.")
            st.error(traceback.format_exc())

if __name__ == "__main__":
    main()
