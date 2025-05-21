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
import logging

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="Discord Message Search",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📡 GenAI Pathfinder Discord Bot - v0.2")

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
    /* Custom sidebar styling can be added here if needed */
    
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
    """Format a summary of messages with improved organization, spacing, and Markdown."""
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

    # Add timeframe if available
    timestamps = [msg.get("timestamp") for msg in messages if msg.get("timestamp")]
    if timestamps:
        start_date = min(timestamps)
        end_date = max(timestamps)
        summary += f"**Timeframe:** `{start_date}` → `{end_date}`\n\n"

    # Add messages by author
    for author_key, author_data in author_messages.items():
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
                urls = re.findall(r'https?://[^\s<>"\']+|www\.[^\s<>"\']+', content)
                external_links = [url for url in urls if url not in external_links]
            # Format the message with better visual hierarchy
            summary += f"<div style='margin-bottom: 1.5em;'>"
            summary += f"<span style='font-size:1.1em; color:#1E88E5;'><b>{timestamp}</b></span>\n\n"
            summary += f"<div style='margin: 0.5em 0 1em 0; font-family: 'Segoe UI', 'Arial', sans-serif;'>{content}</div>\n"
            if jump_url:
                summary += f"<a href='{jump_url}' target='_blank' style='color:#1565C0;'>🔗 Jump to message</a><br>\n"
            if external_links:
                summary += "<div style='margin-top:0.5em;'><b>External Links:</b><ul style='margin:0.2em 0 0.5em 1.2em;'>"
                for url in external_links:
                    summary += f"<li><a href='{url}' target='_blank' style='color:#1565C0;'>{url}</a></li>"
                summary += "</ul></div>"
            summary += "</div>\n<hr style='border:0;border-top:1px solid #e0e0e0;margin:1.5em 0;'>\n"
    return summary

def main():
    logger.debug("Starting main function")
    # Sidebar with a nice header
    with st.sidebar:
        st.markdown("### 🔧 Search Options")
        st.markdown("---")
        
        # Channel filter
        st.markdown("#### 📢 Channel")
        logger.debug("Fetching channels")
        channels = get_channels()
        logger.debug(f"Found {len(channels)} channels")
        channel_names = [ch['name'] for ch in channels]
        selected_channel = st.selectbox(
            "Select a channel to filter results",
            ["All Channels"] + channel_names,
            label_visibility="collapsed",
            key="channel_selector"
        )
        logger.debug(f"Selected channel: {selected_channel}")
        
        st.markdown("---")
        
        # Output format
        st.markdown("#### 📝 Output Format")
        output_format = st.radio(
            "Choose how you want to see the results",
            ["Formatted Text", "JSON"],
            label_visibility="collapsed"
        )
        logger.debug(f"Selected output format: {output_format}")
        
        st.markdown("---")
        
        k_results = st.slider("Number of Results", 1, 20, 5)
        logger.debug(f"Selected number of results: {k_results}")

    # Main search interface
    st.markdown("### 🔍 Search Messages")
    query = st.text_input(
        "Enter your search query",
        placeholder="e.g., 'Show me messages about AI from the last week'",
        label_visibility="collapsed"
    )
    
    if st.button("🔍 Search", key="search_button") and query:
        logger.debug(f"Search button clicked with query: {query}")
        try:
            channel_id = None
            if selected_channel != "All Channels":
                channel_id = next((ch['id'] for ch in channels if ch['name'] == selected_channel), None)
                logger.debug(f"Resolved channel_id: {channel_id}")
            
            with st.spinner("🔍 Searching messages..."):
                try:
                    logger.debug("Attempting to parse timeframe from query")
                    start_dt, end_dt = parse_timeframe(query)
                    logger.debug(f"Parsed timeframe: {start_dt} to {end_dt}")
                    st.markdown(f"**🕒 Timeframe:** {start_dt.date()} → {end_dt.date()}")
                    
                    logger.debug("Calling summarize_messages")
                    results = summarize_messages(
                        start_iso=start_dt.isoformat(),
                        end_iso=end_dt.isoformat(),
                        channel_id=channel_id,
                        as_json=(output_format == "JSON")
                    )
                    logger.debug(f"Summarize results type: {type(results)}")
                except ValueError as e:
                    logger.debug(f"Timeframe parsing failed: {str(e)}")
                    if "No timeframe specified" in str(e):
                        logger.debug("Falling back to search_messages")
                        results = search_messages(
                            query=query,
                            channel_id=channel_id,
                            k=k_results
                        )
                        logger.debug(f"Search results type: {type(results)}")
                    else:
                        raise
                
                logger.debug("Formatting results")
                if isinstance(results, str):
                    formatted_results = results
                elif isinstance(results, dict):
                    formatted_results = json.dumps(results, indent=2) if output_format == "JSON" else format_summary(results.get("messages", []))
                elif isinstance(results, list):
                    formatted_results = format_summary(results)
                else:
                    formatted_results = str(results)
                
                logger.debug("Displaying results")
                st.markdown("### 📝 Results")
                st.markdown('<div class="message-box">', unsafe_allow_html=True)
                if output_format == "JSON":
                    st.json(results)
                    copy_text = json.dumps(results, indent=2)
                else:
                    st.markdown(formatted_results, unsafe_allow_html=True)
                    copy_text = formatted_results
                st.markdown('</div>', unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Improved copy-to-clipboard
                st.markdown("""
<textarea id="copyArea" style="width:100%;height:200px;border-radius:6px;padding:8px;font-family:monospace;">{}</textarea>
<button onclick="navigator.clipboard.writeText(document.getElementById('copyArea').value)" style="margin-top:8px;padding:6px 16px;border:none;border-radius:5px;background:#1E88E5;color:white;font-weight:bold;cursor:pointer;">📋 Copy Output</button>
""".format(copy_text.replace("</textarea>", "&lt;/textarea&gt;")), unsafe_allow_html=True)
                logger.debug("Results displayed successfully")
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}", exc_info=True)
            st.error(f"❌ Error processing query: {str(e)}")
            st.error("Debug info: Please check if the channel exists and contains messages in the specified timeframe.")
            st.error(traceback.format_exc())

if __name__ == "__main__":
    main()
