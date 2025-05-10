import streamlit as st
import json
from datetime import datetime, timedelta
from tools import get_channels, search_messages, summarize_messages
from rag_engine import get_agent_answer, get_answer
import pyperclip
from typing import Dict, Any, Optional

# Page config
st.set_page_config(
    page_title="Discord Message Search",
    page_icon="��",
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
    
    /* RAG context */
    .rag-context {
        background-color: #f5f9ff;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 1px solid #e3f2fd;
        font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
    }
    
    /* Copy button */
    .copy-button {
        background-color: #4CAF50;
        color: white;
        padding: 0.5rem 1rem;
        border: none;
        border-radius: 5px;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .copy-button:hover {
        background-color: #43A047;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
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
    """Format a message for display with jump URL."""
    author = message.get('author', {}).get('username', 'Unknown')
    timestamp = message.get('timestamp', '')
    content = message.get('content', '')
    jump_url = message.get('jump_url', '')
    
    formatted = f"**{author}** (_{timestamp}_):\n{content}"
    if jump_url:
        formatted += f"\n[🔗 View Message]({jump_url})"
    return formatted

def copy_to_clipboard(text: str):
    """Copy text to clipboard and show success message."""
    pyperclip.copy(text)
    st.success("✅ Copied to clipboard!")

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
        
        # Channel filter with emoji
        st.markdown("#### 📢 Channel")
        channels = get_channels()
        channel_names = [ch['name'] for ch in channels]
        selected_channel = st.selectbox(
            "Select a channel to filter results",
            ["All Channels"] + channel_names,
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # Output format with emoji
        st.markdown("#### 📝 Output Format")
        output_format = st.radio(
            "Choose how you want to see the results",
            ["Formatted Text", "JSON"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # Additional options with emoji
        st.markdown("#### ⚙️ Additional Options")
        show_rag = st.checkbox("Show RAG Context", value=False)
        k_results = st.slider("Number of Results", 1, 20, 5)
    
    # Main search interface with a nice container
    st.markdown("### 🔍 Search Messages")
    query = st.text_input(
        "Enter your search query",
        placeholder="e.g., 'Show me messages about AI from the last week'",
        label_visibility="collapsed"
    )
    
    if query:
        try:
            # Get channel ID if a channel is selected
            channel_id = None
            if selected_channel != "All Channels":
                channel_id = next(
                    (ch['id'] for ch in channels if ch['name'] == selected_channel),
                    None
                )
            
            # Process the query with a nice loading animation
            with st.spinner("🔍 Searching messages..."):
                if output_format == "JSON":
                    # Get raw results in JSON format
                    results = get_answer(
                        query,
                        k=k_results,
                        as_json=True,
                        channel_id=channel_id
                    )
                    
                    # Display JSON with copy button in a nice container
                    st.markdown("### 📊 Results")
                    st.markdown('<div class="message-box">', unsafe_allow_html=True)
                    st.json(results)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    if st.button("📋 Copy JSON"):
                        copy_to_clipboard(json.dumps(results, indent=2))
                else:
                    # Get formatted results
                    results = get_agent_answer(query, channel_id)
                    
                    # Display results with copy button in a nice container
                    st.markdown("### 📝 Results")
                    st.markdown('<div class="message-box">', unsafe_allow_html=True)
                    st.markdown(results)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    if st.button("📋 Copy Results"):
                        copy_to_clipboard(results)
                
                # Show RAG context if requested
                if show_rag:
                    st.markdown("### 🔍 RAG Context")
                    with st.expander("View RAG Context", expanded=True):
                        rag_results = get_answer(
                            query,
                            k=k_results,
                            return_matches=True,
                            channel_id=channel_id
                        )
                        if isinstance(rag_results, tuple):
                            _, matches = rag_results
                            for match in matches:
                                st.markdown(
                                    f'<div class="rag-context">{format_message(match)}</div>',
                                    unsafe_allow_html=True
                                )
        
        except Exception as e:
            st.error(f"❌ Error processing query: {str(e)}")

if __name__ == "__main__":
    main()
