import streamlit as st
import json
from datetime import datetime

from rag_engine import get_answer, search_messages, safe_jump_url
from db import SessionLocal, Message  # Import the database session and model

# Streamlit page configuration
st.set_page_config(page_title="GenAI Discord RAG", layout="wide")
st.title("📡 GenAI Pathfinder Discord Bot - v1.0")

# Sidebar: Data refresh instructions
st.sidebar.info(
    "To refresh data, run in your terminal:"
    "```bash"
    "python fetch_messages.py"
    "python embed_store.py"
    "```"
)

# Function to fetch distinct channels
def get_channels():
    session = SessionLocal()
    channels = session.query(Message.channel_name, Message.channel_id).distinct().all()
    session.close()
    return [{"name": ch[0], "id": ch[1]} for ch in channels if ch[0]]  # Filter out empty names

# Fetch channels for the dropdown
channels = get_channels()
channel_options = {ch["name"]: ch["id"] for ch in channels}  # Map channel names to IDs

# Sidebar filters for Search tab
guild_input = st.sidebar.text_input("Guild ID (optional)")
selected_channel = st.sidebar.selectbox("Channel (optional)", [""] + list(channel_options.keys()))
selected_channel_id = channel_options.get(selected_channel) if selected_channel else None
author_input = st.sidebar.text_input("Author name filter (optional)")
keyword_input = st.sidebar.text_input(
    "Keyword filter (optional)",
    help="Exact keyword pre-filter before semantic rerank"
)

# Main tabs
tab1, tab2, tab3 = st.tabs(["💬 Ask", "🔍 Search", "📚 History"])

# --- Tab 1: RAG Query ---
with tab1:
    st.header("RAG Query Interface")
    query = st.text_input("Natural-language question:")
    as_json = st.checkbox("Return raw JSON answer", value=False)
    show_context = st.checkbox("Show RAG context snippets", value=False)

    if st.button("Run RAG Query") and query:
        try:
            answer, matches = get_answer(
                query, k=5, as_json=as_json, return_matches=True
            )
            # Display answer
            if as_json:
                st.json(json.loads(answer))
            else:
                st.markdown(f"**Answer:** {answer}")

            # Display context snippets
            if show_context:
                st.markdown("**Context snippets:**")
                for m in matches:
                    author = (
                        m.get('author', {}).get('display_name')
                        or m.get('author', {}).get('username')
                        or 'Unknown'
                    )
                    ts = m.get('timestamp', '')
                    ch = m.get('channel_name', m.get('channel_id', ''))
                    content = m.get('content', '').replace("", " ")[:200] + '...'
                    url = safe_jump_url(m)
                    st.markdown(
                        f"- **{author}** (_{ts}_ in **#{ch}**): {content} [🔗]({url})"
                    )

            # Append to history
            record = {
                'timestamp': datetime.now().isoformat(),
                'question': query,
                'answer': answer
            }
            with open('chat_history.jsonl', 'a', encoding='utf-8') as f:
                f.write(json.dumps(record) + '\n')

        except Exception as e:
            st.error(f"❌ Error processing request: {e}")

# --- Tab 2: Hybrid Search ---
with tab2:
    st.header("Hybrid Keyword & Semantic Search")
    search_query = st.text_input("Search query:")
    k = st.slider("Top K results", min_value=1, max_value=10, value=5)

    # Dropdown for channel selection
    selected_channel = st.selectbox("Select a channel:", [""] + list(channel_options.keys()))
    selected_channel_id = channel_options.get(selected_channel) if selected_channel else None

    if st.button("Search Messages") and search_query:
        params = {'query': search_query, 'k': k}
        if keyword_input:
            params['keyword'] = keyword_input
        if guild_input:
            try:
                params['guild_id'] = int(guild_input)
            except ValueError:
                st.warning("Guild ID must be an integer.")
        if selected_channel_id:  # Use the selected channel ID from the dropdown
            params['channel_id'] = selected_channel_id
        if author_input:
            params['author_name'] = author_input

        results = search_messages(**params)
        if results:
            for m in results:
                author = (
                    m.get('author', {}).get('display_name')
                    or m.get('author', {}).get('username')
                    or 'Unknown'
                )
                ts = m.get('timestamp', '')
                ch = m.get('channel_name', m.get('channel_id', ''))
                content = m.get('content', '')[:200] + '...'
                url = safe_jump_url(m)
                st.markdown(
                    f"- **{author}** (_{ts}_ in **#{ch}**): {content} [🔗]({url})"
                )
        else:
            st.info("No messages found matching the criteria.")

# --- Tab 3: History ---
with tab3:
    st.header("📚 Chat History (Last 20)")
    try:
        with open('chat_history.jsonl', 'r', encoding='utf-8') as f:
            lines = f.readlines()
        for entry in reversed(lines[-20:]):
            try:
                rec = json.loads(entry)
                st.markdown(f"**{rec['timestamp']}** — **Q:** {rec['question']}")
                st.markdown(f"**A:** {rec['answer']}")
            except json.JSONDecodeError:
                st.warning("⚠️ Skipping malformed history entry.")
    except FileNotFoundError:
        st.info("No history available yet.")
