import streamlit as st
import json
from datetime import datetime

from rag_engine import get_answer, search_messages, safe_jump_url

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

# Sidebar filters for Search tab
guild_input = st.sidebar.text_input("Guild ID (optional)")
channel_input = st.sidebar.text_input("Channel ID (optional)")
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
                f.write(json.dumps(record) + '')

        except Exception as e:
            st.error(f"❌ Error processing request: {e}")

# --- Tab 2: Hybrid Search ---
with tab2:
    st.header("Hybrid Keyword & Semantic Search")
    search_query = st.text_input("Search query:")
    k = st.slider("Top K results", min_value=1, max_value=10, value=5)

    if st.button("Search Messages") and search_query:
        params = {'query': search_query, 'k': k}
        if keyword_input:
            params['keyword'] = keyword_input
        if guild_input:
            try:
                params['guild_id'] = int(guild_input)
            except ValueError:
                st.warning("Guild ID must be an integer.")
        if channel_input:
            try:
                # allow either numeric ID or channel name
                cid = int(channel_input)
                params['channel_id'] = cid
            except ValueError:
                params['channel_name'] = channel_input
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
            rec = json.loads(entry)
            st.markdown(f"**{rec['timestamp']}** — **Q:** {rec['question']}")
            st.markdown(f"**A:** {rec['answer']}")
    except FileNotFoundError:
        st.info("No history available yet.")
