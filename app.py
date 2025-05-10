import streamlit as st
import json
import re
from datetime import datetime
from typing import List, Dict, Any, Optional

from db import SessionLocal, Message
from rag_engine import get_answer, search_messages, safe_jump_url, get_agent_answer
from time_parser import parse_timeframe
from tools import summarize_messages
from utils import validate_channel_id, validate_channel_name

# ─── Streamlit page configuration ───────────────────────────────────────────────
st.set_page_config(page_title="GenAI Discord Bot", layout="wide")
st.title("📡 Pathfinder – GenAI Discord Bot")

# ─── CACHING HELPERS ─────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def get_channels() -> List[Dict[str, Any]]:
    """Fetch distinct channel names & IDs from the database (cached)."""
    session = SessionLocal()
    try:
        rows = session.query(Message.channel_name, Message.channel_id).distinct().all()
        return [
            {"name": name, "id": cid}
            for name, cid in rows
            if name and validate_channel_name(name) and validate_channel_id(cid)  # skip invalid entries
        ]
    finally:
        session.close()

def log_interaction(query: str, answer: Any, matches: List[Dict[str, Any]]) -> None:
    """Log the interaction to chat_history.jsonl."""
    record = {
        "timestamp": datetime.now().isoformat(),
        "question": query,
        "answer": answer
    }
    with open("chat_history.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

@st.cache_data(show_spinner=False)
def cached_search(
    query: str,
    channel_id: Optional[int],
    k: int
) -> List[Dict[str, Any]]:
    """Cache wrapper around semantic + keyword search."""
    return search_messages(query=query, channel_id=channel_id, k=k)

@st.cache_data(show_spinner=False)
def cached_summarize(
    start_iso: str,
    end_iso: str,
    channel_id: Optional[int],
    as_json: bool
) -> Any:
    """Cache wrapper around time-scoped summarization."""
    return summarize_messages(
        start_iso=start_iso,
        end_iso=end_iso,
        channel_id=channel_id,
        as_json=as_json
    )

@st.cache_data(show_spinner=False)
def read_history(n: int = 20) -> List[Dict[str, Any]]:
    """
    Read the last `n` entries from chat_history.jsonl (cached).
    Silently skips any lines that fail JSON decoding.
    """
    out: List[Dict[str, Any]] = []
    try:
        lines = (
            open("chat_history.jsonl", "r", encoding="utf-8")
            .read()
            .splitlines()[-n:]
        )
    except FileNotFoundError:
        return out

    for line in lines:
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            # skip this malformed line
            continue
        out.append(rec)
    return out

# ─── SIDEBAR ────────────────────────────────────────────────────────────────────
channels = get_channels()
channel_options = {ch["name"]: ch["id"] for ch in channels}

selected_channel = st.sidebar.selectbox(
    "Channel (optional)",
    [""] + list(channel_options.keys())
)
selected_channel_id = (
    channel_options[selected_channel] if selected_channel else None
)

# Validate selected channel ID
if selected_channel_id and not validate_channel_id(selected_channel_id):
    st.error(f"Invalid channel ID format: {selected_channel_id}")
    selected_channel_id = None

# ─── TABS ───────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["💬 Query", "📚 History"])

# ─── TAB 1: QUERY ────────────────────────────────────────────────────────────────
with tab1:
    st.header("Ask or Search Discord Messages")
    mode = st.radio("Mode:", ["Ask (RAG Query)", "Search (Messages)"], index=0)
    query = st.text_input("Enter your query:")
    k = st.slider("Top K results", min_value=1, max_value=10, value=5)

    as_json = False
    show_context = False
    if mode == "Ask (RAG Query)":
        as_json = st.checkbox("Return raw JSON answer", value=False)
        show_context = st.checkbox("Show context snippets", value=False)

    if st.button("Run Query") and query:
        with st.spinner("⏳ Running your query..."):
            try:
                if mode == "Ask (RAG Query)":
                    # Let the agent handle time parsing and query execution
                    answer = get_agent_answer(query)
                    st.markdown("### 📚 Response:")
                    st.write(answer)
                    matches = []  # No matches for RAG queries
                else:
                    # Direct search mode
                    results = search_messages(
                        query=query,
                        k=k,
                        channel_id=selected_channel_id
                    )
                    matches = results
                    answer = None

                # Log the interaction
                log_interaction(query, answer, matches)
                
            except Exception as e:
                st.error(f"❌ Error processing request: {e}")

# ─── TAB 2: HISTORY ──────────────────────────────────────────────────────────────
with tab2:
    st.header("📚 Chat History (Last 20)")
    history = read_history(20)
    if history:
        for rec in reversed(history):
            st.markdown(f"**{rec['timestamp']}** — **Q:** {rec['question']}")
            st.markdown(f"**A:** {rec['answer']}")
    else:
        st.info("No history available yet.")
