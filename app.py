import streamlit as st
import json
import re
from datetime import datetime
from typing import List, Dict, Any, Optional

from db import SessionLocal, Message
from rag_engine import get_answer, search_messages, safe_jump_url
from time_parser import parse_timeframe
from tools import summarize_messages

# ─── Streamlit page configuration ───────────────────────────────────────────────
st.set_page_config(page_title="GenAI Discord Bot", layout="wide")
st.title("📡 Pathfinder – GenAI Discord Bot")

# ─── CACHING HELPERS ─────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def get_channels() -> List[Dict[str, Any]]:
    """Fetch distinct channel names & IDs from the database (cached)."""
    session = SessionLocal()
    rows = session.query(Message.channel_name, Message.channel_id).distinct().all()
    session.close()
    return [
        {"name": name, "id": cid}
        for name, cid in rows
        if name  # skip empty names
    ]

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
    """Read the last `n` entries from chat_history.jsonl (cached)."""
    try:
        lines = open("chat_history.jsonl", "r", encoding="utf-8").read().splitlines()[-n:]
        return [json.loads(l) for l in lines if l.strip()]
    except FileNotFoundError:
        return []

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
                    # 1) Natural-language timeframe detection
                    m = re.search(r"(?:last|past)\s+(?:day|week|month)", query, re.I)
                    if m:
                        expr = m.group(0)
                        try:
                            start_dt, end_dt = parse_timeframe(expr)
                        except ValueError as pe:
                            st.error(f"Could not parse timeframe “{expr}”: {pe}")
                            st.stop()

                        st.markdown(f"**🕒 Parsed timeframe:** {start_dt.date()} → {end_dt.date()}")
                        # 2) Summarize messages in that window
                        summary = cached_summarize(
                            start_iso=start_dt.isoformat(),
                            end_iso=end_dt.isoformat(),
                            channel_id=selected_channel_id,
                            as_json=False
                        )
                        st.markdown(f"### 📚 Summary ({start_dt.date()} → {end_dt.date()}):")
                        st.write(summary)
                        answer = summary
                        matches = []

                    else:
                        # Fallback to standard RAG
                        answer, matches = get_answer(
                            query,
                            k=k,
                            as_json=as_json,
                            return_matches=True,
                            channel_id=selected_channel_id
                        )
                        if as_json:
                            st.json(json.loads(answer))
                        else:
                            st.markdown(f"**Answer:** {answer}")

                        answer = answer  # for history
                        if show_context:
                            st.markdown("**Context snippets:**")
                            for m in matches:
                                author = (
                                    m.get("author", {}).get("display_name")
                                    or m.get("author", {}).get("username")
                                    or "Unknown"
                                )
                                ts = m.get("timestamp", "")
                                ch = m.get("channel_name", "") or m.get("channel_id", "")
                                snippet = m.get("content", "").replace("\n", " ")
                                url = safe_jump_url(m)
                                st.markdown(
                                    f"- **{author}** (_{ts}_ in **#{ch}**): {snippet} [🔗]({url})"
                                )

                else:
                    # Mode == "Search (Messages)"
                    results = cached_search(
                        query=query,
                        channel_id=selected_channel_id,
                        k=k
                    )
                    if results:
                        st.markdown(f"**Found {len(results)} messages:**")
                        for m in results:
                            author = (
                                m.get("author", {}).get("display_name")
                                or m.get("author", {}).get("username")
                                or "Unknown"
                            )
                            ts = m.get("timestamp", "")
                            ch = m.get("channel_name", "") or m.get("channel_id", "")
                            snippet = m.get("content", "")[:200] + "…"
                            url = safe_jump_url(m)
                            st.markdown(
                                f"- **{author}** (_{ts}_ in **#{ch}**): {snippet} [🔗]({url})"
                            )
                        answer = results
                    else:
                        st.info("No messages found matching your criteria.")
                        answer = []

                # ── Append to chat history ─────────────────────────────
                record = {
                    "timestamp": datetime.now().isoformat(),
                    "question": query,
                    "answer": answer
                }
                with open("chat_history.jsonl", "a", encoding="utf-8") as f:
                    f.write(json.dumps(record) + "\n")

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
