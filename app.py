import streamlit as st
import json
import re
from datetime import datetime

from db import SessionLocal, Message
from rag_engine import get_answer, search_messages, safe_jump_url
from time_parser import parse_timeframe
from tools import summarize_messages

# --- Page config ---
st.set_page_config(page_title="GenAI Discord Bot", layout="wide")
st.title("📡 Pathfinder – GenAI Discord Bot")

# --- Helper to fetch channels (cached for performance) ---
@st.cache_data
def get_channels():
    session = SessionLocal()
    rows = session.query(Message.channel_name, Message.channel_id).distinct().all()
    session.close()
    return [
        {"name": name, "id": cid}
        for name, cid in rows
        if name  # skip empty names
    ]

channels = get_channels()
channel_options = {ch["name"]: ch["id"] for ch in channels}
selected_channel = st.sidebar.selectbox(
    "Channel (optional)",
    [""] + list(channel_options.keys())
)
selected_channel_id = channel_options.get(selected_channel) if selected_channel else None

# --- Tabs ---
tab1, tab2 = st.tabs(["💬 Query", "📚 History"])

# --- Tab 1: Query ---
with tab1:
    st.header("Ask or Search Discord Messages")
    query_type = st.radio(
        "Mode:",
        ["Ask (RAG Query)", "Search (Messages)"],
        index=0
    )

    query = st.text_input("Enter your query:")
    k = st.slider("Top K results", min_value=1, max_value=10, value=5)

    # For RAG queries, allow JSON or context previews
    if query_type == "Ask (RAG Query)":
        as_json = st.checkbox("Return raw JSON answer", value=False)
        show_context = st.checkbox("Show RAG context snippets", value=False)

    if st.button("Run Query") and query:
        try:
            if query_type == "Ask (RAG Query)":
                # 1) Detect natural-language timeframe
                m = re.search(r"(?:last|past)\s+(?:day|week|month)", query, re.I)
                if m:
                    expr = m.group(0)
                    start_dt, end_dt = parse_timeframe(expr)
                    st.markdown(f"**🕒 Parsed timeframe:** {start_dt.date()} → {end_dt.date()}")

                    # 2) Summarize via summarize_messages
                    summary = summarize_messages(
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
                    # Fallback: full RAG query
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

            else:  # Search (Messages)
                results = search_messages(
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
                else:
                    st.info("No messages found matching your criteria.")

            # --- Append to chat history ---
            record = {
                "timestamp": datetime.now().isoformat(),
                "question": query,
                "answer": answer if query_type == "Ask (RAG Query)" else results
            }
            with open("chat_history.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")

        except Exception as e:
            st.error(f"❌ Error processing request: {e}")

# --- Tab 2: History ---
with tab2:
    st.header("Chat History (last 20)")
    try:
        with open("chat_history.jsonl", "r", encoding="utf-8") as f:
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
