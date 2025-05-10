import streamlit as st
import json
import re
from datetime import datetime

from rag_engine import get_answer, search_messages, safe_jump_url
from db import SessionLocal, Message
from time_parser import parse_timeframe
from tools import summarize_messages_in_range

# Streamlit page configuration
st.set_page_config(page_title="GenAI Discord Bot", layout="wide")
st.title("📡 Pathfinder - GenAI Discord Bot v2.1")

# Function to fetch distinct channels
def get_channels():
    session = SessionLocal()
    channels = (
        session.query(Message.channel_name, Message.channel_id)
        .distinct()
        .all()
    )
    session.close()
    return [
        {"name": ch[0], "id": ch[1]}
        for ch in channels
        if ch[0]
    ]

# Sidebar filters
channels = get_channels()
channel_options = {c["name"]: c["id"] for c in channels}
selected_channel = st.sidebar.selectbox(
    "Channel (optional)",
    [""] + list(channel_options.keys())
)
selected_channel_id = (
    channel_options[selected_channel]
    if selected_channel else None
)
author_input = st.sidebar.text_input(
    "Author name filter (optional)"
)
keyword_input = st.sidebar.text_input(
    "Keyword filter (optional)",
    help="Exact keyword pre-filter before semantic rerank"
)

# Tabs
tab1, tab2 = st.tabs(["💬 Query", "📚 History"])

# --- Tab 1: Query ---
with tab1:
    st.header("Unified Query Interface")
    query_type = st.radio(
        "Select Query Type:",
        ["Ask (RAG Query)", "Search (Messages)"]
    )

    query = st.text_input("Enter your query:")
    k = st.slider("Top K results", 1, 10, 5)

    if query_type == "Ask (RAG Query)":
        as_json = st.checkbox("Return raw JSON answer", value=False)
        show_context = st.checkbox("Show RAG context snippets", value=False)

    if st.button("Run Query") and query:
        try:
            # ----- Ask (RAG Query) branch -----
            if query_type == "Ask (RAG Query)":
                # 1) Look for a natural-language timeframe
                m = re.search(r"(?:last|past)\s+(?:day|week|month)", query, re.I)
                if m:
                    expr = m.group(0)
                    start_dt, end_dt = parse_timeframe(expr)
                    st.markdown(
                        f"**🕒 Parsed timeframe:** {start_dt.date()} → {end_dt.date()}"
                    )
                    # 2) Delegate to summarization tool
                    summary = summarize_messages_in_range(
                        start_iso=start_dt.isoformat(),
                        end_iso=end_dt.isoformat(),
                        channel_id=selected_channel_id,
                        output_format="text"
                    )
                    st.markdown(f"### 📚 Summary ({start_dt.date()}→{end_dt.date()}):")
                    st.write(summary)
                    answer = summary
                    matches = []
                else:
                    # Fallback: standard RAG query
                    answer, matches = get_answer(
                        query,
                        k=k,
                        as_json=as_json,
                        return_matches=True,
                        channel_id=selected_channel_id
                    )
                    # Display the answer
                    if as_json:
                        st.json(json.loads(answer))
                    else:
                        st.markdown(f"**Answer:** {answer}")

                    # Optionally show context snippets
                    if show_context:
                        st.markdown("**Context snippets:**")
                        for m in matches:
                            author = (
                                m.get("author", {}).get("display_name")
                                or m.get("author", {}).get("username")
                                or "Unknown"
                            )
                            ts = m.get("timestamp", "")
                            ch = m.get("channel_name", m.get("channel_id", ""))
                            snippet = (
                                m.get("content", "")
                                .replace("\n", " ")[:200] + "…"
                            )
                            url = safe_jump_url(m)
                            st.markdown(
                                f"- **{author}** (_{ts}_ in **#{ch}**): {snippet} [🔗]({url})"
                            )

            # ----- Search (Messages) branch -----
            elif query_type == "Search (Messages)":
                params = {"query": query, "k": k}
                if keyword_input:
                    params["keyword"] = keyword_input
                if selected_channel_id:
                    params["channel_id"] = selected_channel_id
                if author_input:
                    params["author_name"] = author_input

                results = search_messages(**params)
                if results:
                    for m in results:
                        author = (
                            m.get("author", {}).get("display_name")
                            or m.get("author", {}).get("username")
                            or "Unknown"
                        )
                        ts = m.get("timestamp", "")
                        ch = m.get("channel_name", m.get("channel_id", ""))
                        snippet = m.get("content", "")[:200] + "…"
                        url = safe_jump_url(m)
                        st.markdown(
                            f"- **{author}** (_{ts}_ in **#{ch}**): {snippet} [🔗]({url})"
                        )
                else:
                    st.info("No messages found matching the criteria.")

            # ----- Append to chat history -----
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
    st.header("📚 Chat History (Last 20)")
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
