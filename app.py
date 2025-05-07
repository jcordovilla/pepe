import streamlit as st
from rag_engine import get_answer
from fetch_messages import update_discord_messages
from embed_store import build_langchain_faiss_index
from datetime import datetime
import json

st.set_page_config(page_title="Discord RAG App", layout="wide")
st.title("📡 GenAI Pathfinder Discord Bot - v0.1")

# ✅ Run full pipeline
if st.button("🧰 Run Full Pipeline (Sync + Embed)"):
    with st.spinner("Running full pipeline..."):
        update_discord_messages()
        build_langchain_faiss_index()
    st.success("✅ Pipeline complete!")

# ✅ Tabs
tab1, tab2 = st.tabs(["💬 Ask", "📚 History"])

with tab1:
    st.title("RAG Query Interface")

    with st.form(key="query_form"):
        query = st.text_input("Enter your query:")
        show_context = st.checkbox("Show RAG Context", value=False)
        submit = st.form_submit_button("Submit")

    if submit and query:
        try:
            if show_context:
                final_answer, context = get_answer(query, return_matches=True)
                st.write("**Answer:**", final_answer)
                st.write("**RAG Context:**")
                for match in context:
                    st.markdown(f"- {match}")
            else:
                final_answer = get_answer(query, return_matches=False)
                st.write("**Answer:**", final_answer)
        except Exception as e:
            st.error(f"❌ Sorry, there was an error processing your request: {e}")

# ✅ History tab
with tab2:
    st.subheader("📚 Previous Questions and Answers")

    try:
        with open("chat_history.jsonl", "r", encoding='utf-8') as f:
            lines = f.readlines()

        for entry in reversed(lines[-20:]):
            record = json.loads(entry)
            st.markdown(f"**🕒 {record['timestamp']}** — **Q:** {record['question']}")
            st.markdown(f"**A:** {record['answer'][:500]}...")
            with st.expander("Show full answer and context"):
                try:
                    st.code(json.dumps(json.loads(record['answer']), indent=2), language="json")
                except:
                    st.markdown(record['answer'])
    except FileNotFoundError:
        st.info("No history yet.")
