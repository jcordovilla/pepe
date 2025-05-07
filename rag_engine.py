import os
import json
from dotenv import load_dotenv
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from openai import OpenAI

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
INDEX_DIR = "index_faiss"
gpt_model = os.getenv("GPT_MODEL", "gpt-4-turbo-2024-04-09")

# Load OpenAI client and embedding model
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# ✅ Load FAISS vector store
def load_vectorstore():
    return FAISS.load_local(INDEX_DIR, embedding_model, allow_dangerous_deserialization=True)

# ✅ Retrieve top k matching messages from FAISS
def get_top_k_matches(query: str, k: int = 5):
    store = load_vectorstore()
    retriever = store.as_retriever(search_kwargs={"k": k})
    docs = retriever.invoke(query)
    return [doc.metadata for doc in docs]

# ✅ Build prompt for OpenAI (used for RAG re-answering)
def build_prompt(matches, question: str, as_json: bool) -> list:
    def safe_jump_url(m):
        url = m.get("jump_url")
        gid = m.get("guild_id")
        cid = m.get("channel_id")
        mid = m.get("message_id") or m.get("id")
        # Only use fallback if all are present and look like IDs
        if not url and all([gid, cid, mid]) and all(str(x).isdigit() for x in [gid, cid, mid]):
            url = f"https://discord.com/channels/{gid}/{cid}/{mid}"
        return url or "https://discord.com"  # fallback to root if nothing valid

    context = "\n\n".join(
        f"**{m['author']}** (_{m['timestamp']}_ in **#{m['channel']}**)\n"
        f"{m['content']}\n"
        f"[🔗 View Message]({safe_jump_url(m)})"
        for m in matches
    )

    instructions = (
        "You are a knowledgeable and versatile assistant specialized in analyzing Discord server data.\n\n"
        "Based on the user’s query, you can:\n"
        "- Search and summarize Discord messages using retrieval-augmented generation (RAG).\n"
        "- Call specific analysis tools to compute statistics, extract feedback, find skills, summarize weekly activity, and more.\n\n"
        "Always choose the most appropriate method:\n"
        "- If a tool matches the user request, prefer using it to ensure accurate, structured answers.\n"
        "- If a direct semantic search is better, use RAG.\n\n"
        "When presenting answers:\n"
        "- Respond in concise, clear natural language unless specifically asked for a JSON output.\n"
        "- If quoting messages, include key fields: author, timestamp, channel, and a brief message snippet.\n"
        "- Group or summarize information when appropriate.\n"
        "- Include clickable links (jump_url) when available.\n\n"
        "If no direct answer is possible, summarize findings or suggest a more specific query."
    )

    if as_json:
        instructions += "\nReturn the results as a JSON array with those fields."

    prompt = f"{instructions}\n\nContext:\n{context}\n\nUser's question: {question}\n"

    return [
        {"role": "system", "content": "You are a Discord message analyst."},
        {"role": "user", "content": prompt}
    ]

# ✅ Main RAG-based answering function
def get_answer(query: str, k: int = 5, as_json: bool = False, return_matches: bool = False):
    try:
        matches = get_top_k_matches(query, k)
        print("🧪 Retrieved matches:")
        for i, m in enumerate(matches):
            print(f"[{i}] message_id: {m.get('message_id')} | channel_id: {m.get('channel_id')} | guild_id: {m.get('guild_id')} | jump_url: {m.get('jump_url')}")

        messages = build_prompt(matches, query, as_json)

        response = openai_client.chat.completions.create(
            model=gpt_model,
            messages=messages,
            temperature=0.7,
            max_tokens=1000,
            **({"response_format": {"type": "json_object"}} if as_json else {})
        )

        answer = response.choices[0].message.content

        # ✅ Insert friendly fallback if no good matches
        if not matches:
            fallback_message = "⚠️ I couldn’t find relevant messages. Try rephrasing your question or being more specific."
            return (fallback_message, []) if return_matches else fallback_message

        return (answer, matches) if return_matches else answer

    except Exception as e:
        error_msg = f"❌ Error during RAG retrieval: {e}"
        return (error_msg, []) if return_matches else error_msg