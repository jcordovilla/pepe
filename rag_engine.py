import os
import json
from dotenv import load_dotenv
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from openai import OpenAI
from utils import build_jump_url
from typing import Optional

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

# ✅ Retrieve top k matching messages from FAISS with optional filters
def get_top_k_matches(
    query: str,
    k: int = 5,
    guild_id: Optional[int] = None,
    channel_id: Optional[int] = None
):
    store = load_vectorstore()
    # Build filter kwargs if provided
    filter_kwargs = {}
    if guild_id is not None:
        filter_kwargs["guild_id"] = guild_id
    if channel_id is not None:
        filter_kwargs["channel_id"] = channel_id

    # Prepare search kwargs
    search_kwargs = {"k": k}
    if filter_kwargs:
        search_kwargs["filter"] = filter_kwargs

    # Perform retrieval
    retriever = store.as_retriever(search_kwargs=search_kwargs)
    docs = retriever.invoke(query)
    return [doc.metadata for doc in docs]

# Helper to safely build jump URLs
def safe_jump_url(metadata: dict) -> str:
    """
    Return a valid Discord jump URL or empty string.
    """
    url = metadata.get("jump_url")
    gid = metadata.get("guild_id")
    cid = metadata.get("channel_id")
    mid = metadata.get("message_id") or metadata.get("id")

    if not url and all(v is not None for v in (gid, cid, mid)):
        try:
            url = build_jump_url(int(gid), int(cid), int(mid))
        except ValueError:
            url = ""
    return url or ""

# ✅ Build prompt for OpenAI (used for RAG re-answering)
def build_prompt(matches, question: str, as_json: bool) -> list:
    context = []
    for m in matches:
        author = m.get("author")
        ts = m.get("timestamp")
        channel = m.get("channel")
        content = m.get("content")
        url = safe_jump_url(m)
        line = f"**{author}** (_{ts}_ in **#{channel}**):\n{content}"
        if url:
            line += f"\n[🔗 View Message]({url})"
        context.append(line)
    context_str = "\n\n".join(context)

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

    prompt = f"{instructions}\n\nContext:\n{context_str}\n\nUser's question: {question}\n"
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
            print(f"[{i}] message_id: {m.get('message_id')} | channel_id: {m.get('channel_id')} | guild_id: {m.get('guild_id')} | jump_url: {safe_jump_url(m)}")

        messages = build_prompt(matches, query, as_json)
        response = openai_client.chat.completions.create(
            model=gpt_model,
            messages=messages,
            temperature=0.7,
            max_tokens=1000,
            **({"response_format": {"type": "json_object"}} if as_json else {})
        )

        answer = response.choices[0].message.content
        if not matches:
            fallback = "⚠️ I couldn’t find relevant messages. Try rephrasing your question or being more specific."
            return (fallback, []) if return_matches else fallback

        return (answer, matches) if return_matches else answer
    except Exception as e:
        error_msg = f"❌ Error during RAG retrieval: {e}"
        return (error_msg, []) if return_matches else error_msg