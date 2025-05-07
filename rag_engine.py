import os
import json
from dotenv import load_dotenv
from typing import Optional, List, Dict, Any
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from openai import OpenAI
from utils import build_jump_url

# Load environment variables and initialize clients
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GPT_MODEL = os.getenv("GPT_MODEL", "gpt-4-turbo-2024-04-09")
INDEX_DIR = "index_faiss"

# Embedding model and OpenAI client
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
openai_client = OpenAI(api_key=OPENAI_API_KEY)


def load_vectorstore() -> FAISS:
    """
    Load the locally saved FAISS index from disk.
    """
    return FAISS.load_local(INDEX_DIR, embedding_model, allow_dangerous_deserialization=True)


def get_top_k_matches(
    query: str,
    k: int = 5,
    guild_id: Optional[int] = None,
    channel_id: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Retrieve the top-k FAISS matches for 'query',
    optionally scoped to a guild and/or channel.
    """
    store = load_vectorstore()
    # Build metadata filters as strings
    filter_kwargs: Dict[str, str] = {}
    if guild_id is not None:
        filter_kwargs["guild_id"] = str(guild_id)
    if channel_id is not None:
        filter_kwargs["channel_id"] = str(channel_id)

    # Create retriever with or without filters
    if filter_kwargs:
        retriever = store.as_retriever(
            search_kwargs={"k": k},
            filter=filter_kwargs
        )
    else:
        retriever = store.as_retriever(search_kwargs={"k": k})

    # Use get_relevant_documents to apply filters correctly
    docs = retriever.get_relevant_documents(query)
    return [doc.metadata for doc in docs]


def safe_jump_url(metadata: Dict[str, Any]) -> str:
    """
    Ensure the metadata contains a valid jump_url, constructing one if missing.
    """
    url = metadata.get("jump_url")
    if url:
        return url
    try:
        gid = int(metadata["guild_id"])
        cid = int(metadata["channel_id"])
        mid = int(metadata.get("message_id") or metadata.get("id"))
        return build_jump_url(gid, cid, mid)
    except Exception:
        return ""


def build_prompt(
    matches: List[Dict[str, Any]],
    question: str,
    as_json: bool
) -> List[Dict[str, str]]:
    """
    Build a ChatML prompt given matched messages and the user question.
    """
    context_lines: List[str] = []
    for m in matches:
        author = m.get("author", {})
        author_name = author.get("display_name") or author.get("username") or str(author)
        ts = m.get("timestamp")
        channel_name = m.get("channel_name") or m.get("channel_id")
        content = m.get("content", "").replace("\n", " ")
        url = safe_jump_url(m)
        line = f"**{author_name}** (_{ts}_ in **#{channel_name}**):\n{content}"
        if url:
            line += f"\n[🔗 View Message]({url})"
        context_lines.append(line)
    context = "\n\n".join(context_lines)

    instructions = (
        "You are a knowledgeable and versatile assistant specialized in analyzing Discord server data.\n\n"
        "Based on the user’s query, you can:\n"
        "- Search and summarize Discord messages using retrieval-augmented generation (RAG).\n"
        "- Call specific analysis tools for structured tasks.\n\n"
        "When presenting answers:\n"
        "- Use clear natural language by default, or JSON if requested.\n"
        "- Include author, timestamp, channel, and message snippets.\n"
        "- Provide jump URLs when available."
    )
    if as_json:
        instructions += "\nReturn the results as a JSON array with those fields."

    prompt = f"{instructions}\n\nContext:\n{context}\n\nUser's question: {question}\n"
    return [
        {"role": "system", "content": "You are a Discord message analyst."},
        {"role": "user",   "content": prompt}
    ]


def get_answer(
    query: str,
    k: int = 5,
    as_json: bool = False,
    return_matches: bool = False
) -> Any:
    """
    Run a RAG-based answer: retrieve matches, build a prompt, and ask OpenAI.
    Returns either a string or (answer, matches) if return_matches=True.
    """
    try:
        matches = get_top_k_matches(query, k) if not return_matches else get_top_k_matches(query, k)
        if not matches and not return_matches:
            return "⚠️ I couldn’t find relevant messages. Try rephrasing your question or being more specific."

        chat_messages = build_prompt(matches, query, as_json)
        response = openai_client.chat.completions.create(
            model=GPT_MODEL,
            messages=chat_messages,
            temperature=0.7,
            max_tokens=1000,
            **({"response_format": {"type": "json_object"}} if as_json else {})
        )
        answer = response.choices[0].message.content
        return (answer, matches) if return_matches else answer
    except Exception as e:
        error_msg = f"❌ Error during RAG retrieval: {e}"
        return (error_msg, []) if return_matches else error_msg