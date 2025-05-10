# rag_engine.py

import os
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
from openai import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from utils.helpers import build_jump_url
from tools import resolve_channel_name  # maps a channel_name → channel_id
from time_parser import parse_timeframe

# ——— Load config & initialize clients ———
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GPT_MODEL       = os.getenv("GPT_MODEL", "gpt-4-turbo-2024-04-09")
INDEX_DIR       = "index_faiss"

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
openai_client   = OpenAI(api_key=OPENAI_API_KEY)


def load_vectorstore() -> FAISS:
    """
    Load the locally saved FAISS index from disk.
    """
    return FAISS.load_local(INDEX_DIR, embedding_model, allow_dangerous_deserialization=True)


def get_top_k_matches(
    query: str,
    k: int = 5,
    guild_id: Optional[int]      = None,
    channel_id: Optional[int]    = None,
    channel_name: Optional[str]  = None
) -> List[Dict[str, Any]]:
    """
    Retrieve the top-k FAISS matches for 'query',
    optionally scoped to a guild, channel_id, or channel_name
    via manual post-filtering.
    """
    store = load_vectorstore()
    # 1) Fetch extra candidates for reliable filtering
    fetch_k = k * 10
    retriever = store.as_retriever(search_kwargs={"k": fetch_k})
    docs = retriever.get_relevant_documents(query)
    metas = [doc.metadata for doc in docs]

    # 2) Resolve a human channel_name to ID if needed
    if channel_name and not channel_id:
        resolved = resolve_channel_name(channel_name, guild_id)
        if resolved is None:
            raise ValueError(f"Unknown channel name: {channel_name}")
        channel_id = resolved

    # 3) Manual metadata filters
    if guild_id is not None:
        metas = [m for m in metas if m.get("guild_id") == str(guild_id)]
    if channel_id is not None:
        metas = [m for m in metas if m.get("channel_id") == str(channel_id)]

    # 4) Return the top-k of whatever remains
    return metas[:k]


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
    Construct a ChatML prompt with context from matching messages.
    """
    context_lines: List[str] = []
    for m in matches:
        author      = m.get("author", {})
        author_name = author.get("display_name") or author.get("username") or "Unknown"
        ts          = m.get("timestamp", "")
        ch_name     = m.get("channel_name") or m.get("channel_id")
        content     = m.get("content", "").replace("\n", " ")
        url         = safe_jump_url(m)
        line = f"**{author_name}** (_{ts}_ in **#{ch_name}**):\n{content}"
        if url:
            line += f"\n[🔗 View Message]({url})"
        context_lines.append(line)

    instructions = (
        "You are a knowledgeable assistant specialized in Discord server data.\n\n"
        "Use RAG to answer the user's question based on the provided context. "
        "Include author, timestamp, channel, snippets, and URLs.\n"
    )
    if as_json:
        instructions += "Return results as a JSON array with fields: guild_id, channel_id, message_id, content, timestamp, author, jump_url.\n"

    return [
        {"role": "system", "content": "You are a Discord message analyst."},
        {"role": "user",   "content": instructions + "\nContext:\n" + "\n\n".join(context_lines)
                                  + f"\n\nUser's question: {question}\n"}
    ]


def get_answer(
    query: str,
    k: int                = 5,
    as_json: bool          = False,
    return_matches: bool   = False,
    guild_id: Optional[int]     = None,
    channel_id: Optional[int]   = None,
    channel_name: Optional[str] = None
) -> Any:
    """
    Perform a RAG-based query: retrieve matches, build a prompt, and ask OpenAI.
    Returns either a string (answer) or (answer, matches) if return_matches=True.
    """
    try:
        matches = get_top_k_matches(
            query, k,
            guild_id=guild_id,
            channel_id=channel_id,
            channel_name=channel_name
        )
        if not matches and not return_matches:
            return "⚠️ I couldn't find relevant messages. Try rephrasing your question or being more specific."

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
        err = f"❌ Error during RAG retrieval: {e}"
        return (err, []) if return_matches else err


def get_agent_answer(query: str) -> str:
    """
    High-level function that handles time parsing and query execution.
    This is the main entry point for RAG queries from the app.
    
    Args:
        query: The user's query, which may contain time references
        
    Returns:
        str: The answer to the query
    """
    try:
        # Parse any time references in the query
        timeframe = parse_timeframe(query)
        if timeframe:
            start_iso, end_iso = timeframe
            # If we have a time range, use summarize_messages
            from tools import summarize_messages
            return summarize_messages(
                start_iso=start_iso,
                end_iso=end_iso,
                as_json=False
            )
        
        # Otherwise, use regular RAG search
        return get_answer(query)
        
    except Exception as e:
        return f"❌ Error processing query: {e}"


# ——— Convenience aliases for the app ———
search_messages    = get_top_k_matches
discord_rag_search = get_answer
