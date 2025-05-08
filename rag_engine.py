import os
import json
from dotenv import load_dotenv
from typing import Optional, List, Dict, Any
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from openai import OpenAI
from utils import build_jump_url
from rapidfuzz import process, fuzz
from typing import Optional
from utils.logger import setup_logging
setup_logging()

import logging
log = logging.getLogger(__name__)

# Load environment variables and initialize clients
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GPT_MODEL = os.getenv("GPT_MODEL", "gpt-4-turbo-2024-04-09")
INDEX_DIR = "index_faiss"

# Embedding model and OpenAI client
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Fuzzy string matching
def find_author_id(name_query: str) -> Optional[int]:
    """
    Fuzzy‐match a user’s input against known author usernames
    and return the best matching author_id (or None if no good match).
    """
    # Load all message metadata
    from embed_store import flatten_messages
    all_msgs = flatten_messages("discord_messages_v2.json")

    # Build a map: username → author_id
    author_map: Dict[str, int] = {}
    for _, meta in all_msgs:
        author = meta.get("author", {})
        uname = author.get("username")
        aid   = author.get("id")
        if uname and aid and uname not in author_map:
            author_map[uname] = aid

    if not author_map:
        return None

    # Fuzzy‐match the query against those usernames
    best, score, _ = process.extractOne(
        name_query,
        list(author_map.keys()),
        scorer=fuzz.WRatio
    )
    # You can tweak this threshold; 60 is a reasonable start
    if score < 60:
        return None

    return author_map[best]

def search_messages(
    query: str,
    keyword: Optional[str]     = None,
    guild_id: Optional[int]     = None,
    channel_id: Optional[int]   = None,
    author_name: Optional[str]  = None,
    k: int                      = 5
) -> List[Dict[str, Any]]:
    """
    Hybrid search:
      1) Pre-filter messages containing `keyword` (exact, case-insensitive)
      2) Optionally scope by guild_id & channel_id
      3) Rerank the survivors semantically via FAISS
      4) Return top-k metadata dicts
    """
    # 1) Load all raw texts+metadata
    from embed_store import flatten_messages
    all_msgs = flatten_messages("discord_messages_v2.json")

    # 2) Keyword & metadata filter
    candidates = []
    for text, meta in all_msgs:
        # 1) keyword filter (if any)
        if keyword and keyword.lower() not in text.lower():
            continue
        # 2) guild/channel scope
        if guild_id and meta["guild_id"] != str(guild_id):
            continue
        if channel_id and meta["channel_id"] != str(channel_id):
            continue
        # 3) fuzzy author filter
        if author_name:
            matched_id = find_author_id(author_name)
            if matched_id is None or meta["author"]["id"] != matched_id:
                continue

        candidates.append((text, meta))

    if not candidates:
        return []

    # 3) Split texts & metadatas
    texts, metadatas = zip(*candidates)

    # 4) Build a temporary FAISS index on these candidates
    from langchain_community.vectorstores import FAISS as _FAISS
    temp_store = _FAISS.from_texts(
        texts=list(texts),
        embedding=embedding_model,
        metadatas=list(metadatas)
    )

    # 5) Semantic rerank
    retriever = temp_store.as_retriever(search_kwargs={"k": k})
    docs = retriever.get_relevant_documents(query)

    # 6) Return metadata only
    return [doc.metadata for doc in docs]


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
    optionally filtered by guild and/or channel metadata.
    """
    store = load_vectorstore()

    # Prepare metadata filters as strings
    filters = {}
    if guild_id is not None:
        filters["guild_id"] = str(guild_id)
    if channel_id is not None:
        filters["channel_id"] = str(channel_id)

    # Apply filters directly in retriever
    retriever = store.as_retriever(search_kwargs={
        "k": k * 10,  # fetch more to allow reranking
        "filter": filters
    })

    # Retrieve results from FAISS
    docs = retriever.get_relevant_documents(query)

    # Debug print: show what metadata was retrieved
    log.debug("🔍 Retrieved metadata from FAISS:")
    for doc in docs:
        log.debug(f"📎 guild_id={doc.metadata.get('guild_id')} | channel_id={doc.metadata.get('channel_id')}")

    # Return top-k filtered metadata entries
    return [doc.metadata for doc in docs][:k]


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
    
from typing import List, Dict, Any, Optional


if __name__ == "__main__":
    # Quick local test (adjust guild_id/channel_id as needed)
    log.info(search_messages("ethics", guild_id=1353058864810950737, channel_id=1364361051830747156))

