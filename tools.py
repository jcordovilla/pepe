import os
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from pydantic import BaseModel, Field, ValidationError
from rapidfuzz import process
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from openai import OpenAI

from db import SessionLocal, Message
from utils.helpers import build_jump_url, validate_ids
from time_parser import parse_timeframe
from utils.logger import setup_logging

# Initialize logging
setup_logging()

# Embedding model and FAISS index loader
EMBED_MODEL = OpenAIEmbeddings(model="text-embedding-3-small")
INDEX_DIR = "index_faiss"

def load_vectorstore() -> FAISS:
    """
    Load the locally saved FAISS index.
    """
    return FAISS.load_local(INDEX_DIR, EMBED_MODEL, allow_dangerous_deserialization=True)

class SummarizeParams(BaseModel):
    start_iso: str = Field(..., description="ISO start datetime")
    end_iso: str = Field(..., description="ISO end datetime")
    guild_id: Optional[int] = Field(None, description="Discord guild ID")
    channel_id: Optional[int] = Field(None, description="Discord channel ID")
    as_json: bool = Field(False, description="Return JSON object if True")

class SearchParams(BaseModel):
    query: str = Field(..., description="Search query string")
    keyword: Optional[str] = Field(None, description="Exact keyword pre-filter")
    guild_id: Optional[int] = Field(None, description="Discord guild ID filter")
    channel_id: Optional[int] = Field(None, description="Discord channel ID filter")
    author_name: Optional[str] = Field(None, description="Fuzzy author name filter")
    k: int = Field(5, description="Number of top results to return")

# Tool implementations

def summarize_messages(
    start_iso: str,
    end_iso: str,
    guild_id: Optional[int] = None,
    channel_id: Optional[int] = None,
    as_json: bool = False
) -> Any:
    """
    Summarize Discord messages between two ISO datetimes, optionally scoped by guild and/or channel.
    
    Args:
      start_iso: ISO 8601 start timestamp (inclusive).
      end_iso:   ISO 8601 end   timestamp (inclusive).
      guild_id:  Discord guild ID to filter (optional).
      channel_id: Discord channel ID to filter (optional).
      as_json:   If True, returns a JSON object; otherwise a plain-text summary.
    
    Returns:
      A string summary, or if as_json=True, the parsed JSON object from the LLM.
    """
    # Validate IDs
    if guild_id is not None:
        validate_ids(guild_id=guild_id)
    if channel_id is not None:
        validate_ids(channel_id=channel_id)

    # Load messages from DB
    session = SessionLocal()
    query = session.query(Message).filter(
        Message.timestamp >= start_iso,
        Message.timestamp <= end_iso
    )
    if guild_id is not None:
        query = query.filter(Message.guild_id == guild_id)
    if channel_id is not None:
        query = query.filter(Message.channel_id == channel_id)
    msgs = query.all()
    session.close()

    # Build context snippets
    context_lines = []
    for m in msgs:
        author = m.author.get("username") or str(m.author.get("id"))
        ts = m.timestamp
        text = m.content.replace("\n", " ")
        url = m.jump_url or build_jump_url(m.guild_id, m.channel_id, m.message_id)
        context_lines.append(f"**{author}** ({ts}): {text} [🔗]({url})")
    context = "\n\n".join(context_lines)

    # Assemble RAG prompt
    prompt = (
        f"You are an assistant summarizing Discord messages.\n"
        f"Summarize the following messages between {start_iso} and {end_iso}.\n\n"
        f"{context}"
    )

    # Call the LLM
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model=os.getenv("GPT_MODEL"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=500,
        **({"response_format": {"type": "json_object"}} if as_json else {})
    )

    answer = response.choices[0].message.content
    if as_json:
        # parse string into JSON object
        return json.loads(answer)
    return answer

def search_messages(
    query: str,
    keyword: Optional[str] = None,
    guild_id: Optional[int] = None,
    channel_id: Optional[int] = None,
    author_name: Optional[str] = None,
    k: int = 5
) -> List[Dict[str, Any]]:
    """
    Hybrid keyword + semantic search over Discord messages.

    Args:
      query:       The natural-language query to embed/rerank.
      keyword:     Exact keyword to pre-filter messages (case-insensitive).
      guild_id:    Discord guild ID to restrict search (optional).
      channel_id:  Discord channel ID to restrict search (optional).
      author_name: Fuzzy‐match display name or username to filter by author (optional).
      k:           Number of top results to return.

    Returns:
      A list of up to k message dicts, each containing:
        - guild_id (int)
        - channel_id (int)
        - message_id (int)
        - author (dict)
        - content (str)
        - timestamp (str)
        - jump_url (str)
    """
    # 1) Validate IDs
    if guild_id is not None:
        validate_ids(guild_id=guild_id)
    if channel_id is not None:
        validate_ids(channel_id=channel_id)

    # 2) Pre‐filter with SQLite
    db = SessionLocal()
    q = db.query(Message)
    if keyword:
        q = q.filter(Message.content.ilike(f"%{keyword}%"))
    if guild_id:
        q = q.filter(Message.guild_id == guild_id)
    if channel_id:
        q = q.filter(Message.channel_id == channel_id)
    candidates = q.all()
    db.close()

    if not candidates:
        return []

    # 3) Build temp FAISS index for these candidates
    texts = [m.content for m in candidates]
    metadatas: List[Dict[str,Any]] = []
    for m in candidates:
        metadatas.append({
            "guild_id":    m.guild_id,
            "channel_id":  m.channel_id,
            "message_id":  m.message_id,
            "author":      m.author,
            "content":     m.content,
            "timestamp":   str(m.timestamp),
            "jump_url":    m.jump_url or build_jump_url(m.guild_id, m.channel_id, m.message_id)
        })

    temp_store = FAISS.from_texts(texts=texts, embedding=EMBED_MODEL, metadatas=metadatas)
    retriever = temp_store.as_retriever(search_kwargs={"k": k * 2})
    docs = retriever.get_relevant_documents(query)
    results = [d.metadata for d in docs]

    # 4) Fuzzy‐filter by author if requested
    if author_name:
        # build list of candidate author names
        names = [md["author"].get("username", "") for md in results]
        match, score, idx = process.extractOne(author_name, names)
        results = [r for r in results if r["author"].get("username") == match]

    # 5) Return the top‐k
    return results[:k]


def get_most_reacted_messages(
    guild_id: Optional[int] = None,
    channel_id: Optional[int] = None,
    top_n: int = 5
) -> List[Dict[str, Any]]:
    """
    Return the top N messages by total reaction count, optionally scoped to a guild and/or channel.

    Args:
      guild_id:   Discord guild ID to filter (optional).
      channel_id: Discord channel ID to filter (optional).
      top_n:      Number of messages to return (default 5).

    Returns:
      A list of up to top_n dicts, each containing:
        - guild_id (int)
        - channel_id (int)
        - message_id (int)
        - author (dict)
        - content (str)
        - timestamp (str)
        - jump_url (str)
        - total_reactions (int)
    """
    # 1) Validate IDs
    if guild_id is not None:
        validate_ids(guild_id=guild_id)
    if channel_id is not None:
        validate_ids(channel_id=channel_id)

    # 2) Fetch from DB
    session = SessionLocal()
    q = session.query(Message)
    if guild_id is not None:
        q = q.filter(Message.guild_id == guild_id)
    if channel_id is not None:
        q = q.filter(Message.channel_id == channel_id)
    msgs = q.all()
    session.close()

    # 3) Compute reaction totals and sort
    scored = []
    for m in msgs:
        total = sum(r.get("count", 0) for r in m.reactions)
        scored.append((total, m))
    scored.sort(key=lambda x: x[0], reverse=True)

    # 4) Build result list
    results: List[Dict[str, Any]] = []
    for total, m in scored[:top_n]:
        results.append({
            "guild_id":       m.guild_id,
            "channel_id":     m.channel_id,
            "message_id":     m.message_id,
            "author":         m.author,
            "content":        m.content,
            "timestamp":      str(m.timestamp),
            "jump_url":       m.jump_url or build_jump_url(m.guild_id, m.channel_id, m.message_id),
            "total_reactions": total
        })

    return results

def find_users_by_skill(skill: str) -> List[Dict[str, Any]]:
    """
    Identify authors whose messages contain a given skill keyword.
    """
    session = SessionLocal()
    q = session.query(Message).filter(Message.content.ilike(f"%{skill}%"))
    msgs = q.all()
    session.close()

    # Deduplicate by author ID
    by_author: Dict[int, Dict[str, Any]] = {}
    for m in msgs:
        aid = m.author.get('id')
        if aid not in by_author:
            by_author[aid] = {
                "author_id": aid,
                "username": m.author.get('username'),
                "example_message": m.content,
                "jump_url": m.jump_url or build_jump_url(m.guild_id, m.channel_id, m.message_id)
            }
    return list(by_author.values())


def analyze_message_types(
    guild_id: Optional[int] = None,
    channel_id: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Count message types (text, image, embed, etc.) in a guild/channel.
    """
    session = SessionLocal()
    q = session.query(Message)
    if guild_id:
        q = q.filter(Message.guild_id == guild_id)
    if channel_id:
        q = q.filter(Message.channel_id == channel_id)
    msgs = q.all()
    session.close()

    counts: Dict[str, int] = {}
    for m in msgs:
        mtype = m.__dict__.get('type', 'unknown')  # assume type field if present
        counts[mtype] = counts.get(mtype, 0) + 1
    return [{"type": t, "count": c} for t, c in counts.items()]


def get_pinned_messages(
    guild_id: Optional[int] = None,
    channel_id: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Retrieve pinned messages in a guild/channel.
    """
    session = SessionLocal()
    q = session.query(Message).filter(Message.is_pinned == True)
    if guild_id:
        q = q.filter(Message.guild_id == guild_id)
    if channel_id:
        q = q.filter(Message.channel_id == channel_id)
    msgs = q.all()
    session.close()

    return [
        {
            "guild_id": m.guild_id,
            "channel_id": m.channel_id,
            "message_id": m.message_id,
            "content": m.content,
            "jump_url": m.jump_url or build_jump_url(m.guild_id, m.channel_id, m.message_id)
        }
        for m in msgs
    ]


def extract_feedback_and_ideas(
    keywords: List[str],
    guild_id: Optional[int] = None,
    channel_id: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Extract messages containing any of the given keywords.
    """
    session = SessionLocal()
    q = session.query(Message)
    if guild_id:
        q = q.filter(Message.guild_id == guild_id)
    if channel_id:
        q = q.filter(Message.channel_id == channel_id)

    msgs = q.all()
    session.close()

    results = []
    for m in msgs:
        content_lower = m.content.lower()
        hits = [kw for kw in keywords if kw.lower() in content_lower]
        if hits:
            results.append({
                "message_id": m.message_id,
                "hits": hits,
                "content": m.content,
                "jump_url": m.jump_url or build_jump_url(m.guild_id, m.channel_id, m.message_id)
            })
    return results
