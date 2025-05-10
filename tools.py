# tools.py

import os
import json
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from db import SessionLocal, Message
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from openai import OpenAI
from utils import (
    validate_channel_id,
    validate_guild_id,
    validate_channel_name
)

# setup
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

EMBED_MODEL = "text-embedding-3-small"
GPT_MODEL   = os.getenv("GPT_MODEL", "gpt-4-turbo")
INDEX_DIR   = "index_faiss"

# init clients
_embedding = OpenAIEmbeddings(model=EMBED_MODEL, openai_api_key=OPENAI_API_KEY)
_openai    = OpenAI(api_key=OPENAI_API_KEY)

def _load_store() -> FAISS:
    """Load the FAISS index from disk."""
    try:
        return FAISS.load_local(INDEX_DIR, _embedding, allow_dangerous_deserialization=True)
    except Exception as e:
        raise RuntimeError(f"Failed to load FAISS index from {INDEX_DIR}: {e}")

def _resolve_channel_name(channel_name: str, guild_id: Optional[int] = None) -> Optional[int]:
    """
    Given a human-friendly channel_name (e.g. "non-coders-learning")
    and an optional guild_id, return the corresponding channel_id
    from the database, or None if not found.
    """
    if not validate_channel_name(channel_name):
        raise ValueError(f"Invalid channel name format: {channel_name}")
        
    if guild_id is not None and not validate_guild_id(guild_id):
        raise ValueError(f"Invalid guild ID format: {guild_id}")
        
    session = SessionLocal()
    try:
        query = session.query(Message.channel_id).filter(Message.channel_name == channel_name)
        if guild_id is not None:
            query = query.filter(Message.guild_id == guild_id)
        result = query.first()
        return result[0] if result else None
    finally:
        session.close()

def search_messages(
    query: str,
    k: int = 5,
    keyword: Optional[str] = None,
    guild_id: Optional[int] = None,
    channel_id: Optional[int] = None,
    channel_name: Optional[str] = None,
    author_name: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Hybrid keyword + semantic search.
    - keyword: exact-match pre-filter
    - semantic rerank top-N via FAISS
    """
    if not query.strip():
        raise ValueError("Query cannot be empty")
        
    # Validate IDs
    if guild_id is not None and not validate_guild_id(guild_id):
        raise ValueError(f"Invalid guild ID format: {guild_id}")
        
    if channel_id is not None and not validate_channel_id(channel_id):
        raise ValueError(f"Invalid channel ID format: {channel_id}")
        
    if channel_name is not None and not validate_channel_name(channel_name):
        raise ValueError(f"Invalid channel name format: {channel_name}")

    # resolve channel_name → channel_id
    if channel_name and not channel_id:
        channel_id = _resolve_channel_name(channel_name, guild_id)
        if not channel_id:
            raise ValueError(f"Unknown channel: {channel_name}")

    # 1) keyword pre-filter via DB
    session = SessionLocal()
    try:
        db_q = session.query(Message)
        if guild_id:
            db_q = db_q.filter(Message.guild_id == guild_id)
        if channel_id:
            db_q = db_q.filter(Message.channel_id == channel_id)
        if author_name:
            db_q = db_q.filter(Message.author["username"].as_string() == author_name)
        if keyword:
            db_q = db_q.filter(Message.content.ilike(f"%{keyword}%"))
        candidates = db_q.limit(k * 5).all()

        if not candidates:
            return []

        # 2) rerank via FAISS
        texts = [m.content for m in candidates]
        metas = [m.__dict__ for m in candidates]
        temp_store = FAISS.from_texts(
            texts=texts,
            embedding=_embedding,
            metadatas=metas
        )
        retriever = temp_store.as_retriever(search_kwargs={"k": k})
        docs = retriever.get_relevant_documents(query)

        # extract metadata dicts
        return [d.metadata for d in docs]
    finally:
        session.close()

def summarize_messages(
    start_iso: str,
    end_iso: str,
    guild_id: Optional[int] = None,
    channel_id: Optional[int] = None,
    channel_name: Optional[str] = None,
    as_json: bool = False
) -> Union[str, Dict[str, Any]]:
    """
    Summarize all messages in [start_iso, end_iso].
    Returns either a text summary or JSON, per `as_json`.
    """
    # Validate IDs
    if guild_id is not None and not validate_guild_id(guild_id):
        raise ValueError(f"Invalid guild ID format: {guild_id}")
        
    if channel_id is not None and not validate_channel_id(channel_id):
        raise ValueError(f"Invalid channel ID format: {channel_id}")
        
    if channel_name is not None and not validate_channel_name(channel_name):
        raise ValueError(f"Invalid channel name format: {channel_name}")

    # validate time inputs
    try:
        start = datetime.fromisoformat(start_iso)
        end = datetime.fromisoformat(end_iso)
        if end < start:
            raise ValueError("End time must be after start time")
    except Exception as e:
        raise ValueError(f"Invalid ISO datetime: {e}")

    # resolve channel_name → channel_id
    if channel_name and not channel_id:
        channel_id = _resolve_channel_name(channel_name, guild_id)
        if not channel_id:
            raise ValueError(f"Unknown channel: {channel_name}")

    # load messages from DB
    session = SessionLocal()
    try:
        q = session.query(Message)
        q = q.filter(Message.timestamp.between(start, end))
        if guild_id:
            q = q.filter(Message.guild_id == guild_id)
        if channel_id:
            q = q.filter(Message.channel_id == channel_id)
        msgs = q.order_by(Message.timestamp).all()

        if not msgs:
            return {"summary": "", "note": "No messages in that timeframe"} if as_json else "⚠️ No messages found in the specified timeframe."

        # build prompt with clear time context
        context = "\n\n".join(
            f"**{m.author['username']}** (_{m.timestamp}_): {m.content}"
            for m in msgs
        )
        instructions = (
            f"Summarize the following Discord messages from {start.date()} to {end.date()}:\n\n"
            "Include author, timestamp, and key points. Return JSON if requested."
        )
        prompt = f"{instructions}\n\n{context}\n\nSummary:"

        # chat completion
        resp = _openai.chat.completions.create(
            model=GPT_MODEL,
            messages=[
                {"role": "system", "content": instructions},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=800,
            **({"response_format": {"type": "json_object"}} if as_json else {})
        )
        result = resp.choices[0].message.content
        return json.loads(result) if as_json else result

    finally:
        session.close()
