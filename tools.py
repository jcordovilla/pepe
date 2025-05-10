# tools.py

import os
import json
from typing import List, Dict, Any, Optional, Literal
from datetime import datetime

from pydantic import BaseModel, Field
from rapidfuzz import process
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from openai import OpenAI

from db import SessionLocal, Message
from utils.helpers import build_jump_url, validate_ids
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


def summarize_messages(
    start_iso: str,
    end_iso: str,
    guild_id: Optional[int] = None,
    channel_id: Optional[int] = None,
    as_json: bool = False
) -> Any:
    """
    Summarize messages between two ISO datetimes via RAG.
    """
    if guild_id is not None:
        validate_ids(guild_id=guild_id)
    if channel_id is not None:
        validate_ids(channel_id=channel_id)

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

    context = []
    for m in msgs:
        author = m.author.get("username") or str(m.author.get("id"))
        ts = m.timestamp
        text = m.content.replace("\n", " ")
        url = m.jump_url or build_jump_url(m.guild_id, m.channel_id, m.message_id)
        context.append(f"**{author}** ({ts}): {text} [🔗]({url})")
    payload = "\n\n".join(context)

    prompt = (
        f"You are an assistant summarizing Discord messages."
        f"\nSummarize the following messages between {start_iso} and {end_iso}."
        f"\n\n{payload}"
    )

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model=os.getenv("GPT_MODEL"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=500,
        **({"response_format": {"type": "json_object"}} if as_json else {})
    )
    answer = response.choices[0].message.content
    return json.loads(answer) if as_json else answer


def search_messages(
    query: str,
    keyword: Optional[str] = None,
    guild_id: Optional[int] = None,
    channel_id: Optional[int] = None,
    channel_name: Optional[str] = None,
    author_name: Optional[str] = None,
    k: int = 5
) -> List[Dict[str, Any]]:
    if channel_name and not channel_id:
        channel_id = resolve_channel_name(channel_name, guild_id)
        if not channel_id:
            raise ValueError(f"Unknown channel: {channel_name}")
    if guild_id is not None:
        validate_ids(guild_id=guild_id)
    if channel_id is not None:
        validate_ids(channel_id=channel_id)

    # Pre-filter in DB
    session = SessionLocal()
    q = session.query(Message)
    if keyword:
        q = q.filter(Message.content.ilike(f"%{keyword}%"))
    if guild_id:
        q = q.filter(Message.guild_id == guild_id)
    if channel_id:
        q = q.filter(Message.channel_id == channel_id)
    candidates = q.all()
    session.close()

    if not candidates:
        return []

    texts = [m.content for m in candidates]
    metadatas = []
    for m in candidates:
        metadatas.append({
            'guild_id': m.guild_id,
            'channel_id': m.channel_id,
            'message_id': m.message_id,
            'author': m.author,
            'content': m.content,
            'timestamp': str(m.timestamp),
            'jump_url': m.jump_url or build_jump_url(m.guild_id, m.channel_id, m.message_id)
        })

    temp_store = FAISS.from_texts(texts=texts, embedding=EMBED_MODEL, metadatas=metadatas)
    docs = temp_store.as_retriever(search_kwargs={'k': k*2}).get_relevant_documents(query)
    results = [d.metadata for d in docs]

    if author_name:
        names = [md['author'].get('username', '') for md in results]
        match, score, idx = process.extractOne(author_name, names)
        results = [r for r in results if r['author'].get('username') == match]

    return results[:k]


def get_most_reacted_messages(
    guild_id: Optional[int] = None,
    channel_id: Optional[int] = None,
    top_n: int = 5
) -> List[Dict[str, Any]]:
    if guild_id is not None:
        validate_ids(guild_id=guild_id)
    if channel_id is not None:
        validate_ids(channel_id=channel_id)
    session = SessionLocal()
    q = session.query(Message)
    if guild_id:
        q = q.filter(Message.guild_id == guild_id)
    if channel_id:
        q = q.filter(Message.channel_id == channel_id)
    msgs = q.all()
    session.close()

    scored = sorted(((sum(r.get('count',0) for r in m.reactions), m) for m in msgs),
                    key=lambda x: x[0], reverse=True)
    results = []
    for total, m in scored[:top_n]:
        results.append({
            'guild_id': m.guild_id,
            'channel_id': m.channel_id,
            'message_id': m.message_id,
            'author': m.author,
            'content': m.content,
            'timestamp': str(m.timestamp),
            'jump_url': m.jump_url or build_jump_url(m.guild_id, m.channel_id, m.message_id),
            'total_reactions': total
        })
    return results


def find_users_by_skill(
    skill: str,
    guild_id: Optional[int] = None,
    channel_id: Optional[int] = None
) -> List[Dict[str, Any]]:
    if guild_id is not None:
        validate_ids(guild_id=guild_id)
    if channel_id is not None:
        validate_ids(channel_id=channel_id)
    session = SessionLocal()
    q = session.query(Message).filter(Message.content.ilike(f"%{skill}%"))
    if guild_id:
        q = q.filter(Message.guild_id == guild_id)
    if channel_id:
        q = q.filter(Message.channel_id == channel_id)
    msgs = q.all()
    session.close()

    authors = {}
    for m in msgs:
        aid = m.author.get('id')
        if aid not in authors:
            authors[aid] = {
                'author_id': aid,
                'username': m.author.get('username'),
                'example_message': m.content,
                'jump_url': m.jump_url or build_jump_url(m.guild_id, m.channel_id, m.message_id)
            }
    return list(authors.values())


def summarize_messages_in_range(
    start_iso: str,
    end_iso: str,
    guild_id: Optional[int] = None,
    channel_id: Optional[int] = None,
    channel_name: Optional[str] = None,
    output_format: Literal['text','json'] = 'text'
) -> Any:
    if channel_name and not channel_id:
        channel_id = resolve_channel_name(channel_name, guild_id)
        if not channel_id:
            raise ValueError(f"Unknown channel: {channel_name}")
    start_dt = datetime.fromisoformat(start_iso)
    end_dt = datetime.fromisoformat(end_iso)

    session = SessionLocal()
    q = session.query(Message).filter(
        Message.timestamp >= start_dt,
        Message.timestamp <= end_dt
    )
    if guild_id:
        q = q.filter(Message.guild_id == guild_id)
    if channel_id:
        q = q.filter(Message.channel_id == channel_id)
    msgs = q.all()
    session.close()

    if output_format == 'json':
        return json.dumps({} if not msgs else summarize_messages(start_iso, end_iso, guild_id, channel_id, as_json=True))

    if not msgs:
        return f"No messages found between {start_iso} and {end_iso}."

    lines = [f"Messages from {start_dt.date()} to {end_dt.date()}:]"]
    for m in msgs:
        author = m.author.get('username') or str(m.author.get('id'))
        ts = m.timestamp.isoformat()
        text = m.content.replace("\n", " ")
        url = m.jump_url or build_jump_url(m.guild_id, m.channel_id, m.message_id)
        lines.append(f"**{author}** ({ts}): {text} [🔗]({url})")
    return "\n".join(lines)


def resolve_channel_name(
    channel_name: str,
    guild_id: Optional[int] = None
) -> Optional[int]:
    session = SessionLocal()
    q = session.query(Message.channel_id, Message.channel_name).distinct()
    if guild_id:
        q = q.filter(Message.guild_id == guild_id)
    q = q.filter(Message.channel_name.ilike(channel_name.strip("#")))
    result = q.first()
    session.close()
    return result[0] if result else None
