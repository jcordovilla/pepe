# embed_store.py

import os
from db import SessionLocal, Message
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from typing import List, Dict, Any

INDEX_DIR = "index_faiss"

def build_langchain_faiss_index():
    """
    Read all messages from the SQLite DB, embed them, and save a FAISS index.
    """
    print("📚 Loading messages from SQLite DB...")
    session = SessionLocal()
    rows = session.query(Message).all()
    session.close()

    texts = []
    metadatas = []
    for m in rows:
        texts.append(m.content)
        metadatas.append({
            "channel_id":   str(m.channel_id),
            "channel_name": m.channel_name,           # plain name from Discord API
            "message_id":   str(m.message_id),
            "content":      m.content,
            "timestamp":    m.timestamp.isoformat(),
            "author":       m.author,
            "mention_ids":  m.mention_ids,
            "reactions":    m.reactions,
            "jump_url":     m.jump_url
        })

    print(f"🧠 Embedding {len(texts)} messages with OpenAI...")
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore     = FAISS.from_texts(
        texts=texts,
        embedding=embedding_model,
        metadatas=metadatas
    )

    os.makedirs(INDEX_DIR, exist_ok=True)
    print(f"💾 Saving index to {INDEX_DIR}/")
    vectorstore.save_local(INDEX_DIR)
    print("✅ Index saved.")

def flatten_messages(db_path: str) -> List[tuple]:
    """
    Load all messages from the SQLite database and return them as a list of (text, metadata) tuples.
    """
    print(f"📂 Loading messages from {db_path}...")
    session = SessionLocal()
    rows = session.query(Message).all()
    session.close()

    messages = []
    for m in rows:
        text = m.content
        meta = {
            "guild_id": m.guild_id,
            "channel_id": m.channel_id,
            "channel_name": m.channel_name,
            "message_id": m.message_id,
            "timestamp": m.timestamp.isoformat(),
            "author": m.author,
            "mention_ids": m.mention_ids,
            "reactions": m.reactions,
            "jump_url": m.jump_url,
        }
        messages.append((text, meta))  # Return as a tuple (text, metadata)
    return messages

if __name__ == "__main__":
    build_langchain_faiss_index()
