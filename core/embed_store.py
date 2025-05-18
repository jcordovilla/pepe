# embed_store.py

import os
from db import SessionLocal, Message
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from typing import List, Dict, Any

INDEX_DIR = "index_faiss"

def build_langchain_faiss_index():
    """
    Incrementally embed new messages from the SQLite DB and update the FAISS index.
    """
    print("📚 Loading messages from SQLite DB...")
    session = SessionLocal()
    rows = session.query(Message).all()
    session.close()

    # Prepare all messages and metadata
    texts = []
    metadatas = []
    ids = []
    for m in rows:
        texts.append(m.content)
        metadatas.append({
            "guild_id":     str(m.guild_id),
            "channel_id":   str(m.channel_id),
            "channel_name": m.channel_name,
            "message_id":   str(m.message_id),
            "content":      m.content,
            "timestamp":    m.timestamp.isoformat(),
            "author":       m.author,
            "mention_ids":  m.mention_ids,
            "reactions":    m.reactions,
            "jump_url":     m.jump_url
        })
        ids.append(str(m.message_id))

    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    os.makedirs(INDEX_DIR, exist_ok=True)
    index_path = os.path.join(INDEX_DIR, "index.faiss")
    pkl_path = os.path.join(INDEX_DIR, "index.pkl")

    # Try to load existing index
    if os.path.exists(index_path) and os.path.exists(pkl_path):
        print(f"🔄 Loading existing FAISS index from {INDEX_DIR}/")
        vectorstore = FAISS.load_local(INDEX_DIR, embedding_model, allow_dangerous_deserialization=True)
        # Get all message_ids already in the index
        existing_ids = set()
        for meta in vectorstore.docstore._dict.values():
            mid = meta.metadata.get("message_id")
            if mid:
                existing_ids.add(str(mid))
        # Find new messages
        new_texts = []
        new_metadatas = []
        for text, meta, mid in zip(texts, metadatas, ids):
            if mid not in existing_ids:
                new_texts.append(text)
                new_metadatas.append(meta)
        if new_texts:
            print(f"🧠 Embedding {len(new_texts)} new messages with OpenAI...")
            vectorstore.add_texts(new_texts, metadatas=new_metadatas)
            print(f"💾 Saving updated index to {INDEX_DIR}/")
            vectorstore.save_local(INDEX_DIR)
            print("✅ Index updated.")
        else:
            print("✅ No new messages to embed. Index is up to date.")
    else:
        print(f"🧠 Embedding {len(texts)} messages with OpenAI (full rebuild)...")
        vectorstore = FAISS.from_texts(
            texts=texts,
            embedding=embedding_model,
            metadatas=metadatas
        )
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
