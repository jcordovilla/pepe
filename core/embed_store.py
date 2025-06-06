# embed_store.py

import os
import numpy as np
import pickle
import faiss
from db import SessionLocal, Message
from core.ai_client import get_ai_client
from core.config import get_config
from typing import List, Dict, Any

def build_faiss_index():
    """
    Incrementally embed new messages from the SQLite DB and update the FAISS index using local embeddings.
    """
    config = get_config()
    ai_client = get_ai_client()
    INDEX_DIR = config.faiss_index_path
    
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

    os.makedirs(INDEX_DIR, exist_ok=True)
    index_path = os.path.join(INDEX_DIR, "index.faiss")
    pkl_path = os.path.join(INDEX_DIR, "index.pkl")

    # Try to load existing index
    if os.path.exists(index_path) and os.path.exists(pkl_path):
        print(f"🔄 Loading existing FAISS index from {INDEX_DIR}/")
        # Load existing index
        index = faiss.read_index(index_path)
        with open(pkl_path, "rb") as f:
            existing_metadatas = pickle.load(f)
        
        # Get all message_ids already in the index
        existing_ids = set()
        for meta in existing_metadatas:
            mid = meta.get("message_id")
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
            print(f"🧠 Embedding {len(new_texts)} new messages with local model...")
            new_embeddings = ai_client.create_embeddings(new_texts)
            
            # Add to existing index
            index.add(new_embeddings.astype('float32'))
            existing_metadatas.extend(new_metadatas)
            
            print(f"💾 Saving updated index to {INDEX_DIR}/")
            faiss.write_index(index, index_path)
            with open(pkl_path, "wb") as f:
                pickle.dump(existing_metadatas, f)
            print("✅ Index updated.")
        else:
            print("✅ No new messages to embed. Index is up to date.")
    else:
        print(f"🧠 Embedding {len(texts)} messages with local model (full rebuild)...")
        
        # Create embeddings for all texts
        embeddings = ai_client.create_embeddings(texts)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings.astype('float32'))
        index.add(embeddings.astype('float32'))
        
        print(f"💾 Saving index to {INDEX_DIR}/")
        faiss.write_index(index, index_path)
        with open(pkl_path, "wb") as f:
            pickle.dump(metadatas, f)
        print("✅ Index saved.")

# Legacy function name for compatibility
build_langchain_faiss_index = build_faiss_index

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
