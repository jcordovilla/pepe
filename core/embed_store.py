# embed_store.py

import os
import numpy as np
import json
import faiss
from db import SessionLocal, Message
from core.config import get_config
from typing import List, Dict, Any
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_sentence_transformer():
    """Get the sentence transformer model."""
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer("msmarco-distilbert-base-v4")
    except ImportError:
        raise ImportError("sentence-transformers package is required. Install with: pip install sentence-transformers")

def build_faiss_index():
    """
    Build FAISS index from SQLite DB messages using msmarco-distilbert-base-v4 embeddings.
    Creates standardized format compatible with enhanced RAG engine.
    """
    config = get_config()
    INDEX_DIR = config.faiss_index_path
    
    print("📚 Loading messages from SQLite DB...")
    session = SessionLocal()
    rows = session.query(Message).all()
    session.close()

    if not rows:
        print("❌ No messages found in database.")
        return

    # Prepare all messages and metadata
    texts = []
    metadata_dict = {}
    id_mapping = []
    
    for m in rows:
        # Use cleaned content if available, otherwise raw content
        content = getattr(m, 'cleaned_content', m.content) or m.content
        texts.append(content)
        
        message_id = str(m.message_id)
        metadata_dict[message_id] = {
            "content": m.content,
            "cleaned_content": content,
            "timestamp": m.timestamp.isoformat(),
            "channel_id": m.channel_id,
            "guild_id": m.guild_id,
            "author_id": str(getattr(m, 'author_id', '')),
            "message_type": getattr(m, 'message_type', 'MessageType.default'),
            "is_pinned": getattr(m, 'is_pinned', False),
            "has_embeds": getattr(m, 'has_embeds', False),
            "has_attachments": getattr(m, 'has_attachments', False),
            "has_reply_context": getattr(m, 'has_reply_context', False),
            "content_length": len(content),
            "reaction_count": getattr(m, 'reaction_count', 0),
            "channel_name": getattr(m, 'channel_name', ''),
            "author": getattr(m, 'author', {}),
            "mention_ids": getattr(m, 'mention_ids', []),
            "reactions": getattr(m, 'reactions', []),
            "jump_url": getattr(m, 'jump_url', '')
        }
        id_mapping.append(int(m.message_id))

    os.makedirs(INDEX_DIR, exist_ok=True)
    index_path = os.path.join(INDEX_DIR, "faiss_index.index")
    metadata_path = os.path.join(INDEX_DIR, "metadata.json")

    # Load sentence transformer model
    print(f"🧠 Loading msmarco-distilbert-base-v4 model...")
    model = get_sentence_transformer()
    
    # Create embeddings
    print(f"🧠 Embedding {len(texts)} messages...")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    
    # Create FAISS index
    dimension = embeddings.shape[1]
    print(f"📊 Creating FAISS index with dimension {dimension}")
    index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
    
    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(embeddings.astype('float32'))
    index.add(embeddings.astype('float32'))
    
    # Save index
    print(f"💾 Saving FAISS index to {index_path}")
    faiss.write_index(index, index_path)
    
    # Save metadata in standardized format
    metadata_structure = {
        "metadata": metadata_dict,
        "id_mapping": id_mapping,
        "config": {
            "model_name": "msmarco-distilbert-base-v4",
            "index_type": "flat",
            "dimension": dimension,
            "normalize_embeddings": True,
            "created_at": datetime.now().isoformat(),
            "total_messages": len(texts)
        }
    }
    
    print(f"💾 Saving metadata to {metadata_path}")
    with open(metadata_path, "w", encoding='utf-8') as f:
        json.dump(metadata_structure, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Discord message index created successfully!")
    print(f"   - {len(texts)} messages indexed")
    print(f"   - Dimension: {dimension}")
    print(f"   - Model: msmarco-distilbert-base-v4")
    print(f"   - Location: {INDEX_DIR}")

# Legacy function name for compatibility
build_langchain_faiss_index = build_faiss_index

def flatten_messages(db_path: str = None) -> List[tuple]:
    """
    Load all messages from the SQLite database and return them as a list of (text, metadata) tuples.
    Updated to use standardized metadata format.
    """
    print(f"📂 Loading messages from database...")
    session = SessionLocal()
    rows = session.query(Message).all()
    session.close()

    messages = []
    for m in rows:
        # Use cleaned content if available, otherwise raw content
        content = getattr(m, 'cleaned_content', m.content) or m.content
        
        meta = {
            "guild_id": m.guild_id,
            "channel_id": m.channel_id,
            "channel_name": getattr(m, 'channel_name', ''),
            "message_id": m.message_id,
            "content": m.content,
            "cleaned_content": content,
            "timestamp": m.timestamp.isoformat(),
            "author": getattr(m, 'author', {}),
            "author_id": str(getattr(m, 'author_id', '')),
            "mention_ids": getattr(m, 'mention_ids', []),
            "reactions": getattr(m, 'reactions', []),
            "jump_url": getattr(m, 'jump_url', ''),
            "message_type": getattr(m, 'message_type', 'MessageType.default'),
            "is_pinned": getattr(m, 'is_pinned', False),
            "has_embeds": getattr(m, 'has_embeds', False),
            "has_attachments": getattr(m, 'has_attachments', False),
            "has_reply_context": getattr(m, 'has_reply_context', False),
            "content_length": len(content),
            "reaction_count": getattr(m, 'reaction_count', 0)
        }
        messages.append((content, meta))  # Return as a tuple (text, metadata)
    
    print(f"📊 Loaded {len(messages)} messages")
    return messages

if __name__ == "__main__":
    build_faiss_index()
