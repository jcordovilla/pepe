import os
from db import SessionLocal, Message
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# Directory where FAISS index will be stored
INDEX_DIR = "index_faiss"

def build_langchain_faiss_index():
    """
    Read all messages from the SQLite database, embed them, and save a FAISS index.
    """
    print("📚 Loading messages from SQLite DB...")
    # Open a DB session and fetch all messages
    session = SessionLocal()
    rows = session.query(Message).all()
    session.close()

    # Prepare texts and metadata lists
    texts = []
    metadatas = []
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

    print(f"🧠 Embedding {len(texts)} messages with OpenAI...")
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.from_texts(texts=texts, embedding=embedding_model, metadatas=metadatas)

    # Ensure index directory exists
    os.makedirs(INDEX_DIR, exist_ok=True)
    print(f"💾 Saving index to {INDEX_DIR}/")
    vectorstore.save_local(INDEX_DIR)
    print("✅ Index saved.")


if __name__ == "__main__":
    build_langchain_faiss_index()
