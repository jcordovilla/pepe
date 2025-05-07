import os
import json
from dotenv import load_dotenv
from typing import List, Tuple, Any
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()
DATA_FILE = "discord_messages_v2.json"  # use migrated ID-first file
INDEX_DIR = "index_faiss"


def flatten_messages(json_path: str) -> List[Tuple[str, dict]]:
    """
    Flatten the v2 JSON structure (guilds → channels → messages) into a list of (content, full_metadata).
    Extracts all fields from each message to preserve comprehensive metadata.
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    results: List[Tuple[str, dict]] = []
    guilds = data.get("guilds", {})
    for gid, guild in guilds.items():
        for cid, channel in guild.get("channels", {}).items():
            for msg in channel.get("messages", []):
                content = msg.get("content", "").strip()
                if not content:
                    continue

                # Prepare full metadata by copying the message dict
                full_meta: dict = dict(msg)

                # Ensure IDs are strings
                for key in ["guild_id", "channel_id", "message_id"]:
                    if key in full_meta:
                        full_meta[key] = str(full_meta[key])

                # Default jump_url if missing
                if not full_meta.get("jump_url"):
                    g = full_meta.get("guild_id")
                    c = full_meta.get("channel_id")
                    m_id = full_meta.get("message_id")
                    full_meta["jump_url"] = f"https://discord.com/channels/{g}/{c}/{m_id}"

                # Attach channel and guild names
                full_meta["channel_name"] = channel.get("name")
                full_meta["guild_name"] = guild.get("name")

                results.append((content, full_meta))
    return results


def build_langchain_faiss_index():
    print("📚 Loading messages with full metadata...")
    texts_and_metadata = flatten_messages(DATA_FILE)
    texts = [t for t, _ in texts_and_metadata]
    metadatas = [m for _, m in texts_and_metadata]

    print(f"🧠 Embedding {len(texts)} messages with OpenAI...")
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.from_texts(texts=texts, embedding=embedding_model, metadatas=metadatas)

    os.makedirs(INDEX_DIR, exist_ok=True)
    print(f"💾 Saving index to {INDEX_DIR}/")
    vectorstore.save_local(INDEX_DIR)
    print("✅ Index saved.")


if __name__ == "__main__":
    build_langchain_faiss_index()