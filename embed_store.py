# embed_store.py
import os
import json
from dotenv import load_dotenv
from typing import List, Tuple, Any, Dict, Union
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()
DATA_FILE = "discord_messages_v2.json"
INDEX_DIR = "index_faiss"


def flatten_messages(json_path: str) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Flatten the v2 JSON structure (which can be either direct guild mappings or nested under "guilds")
    into a list of (content, full_metadata). Handles multiple structural variants gracefully.
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data: Dict[str, Any] = json.load(f)

    # Support top-level 'guilds' key or direct guild entries
    root = data.get("guilds", data)
    results: List[Tuple[str, Dict[str, Any]]] = []

    for gid_key, guild_val in root.items():
        # Determine channel mapping and optional guild metadata
        if isinstance(guild_val, dict) and "channels" in guild_val:
            channels = guild_val["channels"]
            guild_name = guild_val.get("name")
        elif isinstance(guild_val, dict):
            channels = guild_val
            guild_name = None
        else:
            continue

        for cid_key, channel_val in channels.items():
            # Determine messages list and channel name
            if isinstance(channel_val, dict) and "messages" in channel_val:
                messages = channel_val["messages"]
                channel_name = channel_val.get("name")
            elif isinstance(channel_val, list):
                messages = channel_val
                channel_name = cid_key
            else:
                continue

            for msg in messages:
                content = msg.get("content", "").strip()
                if not content:
                    continue

                # Copy full metadata
                full_meta = {**msg}

                # Ensure ID fields are strings
                for key in ("guild_id", "channel_id", "message_id"):
                    if key in full_meta:
                        full_meta[key] = str(full_meta[key])

                # Default jump_url if missing
                if not full_meta.get("jump_url"):
                    g = full_meta.get("guild_id")
                    c = full_meta.get("channel_id")
                    m_id = full_meta.get("message_id")
                    full_meta["jump_url"] = f"https://discord.com/channels/{g}/{c}/{m_id}"

                # Attach guild and channel names
                if guild_name:
                    full_meta["guild_name"] = guild_name
                else:
                    gm = msg.get("guild", {}).get("name")
                    if gm:
                        full_meta["guild_name"] = gm
                full_meta["channel_name"] = channel_name

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