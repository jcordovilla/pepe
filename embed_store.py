import os
import json
from dotenv import load_dotenv
from typing import List, Tuple
import copy
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()
DATA_FILE = "discord_messages.json"
INDEX_DIR = "index_faiss"

def flatten_messages(json_path: str) -> List[Tuple[str, dict]]:
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    texts_and_meta = []
    for guild, channels in data.items():
        for channel, messages in channels.items():
            for msg in messages:
                content = msg["content"].strip()
                if not content:
                    continue

                # Extract IDs
                message_id = msg.get("message_id") or msg.get("id")
                channel_id = msg.get("channel_id")
                guild_id = msg.get("guild_id")

                # Guard against corrupted or missing IDs
                if not all([guild_id, channel_id, message_id]) or not all(str(x).isdigit() for x in [guild_id, channel_id, message_id]):
                    print("⛔ Skipping message with invalid IDs:", {
                        "guild_id": guild_id,
                        "channel_id": channel_id,
                        "message_id": message_id
                    })
                    continue

                jump_url = msg.get("jump_url") or f"https://discord.com/channels/{guild_id}/{channel_id}/{message_id}"

                metadata = {
                    "guild": guild,
                    "channel": channel,
                    "author": copy.deepcopy(msg.get("author")),
                    "timestamp": msg["timestamp"],
                    "message_id": message_id,
                    "guild_id": guild_id,
                    "channel_id": channel_id,
                    "jump_url": jump_url,
                    "content": content,
                    "reactions": copy.deepcopy(msg.get("reactions")),
                    "mentions": msg.get("mentions"),
                    "mention_everyone": msg.get("mention_everyone"),
                    "mention_roles": msg.get("mention_roles"),
                    "attachments": msg.get("attachments"),
                    "embeds": msg.get("embeds"),
                    "pinned": msg.get("pinned"),
                    "flags": msg.get("flags"),
                    "type": msg.get("type")
                }
                texts_and_meta.append((content, metadata))
    return texts_and_meta

def build_langchain_faiss_index():
    print("📚 Loading messages...")
    texts_and_metadata = flatten_messages(DATA_FILE)
    texts = [t[0] for t in texts_and_metadata]
    metadatas = [t[1] for t in texts_and_metadata]

    print(f"🧠 Embedding {len(texts)} messages with OpenAI...")
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.from_texts(texts=texts, embedding=embedding_model, metadatas=metadatas)

    print(f"💾 Saving index to {INDEX_DIR}/")
    vectorstore.save_local(INDEX_DIR)
    print("✅ Index saved.")

if __name__ == "__main__":
    build_langchain_faiss_index()
