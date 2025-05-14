import json
import time  # Add this import
from db.db import SessionLocal, Message, Resource  # Updated import paths
from core.resource_detector import detect_resources
from tqdm import tqdm  # Already imported

def author_to_dict(author):
    if isinstance(author, dict):
        return author
    if hasattr(author, '__dict__'):
        return vars(author)
    return author

def main():
    session = SessionLocal()
    try:
        # Query all messages in the database
        messages = session.query(Message).order_by(Message.timestamp.desc()).all()

        # Timer start
        start_time = time.time()

        # For test runs: only process a few messages
        import os
        if os.getenv("BATCH_DETECT_TEST", "0") == "1":
            messages = messages[:50]  # Only process the first 50 messages

        new_resources = []
        for msg in tqdm(messages, desc="Detecting resources", unit="msg"):
            # Patch channel if missing
            if not hasattr(msg, 'channel'):
                class DummyChannel:
                    def __init__(self, name, topic=None):
                        self.name = name
                        self.topic = topic
                channel_name = getattr(msg, 'channel_name', None) or getattr(msg, 'channel', None)
                channel_topic = getattr(msg, 'channel_topic', None)
                msg.channel = DummyChannel(channel_name, channel_topic)
            # Patch author if dict or JSON string
            if hasattr(msg, 'author'):
                if isinstance(msg.author, str):
                    try:
                        import json as _json
                        msg.author = _json.loads(msg.author)
                    except Exception:
                        pass
                if isinstance(msg.author, dict):
                    class DummyAuthor:
                        def __init__(self, d):
                            for k, v in d.items():
                                setattr(self, k, v)
                    msg.author = DummyAuthor(msg.author)
            # Ensure msg.author is always JSON-serializable for DB
            if hasattr(msg, 'author') and type(msg.author).__name__ == 'DummyAuthor':
                msg.author = vars(msg.author)
            # Patch attachments if string (rare in DB, but for compatibility)
            if hasattr(msg, 'attachments') and isinstance(msg.attachments, str):
                try:
                    att_list = json.loads(msg.attachments)
                    class DummyAttachment:
                        def __init__(self, d):
                            for k, v in d.items():
                                setattr(self, k, v)
                    msg.attachments = [DummyAttachment(a) for a in att_list]
                except Exception:
                    msg.attachments = []
            elif not hasattr(msg, 'attachments'):
                msg.attachments = []

            detected = detect_resources(msg)
            for res in detected:
                # Check if resource already exists (by url and message_id)
                exists = session.query(Resource).filter_by(
                    url=res["url"],
                    message_id=str(getattr(msg, "message_id", None) or getattr(msg, "id", None))
                ).first()
                if exists:
                    # Backfill name/description/jump_url if missing and new values are available
                    updated = False
                    if not exists.name and res.get("name"):
                        exists.name = res.get("name")
                        updated = True
                    if not exists.description and res.get("description"):
                        exists.description = res.get("description")
                        updated = True
                    if not exists.jump_url and res.get("jump_url"):
                        exists.jump_url = res.get("jump_url")
                        updated = True
                    if updated:
                        session.add(exists)
                    continue
                resource_obj = Resource(
                    message_id=str(getattr(msg, "message_id", None) or getattr(msg, "id", None)),
                    guild_id=str(getattr(msg, "guild_id", None)),
                    channel_id=str(getattr(msg, "channel_id", None)),
                    url=res["url"],
                    type=res["type"],
                    tag=res["tag"],
                    author=json.dumps(author_to_dict(getattr(msg, "author", None)), default=str),
                    author_display=res.get("author"),
                    channel_name=res.get("channel"),
                    timestamp=getattr(msg, "timestamp", None),
                    context_snippet=res.get("context_snippet"),
                    name=res.get("name"),
                    description=res.get("description"),
                    jump_url=res.get("jump_url"),
                    meta=None
                )
                session.add(resource_obj)
                new_resources.append(res)
        session.commit()
        elapsed = time.time() - start_time
        print(f"Detection completed in {elapsed:.2f} seconds.")
        print(json.dumps(new_resources, indent=2, default=str))
    finally:
        session.close()

if __name__ == "__main__":
    main()