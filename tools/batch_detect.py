import json
from db import Session, engine, Message, Resource
from core.resource_detector import detect_resources

def main():
    session = Session()
    try:
        # Query all messages in the database
        messages = session.query(Message).order_by(Message.timestamp.desc()).all()

        new_resources = []
        for msg in messages:
            detected = detect_resources(msg)
            for res in detected:
                # Check if resource already exists (by url and message_id)
                exists = session.query(Resource).filter_by(url=res["url"], message_id=str(getattr(msg, "message_id", None) or getattr(msg, "id", None))).first()
                if exists:
                    continue
                resource_obj = Resource(
                    message_id=str(getattr(msg, "message_id", None) or getattr(msg, "id", None)),
                    guild_id=str(getattr(msg, "guild_id", None)),
                    channel_id=str(getattr(msg, "channel_id", None)),
                    url=res["url"],
                    type=res["type"],
                    tag=res["tag"],
                    author=json.dumps(getattr(msg, "author", None), default=str),
                    author_display=res.get("author"),
                    channel_name=res.get("channel"),
                    timestamp=getattr(msg, "timestamp", None),
                    context_snippet=res.get("context_snippet"),
                    meta=None
                )
                session.add(resource_obj)
                new_resources.append(res)
        session.commit()
        print(json.dumps(new_resources, indent=2, default=str))
    finally:
        session.close()

if __name__ == "__main__":
    main()