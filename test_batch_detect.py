import json
from db import Session, engine  # Assumes Session and engine are defined in db.py
from db import Message         # Assumes Message is your SQLAlchemy message model
from resource_detector import detect_resources

def main():
    session = Session()
    try:
        # Query 10 most recent messages
        messages = (
            session.query(Message)
            .order_by(Message.timestamp.desc())
            .limit(10)
            .all()
        )

        all_resources = []
        for msg in messages:
            detected = detect_resources(msg)
            if detected:
                all_resources.extend(detected)

        print(json.dumps(all_resources, indent=2, default=str))
    finally:
        session.close()

if __name__ == "__main__":
    main()