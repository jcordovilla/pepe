import json
import time
import random
import hashlib
import os
import logging
from datetime import datetime
from difflib import SequenceMatcher

from db.db import SessionLocal, Message, Resource
from core.resource_detector import detect_resources
from tqdm import tqdm


def parse_timestamp(ts):
    """Simple timestamp parser."""
    if isinstance(ts, datetime) or ts is None:
        return ts
    try:
        return datetime.fromisoformat(ts.replace('Z', '+00:00'))
    except Exception:
        return None


def simple_deduplicate(resources):
    """Simple deduplication based on title and description similarity."""
    unique = []
    for res in resources:
        is_dup = False
        for u in unique:
            title_sim = SequenceMatcher(None, 
                                     (res.get('name') or '').lower(), 
                                     (u.get('name') or '').lower()).ratio()
            desc_sim = SequenceMatcher(None, 
                                    (res.get('description') or '').lower(), 
                                    (u.get('description') or '').lower()).ratio()
            if title_sim > 0.92 and desc_sim > 0.85:
                is_dup = True
                break
        if not is_dup:
            unique.append(res)
    return unique


def setup_message_objects(messages):
    """Simplified message object setup."""
    for msg in messages:
        # Setup channel
        if not hasattr(msg, 'channel'):
            class SimpleChannel:
                def __init__(self, name):
                    self.name = name
            channel_name = getattr(msg, 'channel_name', None)
            msg.channel = SimpleChannel(channel_name)
        
        # Setup author 
        if hasattr(msg, 'author') and isinstance(msg.author, str):
            try:
                msg.author = json.loads(msg.author)
            except:
                pass
        
        # Setup attachments
        if not hasattr(msg, 'attachments'):
            msg.attachments = []
    
    return messages

def main():
    """Simplified batch detection process."""
    # Setup logging
    log_dir = os.path.join(os.path.dirname(__file__), '../data/resources')
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, 'batch_detect.log')
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
    )
    logging.info('Simplified batch detect started.')

    session = SessionLocal()
    try:
        # Get unprocessed messages
        messages = session.query(Message)\
            .filter_by(resource_detected=0)\
            .order_by(Message.timestamp.desc())\
            .all()

        logging.info(f"Found {len(messages)} messages to process")
        
        # Test mode: limit processing
        is_test_mode = os.getenv("BATCH_DETECT_TEST", "0") == "1"
        if is_test_mode and len(messages) > 100:
            messages = random.sample(messages, 100)

        # Setup message objects
        messages = setup_message_objects(messages)
        
        start_time = time.time()
        new_resources = []
        
        # Process each message
        for msg in tqdm(messages, desc="Detecting resources", unit="msg"):
            detected = detect_resources(msg)
            logging.info(f"Message ID {getattr(msg, 'id', None)}: Detected {len(detected)} resources.")
            
            # Assign unique IDs
            for idx, res in enumerate(detected):
                msg_id = getattr(msg, 'id', None)
                if msg_id is not None:
                    unique_id = f"{msg_id}-{idx+1}"
                else:
                    hash_input = (res.get("url") or "") + (res.get("name") or "")
                    unique_id = hashlib.md5(hash_input.encode('utf-8')).hexdigest()
                res["id"] = unique_id
                
            new_resources.extend(detected)
            
            # Mark as processed if resources found
            if detected:
                msg.resource_detected = 1
                session.add(msg)
        
        # Deduplicate resources
        new_resources = simple_deduplicate(new_resources)
        
        # Save to database
        for res in new_resources:
            # Check if already exists
            exists = session.query(Resource).filter_by(
                url=res["url"],
                message_id=str(res.get("message_id", None))
            ).first()
            
            if not exists:
                resource_obj = Resource(
                    message_id=str(res.get("message_id", None)),
                    guild_id=str(res.get("guild_id", None)),
                    channel_id=str(res.get("channel_id", None)),
                    url=res["url"],
                    type=res["type"],
                    tag=res["tag"],
                    author=json.dumps(res.get("author", None), default=str),
                    author_display=res.get("author"),
                    channel_name=res.get("channel"),
                    timestamp=parse_timestamp(res.get("timestamp", None)),
                    context_snippet=res.get("context_snippet"),
                    name=res.get("name"),
                    description=res.get("description"),
                    jump_url=res.get("jump_url"),
                    meta=None,
                )
                session.add(resource_obj)
        
        session.commit()
        elapsed = time.time() - start_time
        print(f"Detection completed in {elapsed:.2f} seconds.")
        print(f"Found {len(new_resources)} new resources.")
        
    finally:
        session.close()
    
    logging.info(f"Batch detect finished. Total new resources: {len(new_resources)}")


if __name__ == "__main__":
    main()