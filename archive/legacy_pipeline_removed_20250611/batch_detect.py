import json
import time
import random
import hashlib
import os
import logging
from datetime import datetime
from difflib import SequenceMatcher
from typing import List, Dict, Any

from db.db import SessionLocal, Message, Resource, get_db_session, execute_query
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
    """Optimized batch detection process with improved performance and error handling."""
    # Setup logging
    log_dir = os.path.join(os.path.dirname(__file__), '../data/resources')
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, 'batch_detect.log')
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
    )
    logging.info('Optimized batch detect started.')

    # Performance metrics
    start_time = time.time()
    total_messages = 0
    total_resources = 0
    processed_messages = 0
    batch_size = 1000  # Process messages in batches to avoid memory issues
    
    try:
        # Get total count first for progress tracking
        def get_message_count(session):
            return session.query(Message).filter_by(resource_detected=0).count()
        
        total_messages = execute_query(get_message_count)
        logging.info(f"Found {total_messages} messages to process")
        
        if total_messages == 0:
            print("No unprocessed messages found.")
            return
        
        # Test mode: limit processing
        is_test_mode = os.getenv("BATCH_DETECT_TEST", "0") == "1"
        if is_test_mode and total_messages > 100:
            total_messages = 100
            logging.info(f"Test mode: limiting to {total_messages} messages")

        # Process messages in batches to avoid memory issues
        offset = 0
        all_new_resources = []
        
        with tqdm(total=total_messages, desc="Processing messages", unit="msg") as pbar:
            while offset < total_messages:
                # Get batch of messages
                def get_message_batch(session):
                    query = session.query(Message)\
                        .filter_by(resource_detected=0)\
                        .order_by(Message.timestamp.desc())\
                        .offset(offset)\
                        .limit(batch_size)
                    
                    if is_test_mode and offset + batch_size > total_messages:
                        query = query.limit(total_messages - offset)
                    
                    return query.all()
                
                messages = execute_query(get_message_batch)
                
                if not messages:
                    break
                
                logging.info(f"Processing batch {offset}-{offset + len(messages)}")
                
                # Setup message objects
                messages = setup_message_objects(messages)
                
                # Process batch
                batch_resources = []
                processed_message_ids = []
                
                for msg in messages:
                    try:
                        detected = detect_resources(msg)
                        
                        # Assign unique IDs and add required fields
                        for idx, res in enumerate(detected):
                            msg_id = getattr(msg, 'id', None)
                            if msg_id is not None:
                                unique_id = f"{msg_id}-{idx+1}"
                            else:
                                hash_input = (res.get("url") or "") + (res.get("name") or "")
                                unique_id = hashlib.md5(hash_input.encode('utf-8')).hexdigest()
                            
                            res["id"] = unique_id
                            res["message_id"] = msg_id
                            res["guild_id"] = getattr(msg, 'guild_id', None)
                            res["channel_id"] = getattr(msg, 'channel_id', None)
                            
                        batch_resources.extend(detected)
                        
                        if detected:
                            processed_message_ids.append(msg_id)
                            
                        processed_messages += 1
                        pbar.update(1)
                        
                    except Exception as e:
                        logging.error(f"Error processing message {getattr(msg, 'id', 'unknown')}: {e}")
                        processed_messages += 1
                        pbar.update(1)
                        continue
                
                # Deduplicate batch resources
                batch_resources = simple_deduplicate(batch_resources)
                all_new_resources.extend(batch_resources)
                
                # Update database in batch
                def update_batch(session):
                    # Mark messages as processed
                    if processed_message_ids:
                        session.query(Message)\
                            .filter(Message.id.in_(processed_message_ids))\
                            .update({Message.resource_detected: 1}, synchronize_session=False)
                    
                    # Get existing resource URLs to avoid duplicates
                    existing_urls = set()
                    if batch_resources:
                        urls = [r["url"] for r in batch_resources]
                        existing = session.query(Resource.url, Resource.message_id)\
                            .filter(Resource.url.in_(urls)).all()
                        existing_urls = set(f"{url}:{msg_id}" for url, msg_id in existing)
                    
                    # Add new resources
                    new_count = 0
                    for res in batch_resources:
                        resource_key = f"{res['url']}:{res.get('message_id', None)}"
                        if resource_key not in existing_urls:
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
                            new_count += 1
                    
                    return new_count
                
                try:
                    new_count = execute_query(update_batch)
                    total_resources += new_count
                    logging.info(f"Batch {offset}-{offset + len(messages)}: {len(batch_resources)} resources detected, {new_count} new resources saved")
                except Exception as e:
                    logging.error(f"Error updating batch {offset}-{offset + len(messages)}: {e}")
                
                offset += len(messages)
        
        # Deduplicate all resources globally
        logging.info("Performing global deduplication...")
        all_new_resources = simple_deduplicate(all_new_resources)
        
        elapsed = time.time() - start_time
        messages_per_second = processed_messages / elapsed if elapsed > 0 else 0
        
        # Performance summary
        print(f"\n{'='*60}")
        print(f"BATCH DETECTION COMPLETED")
        print(f"{'='*60}")
        print(f"📊 Messages processed: {processed_messages:,}")
        print(f"🔗 Resources detected: {len(all_new_resources):,}")
        print(f"💾 New resources saved: {total_resources:,}")
        print(f"⏱️  Processing time: {elapsed:.2f} seconds")
        print(f"🚀 Speed: {messages_per_second:.1f} messages/second")
        print(f"📈 Resource rate: {(len(all_new_resources)/processed_messages*100):.1f}% of messages had resources")
        print(f"{'='*60}")
        
        logging.info(f"Batch detect completed: {processed_messages} messages, {len(all_new_resources)} resources in {elapsed:.2f}s")
        
    except Exception as e:
        elapsed = time.time() - start_time
        logging.error(f"Batch detect failed after {elapsed:.2f}s: {e}")
        print(f"❌ Batch detection failed: {e}")
        raise


if __name__ == "__main__":
    main()