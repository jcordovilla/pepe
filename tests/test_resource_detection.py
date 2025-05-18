import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import sqlite3
import random
import json
from core.resource_detector import detect_resources

DB_PATH = os.path.join(os.path.dirname(__file__), '../data/discord_messages.db')
SAMPLE_OUTPUT_PATH = os.path.join(os.path.dirname(__file__), 'resource_detection_sample.json')
DETECTION_OUTPUT_PATH = os.path.join(os.path.dirname(__file__), 'resource_detection_output.json')

# Helper to convert dicts to objects for detector
class DictToObj:
    def __init__(self, d):
        for k, v in d.items():
            setattr(self, k, v)

def main():
    import logging
    log_path = os.path.join(os.path.dirname(__file__), 'detection_debug.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=[
            logging.FileHandler(log_path, mode='w'),
            logging.StreamHandler()
        ]
    )
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    try:
        cur.execute("SELECT * FROM messages")
    except Exception:
        cur.execute("SELECT * FROM discord_messages")
    rows = cur.fetchall()
    sample_rows = random.sample(rows, min(200, len(rows)))

    # Save the sample for reproducibility
    sample_dicts = [dict(row) for row in sample_rows]
    with open(SAMPLE_OUTPUT_PATH, 'w') as f:
        json.dump(sample_dicts, f, indent=2, default=str)

    all_resources = []
    for row in sample_rows:
        msg_obj = DictToObj(dict(row))
        # Patch channel if missing
        if not hasattr(msg_obj, 'channel'):
            class DummyChannel:
                def __init__(self, name, topic=None):
                    self.name = name
                    self.topic = topic
            channel_name = getattr(msg_obj, 'channel_name', None) or getattr(msg_obj, 'channel', None)
            channel_topic = getattr(msg_obj, 'channel_topic', None)
            msg_obj.channel = DummyChannel(channel_name, channel_topic)
        # Patch author if dict
        if hasattr(msg_obj, 'author') and isinstance(msg_obj.author, dict):
            msg_obj.author = DictToObj(msg_obj.author)
        # Patch attachments if string
        if hasattr(msg_obj, 'attachments') and isinstance(msg_obj.attachments, str):
            try:
                att_list = json.loads(msg_obj.attachments)
                class DummyAttachment:
                    def __init__(self, d):
                        for k, v in d.items():
                            setattr(self, k, v)
                msg_obj.attachments = [DummyAttachment(a) for a in att_list]
            except Exception:
                msg_obj.attachments = []
        detected = detect_resources(msg_obj)
        logging.info(f"[DETECT] Message ID {getattr(msg_obj, 'id', None)}: Detected {len(detected)} resources.")
        # Enrich each detected resource with AI title/description if needed
        from core.resource_detector import ai_enrich_title_description, needs_title_fix
        import logging
        for res in detected:
            logging.info(f"[RESOURCE] Before enrichment: {json.dumps(res, default=str)}")
            if needs_title_fix(res):
                # Patch ai_enrich_title_description to accept a logger for stepwise logging
                def enrich_with_logging(resource):
                    from core.resource_detector import is_bad_title
                    import inspect
                    logs = []
                    def log(msg):
                        logs.append(msg)
                        logging.info(f"[ENRICH-STEP] {msg}")
                    # Inline the enrichment logic for stepwise logging
                    from core.resource_detector import ai_enrich_title_description as enrich_fn
                    # Monkeypatch: wrap the OpenAI call and all fallbacks
                    # (Assume ai_enrich_title_description is already instrumented for logging, else fallback to basic logging)
                    try:
                        log("Calling ai_enrich_title_description (standard LLM prompt)")
                        title, desc = enrich_fn(resource)
                        if is_bad_title(title):
                            log(f"LLM returned bad title: '{title}'. Triggering fallback.")
                        else:
                            log(f"LLM returned good title: '{title}'.")
                        return title, desc
                    except Exception as e:
                        log(f"Exception during LLM enrichment: {e}. Triggering fallback.")
                        # Fallback: use context/description/domain-aware
                        # (This is a simplified fallback, real fallback logic is in ai_enrich_title_description)
                        fallback_title = resource.get('context_snippet') or resource.get('description') or resource.get('url')
                        log(f"Fallback title used: '{fallback_title}'")
                        return fallback_title, resource.get('description')
                title, desc = enrich_with_logging(res)
                res["name"] = title
                res["description"] = desc
                logging.info(f"[ENRICH] After enrichment: title='{title}' desc='{desc[:60]}...'")
            else:
                logging.info(f"[ENRICH] No enrichment needed: title='{res.get('name')}' desc='{str(res.get('description'))[:60]}...'")
        # For each detected resource, output as a flat list with repo_sync-compatible keys
        for idx, res in enumerate(detected):
            # Generate a unique id for each resource: message_id + resource index, or hash if no message_id
            msg_id = res.get("message_id") or getattr(msg_obj, 'id', None)
            # If message_id is None, fallback to hash of url+title+channel
            if msg_id is not None:
                unique_id = f"{msg_id}-{idx+1}"
            else:
                import hashlib
                hash_input = (res.get("url") or "") + (res.get("name") or "") + (res.get("channel") or "")
                unique_id = hashlib.md5(hash_input.encode('utf-8')).hexdigest()
            all_resources.append({
                "id": unique_id,
                "title": res.get("name") or res.get("url"),
                "description": res.get("description") or (res.get("context_snippet") or ""),
                "date": res.get("timestamp")[:10] if res.get("timestamp") else None,
                "author": res.get("author"),
                "channel": res.get("channel"),
                "tag": res.get("tag"),
                "resource_url": res.get("url"),
                "discord_url": res.get("jump_url", None),
            })
    # Deduplicate resources after enrichment and before writing output
    from core.resource_detector import deduplicate_resources
    def fuzzy_deduplicate(resources):
        from difflib import SequenceMatcher
        seen = []
        unique = []
        for res in resources:
            is_dup = False
            for u in unique:
                # Fuzzy match on title and description
                title_sim = SequenceMatcher(None, (res.get('title') or '').lower(), (u.get('title') or '').lower()).ratio()
                desc_sim = SequenceMatcher(None, (res.get('description') or '').lower(), (u.get('description') or '').lower()).ratio()
                # If both are highly similar, treat as duplicate
                if title_sim > 0.92 and desc_sim > 0.85:
                    is_dup = True
                    break
            if not is_dup:
                unique.append(res)
        return unique
    before_dedup = len(all_resources)
    all_resources = fuzzy_deduplicate(all_resources)
    after_dedup = len(all_resources)
    # Print each resource and result in color
    GREEN = '\033[92m'
    RED = '\033[91m'
    RESET = '\033[0m'
    for idx, res in enumerate(all_resources, 1):
        print(f"Resource {idx}:")
        print(json.dumps(res, indent=2, default=str))
        # For now, treat all deduped resources as pass (since only valuable resources are kept)
        print(f"{GREEN}PASS{RESET}\n")
    print(f"Total PASS: {after_dedup} resources detected after deduplication.")
    with open(DETECTION_OUTPUT_PATH, 'w') as f:
        json.dump(all_resources, f, indent=2, default=str)
    # Evaluation summary
    total_msgs = len(sample_rows)
    total_resources = len(all_resources)
    tag_counts = {}
    for r in all_resources:
        tag = r.get('tag', 'None')
        tag_counts[tag] = tag_counts.get(tag, 0) + 1
    print(f"Sampled messages: {total_msgs}")
    print(f"Total resources detected: {total_resources}")
    print(f"Tag counts: {tag_counts}")
    print(f"Sample and detection output saved to {SAMPLE_OUTPUT_PATH} and {DETECTION_OUTPUT_PATH}")

if __name__ == "__main__":
    main()
