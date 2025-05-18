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
        # Enrich each detected resource with AI title/description if needed
        from core.resource_detector import ai_enrich_title_description, needs_title_fix
        for res in detected:
            if needs_title_fix(res):
                title, desc = ai_enrich_title_description(res)
                res["name"] = title
                res["description"] = desc
        # For each detected resource, output as a flat list with repo_sync-compatible keys
        for res in detected:
            all_resources.append({
                "id": res.get("message_id") or getattr(msg_obj, 'id', None),
                "title": res.get("name") or res.get("url"),
                "description": res.get("description") or (res.get("context_snippet") or ""),
                "date": res.get("timestamp")[:10] if res.get("timestamp") else None,
                "author": res.get("author"),
                "channel": res.get("channel"),
                "tag": res.get("tag"),
                "resource_url": res.get("url"),
                "discord_url": res.get("jump_url", None),
            })

    # Deduplicate resources before writing output
    from core.resource_detector import deduplicate_resources
    all_resources = deduplicate_resources(all_resources)

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
