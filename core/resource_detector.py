import re
import json
from datetime import datetime
from typing import List, Dict, Any
from core.classifier import classify_resource  # <-- Import the classifier
from urllib.parse import urlparse, urlunparse

# Example: from tools.time_parser import parse_time  # Uncomment if you want to use your time parser

URL_REGEX = re.compile(
    r'(https?://[^\s]+)'
)

TRASH_PATTERNS = [
    r'discord\\.com/channels/',
    r'discord\\.com/events/',
    r'zoom\\.us',
    r'meet\\.google\\.com',
    r'teams\\.microsoft\\.com',
    r'webex\\.com',
    r'gotomeeting\\.com',
    r'calendar\\.google\\.com',
    r'calendar\\.outlook',
    r'facebook\\.com/events',
    r'\\.(png|jpg|jpeg|gif|webp|svg)$',
    r'cdn\\.discordapp\\.com',
    r'giphy\\.com',
    r'tenor\\.com',
    # Add more as needed
]

def is_trash_url(url):
    return any(re.search(pat, url) for pat in TRASH_PATTERNS)

def ai_vet_resource(resource: dict, log_decision: bool = False) -> dict:
    """
    Use local AI to vet if a resource is valuable (not a meeting, internal, or spam).
    Returns a dict: {'is_valuable': bool, 'name': str, 'description': str}
    If log_decision is True, print the AI's answer and reason for each resource.
    """
    from core.ai_client import get_ai_client
    
    ai_client = get_ai_client()
    system_prompt = (
        """
        You are a filter for a knowledge resource library. 
        Given a link and its context, decide if it is a valuable resource (e.g., paper, tool, tutorial, news article, blog) 
        or if it is a meeting invite, internal navigation, or irrelevant. 
        If valuable, reply in the following JSON format: 
        {"valuable": true, "name": "<resource name>", "description": "<brief description>"}
        If not valuable, reply: {"valuable": false}
        The name should be a concise title for the resource. The description should be 1-2 sentences summarizing its content or value.
        """
    )
    user_content = (
        f"Resource info:\n"
        f"URL: {resource.get('url')}\n"
        f"Type: {resource.get('type')}\n"
        f"Author: {resource.get('author')}\n"
        f"Channel: {resource.get('channel')}\n"
        f"Context: {resource.get('context_snippet')}\n"
    )
    try:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]
        
        answer = ai_client.chat_completion(
            messages,
            temperature=0.0,
            max_tokens=150
        )
        
        if log_decision:
            print(f"AI vetting for resource: {resource.get('url') or '[no url]'} => {answer}")
        
        import json as _json
        try:
            parsed = _json.loads(answer)
            is_valuable = parsed.get("valuable", False)
            name = parsed.get("name")
            description = parsed.get("description")
            return {"is_valuable": is_valuable, "name": name, "description": description}
        except Exception:
            # fallback: treat as not valuable
            return {"is_valuable": False, "name": None, "description": None}
    except Exception as e:
        print(f"AI vetting error: {e}. Defaulting to False.")
        return {"is_valuable": False, "name": None, "description": None}

# Helper to determine resource type from URL or filename
def get_resource_type(url: str) -> str:
    if url.endswith('.pdf'):
        return 'pdf'
    if url.endswith('.mp4'):
        return 'video'
    if 'youtube.com' in url or 'youtu.be' in url:
        return 'youtube'
    if 'drive.google.com' in url:
        return 'drive'
    if 'github.com' in url:
        return 'github'
    if url.endswith('.png') or url.endswith('.jpg') or url.endswith('.jpeg') or url.endswith('.gif'):
        return 'image'
    if url.endswith('.zip'):
        return 'archive'
    return 'link'

def detect_resources(message) -> List[Dict[str, Any]]:
    """
    Detect resources in a Discord message object.
    Returns a list of dicts with keys: url, type, timestamp, author, channel, message_id, context_snippet, tag
    """
    # Basic filtering: skip bots/system messages
    if getattr(message, "author", None) and (getattr(message.author, "bot", False) or getattr(message.author, "system", False)):
        return []

    resources = []

    def normalize_timestamp(ts):
        if hasattr(ts, "isoformat"):
            return str(ts.isoformat())
        return str(ts)

    def get_author_display_name(author):
        if author is None:
            return None
        if isinstance(author, str):
            try:
                author_dict = json.loads(author)
                return author_dict.get('username') or author_dict.get('name')
            except Exception:
                return author
        # Prefer display_name or global_name if available
        if hasattr(author, 'display_name') and getattr(author, 'display_name', None):
            return getattr(author, 'display_name')
        if hasattr(author, 'global_name') and getattr(author, 'global_name', None):
            return getattr(author, 'global_name')
        if hasattr(author, 'nick') and getattr(author, 'nick', None):
            return getattr(author, 'nick')
        if hasattr(author, 'username') and getattr(author, 'username', None):
            return getattr(author, 'username')
        if hasattr(author, 'name'):
            return getattr(author, 'name')
        if isinstance(author, dict):
            return (
                author.get('display_name') or
                author.get('global_name') or
                author.get('nick') or
                author.get('username') or
                author.get('name')
            )
        return str(author)

    def format_timestamp(ts):
        # Return 'YYYY-MM-DD HH:MM' if possible
        if hasattr(ts, 'strftime'):
            return ts.strftime('%Y-%m-%d %H:%M')
        try:
            # Try parsing if it's a string
            from datetime import datetime
            # Handle 'Z' suffix for UTC
            if isinstance(ts, str) and ts.endswith('Z'):
                ts = ts.replace('Z', '+00:00')
            dt = datetime.fromisoformat(ts)
            return dt.strftime('%Y-%m-%d %H:%M')
        except Exception:
            return str(ts)

    def get_jump_url(msg):
        # Try to get jump_url attribute, else build from known fields
        if hasattr(msg, 'jump_url') and getattr(msg, 'jump_url', None):
            return msg.jump_url
        # Try to build from known fields if possible
        guild_id = getattr(msg, 'guild_id', None)
        channel_id = getattr(msg, 'channel_id', None)
        message_id = getattr(msg, 'message_id', None) or getattr(msg, 'id', None)
        if guild_id and channel_id and message_id:
            return f"https://discord.com/channels/{guild_id}/{channel_id}/{message_id}"
        return None

    # --- IMPROVEMENT: Gather more context ---
    content = getattr(message, "content", "")
    channel_topic = getattr(getattr(message, "channel", None), "topic", "")
    thread_name = getattr(getattr(message, "thread", None), "name", "")
    context_snippet = content
    if channel_topic:
        context_snippet += f"\nChannel topic: {channel_topic}"
    if thread_name:
        context_snippet += f"\nThread: {thread_name}"
    context_snippet = context_snippet[:500]  # Use more context, up to 500 chars

    # Extract URLs from message content
    urls = URL_REGEX.findall(content)
    for idx, url in enumerate(urls):
        if is_trash_url(url):
            continue  # Skip trash links
        resource = {
            "url": url,
            "type": get_resource_type(url),
            "timestamp": format_timestamp(getattr(message, "timestamp", None)),
            "author": get_author_display_name(getattr(message, "author", None)),
            "channel": getattr(message, "channel", None).name if getattr(message, "channel", None) else None,
            "jump_url": get_jump_url(message),
            "context_snippet": context_snippet,
            "message_id": getattr(message, "id", None),
            # Unique resource id: message_id + index (1-based)
            "resource_id": f"{getattr(message, 'id', 'noid')}-{idx+1}"
        }
        vet_result = ai_vet_resource(resource, log_decision=True)
        if not vet_result["is_valuable"]:
            continue
        resource["name"] = vet_result["name"]
        resource["description"] = vet_result["description"]
        if needs_title_fix(resource):
            title, desc = ai_enrich_title_description(resource)
            resource["name"], resource["description"] = title, desc
        resource["tag"] = classify_resource(resource)
        resources.append(resource)

    # --- IMPROVEMENT: Attachment handling ---
    attachments = getattr(message, "attachments", [])
    for aidx, att in enumerate(attachments):
        url = getattr(att, "url", None)
        filename = getattr(att, "filename", "")
        content_type = getattr(att, "content_type", "")
        if url and not is_trash_url(url):
            resource_type = get_resource_type(url)
            # Use filename/content_type to improve type detection
            if not resource_type or resource_type == "link":
                if filename.endswith('.pdf'):
                    resource_type = 'pdf'
                elif filename.endswith('.mp4'):
                    resource_type = 'video'
                elif filename.endswith('.png') or filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.gif'):
                    resource_type = 'image'
                elif filename.endswith('.zip'):
                    resource_type = 'archive'
                elif 'pdf' in content_type:
                    resource_type = 'pdf'
                elif 'image' in content_type:
                    resource_type = 'image'
                elif 'video' in content_type:
                    resource_type = 'video'
            resource = {
                "url": url,
                "type": resource_type,
                "timestamp": format_timestamp(getattr(message, "timestamp", None)),
                "author": get_author_display_name(getattr(message, "author", None)),
                "channel": getattr(message, "channel", None).name if getattr(message, "channel", None) else None,
                "jump_url": get_jump_url(message),
                "context_snippet": context_snippet,
                "message_id": getattr(message, "id", None),
                # Unique resource id for attachment: message_id + aidx offset from url count
                "resource_id": f"{getattr(message, 'id', 'noid')}-att{aidx+1}"
            }
            vet_result = ai_vet_resource(resource)
            if not vet_result["is_valuable"]:
                continue
            resource["name"] = vet_result["name"]
            resource["description"] = vet_result["description"]
            if needs_title_fix(resource):
                title, desc = ai_enrich_title_description(resource)
                resource["name"], resource["description"] = title, desc
            resource["tag"] = classify_resource(resource)
            resources.append(resource)

    # --- IMPROVEMENT: Pinned message boost & non-URL resource detection ---
    is_pinned = getattr(message, "is_pinned", False)
    if is_pinned and not urls and content:
        keywords = ["tool", "tutorial", "paper", "event", "job", "course", "resource"]
        if any(kw in content.lower() for kw in keywords):
            resource = {
                "url": None,
                "type": "pinned-text",
                "timestamp": format_timestamp(getattr(message, "timestamp", None)),
                "author": get_author_display_name(getattr(message, "author", None)),
                "channel": getattr(message, "channel", None).name if getattr(message, "channel", None) else None,
                "jump_url": get_jump_url(message),
                "context_snippet": context_snippet
            }
            vet_result = ai_vet_resource(resource)
            if vet_result["is_valuable"]:
                resource["name"] = vet_result["name"]
                resource["description"] = vet_result["description"]
                if needs_title_fix(resource):
                    title, desc = ai_enrich_title_description(resource)
                    resource["name"], resource["description"] = title, desc
                resource["tag"] = classify_resource(resource)
                resources.append(resource)

    return resources

def needs_title_fix(resource):
    """
    Return True if the resource's name/title is missing or is a bad title (URL, domain, slug, hash, generic, etc.).
    Triggers enrichment for all cases where the output title would not be human-readable.
    """
    name = resource.get("name", "")
    if not name or is_bad_title(name):
        return True
    return False

def ai_enrich_title_description(resource):
    """
    Use local AI to generate a high-quality, human-readable title and description for a resource, with robust fallback logic.
    - If the LLM returns a title that is a URL or domain, always use a robust fallback.
    - Fallback: use message context (first non-empty, non-URL line) as title if LLM fails.
    - Fallback: use domain-aware label (e.g. 'WSJ Article', 'Axios Article') as last resort.
    - For Google Drive/docs, synthesize a title from the description if all else fails.
    """
    from core.ai_client import get_ai_client
    import json as _json
    import re
    from urllib.parse import urlparse, unquote
    
    ai_client = get_ai_client()

    def is_bad_title(title):
        if not title:
            return True
        title = title.strip()
        if re.match(r'https?://', title) or re.match(r'^[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', title):
            return True
        if re.search(r'https?://|www\\.|\.[a-z]{2,}', title):
            return True
        if ('-' in title or '_' in title) and title.lower() == title and ' ' not in title:
            return True
        if re.match(r'^[a-f0-9\-]{10,}$', title):
            return True
        if len(re.sub(r'[^a-zA-Z]', '', title)) < 3:
            return True
        if title.lower() in ("resource", "untitled", "index"):
            return True
        return False

    def postprocess_title(title):
        if not title:
            return title
        title = title.strip().strip('-:|')
        title = re.sub(r'^(https?://)?(www\\.)?[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}/?', '', title)
        title = re.sub(r'[-_][a-f0-9]{6,}$', '', title)
        if title.isupper() or title.islower():
            title = title.title()
        return title.strip()

    # --- LLM attempt ---
    message_info = []
    if resource.get('author'):
        message_info.append(f"Author: {resource.get('author')}")
    if resource.get('channel'):
        message_info.append(f"Channel: {resource.get('channel')}")
    if resource.get('timestamp'):
        message_info.append(f"Timestamp: {resource.get('timestamp')}")
    if resource.get('context_snippet'):
        message_info.append(f"Context: {resource.get('context_snippet')}")
    message_info_str = '\n'.join(message_info)

    base_prompt = f"""
You are an expert research assistant for a public resource library. Given a Discord message and a resource URL, your job is to:
- Use your knowledge (as of cutoff date) to infer the real, human-readable title of the resource as it appears on the web (e.g., the article, paper, or video title), not just a summary or the URL/domain.
- If the resource is a news article, paper, blog, or video, use the actual title as published, if possible.
- If you cannot find the real title, generate a concise, human-readable title based on the context and topic, but never use the raw URL, domain, or any part of the URL as the title.
- Never use the Discord message author's username or display name as the resource's author or in the title or description. Only use the real author/title as published on the web resource itself.
- Write a 1-2 sentence description summarizing the resource's content or value. The description must not mention the Discord author unless they are the actual author of the resource.
- Respond in JSON with keys 'title' and 'description'.
- Never use the URL, domain, or any part of the URL as the title. If you cannot find the real title, create a plausible, readable one from the context and topic only.

Message info:
{message_info_str}

Resource URL:
{resource.get('url','')}
"""

    strict_prompt = base_prompt + "\n\nIMPORTANT: The title and description must never be a URL, domain, or any part of the URL. Never use the Discord message author as the resource author, in the title, or in the description. If you cannot find the real title, create a plausible, readable one from the context and topic only."

    for attempt, prompt in enumerate([base_prompt, strict_prompt]):
        try:
            messages = [{"role": "user", "content": prompt}]
            
            content = ai_client.chat_completion(
                messages,
                temperature=0.5 if attempt == 0 else 0.2,
                max_tokens=200
            )
            
            data = _json.loads(content)
            title = postprocess_title(data.get("title"))
            desc = data.get("description")
            if title and not is_bad_title(title):
                return title, desc
        except Exception:
            pass

    # --- Fallback: try message context and description lines ---
    context = resource.get("context_snippet", "")
    description = resource.get("description", "")
    tried_lines = set()
    for source in [context, description]:
        if not source:
            continue
        for line in source.split('\n'):
            line = line.strip()
            if not line or is_bad_title(line) or line in tried_lines:
                continue
            tried_lines.add(line)
            if len(line) > 6:
                return line[:80].strip(), description or context

    # --- Fallback: Google Drive/docs special handling ---
    url = resource.get("url", "")
    if url:
        parsed = urlparse(url)
        domain = parsed.netloc.replace('www.', '')
        path = parsed.path
        # Google Drive or Docs
        if any(d in domain for d in ["drive.google.com", "docs.google.com", "docs.microsoft.com", "onedrive.live.com"]):
            # Try to synthesize a title from the description
            desc = description or context
            # Try to extract a session/topic from the description
            session_match = re.search(r'session on [\"\']?([^\"\']+)[\"\']?', desc, re.IGNORECASE)
            if session_match:
                session_title = session_match.group(1).strip()
                if not is_bad_title(session_title):
                    return f"Session Recording: {session_title}", desc
            # Fallback: use first 10 words of description
            words = desc.split()
            if words:
                fallback_title = "Session Recording: " + " ".join(words[:10])
                if not is_bad_title(fallback_title):
                    return fallback_title, desc
            # Last fallback for docs: label by type
            if "document" in domain or "docs" in domain:
                return "Google Doc", desc
            if "drive" in domain:
                return "Google Drive Resource", desc
            if "onedrive" in domain:
                return "OneDrive Resource", desc

    # --- Fallback: domain-aware label ---
    if url:
        parsed = urlparse(url)
        domain = parsed.netloc.replace('www.', '')
        domain_labels = {
            'wsj.com': 'WSJ Article',
            'axios.com': 'Axios Article',
            'reuters.com': 'Reuters Article',
            'forbes.com': 'Forbes Article',
            'arxiv.org': 'arXiv Paper',
            'pewresearch.org': 'Pew Research Report',
            'digitalocean.com': 'DigitalOcean Guide',
            'tortoisemedia.com': 'Tortoise Media Report',
            'microsoft.com': 'Microsoft Blog',
            'medium.com': 'Medium Article',
            'substack.com': 'Substack Post',
            'linkedin.com': 'LinkedIn Post',
            'youtube.com': 'YouTube Video',
            'youtu.be': 'YouTube Video',
            'bloomberg.com': 'Bloomberg Article',
            'nytimes.com': 'NYT Article',
            'bbc.com': 'BBC Article',
            'cnn.com': 'CNN Article',
            'theguardian.com': 'Guardian Article',
            'washingtonpost.com': 'Washington Post Article',
            'nature.com': 'Nature Article',
            'sciencemag.org': 'Science Magazine Article',
            'github.com': 'GitHub Repository',
            'docs.google.com': 'Google Doc',
            'drive.google.com': 'Google Drive Resource',
            'onedrive.live.com': 'OneDrive Resource',
        }
        for d, label in domain_labels.items():
            if d in domain:
                return label, description or url
        # fallback: use domain as label
        if domain:
            return domain.title() + " Resource", description or url
    # --- Last resort ---
    return "Resource", description or url

def normalize_url(url: str) -> str:
    """Strip query parameters, fragments, and trailing punctuation from a URL."""
    if not url:
        return None
    try:
        parsed = urlparse(url)
        parsed = parsed._replace(query='', fragment='')
        clean = urlunparse(parsed).rstrip(') ').strip()
        return clean
    except Exception:
        return url.strip()

def normalize_title(title: str) -> str:
    """Lowercase, remove non-alphanumeric chars, and collapse whitespace in titles."""
    if not title:
        return None
    t = title.lower()
    t = re.sub(r'[^a-z0-9\s]', '', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t

def deduplicate_resources(resources):
    """
    Remove duplicates by normalized URL (canonicalized), normalized title, and normalized description.
    For Google Drive/docs, treat all links to the same file as the same resource.
    If all else fails, deduplicate on normalized description.
    Also, merge resources with the same id, author, date, channel, and tag, and highly similar title/description, even if URLs differ.
    Log merges to data/resources/resource_merge.log.
    """
    import re
    from urllib.parse import urlparse
    from difflib import SequenceMatcher
    import os
    merge_log_path = os.path.join(os.path.dirname(__file__), '../data/resources/resource_merge.log')
    os.makedirs(os.path.dirname(merge_log_path), exist_ok=True)
    def log_merge(item1, item2, reason):
        with open(merge_log_path, 'a') as f:
            f.write(f"MERGE: {item1.get('id')} | {item1.get('title')} <==> {item2.get('title')} | Reason: {reason}\n")
    def fuzzy_sim(a, b):
        if not a or not b:
            return 0.0
        return SequenceMatcher(None, a, b).ratio()
    seen = set()
    unique = []
    for res in resources:
        is_dup = False
        for u in unique:
            # Fuzzy match on title and description
            title_sim = SequenceMatcher(None, (res.get('name') or '').lower(), (u.get('name') or '').lower()).ratio()
            desc_sim = SequenceMatcher(None, (res.get('description') or '').lower(), (u.get('description') or '').lower()).ratio()
            if title_sim > 0.92 and desc_sim > 0.85:
                is_dup = True
                break
        if not is_dup:
            unique.append(res)
    return unique

def is_bad_title(title):
    if not title:
        return True
    title = title.strip()
    # URL or domain
    if re.match(r'https?://', title) or re.match(r'^[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', title):
        return True
    # Contains a URL or domain substring
    if re.search(r'https?://|www\.|\.[a-z]{2,}', title):
        return True
    # Just a slug (all lowercase, dashes/underscores, no spaces)
    if ('-' in title or '_' in title) and title.lower() == title and ' ' not in title:
        return True
    # Looks like a hash/UUID
    if re.match(r'^[a-f0-9\-]{10,}$', title):
        return True
    # Mostly non-alphabetic
    if len(re.sub(r'[^a-zA-Z]', '', title)) < 3:
        return True
    # Fallback: if title is too short or generic
    if title.lower() in ("resource", "untitled", "index"):
        return True
    return False

if __name__ == "__main__":
    import os
    import sqlite3
    import random

    DB_PATH = os.path.join(os.path.dirname(__file__), '../data/discord_messages.db')
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # Try to get all messages from a likely table name
    try:
        cur.execute("SELECT * FROM messages")
    except Exception:
        cur.execute("SELECT * FROM discord_messages")

    rows = cur.fetchall()
    sample_rows = random.sample(rows, min(30, len(rows)))

    class DictToObj:
        def __init__(self, d):
            for k, v in d.items():
                setattr(self, k, v)

    all_resources = []
    for row in sample_rows:
        msg_obj = DictToObj(dict(row))
        if not hasattr(msg_obj, 'channel'):
            class DummyChannel:
                def __init__(self, name, topic=None):
                    self.name = name
                    self.topic = topic
            channel_name = getattr(msg_obj, 'channel_name', None) or getattr(msg_obj, 'channel', None)
            channel_topic = getattr(msg_obj, 'channel_topic', None)
            msg_obj.channel = DummyChannel(channel_name, channel_topic)
        if hasattr(msg_obj, 'author') and isinstance(msg_obj.author, dict):
            msg_obj.author = DictToObj(msg_obj.author)
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
        all_resources.extend(detect_resources(msg_obj))

    print(json.dumps(all_resources, indent=2, default=str))