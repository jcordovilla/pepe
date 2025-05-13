import re
import json
from datetime import datetime
from typing import List, Dict, Any
from core.classifier import classify_resource  # <-- Import the classifier

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

def ai_vet_resource(resource: dict, log_decision: bool = False) -> bool:
    """
    Use OpenAI to vet if a resource is valuable (not a meeting, internal, or spam).
    Returns True if valuable, False if not or on error.
    If log_decision is True, print the AI's answer and reason for each resource.
    """
    from openai import OpenAI
    import os
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("GPT_MODEL", "gpt-4-turbo")
    openai_client = OpenAI(api_key=api_key)
    system_prompt = (
        "You are a filter for a knowledge resource library. "
        "Given a link and its context, decide if it is a valuable resource (e.g., paper, tool, tutorial, news article, blog) "
        "or if it is a meeting invite, internal navigation, or irrelevant. "
        "Reply with 'Yes' if valuable, 'No' if not, and a brief reason."
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
        response = openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            max_tokens=30,
            temperature=0
        )
        answer = response.choices[0].message.content.strip()
        if log_decision:
            print(f"AI vetting for resource: {resource.get('url') or '[no url]'} => {answer}")
        return answer.lower().startswith('yes')
    except Exception as e:
        print(f"AI vetting error: {e}. Defaulting to False.")
        return False

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
    for url in urls:
        if is_trash_url(url):
            continue  # Skip trash links
        resource = {
            "url": url,
            "type": get_resource_type(url),
            "timestamp": format_timestamp(getattr(message, "timestamp", None)),
            "author": get_author_display_name(getattr(message, "author", None)),
            "channel": getattr(message, "channel", None).name if getattr(message, "channel", None) else None,
            "jump_url": get_jump_url(message),
            "context_snippet": context_snippet
        }
        # Always accept known news domains (skip AI vetting for these)
        from core.classifier import classify_resource
        news_domains = [
            "nytimes.com", "bbc.com", "cnn.com", "reuters.com", "theguardian.com", "washingtonpost.com",
            "bloomberg.com", "forbes.com", "wsj.com", "nbcnews.com", "abcnews.go.com", "cbsnews.com",
            "npr.org", "aljazeera.com", "apnews.com", "news.ycombinator.com", "medium.com", "substack.com",
            "blog", "news", "article", "thetimes.com", "ft.com", "bloomberg.com", "politico.com", "axios.com",
            "wired.com", "techcrunch.com", "engadget.com", "arstechnica.com", "nature.com", "sciencemag.org"
        ]
        if any(domain in url for domain in news_domains):
            resource["tag"] = classify_resource(resource)
            resources.append(resource)
            continue
        # Loosen AI vetting: accept if AI says 'yes' OR if the resource type is 'github', 'pdf', 'youtube', 'drive', or 'video'
        auto_accept_types = ["github", "pdf", "youtube", "drive", "video"]
        if resource["type"] in auto_accept_types:
            resource["tag"] = classify_resource(resource)
            resources.append(resource)
            continue
        if not ai_vet_resource(resource, log_decision=True):
            continue
        resource["tag"] = classify_resource(resource)
        resources.append(resource)

    # --- IMPROVEMENT: Attachment handling ---
    attachments = getattr(message, "attachments", [])
    for att in attachments:
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
                "context_snippet": context_snippet
            }
            if not ai_vet_resource(resource):
                continue
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
            # AI vetting: skip if not valuable
            if ai_vet_resource(resource):
                resource["tag"] = classify_resource(resource)
                resources.append(resource)

    return resources

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