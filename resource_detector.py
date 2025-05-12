import re
import json
from datetime import datetime
from typing import List, Dict, Any
from classifier import classify_resource  # <-- Import the classifier

# Example: from time_parser import parse_time  # Uncomment if you want to use your time parser

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

def ai_vet_resource(resource: dict) -> bool:
    """
    Use OpenAI to vet if a resource is valuable (not a meeting, internal, or spam).
    Returns True if valuable, False if not or on error.
    """
    from openai import OpenAI
    import os
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("GPT_MODEL", "gpt-4-turbo")
    openai_client = OpenAI(api_key=api_key)
    system_prompt = (
        "You are a filter for a knowledge resource library. "
        "Given a link and its context, decide if it is a valuable resource (e.g., paper, tool, tutorial) "
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
            max_tokens=10,
            temperature=0
        )
        answer = response.choices[0].message.content.strip().lower()
        return answer.startswith('yes')
    except Exception as e:
        print(f"AI vetting error: {e}. Defaulting to False.")
        return False

# Helper to determine resource type from URL or filename
def get_resource_type(url: str) -> str:
    if url.endswith('.pdf'):
        return 'pdf'
    if url.endswith('.mp4'):
        return 'video'
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

    # Extract URLs from message content
    content = getattr(message, "content", "")
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
            "context_snippet": content[:200]
        }
        # AI vetting: skip if not valuable
        if not ai_vet_resource(resource):
            continue
        resource["tag"] = classify_resource(resource)
        resources.append(resource)

    # Include attachments as resources
    attachments = getattr(message, "attachments", [])
    for att in attachments:
        url = getattr(att, "url", None)
        if url and not is_trash_url(url):
            resource = {
                "url": url,
                "type": get_resource_type(url),
                "timestamp": format_timestamp(getattr(message, "timestamp", None)),
                "author": get_author_display_name(getattr(message, "author", None)),
                "channel": getattr(message, "channel", None).name if getattr(message, "channel", None) else None,
                "jump_url": get_jump_url(message),
                "context_snippet": content[:200]
            }
            if not ai_vet_resource(resource):
                continue
            resource["tag"] = classify_resource(resource)
            resources.append(resource)

    return resources

if __name__ == "__main__":
    # Mock Discord message objects for testing
    class MockAuthor:
        def __init__(self, name, display_name=None, global_name=None, nick=None, bot=False, system=False):
            self.name = name
            self.display_name = display_name
            self.global_name = global_name
            self.nick = nick
            self.bot = bot
            self.system = system

    class MockChannel:
        def __init__(self, name):
            self.name = name

    class MockAttachment:
        def __init__(self, url):
            self.url = url

    class MockMessage:
        def __init__(self, content, author, channel, id, timestamp, attachments=None):
            self.content = content
            self.author = author
            self.channel = channel
            self.id = id
            self.timestamp = timestamp
            self.attachments = attachments or []

    # Example messages
    messages = [
        MockMessage(
            content="Check this out: https://github.com/user/repo and this PDF: https://example.com/file.pdf",
            author=MockAuthor("alice", display_name="Alice D.", global_name="AliceGlobal"),
            channel=MockChannel("general"),
            id=123,
            timestamp=str(datetime.now()),
            attachments=[MockAttachment("https://example.com/image.png")]
        ),
        MockMessage(
            content="I'm a bot, ignore me.",
            author=MockAuthor("botty", bot=True),
            channel=MockChannel("bots"),
            id=124,
            timestamp=str(datetime.now())
        ),
        MockMessage(
            content="Here's a video: https://videos.com/clip.mp4",
            author=MockAuthor("bob"),
            channel=MockChannel("media"),
            id=125,
            timestamp=str(datetime.now())
        ),
    ]

    all_resources = []
    for msg in messages:
        all_resources.extend(detect_resources(msg))

    print(json.dumps(all_resources, indent=2, default=str))