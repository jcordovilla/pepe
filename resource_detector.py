import re
import json
from datetime import datetime
from typing import List, Dict, Any
from classifier import classify_resource  # <-- Import the classifier

# Example: from time_parser import parse_time  # Uncomment if you want to use your time parser

URL_REGEX = re.compile(
    r'(https?://[^\s]+)'
)

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

    # Extract URLs from message content
    content = getattr(message, "content", "")
    urls = URL_REGEX.findall(content)
    for url in urls:
        resource = {
            "url": url,
            "type": get_resource_type(url),
            "timestamp": normalize_timestamp(getattr(message, "timestamp", None)),
            "author": getattr(message, "author", None).name if getattr(message, "author", None) else None,
            "channel": getattr(message, "channel", None).name if getattr(message, "channel", None) else None,
            "message_id": getattr(message, "id", None),
            "context_snippet": content[:200]
        }
        resource["tag"] = classify_resource(resource)
        resources.append(resource)

    # Include attachments as resources
    attachments = getattr(message, "attachments", [])
    for att in attachments:
        url = getattr(att, "url", None)
        if url:
            resource = {
                "url": url,
                "type": get_resource_type(url),
                "timestamp": normalize_timestamp(getattr(message, "timestamp", None)),
                "author": getattr(message, "author", None).name if getattr(message, "author", None) else None,
                "channel": getattr(message, "channel", None).name if getattr(message, "channel", None) else None,
                "message_id": getattr(message, "id", None),
                "context_snippet": content[:200]
            }
            resource["tag"] = classify_resource(resource)
            resources.append(resource)

    return resources

if __name__ == "__main__":
    # Mock Discord message objects for testing
    class MockAuthor:
        def __init__(self, name, bot=False, system=False):
            self.name = name
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
            author=MockAuthor("alice"),
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