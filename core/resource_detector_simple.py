import re
import json
from datetime import datetime
from typing import List, Dict, Any
from core.classifier import classify_resource
from urllib.parse import urlparse, urlunparse

URL_REGEX = re.compile(r'(https?://[^\s]+)')

TRASH_PATTERNS = [
    r'discord\.com/channels/',
    r'discord\.com/events/',
    r'zoom\.us',
    r'meet\.google\.com',
    r'teams\.microsoft\.com',
    r'webex\.com',
    r'gotomeeting\.com',
    r'calendar\.google\.com',
    r'calendar\.outlook',
    r'facebook\.com/events',
    r'\.(png|jpg|jpeg|gif|webp|svg)$',
    r'cdn\.discordapp\.com',
    r'giphy\.com',
    r'tenor\.com',
]

def is_trash_url(url):
    return any(re.search(pat, url) for pat in TRASH_PATTERNS)

def simple_vet_resource(resource: dict) -> dict:
    """Simplified resource vetting using basic URL patterns."""
    url = resource.get('url', '')
    context = resource.get('context_snippet', '')
    
    # Basic valuable resource patterns
    valuable_patterns = [
        'arxiv.org', 'github.com', 'youtube.com', 'youtu.be',
        'medium.com', 'substack.com', 'papers', 'tutorial',
        'docs.', 'documentation', 'research', 'blog',
        'news', 'article', 'report', 'bbc.com', 'cnn.com',
        'nytimes.com', 'reuters.com', 'bloomberg.com'
    ]
    
    # Skip obvious trash
    if not url or is_trash_url(url):
        return {"is_valuable": False, "name": None, "description": None}
    
    # Check if URL or context contains valuable patterns
    is_valuable = any(pattern in url.lower() or pattern in context.lower() 
                     for pattern in valuable_patterns)
    
    if is_valuable:
        # Generate simple name and description
        parsed = urlparse(url)
        domain = parsed.netloc.replace('www.', '')
        name = f"{domain.title()} Resource"
        description = context[:100] + "..." if len(context) > 100 else context
        return {"is_valuable": True, "name": name, "description": description}
    
    return {"is_valuable": False, "name": None, "description": None}

def simple_enrich_title(resource):
    """Simplified title generation using URL structure and context."""
    url = resource.get("url", "")
    context = resource.get("context_snippet", "")
    
    if url:
        parsed = urlparse(url)
        domain = parsed.netloc.replace('www.', '')
        
        # Domain-based title generation
        domain_labels = {
            'arxiv.org': 'arXiv Paper',
            'github.com': 'GitHub Repository', 
            'youtube.com': 'YouTube Video',
            'youtu.be': 'YouTube Video',
            'medium.com': 'Medium Article',
            'substack.com': 'Substack Post',
            'linkedin.com': 'LinkedIn Post',
            'bloomberg.com': 'Bloomberg Article',
            'nytimes.com': 'NYT Article',
            'bbc.com': 'BBC Article',
            'cnn.com': 'CNN Article',
            'reuters.com': 'Reuters Article',
            'theguardian.com': 'Guardian Article',
            'washingtonpost.com': 'Washington Post Article',
            'wsj.com': 'WSJ Article',
            'axios.com': 'Axios Article',
            'forbes.com': 'Forbes Article',
            'docs.google.com': 'Google Doc',
            'drive.google.com': 'Google Drive Resource',
        }
        
        # Generate title based on domain
        for d, label in domain_labels.items():
            if d in domain:
                desc = context[:100] + "..." if len(context) > 100 else context
                return label, desc
        
        # Generic domain-based title
        if domain:
            title = f"{domain.title()} Resource"
            desc = context[:100] + "..." if len(context) > 100 else context
            return title, desc
    
    # Fallback: use first line of context as title
    if context:
        lines = context.split('\n')
        for line in lines:
            line = line.strip()
            if len(line) > 10 and not line.startswith('http'):
                title = line[:80]
                desc = context[:150] + "..." if len(context) > 150 else context
                return title, desc
    
    # Last resort
    return "Resource", context or "No description available"

def get_resource_type(url: str) -> str:
    """Simple resource type detection."""
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
    if url.endswith(('.png', '.jpg', '.jpeg', '.gif')):
        return 'image'
    if url.endswith('.zip'):
        return 'archive'
    return 'link'

def detect_resources(message) -> List[Dict[str, Any]]:
    """Simplified resource detection."""
    # Skip bot messages
    if getattr(message, "author", None) and getattr(message.author, "bot", False):
        return []

    resources = []
    content = getattr(message, "content", "")
    
    # Helper functions
    def get_author_name(author):
        if not author:
            return None
        if isinstance(author, str):
            return author
        return getattr(author, 'display_name', None) or getattr(author, 'username', None) or str(author)
    
    def format_timestamp(ts):
        if hasattr(ts, 'strftime'):
            return ts.strftime('%Y-%m-%d %H:%M')
        return str(ts) if ts else None
    
    def get_jump_url(msg):
        if hasattr(msg, 'jump_url') and getattr(msg, 'jump_url', None):
            return msg.jump_url
        # Try to build from known fields
        guild_id = getattr(msg, 'guild_id', None)
        channel_id = getattr(msg, 'channel_id', None)
        message_id = getattr(msg, 'id', None)
        if guild_id and channel_id and message_id:
            return f"https://discord.com/channels/{guild_id}/{channel_id}/{message_id}"
        return None
    
    # Context snippet
    context_snippet = content[:500]
    
    # Extract URLs from message content
    urls = URL_REGEX.findall(content)
    for idx, url in enumerate(urls):
        if is_trash_url(url):
            continue
            
        resource = {
            "url": url,
            "type": get_resource_type(url),
            "timestamp": format_timestamp(getattr(message, "timestamp", None)),
            "author": get_author_name(getattr(message, "author", None)),
            "channel": getattr(message, "channel", None).name if getattr(message, "channel", None) else None,
            "jump_url": get_jump_url(message),
            "context_snippet": context_snippet,
            "message_id": getattr(message, "id", None),
            "resource_id": f"{getattr(message, 'id', 'noid')}-{idx+1}"
        }
        
        # Simple vetting
        vet_result = simple_vet_resource(resource)
        if not vet_result["is_valuable"]:
            continue
            
        # Simple title generation
        title, desc = simple_enrich_title(resource)
        resource["name"] = title
        resource["description"] = desc
        resource["tag"] = classify_resource(resource, use_llm=False)  # Use regex-only classification
        
        resources.append(resource)
    
    # Handle attachments
    attachments = getattr(message, "attachments", [])
    for aidx, att in enumerate(attachments):
        url = getattr(att, "url", None)
        if url and not is_trash_url(url):
            filename = getattr(att, "filename", "")
            resource = {
                "url": url,
                "type": get_resource_type(url),
                "timestamp": format_timestamp(getattr(message, "timestamp", None)),
                "author": get_author_name(getattr(message, "author", None)),
                "channel": getattr(message, "channel", None).name if getattr(message, "channel", None) else None,
                "jump_url": get_jump_url(message),
                "context_snippet": context_snippet,
                "message_id": getattr(message, "id", None),
                "resource_id": f"{getattr(message, 'id', 'noid')}-att{aidx+1}"
            }
            
            vet_result = simple_vet_resource(resource)
            if vet_result["is_valuable"]:
                title, desc = simple_enrich_title(resource)
                resource["name"] = filename if filename else title
                resource["description"] = desc
                resource["tag"] = classify_resource(resource, use_llm=False)
                resources.append(resource)
    
    return resources

def deduplicate_resources(resources):
    """Simple deduplication based on URL and title similarity."""
    from difflib import SequenceMatcher
    
    unique = []
    for res in resources:
        is_dup = False
        for u in unique:
            # Check URL similarity
            if res.get('url') == u.get('url'):
                is_dup = True
                break
            
            # Check title similarity
            title_sim = SequenceMatcher(None, 
                                     (res.get('name') or '').lower(), 
                                     (u.get('name') or '').lower()).ratio()
            if title_sim > 0.9:
                is_dup = True
                break
                
        if not is_dup:
            unique.append(res)
    
    return unique

# Aliases for backward compatibility
ai_vet_resource = simple_vet_resource
ai_enrich_title_description = simple_enrich_title
needs_title_fix = lambda resource: not resource.get("name")
