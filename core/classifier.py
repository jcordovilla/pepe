import re
import hashlib
from functools import lru_cache
from typing import Dict, Optional
from core.ai_client import get_ai_client

ai_client = get_ai_client()

# Cache for classification results to avoid re-classifying similar resources
@lru_cache(maxsize=1000)
def _classify_by_url_pattern(url: str, resource_type: str, context_hash: str) -> str:
    """
    Cached classification based on URL patterns and context.
    Uses URL and context hash to determine if we've seen this pattern before.
    """
    return _regex_classify(url.lower(), resource_type, context_hash)

def _regex_classify(url: str, resource_type: str, context_hash: str) -> str:
    """Internal regex-based classification logic."""
    # News domains - expanded list for better coverage
    news_domains = [
        "nytimes.com", "bbc.com", "cnn.com", "reuters.com", "theguardian.com", "washingtonpost.com",
        "bloomberg.com", "forbes.com", "wsj.com", "nbcnews.com", "abcnews.go.com", "cbsnews.com",
        "npr.org", "aljazeera.com", "apnews.com", "news.ycombinator.com", "medium.com", "substack.com",
        "thetimes.com", "ft.com", "politico.com", "axios.com", "wired.com", "techcrunch.com", 
        "engadget.com", "arstechnica.com", "nature.com", "sciencemag.org", "hackernews.com"
    ]
    
    # Enhanced pattern matching with priority order
    if any(x in url for x in [".pdf", "arxiv.org", "researchgate.net", "scholar.google", "acm.org", "ieee.org"]):
        return "Paper"
    if any(x in url for x in ["github.com", "gitlab.com", "bitbucket.org"]) and not url.endswith('/'):
        return "Tool"  # Repositories, not user profiles
    if any(x in url for x in ["docs.google.com", "notion.so", "gitbook.io", "confluence"]):
        return "Tutorial"
    if any(x in url for x in ["tutorial", "howto", "guide", "documentation", "learn", "course"]):
        return "Tutorial"
    if any(x in url for x in ["eventbrite", "meetup", "conference", "webinar", "event", "summit"]):
        return "Event"
    if any(x in url for x in ["jobs", "job", "opportunity", "careers", "hiring", "recruiter"]):
        return "Job/Opportunity"
    if any(domain in url for domain in news_domains):
        return "News/Article"
    if resource_type in ["video", "youtube"] or "youtube.com/watch" in url or "youtu.be" in url:
        return "Tutorial"
    if any(x in url for x in ["tool", "app", "software", "platform", "service"]):
        return "Tool"
    
    return "Other"

def classify_resource(resource: dict, use_llm: bool = True) -> str:
    """
    Classify a resource dict into one of:
    Tool, Paper, Tutorial, Event, Job/Opportunity, News/Article, Other.
    Uses local AI if use_llm is True, else regex-based fallback with caching.
    """
    tags = ["Tool", "Paper", "Tutorial", "Event", "Job/Opportunity", "News/Article", "Other"]
    
    # Extract resource info
    url = resource.get('url', '')
    resource_type = resource.get('type', '')
    context = resource.get('context_snippet', '')
    
    # Create context hash for caching (first 100 chars to avoid huge cache keys)
    context_short = context[:100] if context else ''
    context_hash = hashlib.md5(context_short.encode()).hexdigest()[:8]

    # For batch processing, prefer regex classification for speed
    # unless explicitly requesting LLM
    if not use_llm:
        return _classify_by_url_pattern(url, resource_type, context_hash)

    # LLM classification (slower but potentially more accurate)
    if use_llm and ai_client:
        system_prompt = (
            "You are an assistant that classifies shared links into categories. "
            "The possible tags are: Tool, Paper, Tutorial, Event, Job/Opportunity, News/Article, Other. "
            "Assign the most appropriate tag based on the URL, file type, author, and context snippet. "
            "If the resource is a news article, blog post, or media article, use 'News/Article'. "
            "Respond with just the tag name, nothing else."
        )
        user_content = (
            f"URL: {url}\n"
            f"Type: {resource_type}\n"
            f"Context: {context_short}\n"
        )
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ]
            
            response = ai_client.chat_completion(
                messages,
                temperature=0.0,
                max_tokens=20  # Reduced tokens since we just need the tag
            )
            
            # Extract and normalize the tag
            tag = response.strip()
            for t in tags:
                if t.lower() in tag.lower():
                    return t
            
            # Fallback heuristics
            if any(x in tag.lower() for x in ["news", "article", "blog"]):
                return "News/Article"
                
        except Exception as e:
            # Silently fall back to regex classification
            pass
    
    # Fallback to regex classification
    return _classify_by_url_pattern(url, resource_type, context_hash)

if __name__ == "__main__":
    # Example resources for testing
    test_resources = [
        {
            "url": "https://arxiv.org/abs/1234.5678",
            "type": "pdf",
            "author": "alice",
            "channel": "papers",
            "context_snippet": "Check out this new research paper on transformers."
        },
        {
            "url": "https://github.com/example/tool",
            "type": "github",
            "author": "bob",
            "channel": "tools",
            "context_snippet": "A new open-source tool for data analysis."
        },
        {
            "url": "https://example.com/webinar",
            "type": "link",
            "author": "carol",
            "channel": "events",
            "context_snippet": "Join our upcoming webinar on AI ethics."
        },
        {
            "url": "https://company.com/careers",
            "type": "link",
            "author": "dave",
            "channel": "jobs",
            "context_snippet": "We're hiring for several engineering roles!"
        },
        {
            "url": "https://example.com/video.mp4",
            "type": "video",
            "author": "eve",
            "channel": "media",
            "context_snippet": "Tutorial: How to use our platform."
        },
        {
            "url": "https://randomsite.com/interesting-link",
            "type": "link",
            "author": "frank",
            "channel": "general",
            "context_snippet": "Just sharing something interesting."
        }
    ]

    for res in test_resources:
        tag = classify_resource(res, use_llm=False)
        print(f"Resource: {res['url']}\n  Classified as: {tag}\n")