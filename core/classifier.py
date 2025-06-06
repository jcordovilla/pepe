import re
from core.ai_client import get_ai_client

ai_client = get_ai_client()

def classify_resource(resource: dict, use_llm: bool = True) -> str:
    """
    Classify a resource dict into one of:
    Tool, Paper, Tutorial, Event, Job/Opportunity, News/Article, Other.
    Uses local AI if use_llm is True, else regex-based fallback.
    """
    tags = ["Tool", "Paper", "Tutorial", "Event", "Job/Opportunity", "News/Article", "Other"]

    if use_llm:
        system_prompt = (
            "You are an assistant that classifies shared links into categories. "
            "The possible tags are: Tool, Paper, Tutorial, Event, Job/Opportunity, News/Article, Other. "
            "Assign the most appropriate tag based on the URL, file type, author, and context snippet. "
            "If the resource is a news article, blog post, or media article, use 'News/Article'."
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
            
            response = ai_client.chat_completion(
                messages,
                temperature=0.0,
                max_tokens=50
            )
            
            # Extract the tag from the assistant's reply
            tag = response.strip()
            # Normalize to one of the allowed tags
            for t in tags:
                if t.lower() in tag.lower():
                    return t
            # Heuristic: if 'news' or 'article' in tag, return News/Article
            if any(x in tag.lower() for x in ["news", "article", "blog"]):
                return "News/Article"
            return "Other"
        except Exception as e:
            print(f"Local AI error: {e}. Falling back to regex.")
            # Fallback to regex if API fails

    # Regex-based fallback
    url = resource.get("url", "").lower()
    context = resource.get("context_snippet", "").lower()
    news_domains = [
        "nytimes.com", "bbc.com", "cnn.com", "reuters.com", "theguardian.com", "washingtonpost.com",
        "bloomberg.com", "forbes.com", "wsj.com", "nbcnews.com", "abcnews.go.com", "cbsnews.com",
        "npr.org", "aljazeera.com", "apnews.com", "news.ycombinator.com", "medium.com", "substack.com",
        "blog", "news", "article", "thetimes.com", "ft.com", "bloomberg.com", "politico.com", "axios.com",
        "wired.com", "techcrunch.com", "engadget.com", "arstechnica.com", "nature.com", "sciencemag.org"
    ]
    if any(x in url for x in [".pdf", "arxiv.org", "researchgate.net"]):
        return "Paper"
    if any(x in url for x in ["github.com", "tool", "app", "software"]):
        return "Tool"
    if any(x in url for x in ["tutorial", "howto", "guide", "docs"]):
        return "Tutorial"
    if any(x in url for x in ["eventbrite", "meetup", "conference", "webinar", "event"]):
        return "Event"
    if any(x in url for x in ["jobs", "job", "opportunity", "careers", "hiring"]):
        return "Job/Opportunity"
    if any(domain in url for domain in news_domains) or any(word in context for word in ["news", "article", "blog", "newsletter", "press release"]):
        return "News/Article"
    if "video" in resource.get("type", ""):
        return "Tutorial"
    return "Other"

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