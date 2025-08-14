#!/usr/bin/env python3
"""
Fresh Resource Detector - Quality-First Approach

Extracts only high-quality resources from Discord messages,
filtering out junk and focusing on valuable content.
"""

import json
import re
from pathlib import Path
from collections import Counter, defaultdict
from urllib.parse import urlparse
from typing import Dict, List, Any
import sys
from datetime import datetime
import requests
import os
from tqdm import tqdm
import argparse

# Configure tqdm for better progress bar visibility
tqdm.monitor_interval = 0.05  # Update more frequently

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

class FreshResourceDetector:
    def __init__(self, use_fast_model: bool = True):
        # Get configuration for model selection
        from agentic.config.modernized_config import get_modernized_config
        config = get_modernized_config()
        llm_config = config.get("llm", {})
        
        # Model selection for resource detection - use config from .env
        self.fast_model = llm_config.get("fast_model", "phi3:mini")  # Fast, lightweight model
        self.standard_model = llm_config.get("model", "llama3.1:8b")  # Higher quality model from .env
        self.use_fast_model = use_fast_model
        
        # Resource storage and statistics
        self.detected_resources = []
        from collections import defaultdict
        self.stats = defaultdict(int)
        self.stats.update({
            'total_resources': 0,
            'excluded_domains': 0,
            'parsing_errors': 0,
            'internal_documents': 0,
            'ai_analysis_used': 0,
            'ai_analysis_failed': 0,
            'duplicates_detected': 0,
            'content_duplicates_detected': 0,
            'skipped_processed': 0,
            'bot_messages_filtered': 0
        })
        
        # Incremental processing setup
        project_root = Path(__file__).parent.parent
        self.processed_urls_file = project_root / 'data' / 'processed_resources.json'
        self.resource_checkpoint_file = project_root / 'data' / 'resource_checkpoint.json'
        self.processed_urls = self._load_processed_urls()
        self.resource_checkpoint = self._load_resource_checkpoint()
        
        # URL normalization cache for better duplicate detection
        self.normalized_urls_cache = {}
        
        # Content similarity detection
        self.content_similarity_detector = None
        self.similarity_threshold = 0.8
        
        # Simplified category mapping
        self.category_mapping = {
            # Research & Academic
            'arxiv.org': 'Paper',
            'papers.withcode.com': 'Paper',
            'distill.pub': 'Paper',
            'news.mit.edu': 'News/Article',
            'research.google.com': 'Paper',
            
            # Code & Development
            'github.com': 'Tool',
            'stackoverflow.com': 'Tutorial',
            
            # Documentation & Resources
            'docs.google.com': 'Tutorial',
            'drive.google.com': 'Tutorial',
            'cloud.google.com': 'Tutorial',
            
            # AI & ML Resources
            'openai.com': 'Tool',
            'blog.openai.com': 'News/Article',
            'anthropic.com': 'Tool',
            'huggingface.co': 'Tool',
            'tensorflow.org': 'Tutorial',
            'pytorch.org': 'Tutorial',
            'ai.googleblog.com': 'News/Article',
            'nvidia.com': 'News/Article',
            'blogs.microsoft.com': 'News/Article',
            'chatgpt.com': 'Tool',
            
            # Educational Platforms
            'deeplearning.ai': 'Tutorial',
            'coursera.org': 'Tutorial',
            'edx.org': 'Tutorial',
            'udacity.com': 'Tutorial',
            'fast.ai': 'Tutorial',
            'machinelearningmastery.com': 'Tutorial',
            
            # Data Science & Analytics
            'kaggle.com': 'Tool',
            'towardsdatascience.com': 'News/Article',
            
            # Video Content
            'youtube.com': 'Tutorial',
            'youtu.be': 'Tutorial',
            
            # Articles & Blogs
            'medium.com': 'News/Article',
            
            # News & Tech Publications
            'axios.com': 'News/Article',
            'theguardian.com': 'News/Article',
            'reuters.com': 'News/Article',
            'wsj.com': 'News/Article',
            'ft.com': 'News/Article',
            'businessinsider.com': 'News/Article',
            'theverge.com': 'News/Article',
            'techcrunch.com': 'News/Article',
            'venturebeat.com': 'News/Article',
            'wired.com': 'News/Article',
            'arstechnica.com': 'News/Article',
        }
        
        # Domains to exclude (junk and internal documents)
        self.excluded_domains = {
            # Discord and social media
            'cdn.discordapp.com', 'discord.com/channels', 'discordapp.com',
            'tenor.com', 'giphy.com', 'discord.gg',
            'linkedin.com/in', 'linkedin.com/posts',  # Profile links and posts
            'twitter.com/i/', 'facebook.com', 'instagram.com',
            
            # Meeting and communication tools
            'meet.google.com', 'zoom.us', 'us06web.zoom.us', 'mit.zoom.us',
            'fathom.video',  # Meeting recordings
            'teams.microsoft.com', 'teams.live.com',
            
            # URL shorteners (hard to verify quality)
            'tinyurl.com', 'bit.ly', 't.co', 'goo.gl',
            
            # Internal/private community documents
            'docs.google.com',  # Google Docs/Sheets/Slides (usually private)
            'drive.google.com',  # Google Drive (usually private)
            'app.mural.co',  # Mural boards (usually private)
            'trello.com',  # Trello boards (usually private)
            'notion.so',  # Notion pages (usually private)
            'figma.com',  # Figma designs (usually private)
            'miro.com',  # Miro boards (usually private)
            'whimsical.com',  # Whimsical boards (usually private)
            'lucidchart.com',  # Lucidchart (usually private)
            'draw.io',  # Draw.io diagrams (usually private)
            'canva.com',  # Canva designs (usually private)
            'slides.com',  # Slides.com presentations (usually private)
            'prezi.com',  # Prezi presentations (usually private)
            'padlet.com',  # Padlet boards (usually private)
            'jamboard.google.com',  # Google Jamboard (usually private)
            'whiteboard.microsoft.com',  # Microsoft Whiteboard (usually private)
            
            # Video/audio recordings (usually private)
            'youtube.com/live', 'youtube.com/watch?v=', 'youtu.be',
            'vimeo.com', 'dailymotion.com',
            'spotify.com', 'soundcloud.com',
            
            # Specific app links
            'sync-google-calendar-wit-erprmym.gamma.site'
        }
        
        # File extensions that indicate quality resources
        self.quality_extensions = {
            '.pdf': 0.8, '.docx': 0.7, '.doc': 0.7, '.pptx': 0.7,
            '.py': 0.6, '.ipynb': 0.8, '.md': 0.6, '.tex': 0.7
        }
        
        self.unknown_domains = Counter()  # Track unknown domains
        self.unknown_samples = defaultdict(list)  # Sample URLs for each unknown domain
    
    def _load_processed_urls(self) -> set:
        """Load previously processed URLs for incremental processing"""
        if self.processed_urls_file.exists():
            try:
                with open(self.processed_urls_file, 'r') as f:
                    data = json.load(f)
                    return set(data.get('processed_urls', []))
            except Exception as e:
                print(f"⚠️ Warning: Could not load processed URLs: {e}")
        return set()
    
    def _save_processed_urls(self):
        """Save processed URLs for future incremental processing"""
        try:
            # Ensure data directory exists
            self.processed_urls_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Get all processed URLs (including newly detected ones)
            all_urls = self.processed_urls.union(
                {resource['url'] for resource in self.detected_resources}
            )
            
            # Also save normalized URLs for better duplicate detection
            normalized_urls = {self._normalize_url(url) for url in all_urls}
            
            with open(self.processed_urls_file, 'w') as f:
                json.dump({
                    'processed_urls': list(all_urls),
                    'normalized_urls': list(normalized_urls),
                    'last_updated': datetime.now().isoformat(),
                    'total_processed': len(all_urls),
                    'total_normalized': len(normalized_urls)
                }, f, indent=2)
        except Exception as e:
            print(f"⚠️ Warning: Could not save processed URLs: {e}")
    
    def _load_resource_checkpoint(self) -> Dict[str, int]:
        """Load the resource checkpoint to track processed messages."""
        if self.resource_checkpoint_file.exists():
            try:
                with open(self.resource_checkpoint_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️ Warning: Could not load resource checkpoint: {e}")
        return {}
    
    def _save_resource_checkpoint(self):
        """Save the resource checkpoint to track processed messages."""
        try:
            self.resource_checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.resource_checkpoint_file, 'w') as f:
                json.dump(self.resource_checkpoint, f, indent=2)
        except Exception as e:
            print(f"⚠️ Warning: Could not save resource checkpoint: {e}")
    
    def _is_url_processed(self, url: str) -> bool:
        """Check if a URL has already been processed"""
        return url in self.processed_urls
    
    def _normalize_url(self, url: str) -> str:
        """Normalize URL to handle common variations that should be treated as the same resource"""
        if url in self.normalized_urls_cache:
            return self.normalized_urls_cache[url]
        
        try:
            from urllib.parse import urlparse, parse_qs
            
            parsed = urlparse(url)
            
            # Remove common tracking parameters
            query_params = parse_qs(parsed.query)
            filtered_params = {}
            
            for key, values in query_params.items():
                # Keep important parameters, remove tracking ones
                if key.lower() not in ['utm_source', 'utm_medium', 'utm_campaign', 'utm_term', 'utm_content', 
                                     'ref', 'source', 'fbclid', 'gclid', 'msclkid']:
                    filtered_params[key] = values
            
            # Rebuild query string
            new_query = '&'.join([f"{k}={v[0]}" for k, v in filtered_params.items() if v])
            
            # Normalize domain
            domain = parsed.netloc.lower()
            if domain.startswith('www.'):
                domain = domain[4:]
            
            # Handle YouTube URL variations
            if 'youtube.com' in domain or 'youtu.be' in domain:
                if 'youtu.be' in domain:
                    # Convert youtu.be to youtube.com format
                    video_id = parsed.path[1:]  # Remove leading slash
                    normalized = f"https://youtube.com/watch?v={video_id}"
                else:
                    # Keep youtube.com format but normalize
                    normalized = f"https://youtube.com{parsed.path}"
                    if new_query:
                        normalized += f"?{new_query}"
            else:
                # Rebuild URL for other domains
                normalized = f"{parsed.scheme}://{domain}{parsed.path}"
                if new_query:
                    normalized += f"?{new_query}"
                if parsed.fragment:
                    normalized += f"#{parsed.fragment}"
            
            self.normalized_urls_cache[url] = normalized
            return normalized
            
        except Exception:
            # Fallback to simple normalization
            normalized = url.lower().strip()
            self.normalized_urls_cache[url] = normalized
            return normalized
    
    def _is_normalized_url_processed(self, url: str) -> bool:
        """Check if a normalized URL has already been processed"""
        normalized_url = self._normalize_url(url)
        
        # Check against existing processed URLs from file
        for processed_url in self.processed_urls:
            if self._normalize_url(processed_url) == normalized_url:
                return True
        
        # Check against URLs already processed in current session
        for resource in self.detected_resources:
            if self._normalize_url(resource['url']) == normalized_url:
                return True
        
        return False
    
    def _is_message_processed(self, message_id: str) -> bool:
        """Check if a message has already been processed based on its ID."""
        return message_id in self.resource_checkpoint
    
    def _check_content_similarity(self, resource: Dict[str, Any]) -> bool:
        """Check if a resource is similar to existing ones based on content"""
        if not self.content_similarity_detector or not self.detected_resources:
            return False
        
        try:
            # Find similar resources
            similar = self.content_similarity_detector.find_similar_resources(resource, top_k=1)
            if similar and similar[0]['similarity'] >= self.similarity_threshold:
                self.stats['content_duplicates_detected'] += 1
                return True
        except Exception as e:
            # If similarity check fails, continue without it
            pass
        
        return False
    
    def analyze_discord_messages(self, messages_dir: Path, analyze_unknown: bool = False) -> List[Dict[str, Any]]:
        """Analyze Discord messages and extract high-quality resources"""
        
        print("🔍 Fresh Resource Detection - Quality-First Approach")
        print("=" * 60)
        
        json_files = list(messages_dir.glob('*.json'))
        print(f"📁 Found {len(json_files)} message files to analyze")
        
        # Progress bar for file processing
        with tqdm(json_files, desc="📄 Processing files", unit="file", position=0, leave=True) as file_pbar:
            for json_file in file_pbar:
                file_pbar.set_postfix({"file": json_file.name[:30] + "..." if len(json_file.name) > 30 else json_file.name})
                
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    messages = data.get('messages', [])
                    channel_name = data.get('channel_name', 'Unknown')
                    
                    # Progress bar for message processing within each file
                    with tqdm(messages, desc=f"  📝 Messages in {channel_name}", unit="msg", position=1, leave=False) as msg_pbar:
                        for message in msg_pbar:
                            self._analyze_message(message, channel_name, analyze_unknown)
                            msg_pbar.set_postfix({"resources": len(self.detected_resources)})
                    
                except Exception as e:
                    print(f"   ❌ Error processing {json_file.name}: {e}")
        
        # Progress bar for description generation
        if self.detected_resources:
            print(f"\n🤖 Generating AI descriptions for {len(self.detected_resources)} resources...")
            with tqdm(self.detected_resources, desc="🤖 Generating descriptions", unit="resource", position=0, leave=True) as desc_pbar:
                for i, resource in enumerate(desc_pbar):
                    # Re-generate description with progress tracking
                    message = {'content': f"Resource: {resource['url']}", 'author': {'username': resource['author']}}
                    description = self._generate_description(message, resource['url'])
                    self.detected_resources[i]['description'] = description
                    desc_pbar.set_postfix({"domain": resource['domain'][:20]})
        
        # Save processed URLs for incremental processing
        self._save_processed_urls()
        self._save_resource_checkpoint() # Save checkpoint after processing all messages
        
        return self._generate_report(analyze_unknown)
    
    def _is_internal_document(self, url: str, domain: str, message: Dict[str, Any]) -> bool:
        """Check if a URL is an internal/private community document that should be excluded"""
        content = message.get('content', '').lower()
        url_lower = url.lower()
        
        # Step 1: Rule-based filtering for obvious internal documents
        if self._is_obviously_internal(url, domain, content):
            return True
        
        # Step 2: AI-powered analysis for ambiguous cases
        if self._needs_ai_analysis(url, domain, content):
            return self._ai_analyze_resource_privacy(url, domain, message)
        
        # Step 3: Content-based heuristics for edge cases
        return self._content_based_filtering(url, domain, content)
    
    def _is_obviously_internal(self, url: str, domain: str, content: str) -> bool:
        """Rule-based filtering for obviously internal documents"""
        url_lower = url.lower()
        
        # Check for public indicators first - if present, don't filter out
        public_indicators = [
            'tutorial', 'guide', 'how-to', 'educational', 'learning',
            'public', 'open', 'shared', 'published', 'available',
            'resource', 'reference', 'documentation', 'example',
            'demo', 'showcase', 'presentation', 'talk', 'lecture'
        ]
        
        if any(indicator in content for indicator in public_indicators):
            return False  # Don't filter out if it has public indicators
        
        # Definitely internal patterns (no AI needed)
        definitely_internal_patterns = [
            # Google Workspace private patterns
            '/document/d/', '/spreadsheets/d/', '/presentation/d/',
            '/forms/d/', '/drawings/d/', '/sites/d/',
            '/drive/folders/', '/drive/u/', '/drive/d/',
            
            # Collaboration tools with private patterns
            '/t/', '/m/', '/s/',  # Mural
            '/b/', '/c/', '/card/',  # Trello
            '/page/', '/database/', '/workspace/',  # Notion
            '/file/', '/proto/', '/design/',  # Figma
            '/board/', '/app/',  # Miro
            
            # Video recordings
            '/live/', '/embed/', '/v/',  # Live streams
            'youtu.be',  # YouTube short URLs (usually recordings)
        ]
        
        if any(pattern in url_lower for pattern in definitely_internal_patterns):
            return True
        
        # Content indicators that are definitely internal
        definitely_internal_keywords = [
            'internal', 'private', 'confidential', 'draft', 'working',
            'team meeting', 'recording', 'session', 'workshop',
            'kanban', 'planning', 'brainstorming', 'whiteboard',
            'mindmap', 'flowchart', 'wireframe', 'prototype', 'mockup'
        ]
        
        if any(keyword in content for keyword in definitely_internal_keywords):
            return True
        
        return False
    
    def _needs_ai_analysis(self, url: str, domain: str, content: str) -> bool:
        """Determine if a resource needs AI analysis for privacy assessment"""
        url_lower = url.lower()
        
        # Domains that might have both public and private content
        ambiguous_domains = [
            'docs.google.com', 'drive.google.com',  # Google Workspace
            'youtube.com', 'vimeo.com',  # Video platforms
            'medium.com', 'substack.com',  # Publishing platforms
            'github.com', 'gitlab.com',  # Code repositories
            'huggingface.co', 'paperswithcode.com',  # AI/ML platforms
        ]
        
        # URL patterns that are ambiguous
        ambiguous_patterns = [
            '/watch?v=',  # YouTube videos (could be public tutorials)
            '/channel/',  # YouTube channels (could be public)
            '/playlist/',  # YouTube playlists (could be public)
            '/c/',  # YouTube custom URLs (could be public)
            '/user/',  # Various platforms (could be public)
            '/@',  # Modern social media handles (could be public)
        ]
        
        # Content indicators that suggest ambiguity
        ambiguous_keywords = [
            'tutorial', 'guide', 'how-to', 'educational', 'learning',
            'public', 'open', 'shared', 'published', 'available',
            'resource', 'reference', 'documentation', 'example',
            'demo', 'showcase', 'presentation', 'talk', 'lecture'
        ]
        
        # Check if domain is ambiguous
        is_ambiguous_domain = any(amb_domain in domain for amb_domain in ambiguous_domains)
        
        # Check if URL pattern is ambiguous
        is_ambiguous_pattern = any(pattern in url_lower for pattern in ambiguous_patterns)
        
        # Check if content suggests public value
        has_public_indicators = any(keyword in content for keyword in ambiguous_keywords)
        
        return is_ambiguous_domain and (is_ambiguous_pattern or has_public_indicators)
    
    def _ai_analyze_resource_privacy(self, url: str, domain: str, message: Dict[str, Any]) -> bool:
        """Use AI to analyze if a resource is private/internal or public/valuable"""
        content = message.get('content', '')
        
        # Create AI prompt for privacy analysis
        prompt = f"""
Analyze this resource to determine if it should be included in a public resource database.

URL: {url}
Domain: {domain}
Message Context: {content[:200]}

Consider:
1. Is this a private/internal document (team meetings, internal planning, confidential)?
2. Is this a public educational resource (tutorials, guides, open documentation)?
3. Is this valuable for the broader AI/tech community?
4. Would sharing this publicly violate privacy or confidentiality?

Respond with ONLY "PRIVATE" or "PUBLIC" based on your analysis.
"""
        
        try:
            llm_endpoint = os.getenv('LLM_ENDPOINT', 'http://localhost:11434/api/generate')
            llm_model = self.fast_model if self.use_fast_model else self.standard_model
            
            response = requests.post(llm_endpoint, json={
                "model": llm_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,  # Low temperature for consistent decisions
                    "top_p": 0.9,
                    "num_predict": 10,  # Short response
                    "stop": ["\n", " ", "."]  # Stop at first word
                }
            }, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                result = data.get('response', '').strip().upper()
                
                # Track AI decisions for analysis
                self.stats['ai_analyzed'] = self.stats.get('ai_analyzed', 0) + 1
                if result == 'PRIVATE':
                    self.stats['ai_filtered'] = self.stats.get('ai_filtered', 0) + 1
                
                return result == 'PRIVATE'
            else:
                # Fallback to conservative approach if AI fails
                print(f"⚠️ AI analysis failed for {url[:50]}..., using fallback")
                return self._conservative_fallback(url, domain, content)
                
        except Exception as e:
            # Fallback to conservative approach if AI fails
            print(f"⚠️ AI analysis error for {url[:50]}...: {e}")
            return self._conservative_fallback(url, domain, content)
    
    def _conservative_fallback(self, url: str, domain: str, content: str) -> bool:
        """Conservative fallback when AI analysis fails"""
        # When in doubt, be conservative and filter out
        # This prevents accidentally exposing private content
        
        # But allow some obvious public resources through
        public_indicators = [
            'tutorial', 'guide', 'how-to', 'educational', 'learning',
            'public', 'open', 'shared', 'published', 'available',
            'resource', 'reference', 'documentation', 'example'
        ]
        
        if any(indicator in content for indicator in public_indicators):
            return False  # Allow through if it has public indicators
        
        return True  # Filter out by default (conservative)
    
    def _content_based_filtering(self, url: str, domain: str, content: str) -> bool:
        """Content-based heuristics for edge cases"""
        # Check for collaboration tools with internal indicators
        collaboration_domains = [
            'mural.co', 'trello.com', 'notion.so', 'figma.com', 
            'miro.com', 'whimsical.com', 'lucidchart.com', 'canva.com'
        ]
        
        if any(collab_domain in domain for collab_domain in collaboration_domains):
            # Additional content check for collaboration tools
            internal_collab_keywords = [
                'team', 'meeting', 'recording', 'session', 'workshop',
                'board', 'project', 'planning', 'brainstorming',
                'collaboration', 'whiteboard', 'mindmap', 'flowchart'
            ]
            
            if any(keyword in content for keyword in internal_collab_keywords):
                return True
        
        return False

    def _analyze_message(self, message: Dict[str, Any], channel_name: str, analyze_unknown: bool = False):
        """Analyze a single message for resources"""
        content = message.get('content', '')
        if not content:
            return
        
        # Extract URLs from message content
        urls = re.findall(r'https?://[^\s\n\r<>"\']+', content)
        
        for url in urls:
            resource = self._evaluate_url(url, message, channel_name, analyze_unknown)
            if resource:
                self.detected_resources.append(resource)
                self.stats['total_resources'] += 1
                self.stats[f'category_{resource["category"]}'] += 1
    
    def _generate_description(self, message, url):
        """Generate an AI description for the resource using the Llama model."""
        llm_endpoint = os.getenv('LLM_ENDPOINT', 'http://localhost:11434/api/generate')
        llm_model = self.fast_model if self.use_fast_model else self.standard_model
        
        # Extract key content and context for better description generation
        content = message.get('content', '')[:300]  # Increase content length for better context
        domain = urlparse(url).netloc.lower()
        parsed_url = urlparse(url)
        
        # Create more sophisticated prompt based on domain and URL structure
        if 'arxiv.org' in domain:
            prompt_context = f"This is a research paper from arXiv (URL: {url}). Context from message: {content}\n\nGenerate a clear, informative description of this research paper including its key contributions and domain. Keep it under 80 words:"
        elif 'github.com' in domain:
            repo_path = parsed_url.path.strip('/')
            prompt_context = f"This is a GitHub repository: {repo_path} (URL: {url}). Context: {content}\n\nDescribe this code repository, its purpose, and what it implements. Focus on the technical aspects. Keep it under 80 words:"
        elif 'huggingface.co' in domain:
            prompt_context = f"This is a resource from Hugging Face (URL: {url}). Context: {content}\n\nDescribe this AI/ML model, dataset, or tool from Hugging Face, including its capabilities and use cases. Keep it under 80 words:"
        elif 'youtube.com' in domain or 'youtu.be' in domain:
            prompt_context = f"This is a YouTube video (URL: {url}). Context: {content}\n\nDescribe this educational video content and what viewers can learn from it. Keep it under 80 words:"
        elif any(news_domain in domain for news_domain in ['techcrunch.com', 'theverge.com', 'wired.com', 'arstechnica.com']):
            prompt_context = f"This is a tech news article from {domain} (URL: {url}). Context: {content}\n\nSummarize this tech news article and its key points. Keep it under 80 words:"
        elif '.pdf' in url.lower():
            prompt_context = f"This is a PDF document (URL: {url}). Context: {content}\n\nDescribe this document and its likely content based on the context. Keep it under 80 words:"
        else:
            # Generic but more informative prompt
            prompt_context = f"Resource URL: {url}\nDomain: {domain}\nContext from message: {content}\n\nBased on the URL and context, provide a clear, informative description of this resource and its value. Keep it under 80 words:"
        
        try:
            response = requests.post(llm_endpoint, json={
                "model": llm_model,
                "prompt": prompt_context,
                "stream": False,
                "options": {
                    "temperature": 0.2,  # Slightly higher for more natural language
                    "top_p": 0.9,
                    "num_predict": 100,  # Ollama uses num_predict instead of max_tokens
                    "stop": ["\n\n", "---"]  # Stop tokens to prevent rambling
                }
            }, timeout=20)  # Increase timeout for better quality
            
            if response.status_code == 200:
                data = response.json()
                description = data.get('response', '').strip()
                
                # Clean up the description
                if description:
                    # Remove common prefixes that the model might add
                    prefixes_to_remove = [
                        "This is a ", "This appears to be ", "Based on the URL", 
                        "According to the context", "The resource is"
                    ]
                    for prefix in prefixes_to_remove:
                        if description.startswith(prefix):
                            description = description[len(prefix):].lstrip()
                    
                    # Ensure it doesn't exceed reasonable length
                    if len(description) > 300:
                        description = description[:297] + "..."
                    
                    return description
                else:
                    return self._generate_fallback_description(url, domain, content)
            else:
                print(f"⚠️ LLM API error (status {response.status_code}) for {url[:50]}...")
                return self._generate_fallback_description(url, domain, content)
                
        except Exception as e:
            print(f"⚠️ LLM connection error for {url[:50]}...: {e}")
            return self._generate_fallback_description(url, domain, content)
    
    def _generate_fallback_description(self, url, domain, content):
        """Generate a fallback description when LLM is unavailable."""
        parsed_url = urlparse(url)
        
        # Create intelligent fallback descriptions based on URL patterns
        if 'arxiv.org' in domain:
            paper_id = parsed_url.path.split('/')[-1] if parsed_url.path else "unknown"
            return f"Research paper from arXiv - {paper_id}"
        elif 'github.com' in domain:
            repo_path = parsed_url.path.strip('/')
            return f"GitHub repository: {repo_path}"
        elif 'huggingface.co' in domain:
            path_parts = parsed_url.path.strip('/').split('/')
            if len(path_parts) >= 2:
                return f"Hugging Face {path_parts[0]}: {'/'.join(path_parts[1:])}"
            return "Hugging Face AI/ML resource"
        elif 'youtube.com' in domain or 'youtu.be' in domain:
            return "Educational video content from YouTube"
        elif '.pdf' in url.lower():
            filename = parsed_url.path.split('/')[-1] if parsed_url.path else "document"
            return f"PDF document: {filename}"
        elif any(news_domain in domain for news_domain in ['techcrunch.com', 'theverge.com', 'wired.com']):
            return f"Tech news article from {domain}"
        else:
            # Extract meaningful info from URL structure
            if parsed_url.path and len(parsed_url.path) > 1:
                path_parts = [p for p in parsed_url.path.split('/') if p]
                if path_parts:
                    return f"Resource from {domain}: {'/'.join(path_parts[:2])}"
            return f"Resource from {domain}"

    def _evaluate_url(self, url: str, message: Dict[str, Any], channel_name: str, analyze_unknown: bool = False) -> Dict[str, Any]:
        """Evaluate if a URL is a high-quality resource"""
        # Check if URL has already been processed (exact match)
        if self._is_url_processed(url):
            self.stats['skipped_processed'] += 1
            return None
        
        # Check if normalized URL has already been processed (duplicate detection)
        if self._is_normalized_url_processed(url):
            self.stats['duplicates_detected'] += 1
            return None
            
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            if domain.startswith('www.'):
                domain = domain[4:]
            
            # Check for excluded domains
            if any(excluded in domain for excluded in self.excluded_domains):
                self.stats['excluded_domains'] += 1
                return None
            
            # Check for internal/private community documents
            if self._is_internal_document(url, domain, message):
                self.stats['excluded_domains'] += 1
                return None
            
            # Determine category using simplified mapping
            category = None
            for mapped_domain, mapped_category in self.category_mapping.items():
                if mapped_domain in domain:
                    category = mapped_category
                    break
            
            if not category:
                # Check for file extensions
                path = parsed.path.lower()
                for ext in self.quality_extensions.keys():
                    if path.endswith(ext):
                        category = 'Tutorial'  # Documents are typically tutorials
                        break
                
                if not category:
                    self.stats['unknown_domains'] += 1
                    if analyze_unknown:
                        self.unknown_domains[domain] += 1
                        if len(self.unknown_samples[domain]) < 3:
                            self.unknown_samples[domain].append({
                                'url': url,
                                'channel': channel_name,
                                'author': message.get('author', {}).get('username', 'Unknown')
                            })
                    return None
            
            # Extract author and timestamp
            author_display = message.get('author', {}).get('display_name') or message.get('author', {}).get('username', 'Unknown')
            
            # Format timestamp as YYYY-MM-DD
            raw_ts = message.get('timestamp')
            try:
                ts = datetime.fromisoformat(raw_ts)
                ts_str = ts.strftime('%Y-%m-%d')
            except Exception:
                ts_str = raw_ts[:10] if raw_ts else ''
            
            # Extract jump_url from message
            jump_url = message.get('jump_url')
            
            # Generate description
            description = self._generate_description(message, url)
            
            resource = {
                'url': url,
                'domain': domain,
                'category': category,
                'channel_name': channel_name,
                'author': author_display,
                'timestamp': ts_str,
                'jump_url': jump_url,
                'description': description
            }
            
            # Check for content similarity with existing resources
            if self._check_content_similarity(resource):
                return None
            
            return resource
        except Exception as e:
            self.stats['parsing_errors'] += 1
            return None
    
    def _generate_report(self, analyze_unknown: bool = False) -> Dict[str, Any]:
        """Generate comprehensive analysis report"""
        
        # Sort resources by category
        sorted_resources = sorted(self.detected_resources, key=lambda x: x['category'])
        
        # Generate statistics
        category_stats = Counter([r['category'] for r in sorted_resources])
        domain_stats = Counter([r['domain'] for r in sorted_resources])
        channel_stats = Counter([r['channel_name'] for r in sorted_resources])
        
        # Calculate category distribution
        category_distribution = {
            'News/Article': len([r for r in sorted_resources if r['category'] == 'News/Article']),
            'Tool': len([r for r in sorted_resources if r['category'] == 'Tool']),
            'Paper': len([r for r in sorted_resources if r['category'] == 'Paper']),
            'Tutorial': len([r for r in sorted_resources if r['category'] == 'Tutorial']),
            'Event': len([r for r in sorted_resources if r['category'] == 'Event']),
            'Job/Opportunity': len([r for r in sorted_resources if r['category'] == 'Job/Opportunity'])
        }
        
        print("\n📊 Fresh Resource Detection Results")
        print("=" * 40)
        print(f"✅ High-quality resources found: {len(sorted_resources)}")
        print(f"⏭️ Skipped (already processed): {self.stats.get('skipped_processed', 0)}")
        print(f"❌ Excluded (low-quality + internal docs): {self.stats['excluded_domains']}")
        print(f"❓ Unknown domains skipped: {self.stats['unknown_domains']}")
        print(f"⚠️ Parsing errors: {self.stats['parsing_errors']}")
        
        # Show AI analysis statistics
        ai_analyzed = self.stats.get('ai_analyzed', 0)
        ai_filtered = self.stats.get('ai_filtered', 0)
        if ai_analyzed > 0:
            ai_kept = ai_analyzed - ai_filtered
            print(f"🤖 AI analysis: {ai_analyzed} ambiguous resources analyzed")
            print(f"   📊 AI decisions: {ai_kept} kept public, {ai_filtered} filtered private")
        
        print(f"🔒 Smart filtering: Rule-based + AI analysis for privacy protection")
        
        # Show incremental processing info
        if self.processed_urls:
            print(f"📈 Total URLs processed (all time): {len(self.processed_urls)}")
            print(f"🚀 Incremental processing: Enabled")
        
        if analyze_unknown and self.unknown_domains:
            print(f"\n🔍 Unknown Domains Analysis ({len(self.unknown_domains)} unique domains):")
            print("=" * 50)
            
            # Show top unknown domains
            print("📊 Top Unknown Domains:")
            for domain, count in self.unknown_domains.most_common(20):
                print(f"   {domain}: {count} URLs")
                
                # Show sample URLs and context
                samples = self.unknown_samples[domain]
                for i, sample in enumerate(samples[:2]):  # Show first 2 samples
                    print(f"      {i+1}. {sample['url'][:70]}...")
                    print(f"         Channel: {sample['channel']} | Author: {sample['author']}")
                print()
            
            # Analyze domain patterns
            self._analyze_domain_patterns()
        
        print("\n📁 Resource Categories:")
        for category, count in category_stats.most_common():
            print(f"   {category}: {count} resources")
        
        print("\n🌐 Top Domains:")
        for domain, count in domain_stats.most_common(10):
            print(f"   {domain}: {count} resources")
        
        print("\n📺 Top Channels:")
        for channel, count in channel_stats.most_common(5):
            print(f"   {channel}: {count} resources")
        
        print("\n📁 Category Distribution:")
        for category, count in category_distribution.items():
            if count > 0:
                print(f"   {category}: {count} resources")
        
        print("\n📄 Top 10 Resources:")
        for i, resource in enumerate(sorted_resources[:10]):
            print(f"   {i+1}. [{resource['category']}] {resource['url']}")
            print(f"      Author: {resource['author']} | Channel: {resource['channel_name']}")
            print(f"      Description: {resource['description'][:80]}...")
            print()
        
        return {
            'resources': sorted_resources,
            'statistics': {
                'total_found': len(sorted_resources),
                'excluded_count': self.stats['excluded_domains'],
                'unknown_count': self.stats['unknown_domains'],
                'categories': dict(category_stats),
                'domains': dict(domain_stats),
                'channels': dict(channel_stats),
                'category_distribution': category_distribution
            },
            'unknown_domains': dict(self.unknown_domains) if analyze_unknown else {}
        }
    
    def _analyze_domain_patterns(self):
        """Analyze patterns in unknown domains to suggest additions"""
        print("\n🧠 Domain Pattern Analysis:")
        
        # Educational institutions
        edu_domains = [d for d in self.unknown_domains.keys() if '.edu' in d or 'university' in d or 'mit.' in d or 'stanford.' in d]
        if edu_domains:
            print(f"   📚 Educational institutions ({len(edu_domains)}): {', '.join(edu_domains[:5])}...")
        
        # Government/research
        gov_domains = [d for d in self.unknown_domains.keys() if '.gov' in d or '.org' in d and ('research' in d or 'institute' in d)]
        if gov_domains:
            print(f"   🏛️ Government/Research ({len(gov_domains)}): {', '.join(gov_domains[:5])}...")
        
        # Tech companies
        tech_domains = [d for d in self.unknown_domains.keys() if any(tech in d for tech in ['microsoft', 'google', 'amazon', 'meta', 'nvidia', 'anthropic'])]
        if tech_domains:
            print(f"   💻 Tech companies ({len(tech_domains)}): {', '.join(tech_domains[:5])}...")
        
        # News/blogs
        news_domains = [d for d in self.unknown_domains.keys() if any(news in d for news in ['.com', 'blog', 'news', 'techcrunch', 'venturebeat'])]
        high_count_news = [d for d in news_domains if self.unknown_domains[d] >= 3]  # Domains with 3+ mentions
        if high_count_news:
            print(f"   📰 News/Blogs with multiple mentions ({len(high_count_news)}): {', '.join(high_count_news[:5])}...")
        
        # PDF hosts
        pdf_domains = [d for d in self.unknown_domains.keys() if any(pdf in d for pdf in ['assets.', 'cdn.', 'storage.']) and any(url.endswith('.pdf') for url in [s['url'] for s in self.unknown_samples[d]])]
        if pdf_domains:
            print(f"   📄 PDF hosts ({len(pdf_domains)}): {', '.join(pdf_domains[:5])}...")
    
    def _ensure_resources_table_exists(self, conn):
        """Ensure the resources table and indexes exist in the SQLite DB."""
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS resources (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT,
                description TEXT,
                date TEXT,
                author TEXT,
                channel TEXT,
                tag TEXT NOT NULL,
                resource_url TEXT UNIQUE NOT NULL,
                discord_url TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        # Create indexes for fast searching
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_tag ON resources(tag)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_author ON resources(author)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_channel ON resources(channel)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_date ON resources(date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON resources(created_at)")
        conn.commit()

    def save_resources(self, output_path: Path, analyze_unknown: bool = False):
        """Save detected resources to JSON file"""
        report = self._generate_report(analyze_unknown)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"💾 Resources saved to: {output_path}")
        return report

    def save_resources_to_db(self, db_path: Path):
        """Upsert all detected resources into the resources table in the given SQLite DB."""
        import sqlite3
        if not db_path.exists():
            print(f"❌ Resource DB not found: {db_path}")
            return
        try:
            with sqlite3.connect(db_path) as conn:
                self._ensure_resources_table_exists(conn)
                cursor = conn.cursor()
                inserted = 0
                updated = 0
                for resource in self.detected_resources:
                    # Generate title from URL or description
                    title = self._generate_title(resource)
                    
                    # Upsert using ON CONFLICT(url) DO UPDATE (old table structure)
                    cursor.execute(
                        """
                        INSERT INTO resources (url, domain, category, quality_score, channel_name, author, timestamp, jump_url, description, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                        ON CONFLICT(url) DO UPDATE SET
                            domain=excluded.domain,
                            category=excluded.category,
                            quality_score=excluded.quality_score,
                            channel_name=excluded.channel_name,
                            author=excluded.author,
                            timestamp=excluded.timestamp,
                            jump_url=excluded.jump_url,
                            description=excluded.description,
                            updated_at=CURRENT_TIMESTAMP
                        """,
                        (
                            resource.get('url'),
                            resource.get('domain'),
                            resource.get('category'),
                            0.8,  # Default quality score
                            resource.get('channel_name'),
                            resource.get('author'),
                            resource.get('timestamp'),
                            resource.get('jump_url'),
                            resource.get('description'),
                        )
                    )
                    if cursor.rowcount == 1:
                        inserted += 1
                    else:
                        updated += 1
                conn.commit()
                print(f"\n💾 Saved {inserted} new resources, updated {updated} existing resources in DB: {db_path}")
        except sqlite3.OperationalError as e:
            if 'no such table' in str(e):
                print(f"❌ Table 'resources' does not exist in {db_path}. Please create it first.")
            else:
                print(f"❌ SQLite error: {e}")
        except Exception as e:
            print(f"❌ Error saving resources to DB: {e}")
    
    def _generate_title(self, resource: Dict[str, Any]) -> str:
        """Generate a title for the resource based on URL or description."""
        url = resource.get('url', '')
        description = resource.get('description', '')
        
        # Try to extract title from URL path
        try:
            parsed = urlparse(url)
            path_parts = [p for p in parsed.path.split('/') if p]
            
            if 'arxiv.org' in url and len(path_parts) >= 2:
                # For arXiv papers, use the paper ID as title
                return f"arXiv Paper: {path_parts[-1]}"
            elif 'github.com' in url and len(path_parts) >= 2:
                # For GitHub repos, use owner/repo as title
                return f"GitHub: {'/'.join(path_parts[:2])}"
            elif 'huggingface.co' in url and len(path_parts) >= 2:
                # For Hugging Face, use the model/dataset name
                return f"Hugging Face: {'/'.join(path_parts[:2])}"
            elif path_parts:
                # Use the last meaningful path component
                return path_parts[-1].replace('-', ' ').replace('_', ' ').title()
        except:
            pass
        
        # Fallback to first sentence of description
        if description:
            first_sentence = description.split('.')[0]
            if len(first_sentence) > 10:
                return first_sentence[:100] + ('...' if len(first_sentence) > 100 else '')
        
        # Final fallback
        return f"Resource from {resource.get('domain', 'unknown')}"
    
    def export_resources_from_db(self, db_path: Path) -> Dict[str, Any]:
        """Export resources from database in the new JSON structure."""
        import sqlite3
        
        if not db_path.exists():
            print(f"❌ Resource DB not found: {db_path}")
            return {"resources": []}
        
        try:
            with sqlite3.connect(db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("""
                    SELECT id, url, domain, category, quality_score, channel_name, author, timestamp, jump_url, description
                    FROM resources 
                    ORDER BY id ASC
                """)
                
                resources = []
                for i, row in enumerate(cursor, 1):  # Start enumeration from 1
                    # Generate title from URL or description
                    title = self._generate_title_from_url(row['url'], row['description'])
                    
                    # Map old category to new tag
                    tag = self._map_category_to_tag(row['category'])
                    
                    # Fix channel name encoding - just use as is since it's already correct
                    channel_name = row['channel_name'] or ''
                    
                    # Fix jump URL - check if it's actually null or empty
                    discord_url = row['jump_url']
                    if discord_url and discord_url.strip():
                        discord_url = discord_url
                    else:
                        discord_url = None
                    
                    resource = {
                        "id": i,  # Use incremental ID starting from 1
                        "title": title,
                        "description": row['description'],
                        "date": row['timestamp'],
                        "author": row['author'],
                        "channel": channel_name,
                        "tag": tag,
                        "resource_url": row['url'],
                        "discord_url": discord_url
                    }
                    resources.append(resource)
                
                return {
                    "export_date": datetime.now().isoformat(),
                    "total_resources": len(resources),
                    "resources": resources
                }
                
        except Exception as e:
            print(f"❌ Error exporting from DB: {e}")
            return {"resources": []}
    
    def _generate_title_from_url(self, url: str, description: str) -> str:
        """Generate a title from URL or description for export using LLM when possible."""
        try:
            parsed = urlparse(url)
            path_parts = [p for p in parsed.path.split('/') if p]
            
            if 'arxiv.org' in url and len(path_parts) >= 2:
                return f"arXiv Paper: {path_parts[-1]}"
            elif 'github.com' in url and len(path_parts) >= 2:
                return f"GitHub: {'/'.join(path_parts[:2])}"
            elif 'huggingface.co' in url and len(path_parts) >= 2:
                return f"Hugging Face: {'/'.join(path_parts[:2])}"
            elif 'youtube.com' in url or 'youtu.be' in url:
                # For YouTube videos, generate concise titles from description
                if description and len(description) > 20:
                    return self._extract_youtube_title(description)
                # Fallback for YouTube videos
                return "YouTube Video"
            elif description and len(description) > 30:
                # Use LLM for other resources with good descriptions
                return self._generate_llm_title(url, description)
            elif path_parts:
                return path_parts[-1].replace('-', ' ').replace('_', ' ').title()
        except:
            pass
        
        # Fallback to first sentence of description
        if description:
            first_sentence = description.split('.')[0]
            if len(first_sentence) > 10:
                return first_sentence[:100] + ('...' if len(first_sentence) > 100 else '')
        
        return f"Resource from {urlparse(url).netloc}"
    
    def _generate_llm_title(self, url: str, description: str) -> str:
        """Generate a title using LLM for non-YouTube resources."""
        try:
            llm_endpoint = os.getenv('LLM_ENDPOINT', 'http://localhost:11434/api/generate')
            llm_model = self.fast_model
            
            domain = urlparse(url).netloc
            
            prompt = f"""Given this resource description, generate a concise, informative title (max 50 characters):

URL: {url}
Domain: {domain}
Description: {description[:300]}

Generate a short, descriptive title that captures the main topic or subject. Focus on the key technology, tool, or concept being discussed. Keep it simple and direct.

Important: Do not include words like "This", "The", "A", "An" at the end. End with the main subject.

Title:"""
            
            response = requests.post(llm_endpoint, json={
                'model': llm_model,
                'prompt': prompt,
                'stream': False,
                'options': {
                    'temperature': 0.3,
                    'top_p': 0.9,
                    'max_tokens': 60
                }
            }, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                title = result.get('response', '').strip()
                
                # Clean up the title
                title = title.replace('"', '').replace("'", '').strip()
                
                # Remove newlines and extra whitespace
                title = re.sub(r'\n+', ' ', title)
                title = re.sub(r'\s+', ' ', title)
                
                # Remove markdown formatting
                title = re.sub(r'#{1,6}\s*', '', title)  # Remove headers
                title = re.sub(r'\*\*(.*?)\*\*', r'\1', title)  # Remove bold
                title = re.sub(r'\*(.*?)\*', r'\1', title)  # Remove italic
                title = re.sub(r'`(.*?)`', r'\1', title)  # Remove code
                
                # Clean up any remaining formatting
                title = title.strip()
                
                # Remove meaningless endings
                meaningless_endings = [' this', ' the', ' a', ' an', ' that', ' these', ' those']
                title_lower = title.lower()
                for ending in meaningless_endings:
                    if title_lower.endswith(ending):
                        title = title[:-len(ending)].strip()
                        break
                
                # Ensure it's not too long and has clean endings
                if len(title) > 50:
                    # Try to find a natural break point (space, comma, period)
                    for i in range(47, 40, -1):
                        if title[i] in ' .,;:':
                            title = title[:i].strip()
                            break
                    else:
                        # If no natural break found, truncate at word boundary
                        words = title[:47].split()
                        if len(words) > 1:
                            title = ' '.join(words[:-1])
                        else:
                            title = title[:47]
                
                # Ensure it's not empty or too short
                if len(title) > 5:
                    return title
            
        except Exception as e:
            print(f"⚠️ LLM title generation failed for {url}: {e}")
        
        # Fallback to rule-based extraction
        return self._generate_title_fallback(url, description)
    
    def _generate_title_fallback(self, url: str, description: str) -> str:
        """Fallback method for generating titles using rules."""
        try:
            parsed = urlparse(url)
            path_parts = [p for p in parsed.path.split('/') if p]
            
            if path_parts:
                return path_parts[-1].replace('-', ' ').replace('_', ' ').title()
        except:
            pass
        
        # Fallback to first sentence of description
        if description:
            first_sentence = description.split('.')[0]
            if len(first_sentence) > 10:
                return first_sentence[:100] + ('...' if len(first_sentence) > 100 else '')
        
        return f"Resource from {urlparse(url).netloc}"
    
    def _extract_youtube_title(self, description: str) -> str:
        """Extract a concise, meaningful title from YouTube video description using LLM."""
        if not description or len(description) < 20:
            return "YouTube Video"
        
        try:
            # Use the fast LLM to generate a concise title
            llm_endpoint = os.getenv('LLM_ENDPOINT', 'http://localhost:11434/api/generate')
            llm_model = self.fast_model
            
            prompt = f"""Given this YouTube video description, generate a concise, informative title (max 40 characters):

Description: {description[:300]}

Generate a short, descriptive title that captures the main topic or subject. Focus on the key technology, tool, or concept being discussed. Keep it simple and direct. Examples:
- "Python Tutorial" 
- "Discord Bot Setup"
- "AI Code Review"
- "Origami Crane Guide"

Important: Do not include words like "This", "The", "A", "An" at the end. End with the main subject.

Title:"""
            
            response = requests.post(llm_endpoint, json={
                'model': llm_model,
                'prompt': prompt,
                'stream': False,
                'options': {
                    'temperature': 0.3,
                    'top_p': 0.9,
                    'max_tokens': 50
                }
            }, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                title = result.get('response', '').strip()
                
                # Clean up the title
                title = title.replace('"', '').replace("'", '').strip()
                
                # Remove newlines and extra whitespace
                title = re.sub(r'\n+', ' ', title)
                title = re.sub(r'\s+', ' ', title)
                
                # Remove markdown formatting
                title = re.sub(r'#{1,6}\s*', '', title)  # Remove headers
                title = re.sub(r'\*\*(.*?)\*\*', r'\1', title)  # Remove bold
                title = re.sub(r'\*(.*?)\*', r'\1', title)  # Remove italic
                title = re.sub(r'`(.*?)`', r'\1', title)  # Remove code
                
                # Clean up any remaining formatting
                title = title.strip()
                
                # Remove meaningless endings
                meaningless_endings = [' this', ' the', ' a', ' an', ' that', ' these', ' those']
                title_lower = title.lower()
                for ending in meaningless_endings:
                    if title_lower.endswith(ending):
                        title = title[:-len(ending)].strip()
                        break
                
                # Ensure it's not too long and has clean endings
                if len(title) > 40:
                    # Try to find a natural break point (space, comma, period)
                    for i in range(37, 30, -1):
                        if title[i] in ' .,;:':
                            title = title[:i].strip()
                            break
                    else:
                        # If no natural break found, truncate at word boundary
                        words = title[:37].split()
                        if len(words) > 1:
                            title = ' '.join(words[:-1])
                        else:
                            title = title[:37]
                
                # Ensure it's not empty or too short
                if len(title) > 5:
                    return title
            
        except Exception as e:
            print(f"⚠️ LLM title generation failed: {e}")
        
        # Fallback to rule-based extraction
        return self._extract_youtube_title_fallback(description)
    
    def _extract_youtube_title_fallback(self, description: str) -> str:
        """Fallback method for extracting YouTube titles using rules."""
        import re
        
        # Remove common YouTube prefixes
        description = re.sub(r'^This YouTube (tutorial|video|interview|playlist)', '', description, flags=re.IGNORECASE)
        description = re.sub(r'^In the YouTube (tutorial|video|interview|playlist)', '', description, flags=re.IGNORECASE)
        description = re.sub(r'^In this YouTube (tutorial|video|interview|playlist)', '', description, flags=re.IGNORECASE)
        
        # Clean up the description
        description = description.strip()
        
        # Take first sentence and clean it up
        first_sentence = description.split('.')[0]
        
        # Remove common phrases that make titles verbose
        first_sentence = re.sub(r'guides users through the process of', 'tutorial on', first_sentence, flags=re.IGNORECASE)
        first_sentence = re.sub(r'demonstrates the process of', 'tutorial on', first_sentence, flags=re.IGNORECASE)
        first_sentence = re.sub(r'features? (a |an )?', '', first_sentence, flags=re.IGNORECASE)
        first_sentence = re.sub(r'discusses? (how |the |about )?', '', first_sentence, flags=re.IGNORECASE)
        first_sentence = re.sub(r'shares? (a |an )?', '', first_sentence, flags=re.IGNORECASE)
        first_sentence = re.sub(r'introduces? (a |an )?', '', first_sentence, flags=re.IGNORECASE)
        first_sentence = re.sub(r'explores? (the |about )?', '', first_sentence, flags=re.IGNORECASE)
        
        # Clean up extra whitespace and punctuation
        first_sentence = re.sub(r'\s+', ' ', first_sentence).strip()
        first_sentence = re.sub(r'^[,\s]+', '', first_sentence)
        
        # Extract key topic - look for specific technologies, tools, or concepts
        tech_patterns = [
            r'(Python|JavaScript|Java|C\+\+|React|Vue|Angular|Node\.js|Django|Flask|TensorFlow|PyTorch|OpenAI|ChatGPT|Discord|GitHub|Replit|Shopify|AI|ML|Machine Learning|Deep Learning|Data Science|Origami|Crane|TED|Codex|CLI)',
            r'(tutorial|guide|course|lesson|workshop|demo|introduction|overview|basics|fundamentals)',
            r'(creating|building|developing|implementing|setting up|configuring|deploying)'
        ]
        
        # Try to find a concise title based on key concepts
        for pattern in tech_patterns:
            matches = re.findall(pattern, first_sentence, flags=re.IGNORECASE)
            if matches:
                # Create a concise title from the key concepts
                key_concepts = [m for m in matches if len(m) > 2]  # Filter out short words
                if key_concepts:
                    # Take first 2-3 key concepts
                    title_parts = key_concepts[:3]
                    concise_title = ' '.join(title_parts)
                    if len(concise_title) > 10 and len(concise_title) < 50:
                        return concise_title.title()
        
        # Fallback: take first meaningful phrase (up to 40 chars)
        words = first_sentence.split()
        meaningful_words = [w for w in words if len(w) > 2 and not w.lower() in ['the', 'and', 'for', 'with', 'that', 'this', 'from', 'they', 'will', 'can', 'are', 'was', 'were', 'have', 'has', 'had', 'been', 'being']]
        
        if meaningful_words:
            concise_title = ' '.join(meaningful_words[:6])  # Take first 6 meaningful words
            if len(concise_title) > 10:
                return concise_title[:40] + ('...' if len(concise_title) > 40 else '')
        
        # Final fallback
        if len(first_sentence) > 10:
            return first_sentence[:40] + ('...' if len(first_sentence) > 40 else '')
        
        return "YouTube Video"
    
    def _map_category_to_tag(self, old_category: str) -> str:
        """Map old category names to new tag names."""
        category_mapping = {
            'Research Papers': 'Paper',
            'AI Research': 'Paper',
            'AI Resources': 'Tool',
            'AI Models': 'Tool',
            'Code Repositories': 'Tool',
            'Technical Q&A': 'Tutorial',
            'ML Documentation': 'Tutorial',
            'AI Education': 'Tutorial',
            'Online Courses': 'Tutorial',
            'Educational Videos': 'Tutorial',
            'Articles': 'News/Article',
            'Tech News': 'News/Article',
            'News & Analysis': 'News/Article',
            'Business News': 'News/Article',
            'Academic News': 'News/Article',
            'Documents': 'Tutorial',
            'Documentation': 'Tutorial',
            'Shared Documents': 'Tutorial',
            'AI/GPU Technology': 'News/Article',
            'AI Tools': 'Tool',
            'Tech Documentation': 'Tutorial',
            'Data Science': 'Tool',
            'Research Visualization': 'Paper',
            'Research': 'Paper'
        }
        
        return category_mapping.get(old_category, 'News/Article')

def main():
    parser = argparse.ArgumentParser(description='Fresh Resource Detector - Quality-First Approach')
    parser.add_argument('--fast-model', action='store_true', 
                       help='Use fast model (phi3:mini) for faster processing')
    parser.add_argument('--standard-model', action='store_true',
                       help='Use standard model (llama3.1:8b) for better quality')
    parser.add_argument('--reset-cache', action='store_true',
                       help='Reset processed URLs cache and reprocess all resources')
    args = parser.parse_args()

    use_fast_model = args.fast_model or (not args.standard_model)  # Default to fast model

    if args.reset_cache:
        cache_file = project_root / 'data' / 'processed_resources.json'
        checkpoint_file = project_root / 'data' / 'resource_checkpoint.json'
        if cache_file.exists():
            cache_file.unlink()
            print("🗑️ Reset processed URLs cache")
        if checkpoint_file.exists():
            checkpoint_file.unlink()
            print("🗑️ Reset resource checkpoint")

    detector = FreshResourceDetector(use_fast_model=use_fast_model)
    model_name = detector.fast_model if use_fast_model else detector.standard_model
    print(f"🤖 Using model: {model_name} ({'fast' if use_fast_model else 'standard'} mode)")

    db_path = project_root / 'data' / 'discord_messages.db'
    if not db_path.exists():
        print(f"❌ Message database not found: {db_path}")
        print("💡 Run 'pepe-admin sync' first to fetch Discord messages")
        return

    print("🔍 Running resource detection from SQLite database...")
    print("📊 Loading messages from database...")
    print("⏳ This may take a while - you'll see progress bars below...")
    print("-" * 60)
    
    import sqlite3
    messages = []
    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            # Get new messages only (after last processed message)
            last_processed_id = detector.resource_checkpoint.get('last_processed_message_id')
            
            if last_processed_id:
                print(f"🔄 Incremental processing from message {last_processed_id[:8]}...")
                cursor = conn.execute("""
                    SELECT * FROM messages 
                    WHERE content IS NOT NULL AND content != ''
                    AND message_id > ?
                    ORDER BY timestamp ASC
                """, [last_processed_id])
                
                count_cursor = conn.execute("""
                    SELECT COUNT(*) as count FROM messages 
                    WHERE content IS NOT NULL AND content != ''
                    AND message_id > ?
                """, [last_processed_id])
            else:
                print(f"📥 Full processing (no checkpoint found)...")
                cursor = conn.execute("""
                    SELECT * FROM messages 
                    WHERE content IS NOT NULL AND content != ''
                    ORDER BY timestamp ASC
                """)
                
                count_cursor = conn.execute("""
                    SELECT COUNT(*) as count FROM messages 
                    WHERE content IS NOT NULL AND content != ''
                """)
            
            new_messages = count_cursor.fetchone()['count']
            print(f"📊 Found {new_messages:,} new messages to analyze")
            
            # Load messages with progress bar
            with tqdm(total=new_messages, desc="📥 Loading messages", unit="msg", position=0, leave=True) as load_pbar:
                for row in cursor:
                    # Check if message is from a bot
                    is_bot = False
                    try:
                        raw_data = json.loads(row['raw_data']) if row['raw_data'] else {}
                        author_data = raw_data.get('author', {})
                        is_bot = author_data.get('bot', False)
                    except:
                        pass
                    
                    if is_bot:
                        detector.stats['bot_messages_filtered'] += 1
                        load_pbar.update(1)
                        continue
                    
                    message_dict = {
                        'content': row['content'],
                        'author': {
                            'username': row['author_username'],
                            'display_name': row['author_display_name'] or row['author_username']
                        },
                        'timestamp': row['timestamp'],
                        'message_id': row['message_id'],
                        'channel_id': row['channel_id'],
                        'channel_name': row['channel_name'],
                        'jump_url': row['jump_url']
                    }
                    messages.append(message_dict)
                    load_pbar.update(1)
                    load_pbar.set_postfix({"loaded": len(messages), "bots_filtered": detector.stats['bot_messages_filtered']})
                    
    except Exception as e:
        print(f"❌ Error reading database: {e}")
        return

    print(f"✅ Loaded {len(messages):,} messages successfully")
    
    if new_messages == 0:
        print("✅ No new messages to analyze - resource detection is up to date")
        return
    
    print("🔍 Starting resource analysis...")

    # Analyze messages with progress bar and incremental logic
    skipped = 0
    processed = 0
    resources_found = 0
    urls_extracted = 0
    last_processed_message_id = None
    
    print(f"\n🔍 Analyzing {len(messages):,} messages for high-quality resources...")
    print("📊 Progress indicators will show:")
    print("   • Message analysis progress")
    print("   • URL extraction and evaluation")
    print("   • Resource quality assessment")
    print("   • Real-time resource detection stats")
    print("-" * 60)
    
    # Initialize detailed tracking
    recent_resources = []
    domain_stats = defaultdict(int)
    category_stats = defaultdict(int)
    
    with tqdm(messages, desc="📝 Analyzing messages", unit="msg", position=0, leave=True) as msg_pbar:
        for i, message in enumerate(msg_pbar):
            content = message.get('content', '')
            message_id = message.get('message_id', '')
            channel_name = message.get('channel_name', 'Unknown')
            
            # Track the last processed message ID for checkpointing
            last_processed_message_id = message_id
            
            if not content:
                continue
                
            urls = re.findall(r'https?://[^\s\n\r<>"]+', content)
            urls_extracted += len(urls)
            new_url_found = False
            
            for url in urls:
                if detector._is_url_processed(url):
                    skipped += 1
                    continue
                    
                resource = detector._evaluate_url(url, message, channel_name, analyze_unknown=False)
                if resource:
                    detector.detected_resources.append(resource)
                    detector.stats['total_resources'] += 1
                    detector.stats[f'category_{resource["category"]}'] += 1
                    processed += 1
                    resources_found += 1
                    new_url_found = True
                    
                    # Track recent resources for display
                    recent_resources.append({
                        'url': resource['url'][:50] + '...' if len(resource['url']) > 50 else resource['url'],
                        'category': resource['category'],
                        'domain': resource['domain']
                    })
                    
                    # Keep only last 3 recent resources
                    if len(recent_resources) > 3:
                        recent_resources.pop(0)
                    
                    # Update domain and category stats
                    domain_stats[resource['domain']] += 1
                    category_stats[resource['category']] += 1
            
            # Update progress bar with detailed stats every 50 messages
            if (i + 1) % 50 == 0 or new_url_found:
                # Get top domains and categories
                top_domains = sorted(domain_stats.items(), key=lambda x: x[1], reverse=True)[:3]
                top_categories = sorted(category_stats.items(), key=lambda x: x[1], reverse=True)[:3]
                
                # Create detailed postfix
                postfix = {
                    "found": resources_found,
                    "skipped": skipped,
                    "urls": urls_extracted,
                    "progress": f"{i+1}/{len(messages)}"
                }
                
                # Add recent resource info if available
                if recent_resources:
                    latest = recent_resources[-1]
                    postfix["latest"] = f"{latest['category']}: {latest['domain']}"
                
                msg_pbar.set_postfix(postfix)
                
                # Force refresh to show updates
                msg_pbar.refresh()
                
                # Show detailed summary every 500 messages
                if (i + 1) % 500 == 0:
                    print(f"\n📊 Progress Summary at {i+1:,}/{len(messages):,} messages:")
                    print(f"   🔍 Resources found: {resources_found}")
                    print(f"   ⏭️ URLs skipped: {skipped}")
                    print(f"   🔗 URLs extracted: {urls_extracted}")
                    
                    if top_domains:
                        print(f"   🌐 Top domains: {', '.join([f'{d}({c})' for d, c in top_domains])}")
                    if top_categories:
                        print(f"   📁 Top categories: {', '.join([f'{c}({n})' for c, n in top_categories])}")
                    
                    if recent_resources:
                        print(f"   🆕 Recent finds:")
                        for res in recent_resources[-2:]:  # Show last 2
                            print(f"      • {res['category']} from {res['domain']}")
                    
                    print("-" * 40)

    print(f"\n✅ Analysis complete!")
    print(f"📊 Results: {resources_found} new resources found, {skipped} URLs skipped (already processed)")
    
    # Show incremental processing summary
    if detector.resource_checkpoint.get('last_processed_message_id'):
        print(f"🔄 Incremental processing: Continuing from previous checkpoint")
        print(f"   📍 Last processed message: {detector.resource_checkpoint['last_processed_message_id'][:8]}...")
        print(f"   📈 New messages processed: {len(messages):,}")
    else:
        print(f"🔄 Full processing: No previous checkpoint found")
        print(f"   📈 Total messages processed: {len(messages):,}")

    # Save checkpoint with last processed message ID
    if last_processed_message_id:
        detector.resource_checkpoint['last_processed_message_id'] = last_processed_message_id
        detector._save_resource_checkpoint()
        print(f"💾 Saved checkpoint: {last_processed_message_id[:8]}...")

    # Progress bar for description generation (for any resources missing description)
    resources_to_describe = [r for r in detector.detected_resources if not r.get('description') or r['description'].startswith('AI/ML resource from')]
    if resources_to_describe:
        # Limit to first 50 resources to prevent getting stuck on large datasets
        if len(resources_to_describe) > 50:
            print(f"⚠️ Limiting description generation to first 50 resources (found {len(resources_to_describe)} total)")
            resources_to_describe = resources_to_describe[:50]
            
        print(f"\n🤖 Generating AI descriptions using {model_name} for {len(resources_to_describe)} new resources...")
        print("📡 This will use the local Ollama server for intelligent descriptions")
        print("🔄 Progress will show AI generation vs fallback descriptions")
        print("-" * 60)
        
        # Test LLM connection first
        llm_available = False
        try:
            test_response = requests.get('http://localhost:11434/api/tags', timeout=5)
            if test_response.status_code == 200:
                llm_available = True
                print("✅ Ollama server is connected and ready")
            else:
                print("⚠️ Ollama server not responding - using intelligent fallback descriptions")
        except Exception:
            print("⚠️ Ollama server not available - using intelligent fallback descriptions")
        
        llm_success_count = 0
        fallback_count = 0
        
        with tqdm(resources_to_describe, desc="🤖 Generating descriptions", unit="resource", position=0, leave=True) as desc_pbar:
            for i, resource in enumerate(desc_pbar):
                # Create a proper message structure for better context
                original_message = {
                    'content': f"Shared by {resource['author']} in #{resource['channel_name']}: {resource['url']}",
                    'author': {'username': resource['author']}
                }
                
                old_description = resource.get('description', '')
                description = detector._generate_description(original_message, resource['url'])
                resource['description'] = description
                
                # Track success/fallback
                if description != old_description and not description.startswith('AI/ML resource from'):
                    # Check if it's an LLM-generated description vs fallback
                    if any(indicator in description.lower() for indicator in ['research', 'repository', 'implements', 'provides', 'framework', 'tool', 'paper', 'article']):
                        llm_success_count += 1
                    else:
                        fallback_count += 1
                else:
                    fallback_count += 1
                
                # Enhanced postfix with more detailed information
                desc_pbar.set_postfix({
                    "domain": resource['domain'][:15],
                    "category": resource['category'][:10],
                    "AI": llm_success_count,
                    "fallback": fallback_count,
                    "success_rate": f"{(llm_success_count / (i + 1)) * 100:.0f}%"
                })
                
                # Force update every 5 resources for better feedback
                if (i + 1) % 5 == 0:
                    desc_pbar.refresh()
        
        print(f"✅ Description generation complete:")
        print(f"   🤖 AI-generated descriptions: {llm_success_count}")
        print(f"   🔄 Intelligent fallback descriptions: {fallback_count}")
        if llm_success_count > 0:
            print(f"   📊 LLM success rate: {(llm_success_count / len(resources_to_describe)) * 100:.1f}%")

    print("\n💾 Saving processed URLs for incremental processing...")
    detector._save_processed_urls()
    print("✅ Processed URLs saved")

    print("\n💾 Saving resources to database...")
    # Save to SQLite DB FIRST
    db_resource_path = project_root / 'data' / 'resources.db'
    detector.save_resources_to_db(db_resource_path)

    print("\n📄 Creating detailed resources report...")
    # Create the detailed report file that pepe-admin expects
    detailed_report_path = project_root / 'data' / 'optimized_fresh_resources.json'
    print(f"   📤 Creating detailed report: {detailed_report_path.name}")
    
    # Generate and save the detailed report
    detailed_report = detector._generate_report(analyze_unknown=False)
    with open(detailed_report_path, 'w', encoding='utf-8') as f:
        json.dump(detailed_report, f, indent=2, ensure_ascii=False, default=str)
    print("   ✅ Detailed report created")

    print("\n📄 Creating JSON export from database...")
    # Create JSON export from database
    export_path = project_root / 'data' / 'resources-data.json'
    print(f"   📤 Creating export file: {export_path.name}")
    
    # Export from database with new structure
    export_data = detector.export_resources_from_db(db_resource_path)
    with open(export_path, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)
    print("   ✅ Export file created")

    print(f"\n🎯 FINAL OPTIMIZED RESULTS:")
    print("=" * 60)
    print(f"✅ Found {resources_found} high-quality resources!")
    print(f"⏭️ Skipped (already processed): {skipped}")
    print(f"❌ Excluded {detector.stats['excluded_domains']} low-quality URLs")
    print(f"❓ Unknown domains: {detector.stats['unknown_domains']}")
    print(f"🤖 Bot messages filtered: {detector.stats['bot_messages_filtered']}")
    print(f"🔗 Total URLs extracted: {urls_extracted}")
    print(f"📝 Messages analyzed: {len(messages):,}")

    # Show category distribution
    category_stats = Counter([r['category'] for r in detector.detected_resources])
    print(f"\n📁 Resource Categories Found:")
    for category, count in sorted(category_stats.items(), key=lambda x: x[1], reverse=True):
        print(f"   {category}: {count} resources")
    
    print(f"\n💡 RECOMMENDATION: PROCEED")
    print("🚀 Resources have been saved to database and exported to JSON")
    
    print(f"\n📄 Files created:")
    print(f"   • Database: {db_resource_path}")
    print(f"   • Detailed report: {detailed_report_path}")
    print(f"   • Export file: {export_path}")
    
    return {"resources": detector.detected_resources, "statistics": detector.stats}

if __name__ == '__main__':
    main() 