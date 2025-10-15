#!/usr/bin/env python3
"""
Fresh Resource Detector - Quality-First Approach with GPT-5 Enhancement

Extracts only high-quality resources from Discord messages,
filtering out junk and focusing on valuable content.

Phase 1 & 2 Enhancements:
- Intelligent title extraction and generation
- Rich contextual descriptions using GPT-5 mini
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
import asyncio

# Configure tqdm for better progress bar visibility
tqdm.monitor_interval = 0.05  # Update more frequently

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import new enrichment services
try:
    from agentic.services.resource_enrichment import ResourceEnrichment
    GPT5_AVAILABLE = True
except ImportError:
    GPT5_AVAILABLE = False
    print("⚠️ ResourceEnrichment not available - will use fallback methods")

class FreshResourceDetector:
    def __init__(self, use_fast_model: bool = True, use_gpt5: bool = False):
        # Model selection for resource detection
        self.use_fast_model = use_fast_model
        self.fast_model = os.getenv('LLM_FAST_MODEL', 'phi3:mini')  # Smaller, faster model
        self.standard_model = os.getenv('LLM_MODEL', 'llama3.1:8b')   # Standard model
        
        # NEW: Enrichment service (defaults to local LLM)
        # Use --use-openai flag to enable OpenAI API (requires OPENAI_API_KEY)
        self.use_gpt5 = use_gpt5 and GPT5_AVAILABLE
        self.enrichment = ResourceEnrichment(use_gpt5=self.use_gpt5) if GPT5_AVAILABLE else None
        
        if self.use_gpt5 and self.enrichment:
            print("✅ Using OpenAI API for enrichment (GPT-4o-mini)")
        else:
            print("✅ Using local LLM for enrichment (free, works great)")
        
        # Incremental processing tracking
        self.processed_urls_file = Path('data/processed_resources.json')
        self.processed_urls = self._load_processed_urls()
        
        # High-quality domains (curated list)
        self.high_quality_domains = {
            # Research & Academic
            'arxiv.org': {'category': 'Research Papers', 'score': 0.95},
            'papers.withcode.com': {'category': 'Research Papers', 'score': 0.85},
            'distill.pub': {'category': 'Research Visualization', 'score': 0.90},
            'news.mit.edu': {'category': 'Academic News', 'score': 0.85},
            'research.google.com': {'category': 'Research', 'score': 0.85},
            
            # Code & Development
            'github.com': {'category': 'Code Repositories', 'score': 0.90},
            'stackoverflow.com': {'category': 'Technical Q&A', 'score': 0.80},
            
            # Documentation & Resources
            'docs.google.com': {'category': 'Documentation', 'score': 0.85},
            'drive.google.com': {'category': 'Shared Documents', 'score': 0.80},
            'cloud.google.com': {'category': 'Technical Documentation', 'score': 0.80},
            
            # AI & ML Resources
            'openai.com': {'category': 'AI Resources', 'score': 0.90},
            'blog.openai.com': {'category': 'AI Research', 'score': 0.85},
            'anthropic.com': {'category': 'AI Research', 'score': 0.90},
            'huggingface.co': {'category': 'AI Models', 'score': 0.90},
            'tensorflow.org': {'category': 'ML Documentation', 'score': 0.85},
            'pytorch.org': {'category': 'ML Documentation', 'score': 0.85},
            'ai.googleblog.com': {'category': 'AI Research', 'score': 0.80},
            'nvidia.com': {'category': 'AI/GPU Technology', 'score': 0.80},
            'blogs.microsoft.com': {'category': 'Tech Documentation', 'score': 0.75},
            'chatgpt.com': {'category': 'AI Tools', 'score': 0.75},
            
            # Educational Platforms
            'deeplearning.ai': {'category': 'AI Education', 'score': 0.85},
            'coursera.org': {'category': 'Online Courses', 'score': 0.75},
            'edx.org': {'category': 'Online Courses', 'score': 0.75},
            'udacity.com': {'category': 'Online Courses', 'score': 0.75},
            'fast.ai': {'category': 'AI Education', 'score': 0.80},
            'machinelearningmastery.com': {'category': 'ML Education', 'score': 0.75},
            
            # Data Science & Analytics
            'kaggle.com': {'category': 'Data Science', 'score': 0.80},
            'towardsdatascience.com': {'category': 'Data Science', 'score': 0.75},
            
            # Video Content
            'youtube.com': {'category': 'Educational Videos', 'score': 0.75},
            'youtu.be': {'category': 'Educational Videos', 'score': 0.75},
            
            # Articles & Blogs
            'medium.com': {'category': 'Articles', 'score': 0.70},
            
            # News & Tech Publications (High Quality)
            'axios.com': {'category': 'Tech News', 'score': 0.80},
            'theguardian.com': {'category': 'News & Analysis', 'score': 0.75},
            'reuters.com': {'category': 'News & Analysis', 'score': 0.80},
            'wsj.com': {'category': 'Business News', 'score': 0.80},
            'ft.com': {'category': 'Financial News', 'score': 0.80},
            'businessinsider.com': {'category': 'Business News', 'score': 0.70},
            'theverge.com': {'category': 'Tech News', 'score': 0.75},
            'techcrunch.com': {'category': 'Tech News', 'score': 0.75},
            'venturebeat.com': {'category': 'Tech News', 'score': 0.70},
            'wired.com': {'category': 'Tech News', 'score': 0.75},
            'arstechnica.com': {'category': 'Tech News', 'score': 0.75},
            
            # Productivity & Collaboration Tools (with valuable content)
            'app.mural.co': {'category': 'Collaboration Boards', 'score': 0.60},
            'trello.com': {'category': 'Project Management', 'score': 0.60},
            'notion.so': {'category': 'Documentation', 'score': 0.70}
        }
        
        # Domains to exclude (junk)
        self.excluded_domains = {
            'cdn.discordapp.com', 'discord.com/channels', 'discordapp.com',
            'tenor.com', 'giphy.com', 'discord.gg',
            'meet.google.com', 'zoom.us', 'us06web.zoom.us', 'mit.zoom.us',
            'linkedin.com/in', 'linkedin.com/posts',  # Profile links and posts
            'twitter.com/i/', 'facebook.com', 'instagram.com',
            'fathom.video',  # Meeting recordings
            'tinyurl.com', 'bit.ly',  # URL shorteners (hard to verify quality)
            'sync-google-calendar-wit-erprmym.gamma.site'  # Specific app link
        }
        
        # File extensions that indicate quality resources
        self.quality_extensions = {
            '.pdf': 0.8, '.docx': 0.7, '.doc': 0.7, '.pptx': 0.7,
            '.py': 0.6, '.ipynb': 0.8, '.md': 0.6, '.tex': 0.7
        }
        
        self.detected_resources = []
        self.stats = defaultdict(int)
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
            
            # Get all processed URLs
            all_urls = self.processed_urls.union(
                {resource['url'] for resource in self.detected_resources}
            )
            
            with open(self.processed_urls_file, 'w') as f:
                json.dump({
                    'processed_urls': list(all_urls),
                    'last_updated': datetime.now().isoformat(),
                    'total_processed': len(all_urls)
                }, f, indent=2)
        except Exception as e:
            print(f"⚠️ Warning: Could not save processed URLs: {e}")
    
    def _is_url_processed(self, url: str) -> bool:
        """Check if a URL has already been processed"""
        return url in self.processed_urls
    
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
        
        return self._generate_report(analyze_unknown)
    
    def _is_bot_author(self, message: Dict[str, Any]) -> bool:
        """Check if author is a bot using the proper bot flag from raw_data"""
        # First try to get bot status from raw_data (most reliable)
        raw_data = message.get('raw_data')
        if raw_data:
            try:
                raw_json = json.loads(raw_data)
                author_data = raw_json.get('author', {})
                if 'bot' in author_data:
                    return author_data['bot']
            except (json.JSONDecodeError, TypeError):
                pass
        
        # Fallback to string matching if raw_data unavailable or malformed
        author = message.get('author', {})
        if isinstance(author, dict):
            author_name = author.get('username', '').lower()
            return 'pepe' in author_name or 'bot' in author_name
        else:
            author_name = str(author).lower()
            return 'pepe' in author_name or 'bot' in author_name

    def _analyze_message(self, message: Dict[str, Any], channel_name: str, analyze_unknown: bool = False):
        """Analyze a single message for resources"""
        content = message.get('content', '')
        if not content:
            return
        
        # Filter out bot messages using robust bot detection
        if self._is_bot_author(message):
            self.stats['bot_messages_skipped'] = self.stats.get('bot_messages_skipped', 0) + 1
            return
        
        # Filter out test/playground channels
        channel_lower = channel_name.lower()
        if any(keyword in channel_lower for keyword in ['pg', 'playground', 'test']):
            self.stats['test_channel_messages_skipped'] = self.stats.get('test_channel_messages_skipped', 0) + 1
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
            }, timeout=30)  # Increase timeout for better quality
            
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
                    print(f"⚠️ LLM returned empty response for {url[:50]}...")
                    return self._generate_fallback_description(url, domain, content)
            else:
                print(f"⚠️ LLM API error (status {response.status_code}) for {url[:50]}...")
                return self._generate_fallback_description(url, domain, content)
                
        except Exception as e:
            print(f"⚠️ LLM connection error for {url[:50]}...: {e}")
            return self._generate_fallback_description(url, domain, content)
    
    def _generate_fallback_title(self, url: str, domain: str) -> str:
        """Generate fallback title from URL structure"""
        parsed_url = urlparse(url)
        
        # Domain-specific fallback titles
        if 'youtube.com' in domain or 'youtu.be' in domain:
            return "YouTube Video"
        elif 'arxiv.org' in domain:
            paper_id = parsed_url.path.split('/')[-1]
            return f"arXiv Paper: {paper_id}"
        elif 'github.com' in domain:
            path_parts = parsed_url.path.strip('/').split('/')
            if len(path_parts) >= 2:
                return f"GitHub: {path_parts[0]}/{path_parts[1]}"
            return "GitHub Repository"
        elif 'huggingface.co' in domain:
            path_parts = parsed_url.path.strip('/').split('/')
            if len(path_parts) >= 2:
                return f"Hugging Face: {path_parts[0]}/{path_parts[1]}"
            return "Hugging Face AI/ML Resource"
        elif '.pdf' in url.lower():
            filename = parsed_url.path.split('/')[-1]
            return f"PDF: {filename}"
        else:
            return f"Resource from {domain}"
    
    def _generate_fallback_description(self, url, domain, content):
        """Generate a fallback description when LLM is unavailable."""
        parsed_url = urlparse(url)
        
        # Create intelligent fallback descriptions based on URL patterns
        if 'arxiv.org' in domain:
            paper_id = parsed_url.path.split('/')[-1] if parsed_url.path else "unknown"
            return f"Research paper from arXiv ({paper_id}) - Academic research in artificial intelligence and machine learning"
        elif 'github.com' in domain:
            repo_path = parsed_url.path.strip('/')
            return f"GitHub repository: {repo_path} - Open source code and development resources"
        elif 'huggingface.co' in domain:
            path_parts = parsed_url.path.strip('/').split('/')
            if len(path_parts) >= 2:
                return f"Hugging Face {path_parts[0]}: {'/'.join(path_parts[1:])} - AI models, datasets, and machine learning resources"
            return "Hugging Face AI/ML resource - Pre-trained models and datasets for machine learning"
        elif 'youtube.com' in domain or 'youtu.be' in domain:
            return "Educational video content from YouTube - Tutorials, lectures, and educational material"
        elif '.pdf' in url.lower():
            filename = parsed_url.path.split('/')[-1] if parsed_url.path else "document"
            return f"PDF document: {filename} - Research paper, documentation, or educational material"
        elif any(news_domain in domain for news_domain in ['techcrunch.com', 'theverge.com', 'wired.com', 'arstechnica.com']):
            return f"Tech news article from {domain} - Latest technology news and analysis"
        elif 'deeplearning.ai' in domain:
            return f"DeepLearning.AI course - Professional AI and machine learning education"
        elif 'status.' in domain:
            return f"Service status page for {domain} - Real-time system status and updates"
        elif 'docs.google.com' in domain:
            return f"Google Docs presentation - Collaborative document or presentation"
        elif 'hal.science' in domain:
            return f"Research paper from HAL (Hyper Articles en Ligne) - Academic research repository"
        elif 'mural.co' in domain:
            return f"Mural workspace - Collaborative visual workspace for teams"
        elif 'mit.edu' in domain:
            return f"MIT resource - Academic content from Massachusetts Institute of Technology"
        elif 'senedd.wales' in domain:
            return f"Welsh Parliament document - Official government documentation"
        else:
            # Extract meaningful info from URL structure and provide more context
            if parsed_url.path and len(parsed_url.path) > 1:
                path_parts = [p for p in parsed_url.path.split('/') if p]
                if path_parts:
                    return f"Resource from {domain} - {path_parts[0].replace('_', ' ').replace('-', ' ').title()}"
            return f"Resource from {domain} - Web-based content and information"

    def _evaluate_url(self, url: str, message: Dict[str, Any], channel_name: str, analyze_unknown: bool = False) -> Dict[str, Any]:
        """Evaluate if a URL is a high-quality resource"""
        # Check if URL has already been processed
        if self._is_url_processed(url):
            self.stats['skipped_processed'] += 1
            return None
            
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            if domain.startswith('www.'):
                domain = domain[4:]
            if any(excluded in domain for excluded in self.excluded_domains):
                self.stats['excluded_domains'] += 1
                return None
            domain_info = None
            for hq_domain, info in self.high_quality_domains.items():
                if hq_domain in domain:
                    domain_info = info
                    break
            if not domain_info:
                path = parsed.path.lower()
                quality_score = 0
                for ext, score in self.quality_extensions.items():
                    if path.endswith(ext):
                        quality_score = score
                        domain_info = {'category': 'Documents', 'score': score}
                        break
                if not domain_info:
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
            # --- NEW EXPORT STRUCTURE ---
            author_display = message.get('author', {}).get('display_name') or message.get('author', {}).get('username', 'Unknown')
            # Format timestamp as YYYY-MM-DD
            raw_ts = message.get('timestamp')
            try:
                ts = datetime.fromisoformat(raw_ts)
                ts_str = ts.strftime('%Y-%m-%d')
            except Exception:
                ts_str = raw_ts[:10] if raw_ts else ''
            jump_url = message.get('jump_url')
            
            # NEW: Use enhanced enrichment service (Phase 1 & 2)
            if self.enrichment:
                # Run async enrichment in sync context
                enriched = asyncio.run(self.enrichment.enrich_resource(url, message, channel_name))
                title = enriched.get('title', self._generate_fallback_title(url, domain))
                description = enriched.get('description', self._generate_description(message, url))
            else:
                # Fallback to old method
                title = self._generate_fallback_title(url, domain)
                description = self._generate_description(message, url)
            
            resource = {
                'id': len(self.detected_resources) + 1,  # Simple ID
                'resource_url': url,  # HTML expects this field
                'title': title,
                'tag': domain_info['category'],  # HTML expects this field
                'date': ts_str,  # HTML expects this field
                'description': description
            }
            return resource
        except Exception as e:
            self.stats['parsing_errors'] += 1
            return None
    
    def _generate_report(self, analyze_unknown: bool = False) -> Dict[str, Any]:
        """Generate comprehensive analysis report"""
        
        # Sort resources by date (newest first)
        sorted_resources = sorted(self.detected_resources, key=lambda x: x.get('date', ''), reverse=True)
        
        # Generate statistics
        category_stats = Counter([r['tag'] for r in sorted_resources])
        
        # Simple quality distribution based on resource count
        total_resources = len(sorted_resources)
        quality_distribution = {
            'excellent': total_resources,  # All resources are considered high quality
            'high': 0,
            'good': 0,
            'fair': 0
        }
        
        print("\n📊 Fresh Resource Detection Results")
        print("=" * 40)
        print(f"✅ High-quality resources found: {len(sorted_resources)}")
        print(f"⏭️ Skipped (already processed): {self.stats.get('skipped_processed', 0)}")
        print(f"🤖 Bot messages skipped: {self.stats.get('bot_messages_skipped', 0)}")
        print(f"🧪 Test channel messages skipped: {self.stats.get('test_channel_messages_skipped', 0)}")
        print(f"❌ Excluded low-quality URLs: {self.stats['excluded_domains']}")
        print(f"❓ Unknown domains skipped: {self.stats['unknown_domains']}")
        print(f"⚠️ Parsing errors: {self.stats['parsing_errors']}")
        
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
        
        print("\n⭐ Quality Distribution:")
        for level, count in quality_distribution.items():
            print(f"   {level.capitalize()}: {count} resources")
        
        print("\n📄 Top 10 Resources:")
        for i, resource in enumerate(sorted_resources[:10]):
            print(f"   {i+1}. [{resource['tag']}] {resource['resource_url']}")
            print(f"      Date: {resource['date']}")
            print(f"      Description: {resource['description'][:80]}...")
            print()
        
        return {
            'export_date': datetime.now().isoformat(),
            'total_resources': len(sorted_resources),
            'resources': sorted_resources,
            'statistics': {
                'total_found': len(sorted_resources),
                'excluded_count': self.stats['excluded_domains'],
                'unknown_count': self.stats['unknown_domains'],
                'categories': dict(category_stats),
                'quality_distribution': quality_distribution
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
    
    def _deduplicate_resources(self, resources: List[Dict]) -> List[Dict]:
        """Remove duplicate resources based on URL and title similarity"""
        if not resources:
            return resources
            
        seen_urls = set()
        seen_titles = set()
        unique_resources = []
        duplicates_removed = 0
        
        for resource in resources:
            # Check for URL duplicates
            url = resource.get('resource_url', '').strip()
            if url and url in seen_urls:
                duplicates_removed += 1
                continue
            
            # Check for title duplicates (case-insensitive)
            title = resource.get('title', '').strip().lower()
            if title and title in seen_titles:
                duplicates_removed += 1
                continue
            
            # Check for very similar titles (fuzzy matching)
            is_similar = False
            for seen_title in seen_titles:
                # Simple similarity check - if titles are 80%+ similar, consider duplicate
                if len(title) > 10 and len(seen_title) > 10:
                    similarity = self._calculate_similarity(title, seen_title)
                    if similarity > 0.8:
                        is_similar = True
                        break
            
            if is_similar:
                duplicates_removed += 1
                continue
            
            unique_resources.append(resource)
            
            if url:
                seen_urls.add(url)
            if title:
                seen_titles.add(title)
        
        if duplicates_removed > 0:
            print(f"   🧹 Removed {duplicates_removed} duplicate resources")
        
        return unique_resources
    
    def _calculate_similarity(self, str1: str, str2: str) -> float:
        """Calculate simple string similarity ratio"""
        if not str1 or not str2:
            return 0.0
        
        # Simple Jaccard similarity on words
        words1 = set(str1.split())
        words2 = set(str2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0

    def save_resources(self, output_path: Path, analyze_unknown: bool = False):
        """Save detected resources to JSON file with deduplication"""
        report = self._generate_report(analyze_unknown)
        
        # Deduplicate resources before saving
        if 'resources' in report and report['resources']:
            original_count = len(report['resources'])
            report['resources'] = self._deduplicate_resources(report['resources'])
            final_count = len(report['resources'])
            
            if original_count != final_count:
                print(f"   🧹 Deduplication: {original_count} → {final_count} resources ({original_count - final_count} duplicates removed)")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"💾 Resources saved to: {output_path}")
        return report

def main():
    parser = argparse.ArgumentParser(description='Fresh Resource Detector - Quality-First Approach')
    parser.add_argument('--fast-model', action='store_true', 
                       help='Use fast model (phi3:mini) for faster processing')
    parser.add_argument('--standard-model', action='store_true',
                       help='Use standard model (llama3.1:8b) for better quality')
    parser.add_argument('--use-openai', action='store_true',
                       help='Use OpenAI API (gpt-4o-mini) for enrichment instead of local LLM')
    parser.add_argument('--reset-cache', action='store_true',
                       help='Reset processed URLs cache and reprocess all resources')
    args = parser.parse_args()

    use_fast_model = args.fast_model or (not args.standard_model)  # Default to fast model
    use_gpt5 = args.use_openai  # Default to local LLM (free)

    if args.reset_cache:
        cache_file = project_root / 'data' / 'processed_resources.json'
        if cache_file.exists():
            cache_file.unlink()
            print("🗑️ Reset processed URLs cache")

    detector = FreshResourceDetector(use_fast_model=use_fast_model, use_gpt5=use_gpt5)
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
            cursor = conn.execute("""
                SELECT * FROM messages 
                WHERE content IS NOT NULL AND content != ''
                ORDER BY timestamp DESC
            """)
            
            # First, count total messages for progress bar
            count_cursor = conn.execute("""
                SELECT COUNT(*) as count FROM messages 
                WHERE content IS NOT NULL AND content != ''
            """)
            total_messages = count_cursor.fetchone()['count']
            print(f"📊 Found {total_messages:,} messages to analyze")
            
            # Load messages with progress bar
            with tqdm(total=total_messages, desc="📥 Loading messages", unit="msg", position=0, leave=True) as load_pbar:
                for row in cursor:
                    message_dict = {
                        'content': row['content'],
                        'author': {
                            'username': row['author_username'],
                            'display_name': row['author_display_name'] or row['author_username']
                        },
                        'timestamp': row['timestamp'],
                        'message_id': row['message_id'],
                        'channel_id': row['channel_id'],
                        'channel_name': row['channel_name']
                    }
                    messages.append(message_dict)
                    load_pbar.update(1)
                    load_pbar.set_postfix({"loaded": len(messages)})
                    
    except Exception as e:
        print(f"❌ Error reading database: {e}")
        return

    print(f"✅ Loaded {len(messages):,} messages successfully")
    print("\n" + "="*80)
    print("🔍 PHASE 1: RESOURCE EXTRACTION & QUALITY FILTERING")
    print("="*80)

    # Analyze messages with progress bar and incremental logic
    skipped = 0
    processed = 0
    resources_found = 0
    urls_extracted = 0
    
    # Create an async function for batch processing
    async def process_resources_batch(resources_to_process):
        """Process a batch of resources with async enrichment"""
        for resource_data in resources_to_process:
            url, message, channel_name = resource_data
            resource = await process_single_resource(url, message, channel_name)
            if resource:
                detector.detected_resources.append(resource)
        return len([r for r in resources_to_process if r])
    
    async def process_single_resource(url, message, channel_name):
        """Process a single resource with enrichment"""
        if detector._is_url_processed(url):
            return None
            
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            if domain.startswith('www.'):
                domain = domain[4:]
            if any(excluded in domain for excluded in detector.excluded_domains):
                detector.stats['excluded_domains'] += 1
                return None
            domain_info = None
            for hq_domain, info in detector.high_quality_domains.items():
                if hq_domain in domain:
                    domain_info = info
                    break
            if not domain_info:
                path = parsed.path.lower()
                quality_score = 0
                for ext, score in detector.quality_extensions.items():
                    if path.endswith(ext):
                        quality_score = score
                        domain_info = {'category': 'Documents', 'score': score}
                        break
                if not domain_info:
                    detector.stats['unknown_domains'] += 1
                    return None
            
            # Get basic metadata
            author_display = message.get('author', {}).get('display_name') or message.get('author', {}).get('username', 'Unknown')
            raw_ts = message.get('timestamp')
            try:
                ts = datetime.fromisoformat(raw_ts)
                ts_str = ts.strftime('%Y-%m-%d')
            except Exception:
                ts_str = raw_ts[:10] if raw_ts else ''
            jump_url = message.get('jump_url')
            
            # Use enhanced enrichment service (Phase 1 & 2)
            if detector.enrichment:
                enriched = await detector.enrichment.enrich_resource(url, message, channel_name)
                title = enriched.get('title', detector._generate_fallback_title(url, domain))
                description = enriched.get('description', detector._generate_description(message, url))
            else:
                title = detector._generate_fallback_title(url, domain)
                description = detector._generate_description(message, url)
            
            resource = {
                'id': len(detector.detected_resources) + 1,  # Simple ID
                'resource_url': url,  # HTML expects this field
                'title': title,
                'tag': domain_info['category'],  # HTML expects this field
                'date': ts_str,  # HTML expects this field
                'description': description
            }
            detector.stats['total_resources'] += 1
            detector.stats[f'category_{domain_info["category"]}'] += 1
            return resource
        except Exception as e:
            detector.stats['parsing_errors'] += 1
            return None
    
    # Extract URLs first with progress bar
    print("\n📊 Step 1/3: Extracting URLs from messages...")
    urls_to_process = []
    with tqdm(messages, desc="🔗 Extracting URLs", unit="msg", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as url_pbar:
        for message in url_pbar:
            content = message.get('content', '')
            if not content:
                continue
            
            # Filter out bot messages using robust bot detection
            if detector._is_bot_author(message):
                detector.stats['bot_messages_skipped'] = detector.stats.get('bot_messages_skipped', 0) + 1
                continue
            
            # Filter out test/playground channels
            channel_name = message.get('channel_name', '')
            channel_lower = channel_name.lower()
            if any(keyword in channel_lower for keyword in ['pg', 'playground', 'test']):
                detector.stats['test_channel_messages_skipped'] = detector.stats.get('test_channel_messages_skipped', 0) + 1
                continue
            
            urls = re.findall(r'https?://[^\s\n\r<>"]+', content)
            urls_extracted += len(urls)
            for url in urls:
                if not detector._is_url_processed(url):
                    urls_to_process.append((url, message, message['channel_name']))
                else:
                    skipped += 1
            url_pbar.set_postfix({"URLs": urls_extracted, "unique": len(urls_to_process), "skipped": skipped})
    
    print(f"   ✅ Found {urls_extracted:,} total URLs ({len(urls_to_process):,} new, {skipped:,} already processed)")
    
    # Process resources with enrichment
    if urls_to_process:
        print(f"\n📊 Step 2/3: Evaluating and enriching {len(urls_to_process):,} new resources...")
        print("   🌐 Web scraping for metadata (fallback)")
        print("   📝 Message-based extraction (primary)")
        print("   🤖 OpenAI API (GPT-4o-mini)" if detector.use_gpt5 else "   🤖 Local LLM (Ollama)")
        
        # Process in batches to show progress
        batch_size = 1  # Process one at a time for real-time progress
        with tqdm(total=len(urls_to_process), desc="✨ Enriching resources", unit="resource", 
                  bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as enrich_pbar:
            for i in range(0, len(urls_to_process), batch_size):
                batch = urls_to_process[i:i+batch_size]
                # Process each resource individually for better progress tracking
                for url_data in batch:
                    url, message, channel_name = url_data
                    resource = asyncio.run(process_single_resource(url, message, channel_name))
                    if resource:
                        detector.detected_resources.append(resource)
                        resources_found += 1
                        processed += 1
                    
                    # Get enrichment stats
                    if detector.enrichment:
                        enrich_stats = detector.enrichment.get_stats()
                        gpt5_stats = detector.enrichment.gpt5.get_stats() if detector.enrichment.gpt5 else {}
                        
                        postfix = {
                            "found": resources_found,
                            "msg_based": enrich_stats.get('message_based', 0),
                            "scraped": enrich_stats.get('web_scraped', 0)
                        }
                        if gpt5_stats:
                            postfix["OpenAI"] = gpt5_stats.get('gpt5_calls', 0)
                            postfix["cached"] = gpt5_stats.get('gpt5_cached', 0)
                    else:
                        postfix = {"found": resources_found}
                    
                    enrich_pbar.set_postfix(postfix)
                    enrich_pbar.update(1)
        
        print(f"   ✅ Successfully enriched {resources_found:,} high-quality resources")
    
    print(f"\n📊 Step 3/3: Finalizing and saving results...")
    print(f"\n✅ Resource extraction complete!")
    print(f"   📦 Total resources found: {resources_found:,}")
    print(f"   ⏭️  URLs skipped (cached): {skipped:,}")
    print(f"   ❌ URLs filtered out: {detector.stats.get('excluded_domains', 0) + detector.stats.get('unknown_domains', 0):,}")

    # Progress bar for description generation (for any resources missing description)
    resources_to_describe = [r for r in detector.detected_resources if not r.get('description') or r['description'].startswith('AI/ML resource from')]
    if resources_to_describe:
        # Limit to first 50 resources to prevent getting stuck on large datasets
        if len(resources_to_describe) > 50:
            print(f"⚠️ Limiting description generation to first 50 resources (found {len(resources_to_describe)} total)")
            resources_to_describe = resources_to_describe[:50]
            
        print(f"\n🤖 Generating AI descriptions using {model_name} for {len(resources_to_describe)} new resources...")
        print("📡 This will use the local Ollama server for intelligent descriptions")
        
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
                
                desc_pbar.set_postfix({
                    "domain": resource['domain'][:20],
                    "AI": llm_success_count,
                    "fallback": fallback_count,
                    "progress": f"{i+1}/{len(resources_to_describe)}"
                })
                
                # Force update every 5 resources for better feedback
                if (i + 1) % 5 == 0:
                    desc_pbar.refresh()
        
        print(f"✅ Description generation complete:")
        print(f"   🤖 AI-generated descriptions: {llm_success_count}")
        print(f"   🔄 Intelligent fallback descriptions: {fallback_count}")
        if llm_success_count > 0:
            print(f"   📊 LLM success rate: {(llm_success_count / len(resources_to_describe)) * 100:.1f}%")

    print("\n" + "="*80)
    print("💾 PHASE 2: SAVING RESULTS")
    print("="*80)
    
    print("\n📝 Saving checkpoint data...")
    detector._save_processed_urls()
    print("   ✅ Checkpoint saved for incremental updates")

    print("\n📄 Generating final report and saving to files...")
    # Save results to JSON file in docs/ directory
    output_path = project_root / 'docs' / 'resources-data.json'
    
    # Show progress while saving
    with tqdm(total=3, desc="💾 Saving files", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}') as save_pbar:
        save_pbar.set_postfix({"file": "report"})
        report = detector.save_resources(output_path, analyze_unknown=False)
        save_pbar.update(1)

        # Also save a simplified export file with just the resources list
        save_pbar.set_postfix({"file": "export"})
        export_path = project_root / 'data' / 'resources_export.json'
        export_data = {
            'export_date': datetime.now().isoformat(),
            'total_resources': len(report['resources']),
            'resources': report['resources']
        }
        with open(export_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)
        save_pbar.update(1)
        
        save_pbar.set_postfix({"file": "complete"})
        save_pbar.update(1)
    
    print(f"   ✅ Files saved successfully:")

    print("\n" + "="*80)
    print("📊 FINAL SUMMARY")
    print("="*80)
    
    print(f"\n🎯 Resource Discovery:")
    print(f"   ✅ High-quality resources: {report['statistics']['total_found']:,}")
    print(f"   ⏭️  Already processed: {skipped:,}")
    print(f"   🤖 Bot messages skipped: {detector.stats.get('bot_messages_skipped', 0):,}")
    print(f"   🧪 Test channel messages skipped: {detector.stats.get('test_channel_messages_skipped', 0):,}")
    print(f"   ❌ Filtered out: {report['statistics']['excluded_count']:,}")
    print(f"   ❓ Unknown domains: {report['statistics']['unknown_count']:,}")

    # Quality assessment
    stats = report['statistics']
    excellent = stats['quality_distribution']['excellent']
    high = stats['quality_distribution']['high']
    total = stats['total_found']
    quality_percentage = ((excellent + high) / total * 100) if total > 0 else 0
    print(f"\n📊 Quality Assessment:")
    print(f"   Excellent + High Quality: {excellent + high}/{total} ({quality_percentage:.1f}%)")
    if quality_percentage >= 80:
        print(f"🎉 EXCELLENT! {quality_percentage:.1f}% high-quality resources detected!")
        print("✅ This collection is ready for import into your resource database.")
        recommendation = "PROCEED"
    elif quality_percentage >= 60:
        print(f"👍 GOOD! {quality_percentage:.1f}% high-quality resources detected.")
        print("✅ This is a solid foundation for your resource database.")
        recommendation = "PROCEED"
    else:
        print(f"⚠️ MIXED: Only {quality_percentage:.1f}% high-quality resources.")
        print("❓ You may want to review and adjust criteria.")
        recommendation = "REVIEW" if total > 0 else "NO_RESOURCES"

    # Show top categories
    categories = stats['categories']
    print(f"\n📁 Resource Categories Found:")
    for category, count in sorted(categories.items(), key=lambda x: x[1], reverse=True)[:8]:
        print(f"   {category}: {count} resources")
    
    print(f"\n💡 RECOMMENDATION: {recommendation}")
    if recommendation == "PROCEED":
        print("🚀 Next step: Import these optimized resources")
        print(f"   Command: ./pepe-admin resources migrate")
        print(f"   (This will use: {output_path})")
    else:
        print("🔍 Next step: Review results and adjust quality criteria")
    
    print(f"\n📄 Files created:")
    print(f"   • Detailed report: {output_path}")
    print(f"   • Export file: {export_path}")
    
    # NEW: Display enrichment statistics with better formatting
    if detector.enrichment:
        enrichment_stats = detector.enrichment.get_stats()
        gpt5_stats = detector.enrichment.gpt5.get_stats() if detector.enrichment.gpt5 else None
        
        print(f"\n🤖 Enrichment Performance:")
        print(f"   📦 Resources processed: {enrichment_stats['total_processed']:,}")
        print(f"   📝 Message-based extraction: {enrichment_stats['message_based']:,}")
        print(f"   🌐 Web scraping used: {enrichment_stats['web_scraped']:,}")
        print(f"   ✨ Titles from scraping: {enrichment_stats['titles_scraped']:,}")
        print(f"   🤖 Titles from LLM: {enrichment_stats['titles_generated']:,}")
        print(f"   📝 Descriptions from LLM: {enrichment_stats['descriptions_generated']:,}")
        
        if gpt5_stats:
            print(f"\n💰 OpenAI API Usage:")
            api_calls = gpt5_stats['gpt5_calls']
            cached = gpt5_stats['gpt5_cached']
            fallback = gpt5_stats['fallback_calls']
            errors = gpt5_stats['errors']
            
            print(f"   🔵 New API calls: {api_calls:,}")
            print(f"   🟢 Cached responses: {cached:,}")
            print(f"   🟡 Fallback to local LLM: {fallback:,}")
            if errors > 0:
                print(f"   🔴 Errors: {errors:,}")
            
            total_calls = api_calls + cached
            if total_calls > 0:
                cache_rate = (cached / total_calls) * 100
                estimated_cost = api_calls * 0.02  # ~$0.02 per resource
                savings = cached * 0.02
                print(f"\n   📊 Efficiency:")
                print(f"      • Cache hit rate: {cache_rate:.1f}%")
                print(f"      • API cost: ${estimated_cost:.2f}")
                print(f"      • Cache savings: ${savings:.2f}")
                print(f"      • Total saved: ${savings:.2f} ({cached:,} cached calls)")
    
    # Final completion banner
    print("\n" + "="*80)
    print("✅ RESOURCE DETECTION COMPLETE!")
    print("="*80)
    print(f"\n📂 Output files ready:")
    print(f"   {output_path}")
    print(f"   {export_path}")
    
    if recommendation == "PROCEED":
        print(f"\n🚀 Next step: Review and import resources")
        print(f"   The detected resources are ready for use!")
    
    return report

if __name__ == '__main__':
    main() 