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
        # Model selection for resource detection
        self.use_fast_model = use_fast_model
        self.fast_model = os.getenv('LLM_FAST_MODEL', 'phi3:mini')  # Smaller, faster model
        self.standard_model = os.getenv('LLM_MODEL', 'llama3.1:8b')   # Standard model
        
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
            
            # Generate description with progress indication
            description = self._generate_description(message, url)
            
            resource = {
                'url': url,
                'domain': domain,
                'category': domain_info['category'],
                'quality_score': domain_info['score'],
                'channel_name': channel_name,
                'author': author_display,
                'timestamp': ts_str,
                'jump_url': jump_url,
                'description': description
            }
            content_lower = message.get('content', '').lower()
            if any(keyword in content_lower for keyword in ['paper', 'research', 'study', 'tutorial']):
                resource['quality_score'] = min(1.0, resource['quality_score'] + 0.1)
            return resource
        except Exception as e:
            self.stats['parsing_errors'] += 1
            return None
    
    def _generate_report(self, analyze_unknown: bool = False) -> Dict[str, Any]:
        """Generate comprehensive analysis report"""
        
        # Sort resources by quality score
        sorted_resources = sorted(self.detected_resources, key=lambda x: x['quality_score'], reverse=True)
        
        # Generate statistics
        category_stats = Counter([r['category'] for r in sorted_resources])
        domain_stats = Counter([r['domain'] for r in sorted_resources])
        channel_stats = Counter([r['channel_name'] for r in sorted_resources])
        
        # Calculate quality distribution
        quality_distribution = {
            'excellent': len([r for r in sorted_resources if r['quality_score'] >= 0.9]),
            'high': len([r for r in sorted_resources if 0.8 <= r['quality_score'] < 0.9]),
            'good': len([r for r in sorted_resources if 0.7 <= r['quality_score'] < 0.8]),
            'fair': len([r for r in sorted_resources if r['quality_score'] < 0.7])
        }
        
        print("\n📊 Fresh Resource Detection Results")
        print("=" * 40)
        print(f"✅ High-quality resources found: {len(sorted_resources)}")
        print(f"⏭️ Skipped (already processed): {self.stats.get('skipped_processed', 0)}")
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
            print(f"   {i+1}. [{resource['category']}] {resource['url']}")
            print(f"      Quality: {resource['quality_score']:.2f} | Author: {resource['author']}")
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
    
    def save_resources(self, output_path: Path, analyze_unknown: bool = False):
        """Save detected resources to JSON file"""
        report = self._generate_report(analyze_unknown)
        
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
    parser.add_argument('--reset-cache', action='store_true',
                       help='Reset processed URLs cache and reprocess all resources')
    args = parser.parse_args()

    use_fast_model = args.fast_model or (not args.standard_model)  # Default to fast model

    if args.reset_cache:
        cache_file = project_root / 'data' / 'processed_resources.json'
        if cache_file.exists():
            cache_file.unlink()
            print("🗑️ Reset processed URLs cache")

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
    print("🔍 Starting resource analysis...")

    # Analyze messages with progress bar and incremental logic
    skipped = 0
    processed = 0
    resources_found = 0
    urls_extracted = 0
    
    print(f"\n🔍 Analyzing {len(messages):,} messages for high-quality resources...")
    print("📊 Progress indicators will show:")
    print("   • Message analysis progress")
    print("   • URL extraction and evaluation")
    print("   • Resource quality assessment")
    print("-" * 60)
    
    with tqdm(messages, desc="📝 Analyzing messages", unit="msg", position=0, leave=True) as msg_pbar:
        for i, message in enumerate(msg_pbar):
            content = message.get('content', '')
            if not content:
                continue
                
            urls = re.findall(r'https?://[^\s\n\r<>"]+', content)
            urls_extracted += len(urls)
            new_url_found = False
            
            # Show URL extraction progress in postfix
            if urls:
                msg_pbar.set_postfix({
                    "processed": processed,
                    "skipped": skipped,
                    "resources": resources_found,
                    "urls": urls_extracted,
                    "progress": f"{i+1}/{len(messages)}"
                })
            
            for url in urls:
                if detector._is_url_processed(url):
                    skipped += 1
                    continue
                    
                resource = detector._evaluate_url(url, message, message['channel_name'], analyze_unknown=False)
                if resource:
                    detector.detected_resources.append(resource)
                    detector.stats['total_resources'] += 1
                    detector.stats[f'category_{resource["category"]}'] += 1
                    processed += 1
                    resources_found += 1
                    new_url_found = True
            
            # Update progress bar with detailed stats
            msg_pbar.set_postfix({
                "processed": processed,
                "skipped": skipped,
                "resources": resources_found,
                "urls": urls_extracted,
                "progress": f"{i+1}/{len(messages)}"
            })
            
            # Force update every 100 messages to ensure progress is visible
            if (i + 1) % 100 == 0:
                msg_pbar.refresh()

    print(f"\n✅ Analysis complete!")
    print(f"📊 Results: {resources_found} new resources found, {skipped} URLs skipped (already processed)")

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

    print("\n📄 Saving results to files...")
    print("🔄 Creating detailed report and export files...")
    
    # Save results to JSON file with progress indication
    output_path = project_root / 'data' / 'optimized_fresh_resources.json'
    print(f"   📊 Generating detailed report: {output_path.name}")
    report = detector.save_resources(output_path, analyze_unknown=False)
    print("   ✅ Detailed report saved")

    # Also save a simplified export file with just the resources list
    export_path = project_root / 'data' / 'resources_export.json'
    print(f"   📤 Creating export file: {export_path.name}")
    export_data = {
        'export_date': datetime.now().isoformat(),
        'total_resources': len(report['resources']),
        'resources': report['resources']
    }
    with open(export_path, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)
    print("   ✅ Export file created")

    print(f"\n🎯 FINAL OPTIMIZED RESULTS:")
    print("=" * 60)
    print(f"✅ Found {report['statistics']['total_found']} high-quality resources!")
    print(f"⏭️ Skipped (already processed): {skipped}")
    print(f"❌ Excluded {report['statistics']['excluded_count']} low-quality URLs")
    print(f"❓ Unknown domains: {report['statistics']['unknown_count']}")
    print(f"🔗 Total URLs extracted: {urls_extracted}")
    print(f"📝 Messages analyzed: {len(messages):,}")

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
    
    return report

if __name__ == '__main__':
    main() 