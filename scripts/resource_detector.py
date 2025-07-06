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

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

class FreshResourceDetector:
    def __init__(self):
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
    
    def analyze_discord_messages(self, messages_dir: Path, analyze_unknown: bool = False) -> List[Dict[str, Any]]:
        """Analyze Discord messages and extract high-quality resources"""
        
        print("🔍 Fresh Resource Detection - Quality-First Approach")
        print("=" * 60)
        
        json_files = list(messages_dir.glob('*.json'))
        print(f"📁 Found {len(json_files)} message files to analyze")
        
        for i, json_file in enumerate(json_files):
            print(f"📄 Processing {json_file.name} ({i+1}/{len(json_files)})")
            
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                messages = data.get('messages', [])
                channel_name = data.get('channel_name', 'Unknown')
                
                for message in messages:
                    self._analyze_message(message, channel_name, analyze_unknown)
                
            except Exception as e:
                print(f"   ❌ Error processing {json_file.name}: {e}")
        
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
        """Generate an AI description for the resource using the LLM."""
        llm_endpoint = os.getenv('LLM_ENDPOINT', 'http://localhost:11434/api/generate')
        llm_model = os.getenv('LLM_MODEL', 'llama3.1:8b')
        
        # Extract key content for faster processing
        content = message.get('content', '')[:200]  # Limit content length
        domain = urlparse(url).netloc.lower()
        
        # Lean, focused prompt
        prompt = f"AI resource: {url}\nContext: {content}\nBrief AI/ML description (50 words max):"
        
        try:
            response = requests.post(llm_endpoint, json={
                "model": llm_model,
                "prompt": prompt,
                "max_tokens": 80,  # Reduced for faster generation
                "temperature": 0.1,  # Lower for more consistent, faster responses
                "top_p": 0.9,  # Add top_p for better quality/speed balance
                "stream": False  # Ensure no streaming for faster response
            }, timeout=10)  # Reduced timeout
            
            if response.status_code == 200:
                data = response.json()
                description = data.get('response', '').strip()
                return description if description else f"AI/ML resource from {domain}"
            else:
                return f"AI/ML resource from {domain}"
        except Exception as e:
            return f"AI/ML resource from {domain}"

    def _evaluate_url(self, url: str, message: Dict[str, Any], channel_name: str, analyze_unknown: bool = False) -> Dict[str, Any]:
        """Evaluate if a URL is a high-quality resource"""
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
        print(f"❌ Excluded low-quality URLs: {self.stats['excluded_domains']}")
        print(f"❓ Unknown domains skipped: {self.stats['unknown_domains']}")
        print(f"⚠️ Parsing errors: {self.stats['parsing_errors']}")
        
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
    detector = FreshResourceDetector()
    
    # Check if we have the SQLite database
    db_path = project_root / 'data' / 'discord_messages.db'
    
    if not db_path.exists():
        print(f"❌ Message database not found: {db_path}")
        print("💡 Run 'pepe-admin sync' first to fetch Discord messages")
        return
    
    print("🔍 Running resource detection from SQLite database...")
    
    # Read messages from SQLite database
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
            
            for row in cursor:
                # Convert SQLite row to message dict format
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
        
        print(f"📊 Found {len(messages):,} messages to analyze")
        
    except Exception as e:
        print(f"❌ Error reading database: {e}")
        return
    
    # Analyze messages
    print("🔍 Analyzing messages for resources...")
    for message in messages:
        detector._analyze_message(message, message['channel_name'], analyze_unknown=False)
    
    # Save results to JSON file
    output_path = project_root / 'data' / 'optimized_fresh_resources.json'
    report = detector.save_resources(output_path, analyze_unknown=False)
    
    # Also save a simplified export file with just the resources list
    export_path = project_root / 'data' / 'resources_export.json'
    export_data = {
        'export_date': datetime.now().isoformat(),
        'total_resources': len(report['resources']),
        'resources': report['resources']
    }
    
    with open(export_path, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"📄 Export file created: {export_path}")
    
    print(f"\n🎯 FINAL OPTIMIZED RESULTS:")
    print(f"✅ Found {report['statistics']['total_found']} high-quality resources!")
    print(f"❌ Excluded {report['statistics']['excluded_count']} low-quality URLs")
    print(f"❓ Unknown domains: {report['statistics']['unknown_count']}")
    
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
        recommendation = "REVIEW"
    
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