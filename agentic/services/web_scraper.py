#!/usr/bin/env python3
"""
Web scraping utilities for resource metadata extraction
"""

import asyncio
import logging
from typing import Optional, Dict, Any
from urllib.parse import urlparse, parse_qs
import time
import re

import aiohttp
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class WebScraper:
    """Extract metadata and content from web pages"""

    def __init__(self, requests_per_second: float = 2.0):
        self.timeout = aiohttp.ClientTimeout(total=15)
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5'
        }

        # Rate limiting (MEDIUM-PRIORITY IMPROVEMENT)
        self._requests_per_second = requests_per_second
        self._min_interval = 1.0 / requests_per_second
        self._last_request_time: Dict[str, float] = {}  # Per-domain rate limiting
        self._global_last_request = 0.0

    async def _rate_limit(self, domain: str) -> None:
        """Apply rate limiting per domain and globally"""
        current_time = time.time()

        # Global rate limit
        global_elapsed = current_time - self._global_last_request
        if global_elapsed < self._min_interval:
            await asyncio.sleep(self._min_interval - global_elapsed)
            current_time = time.time()

        # Per-domain rate limit (more conservative)
        domain_last = self._last_request_time.get(domain, 0)
        domain_elapsed = current_time - domain_last
        domain_interval = self._min_interval * 2  # Double the interval per domain
        if domain_elapsed < domain_interval:
            await asyncio.sleep(domain_interval - domain_elapsed)

        # Update timestamps
        self._global_last_request = time.time()
        self._last_request_time[domain] = time.time()
    
    async def extract_metadata(self, url: str) -> Dict[str, Any]:
        """Extract title, description, and other metadata from URL"""

        metadata = {
            'url': url,
            'title': None,
            'description': None,
            'author': None,
            'published_date': None,
            'content_preview': None,
            'thumbnail_url': None,
            'domain': urlparse(url).netloc,
            'extraction_method': 'failed'
        }

        try:
            # Domain-specific extraction
            domain = metadata['domain'].lower()

            # Apply rate limiting before making requests
            await self._rate_limit(domain)
            
            if 'youtube.com' in domain or 'youtu.be' in domain:
                return await self._extract_youtube_metadata(url, metadata)
            elif 'arxiv.org' in domain:
                return await self._extract_arxiv_metadata(url, metadata)
            elif 'github.com' in domain:
                return await self._extract_github_metadata(url, metadata)
            elif 'huggingface.co' in domain:
                return await self._extract_huggingface_metadata(url, metadata)
            else:
                return await self._extract_generic_metadata(url, metadata)
                
        except Exception as e:
            logger.warning(f"⚠️ Failed to extract metadata from {url}: {e}")
            metadata['extraction_method'] = 'error'
            return metadata
    
    async def _extract_generic_metadata(self, url: str, metadata: Dict) -> Dict:
        """Extract metadata from generic web page"""
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=self.headers, timeout=self.timeout) as response:
                    if response.status != 200:
                        logger.warning(f"⚠️ HTTP {response.status} for {url}")
                        return metadata
                    
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # Extract title - try multiple sources
                    title = None
                    
                    # 1. Open Graph title
                    og_title = soup.find('meta', property='og:title')
                    if og_title and og_title.get('content'):
                        title = og_title['content']
                    
                    # 2. Twitter card title
                    if not title:
                        twitter_title = soup.find('meta', attrs={'name': 'twitter:title'})
                        if twitter_title and twitter_title.get('content'):
                            title = twitter_title['content']
                    
                    # 3. Standard title tag
                    if not title:
                        title_tag = soup.find('title')
                        if title_tag and title_tag.string:
                            title = title_tag.string.strip()
                    
                    # 4. H1 tag
                    if not title:
                        h1 = soup.find('h1')
                        if h1:
                            title = h1.get_text().strip()
                    
                    metadata['title'] = self._clean_title(title) if title else None
                    
                    # Extract description
                    og_desc = soup.find('meta', property='og:description')
                    if og_desc and og_desc.get('content'):
                        metadata['description'] = og_desc['content']
                    else:
                        meta_desc = soup.find('meta', attrs={'name': 'description'})
                        if meta_desc and meta_desc.get('content'):
                            metadata['description'] = meta_desc['content']
                    
                    # Extract author
                    author_meta = soup.find('meta', attrs={'name': 'author'})
                    if author_meta and author_meta.get('content'):
                        metadata['author'] = author_meta['content']
                    
                    # Extract thumbnail
                    og_image = soup.find('meta', property='og:image')
                    if og_image and og_image.get('content'):
                        metadata['thumbnail_url'] = og_image['content']
                    
                    # Extract content preview (first paragraph)
                    paragraphs = soup.find_all('p')
                    if paragraphs:
                        for p in paragraphs[:5]:
                            text = p.get_text().strip()
                            if len(text) > 100:
                                metadata['content_preview'] = text[:500]
                                break
                    
                    metadata['extraction_method'] = 'generic_html'
                    return metadata
                    
        except Exception as e:
            logger.warning(f"⚠️ Generic extraction failed for {url}: {e}")
            return metadata
    
    async def _extract_youtube_metadata(self, url: str, metadata: Dict) -> Dict:
        """Extract metadata from YouTube URL"""
        
        try:
            # Extract video ID
            video_id = None
            parsed = urlparse(url)
            
            if 'youtube.com' in parsed.netloc:
                query = parse_qs(parsed.query)
                video_id = query.get('v', [None])[0]
            elif 'youtu.be' in parsed.netloc:
                video_id = parsed.path.strip('/')
            
            if not video_id:
                return metadata
            
            # Scrape YouTube page for metadata
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=self.headers, timeout=self.timeout) as response:
                    if response.status != 200:
                        return metadata
                    
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # Extract title from meta tags
                    og_title = soup.find('meta', property='og:title')
                    if og_title and og_title.get('content'):
                        metadata['title'] = og_title['content']
                    
                    # Extract description
                    og_desc = soup.find('meta', property='og:description')
                    if og_desc and og_desc.get('content'):
                        metadata['description'] = og_desc['content']
                    
                    # Extract thumbnail
                    og_image = soup.find('meta', property='og:image')
                    if og_image and og_image.get('content'):
                        metadata['thumbnail_url'] = og_image['content']
                    
                    # Try to extract channel name
                    try:
                        channel_link = soup.find('link', {'itemprop': 'name'})
                        if channel_link and channel_link.get('content'):
                            metadata['author'] = channel_link['content']
                    except:
                        pass
                    
                    metadata['extraction_method'] = 'youtube'
                    return metadata
                    
        except Exception as e:
            logger.warning(f"⚠️ YouTube extraction failed for {url}: {e}")
            return metadata
    
    async def _extract_arxiv_metadata(self, url: str, metadata: Dict) -> Dict:
        """Extract metadata from arXiv paper"""
        
        try:
            # Extract paper ID
            parsed = urlparse(url)
            paper_id = parsed.path.split('/')[-1]
            
            # Use arXiv API
            api_url = f"http://export.arxiv.org/api/query?id_list={paper_id}"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(api_url, timeout=self.timeout) as response:
                    if response.status != 200:
                        return metadata
                    
                    xml_content = await response.text()
                    soup = BeautifulSoup(xml_content, 'xml')
                    
                    entry = soup.find('entry')
                    if entry:
                        title = entry.find('title')
                        if title:
                            metadata['title'] = title.get_text().strip()
                        
                        summary = entry.find('summary')
                        if summary:
                            metadata['description'] = summary.get_text().strip()
                        
                        authors = entry.find_all('author')
                        if authors:
                            author_names = [a.find('name').get_text() for a in authors]
                            metadata['author'] = ', '.join(author_names[:3])
                            if len(author_names) > 3:
                                metadata['author'] += ' et al.'
                        
                        published = entry.find('published')
                        if published:
                            metadata['published_date'] = published.get_text().strip()
                        
                        metadata['extraction_method'] = 'arxiv_api'
                        return metadata
                        
        except Exception as e:
            logger.warning(f"⚠️ arXiv extraction failed for {url}: {e}")
            return metadata
    
    async def _extract_github_metadata(self, url: str, metadata: Dict) -> Dict:
        """Extract metadata from GitHub repository"""
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=self.headers, timeout=self.timeout) as response:
                    if response.status != 200:
                        return metadata
                    
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # Extract repository name and description
                    og_title = soup.find('meta', property='og:title')
                    if og_title and og_title.get('content'):
                        metadata['title'] = og_title['content']
                    
                    og_desc = soup.find('meta', property='og:description')
                    if og_desc and og_desc.get('content'):
                        metadata['description'] = og_desc['content']
                    
                    # Extract README preview
                    readme = soup.find('article', class_=re.compile('markdown-body'))
                    if readme:
                        text = readme.get_text().strip()
                        if len(text) > 100:
                            metadata['content_preview'] = text[:1000]
                    
                    metadata['extraction_method'] = 'github'
                    return metadata
                    
        except Exception as e:
            logger.warning(f"⚠️ GitHub extraction failed for {url}: {e}")
            return metadata
    
    async def _extract_huggingface_metadata(self, url: str, metadata: Dict) -> Dict:
        """Extract metadata from Hugging Face"""
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=self.headers, timeout=self.timeout) as response:
                    if response.status != 200:
                        return metadata
                    
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # Extract title and description
                    og_title = soup.find('meta', property='og:title')
                    if og_title and og_title.get('content'):
                        metadata['title'] = og_title['content']
                    
                    og_desc = soup.find('meta', property='og:description')
                    if og_desc and og_desc.get('content'):
                        metadata['description'] = og_desc['content']
                    
                    metadata['extraction_method'] = 'huggingface'
                    return metadata
                    
        except Exception as e:
            logger.warning(f"⚠️ Hugging Face extraction failed for {url}: {e}")
            return metadata
    
    def _clean_title(self, title: str) -> str:
        """Clean and normalize title"""
        if not title:
            return None
        
        # Remove excessive whitespace
        title = ' '.join(title.split())
        
        # Remove common suffixes
        suffixes_to_remove = [
            ' - YouTube',
            ' | YouTube',
            ' | Twitter',
            ' | LinkedIn',
            ' - Medium',
            ' | Medium'
        ]
        
        for suffix in suffixes_to_remove:
            if title.endswith(suffix):
                title = title[:-len(suffix)].strip()
        
        # Truncate if too long
        if len(title) > 200:
            title = title[:197] + '...'
        
        return title

