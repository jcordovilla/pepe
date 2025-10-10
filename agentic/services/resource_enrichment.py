#!/usr/bin/env python3
"""
Resource Enrichment Service - Phase 1 & 2 Implementation
Provides high-quality title and description generation using GPT-5 mini
"""

import asyncio
import logging
from typing import Dict, Any, Optional
from urllib.parse import urlparse

from .gpt5_service import GPT5Service
from .web_scraper import WebScraper

logger = logging.getLogger(__name__)


class ResourceEnrichment:
    """
    Enhanced resource enrichment with:
    - Phase 1: Intelligent title extraction and generation
    - Phase 2: Rich contextual description generation
    """
    
    def __init__(self, use_gpt5: bool = True):
        self.gpt5 = GPT5Service(use_cache=True) if use_gpt5 else None
        self.scraper = WebScraper()
        self.use_gpt5 = use_gpt5
        
        self.stats = {
            'total_processed': 0,
            'titles_scraped': 0,
            'titles_generated': 0,
            'descriptions_generated': 0,
            'errors': 0
        }
    
    async def enrich_resource(
        self,
        url: str,
        message: Dict[str, Any],
        channel_name: str
    ) -> Dict[str, Any]:
        """
        Complete resource enrichment pipeline
        Returns: {title, description, metadata}
        """
        
        self.stats['total_processed'] += 1
        
        enriched = {
            'url': url,
            'title': None,
            'description': None,
            'scraped_metadata': None,
            'enrichment_method': []
        }
        
        try:
            # Step 1: Scrape web page for metadata
            logger.debug(f"🌐 Scraping metadata from: {url}")
            scraped = await self.scraper.extract_metadata(url)
            enriched['scraped_metadata'] = scraped
            
            # Step 2: Extract or generate title
            title = await self._get_title(url, message, channel_name, scraped)
            enriched['title'] = title
            
            # Step 3: Generate rich description
            description = await self._get_description(url, message, channel_name, scraped, title)
            enriched['description'] = description
            
            return enriched
            
        except Exception as e:
            logger.error(f"❌ Enrichment failed for {url}: {e}")
            self.stats['errors'] += 1
            
            # Fallback to basic info
            enriched['title'] = self._generate_fallback_title(url)
            enriched['description'] = self._generate_fallback_description(url, message)
            enriched['enrichment_method'].append('fallback')
            
            return enriched
    
    async def _get_title(
        self,
        url: str,
        message: Dict,
        channel_name: str,
        scraped: Dict
    ) -> str:
        """
        Phase 1: Intelligent title extraction/generation
        Strategy:
        1. Use scraped title if high quality
        2. Generate with GPT-5 mini if needed
        3. Fallback to URL-based title
        """
        
        # Strategy 1: Use scraped title if it exists and is good quality
        scraped_title = scraped.get('title')
        if scraped_title and self._is_good_title(scraped_title):
            self.stats['titles_scraped'] += 1
            logger.debug(f"✅ Using scraped title: {scraped_title}")
            return scraped_title
        
        # Strategy 2: Generate with GPT-5 mini
        if self.use_gpt5 and self.gpt5:
            try:
                generated_title = await self._generate_title_gpt5(url, message, channel_name, scraped)
                if generated_title and len(generated_title) > 5:
                    self.stats['titles_generated'] += 1
                    logger.debug(f"🤖 Generated title: {generated_title}")
                    return generated_title
            except Exception as e:
                logger.warning(f"⚠️ GPT-5 title generation failed: {e}")
        
        # Strategy 3: Fallback
        fallback_title = self._generate_fallback_title(url, scraped)
        logger.debug(f"🔄 Using fallback title: {fallback_title}")
        return fallback_title
    
    def _is_good_title(self, title: str) -> bool:
        """Check if scraped title is high quality"""
        if not title or len(title) < 10:
            return False
        
        # Check for generic/bad titles
        bad_patterns = [
            'untitled',
            'no title',
            'page not found',
            '404',
            'error',
            'access denied'
        ]
        
        title_lower = title.lower()
        if any(pattern in title_lower for pattern in bad_patterns):
            return False
        
        # Title should have some meaningful content
        if len(title.split()) < 3:
            return False
        
        return True
    
    async def _generate_title_gpt5(
        self,
        url: str,
        message: Dict,
        channel_name: str,
        scraped: Dict
    ) -> str:
        """Generate title using GPT-5 mini"""
        
        domain = urlparse(url).netloc
        message_content = message.get('content', '')[:500]
        author = message.get('author', {}).get('display_name', 'Unknown')
        
        # Include scraped context if available
        scraped_desc = scraped.get('description', '')[:300] if scraped.get('description') else ''
        scraped_preview = scraped.get('content_preview', '')[:300] if scraped.get('content_preview') else ''
        
        context_parts = []
        if message_content:
            context_parts.append(f"Discord message: {message_content}")
        if scraped_desc:
            context_parts.append(f"Page description: {scraped_desc}")
        if scraped_preview:
            context_parts.append(f"Content preview: {scraped_preview}")
        
        context = "\n".join(context_parts)
        
        prompt = f"""Generate a concise, specific title for this resource.

URL: {url}
Domain: {domain}
Shared by: {author} in #{channel_name}
Context:
{context}

Requirements:
- Maximum 10-12 words
- Be specific, not generic (avoid "YouTube Video", "Resource", "Link")
- Capture the actual content/topic
- Format examples:
  * "MIT Research: AI Models Learn Human Sketching Techniques"
  * "OpenAI Introduces ChatGPT Agent for Complex Tasks"
  * "Google Cloud: 101 Real-World Generative AI Use Cases"
  * "Anthropic Achieves ISO 42001 AI Responsibility Certification"

Generate a specific, informative title:"""

        title = await self.gpt5.generate(
            prompt=prompt,
            temperature=0.3,
            max_tokens=30
        )
        
        # Clean up the title
        title = title.strip().strip('"').strip("'")
        
        return title
    
    def _generate_fallback_title(self, url: str, scraped: Dict = None) -> str:
        """Generate fallback title from URL structure"""
        
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        
        # Domain-specific fallback titles
        if 'youtube.com' in domain or 'youtu.be' in domain:
            return "YouTube Video"
        elif 'arxiv.org' in domain:
            paper_id = parsed.path.split('/')[-1]
            return f"arXiv Paper: {paper_id}"
        elif 'github.com' in domain:
            path_parts = parsed.path.strip('/').split('/')
            if len(path_parts) >= 2:
                return f"GitHub: {path_parts[0]}/{path_parts[1]}"
            return "GitHub Repository"
        elif 'huggingface.co' in domain:
            path_parts = parsed.path.strip('/').split('/')
            if len(path_parts) >= 2:
                return f"Hugging Face: {path_parts[0]}/{path_parts[1]}"
            return "Hugging Face AI/ML Resource"
        elif '.pdf' in url.lower():
            filename = parsed.path.split('/')[-1]
            return f"PDF: {filename}"
        else:
            # Generic fallback
            return f"Resource from {domain}"
    
    async def _get_description(
        self,
        url: str,
        message: Dict,
        channel_name: str,
        scraped: Dict,
        title: str
    ) -> str:
        """
        Phase 2: Generate rich, contextual description
        """
        
        # Use scraped description if it's comprehensive
        scraped_desc = scraped.get('description')
        if scraped_desc and len(scraped_desc) > 100:
            # Still enhance it with GPT-5 if available
            if self.use_gpt5 and self.gpt5:
                try:
                    enhanced = await self._enhance_description_gpt5(
                        url, message, channel_name, scraped, title, scraped_desc
                    )
                    if enhanced:
                        self.stats['descriptions_generated'] += 1
                        return enhanced
                except Exception as e:
                    logger.warning(f"⚠️ Description enhancement failed: {e}")
            
            return scraped_desc
        
        # Generate description with GPT-5
        if self.use_gpt5 and self.gpt5:
            try:
                generated = await self._generate_description_gpt5(
                    url, message, channel_name, scraped, title
                )
                if generated:
                    self.stats['descriptions_generated'] += 1
                    return generated
            except Exception as e:
                logger.warning(f"⚠️ GPT-5 description generation failed: {e}")
        
        # Fallback
        return self._generate_fallback_description(url, message, scraped)
    
    async def _generate_description_gpt5(
        self,
        url: str,
        message: Dict,
        channel_name: str,
        scraped: Dict,
        title: str
    ) -> str:
        """Generate rich description using GPT-5 mini"""
        
        domain = urlparse(url).netloc
        message_content = message.get('content', '')[:800]
        author = message.get('author', {}).get('display_name', 'Unknown')
        
        # Build comprehensive context
        context_parts = [f"Title: {title}"]
        
        if message_content:
            context_parts.append(f"Discord message: {message_content}")
        
        if scraped.get('description'):
            context_parts.append(f"Page description: {scraped['description'][:400]}")
        
        if scraped.get('content_preview'):
            context_parts.append(f"Content preview: {scraped['content_preview'][:600]}")
        
        if scraped.get('author'):
            context_parts.append(f"Author: {scraped['author']}")
        
        context = "\n".join(context_parts)
        
        prompt = f"""Generate a detailed, informative description (60-100 words) for this AI/tech resource.

URL: {url}
Domain: {domain}
Shared by: {author} in #{channel_name}

Context:
{context}

Requirements:
1. Start directly with key information (no "This is..." or "This article...")
2. Include specific technical details, frameworks, technologies, or topics
3. Explain the resource's value, use cases, and target audience
4. Mention notable authors, institutions, companies, or dates if relevant
5. Be objective and informative, not promotional
6. Use 60-100 words
7. Write in a clear, professional style

Description:"""

        description = await self.gpt5.generate(
            prompt=prompt,
            temperature=0.4,
            max_tokens=150
        )
        
        return description.strip()
    
    async def _enhance_description_gpt5(
        self,
        url: str,
        message: Dict,
        channel_name: str,
        scraped: Dict,
        title: str,
        existing_description: str
    ) -> str:
        """Enhance an existing description with more context"""
        
        message_content = message.get('content', '')[:500]
        
        prompt = f"""Enhance this resource description with more context and specificity.

Title: {title}
URL: {url}
Current description: {existing_description}
Discord context: {message_content}

Requirements:
1. Keep the core information from the current description
2. Add specific technical details, use cases, or context
3. Make it more informative and actionable
4. 60-100 words
5. Start directly with information (no "This describes...")

Enhanced description:"""

        enhanced = await self.gpt5.generate(
            prompt=prompt,
            temperature=0.3,
            max_tokens=150
        )
        
        return enhanced.strip()
    
    def _generate_fallback_description(
        self,
        url: str,
        message: Dict,
        scraped: Dict = None
    ) -> str:
        """Generate basic fallback description"""
        
        domain = urlparse(url).netloc
        message_content = message.get('content', '')[:200]
        
        # Use scraped description if available
        if scraped and scraped.get('description'):
            return scraped['description']
        
        # Domain-specific fallbacks
        if 'arxiv.org' in domain:
            return "Research paper from arXiv covering AI, machine learning, or computer science topics."
        elif 'github.com' in domain:
            path = urlparse(url).path
            return f"Code repository on GitHub: {path.strip('/')}"
        elif 'youtube.com' in domain or 'youtu.be' in domain:
            return "Educational video content covering AI, technology, or related topics."
        elif 'huggingface.co' in domain:
            return "AI/ML resource from Hugging Face, including models, datasets, or documentation."
        elif any(domain.endswith(news) for news in ['.com', '.org', '.net']):
            if message_content:
                return f"Resource shared in context: {message_content}"
            return f"Content from {domain}"
        
        return f"Resource from {domain}"
    
    def get_stats(self) -> Dict[str, int]:
        """Get enrichment statistics"""
        return self.stats.copy()
    
    def reset_stats(self):
        """Reset statistics"""
        self.stats = {
            'total_processed': 0,
            'titles_scraped': 0,
            'titles_generated': 0,
            'descriptions_generated': 0,
            'errors': 0
        }

