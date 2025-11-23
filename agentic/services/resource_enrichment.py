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
    
    def __init__(self, use_gpt5: bool = False):
        # Default to local LLM (use_gpt5=False) - works great and is free
        # Set use_gpt5=True to use OpenAI API (requires OPENAI_API_KEY)
        self.gpt5 = GPT5Service(use_cache=True) if use_gpt5 else None
        self.scraper = WebScraper()
        self.use_gpt5 = use_gpt5
        
        self.stats = {
            'total_processed': 0,
            'message_based': 0,
            'web_scraped': 0,
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
        Complete resource enrichment pipeline - Hybrid approach
        1. Try message-based extraction first (fast)
        2. Fall back to web scraping only if needed
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
            # Step 1: Try extracting from Discord message context (FAST)
            message_content = message.get('content', '').strip()
            
            has_context = self._has_sufficient_context(message_content, url)
            print(f"   🔍 Message context check: {has_context} (content: {len(message_content)} chars)")
            
            if has_context:
                # Message has good context - use it directly
                logger.debug(f"📝 Using message context for: {url}")
                print(f"   📝 Attempting message-based extraction...")
                title, description = await self._extract_from_message(url, message, channel_name)
                
                if title and description:
                    enriched['title'] = title
                    enriched['description'] = description
                    enriched['enrichment_method'].append('message_extraction')
                    self.stats['message_based'] += 1
                    print(f"   ✅ Message-based extraction succeeded!")
                    return enriched
                else:
                    print(f"   ⚠️ Message-based extraction returned None, falling back to web scraping...")
            
            # Step 2: Fall back to web scraping if message has insufficient context
            logger.debug(f"🌐 Scraping metadata from: {url}")
            scraped = await self.scraper.extract_metadata(url)
            enriched['scraped_metadata'] = scraped
            
            # Step 3: Extract or generate title
            title = await self._get_title(url, message, channel_name, scraped)
            enriched['title'] = title
            
            # Step 4: Generate rich description
            description = await self._get_description(url, message, channel_name, scraped, title)
            enriched['description'] = description
            enriched['enrichment_method'].append('web_scraping')
            self.stats['web_scraped'] += 1
            
            return enriched
            
        except Exception as e:
            logger.error(f"❌ Enrichment failed for {url}: {e}")
            self.stats['errors'] += 1
            
            # Fallback to basic info
            enriched['title'] = self._generate_fallback_title(url)
            enriched['description'] = self._generate_fallback_description(url, message)
            enriched['enrichment_method'].append('fallback')
            
            return enriched
    
    def _has_sufficient_context(self, message_content: str, url: str) -> bool:
        """Check if Discord message has enough context to extract title/description"""
        if not message_content:
            return False
        
        # Remove the URL from message to see how much text remains
        text_without_url = message_content.replace(url, '').strip()
        
        # Need at least 20 characters of text beyond the URL
        if len(text_without_url) < 20:
            return False
        
        # Need at least 3 words
        word_count = len(text_without_url.split())
        if word_count < 3:
            return False
        
        return True
    
    async def _extract_from_message(
        self,
        url: str,
        message: Dict[str, Any],
        channel_name: str
    ) -> tuple[str, str]:
        """
        Extract title and description from Discord message using GPT-5 mini
        Returns: (title, description)
        """
        
        if not self.use_gpt5 or not self.gpt5:
            return None, None
        
        message_content = message.get('content', '')
        author = message.get('author', {}).get('display_name', 'Unknown')
        domain = urlparse(url).netloc
        
        # Optimized prompt for GPT-5-mini - structured and concise
        prompt = f"""Extract title and description from this Discord-shared resource.

URL: {url}
Message: "{message_content}"
Channel: #{channel_name}

Output format (strict):
TITLE: <5-10 words, factual, specific to content>
DESCRIPTION: <40-60 words, what it is + why it matters>

Rules:
- No marketing language or superlatives
- Be specific (not "YouTube Video" but "Tutorial on Fine-tuning LLMs")
- Base on message context, don't invent details"""

        try:
            # GPT-5-mini uses reasoning tokens + output tokens
            # Need MUCH higher limit: reasoning uses 200-500 tokens, then output needs space
            response = await self.gpt5.generate(
                prompt=prompt,
                temperature=1.0,
                max_tokens=800  # High limit for reasoning + output
            )
            
            print(f"   🤖 GPT-5 Response (full): [{response}]")
            print(f"   📏 Response length: {len(response)} chars")
            
            # Parse response - handle both plain and markdown formatting
            lines = response.strip().split('\n')
            title = None
            description = None
            
            for line in lines:
                # Handle both "TITLE:" and "**TITLE:**" formats
                line_clean = line.strip().replace('**', '').replace('*', '')
                
                if line_clean.startswith('TITLE:'):
                    title = line_clean.replace('TITLE:', '').strip()
                elif line_clean.startswith('DESCRIPTION:'):
                    description = line_clean.replace('DESCRIPTION:', '').strip()
            
            print(f"   📋 Parsed - Title: {title}, Description: {description[:50] if description else None}...")
            
            if title and description:
                self.stats['titles_generated'] += 1
                self.stats['descriptions_generated'] += 1
                return title, description
            
            print(f"   ⚠️ Failed to parse title/description from GPT-5 response")
            return None, None
            
        except Exception as e:
            logger.warning(f"⚠️ Message extraction failed: {e}")
            print(f"   ❌ Exception in message extraction: {e}")
            return None, None
    
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
        
        prompt = f"""Generate a specific title for this resource (5-10 words max).

URL: {url}
Context:
{context}

Good examples:
- "Fine-tuning Llama 3 for Domain-Specific Tasks"
- "Google DeepMind Paper on Multimodal Reasoning"
- "Hugging Face Dataset for Code Generation"

Bad examples (too generic):
- "YouTube Video" / "GitHub Repository" / "Interesting Article"

Title:"""

        title = await self.gpt5.generate(
            prompt=prompt,
            temperature=1.0,  # GPT-5-mini requires default temperature
            max_tokens=600  # High limit for reasoning model + output
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
        
        prompt = f"""Write a 50-80 word description for this resource.

URL: {url}
{context}

Guidelines:
- Start with the main topic/content (not "This resource...")
- Include: what it covers, key technologies/concepts, target audience
- Be factual and specific, not promotional
- Mention authors/organizations if notable

Description:"""

        description = await self.gpt5.generate(
            prompt=prompt,
            temperature=1.0,  # GPT-5-mini requires default temperature
            max_tokens=1000  # High limit for reasoning model + output
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
        
        prompt = f"""Improve this description (50-80 words).

Title: {title}
Current: {existing_description}
Additional context: {message_content}

Keep core info, add specificity. Start directly with content, not "This..."

Enhanced:"""

        enhanced = await self.gpt5.generate(
            prompt=prompt,
            temperature=1.0,  # GPT-5-mini requires default temperature
            max_tokens=400  # Increased for reasoning tokens + output
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
            'message_based': 0,
            'web_scraped': 0,
            'titles_scraped': 0,
            'titles_generated': 0,
            'descriptions_generated': 0,
            'errors': 0
        }

