#!/usr/bin/env python3
"""
PDF Analyzer Service

Two-tier PDF analysis for resource detection:
- Tier 1: Text extraction and metadata (no LLM)
- Tier 2: Content analysis and description generation (LLM)

Supports both local LLM (deepseek-r1:8b) and OpenAI API modes.
"""

import asyncio
import hashlib
import logging
import os
import re
from dataclasses import dataclass
from io import BytesIO
from typing import Dict, Any, Optional, Tuple
from urllib.parse import urlparse

import aiohttp
import PyPDF2

logger = logging.getLogger(__name__)


@dataclass
class PDFAnalysisResult:
    """Result of PDF analysis"""
    url: str
    is_useful: bool  # Pass/fail decision
    document_type: str  # research_paper, tutorial, whitepaper, slides, report, other
    title: Optional[str]
    description: Optional[str]
    page_count: int
    text_length: int
    confidence: float  # 0.0 - 1.0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'url': self.url,
            'is_useful': self.is_useful,
            'document_type': self.document_type,
            'title': self.title,
            'description': self.description,
            'page_count': self.page_count,
            'text_length': self.text_length,
            'confidence': self.confidence,
            'error': self.error
        }


class PDFAnalyzer:
    """
    PDF analysis service for resource detection workflow.

    Usage:
        analyzer = PDFAnalyzer(use_openai=False)  # Use local LLM
        result = await analyzer.analyze_pdf(pdf_bytes, url)

        if result.is_useful:
            # Include in resource library with result.description
    """

    def __init__(self, use_openai: bool = False):
        self.use_openai = use_openai

        # LLM configuration
        if use_openai:
            self.api_key = os.getenv('OPENAI_API_KEY')
            self.model = os.getenv('OPENAI_MODEL', 'gpt-5-mini-2025-08-07')
            self.api_url = "https://api.openai.com/v1/chat/completions"
        else:
            self.llm_endpoint = os.getenv('LLM_ENDPOINT', 'http://localhost:11434/api/generate')
            self.model = os.getenv('LLM_MODEL', 'deepseek-r1:8b')

        # Analysis settings
        self.max_pages_to_extract = 5
        self.max_text_for_llm = 2000  # chars
        self.min_text_for_analysis = 100  # chars - skip if less

        # Cache for results
        self._cache: Dict[str, PDFAnalysisResult] = {}

        logger.info(f"📄 PDF Analyzer initialized: {'OpenAI' if use_openai else 'Local LLM'} mode, model={self.model}")

    async def analyze_pdf(self, pdf_content: bytes, url: str) -> PDFAnalysisResult:
        """
        Analyze a PDF document.

        Args:
            pdf_content: Raw PDF bytes
            url: Source URL (for caching and context)

        Returns:
            PDFAnalysisResult with pass/fail decision and description
        """
        # Check cache
        cache_key = hashlib.md5(url.encode()).hexdigest()
        if cache_key in self._cache:
            logger.debug(f"📦 PDF analysis cache hit: {url}")
            return self._cache[cache_key]

        try:
            # Tier 1: Extract text and metadata
            extraction = self._extract_pdf_content(pdf_content, url)

            if extraction['error']:
                return PDFAnalysisResult(
                    url=url,
                    is_useful=False,
                    document_type='unknown',
                    title=None,
                    description=None,
                    page_count=0,
                    text_length=0,
                    confidence=0.0,
                    error=extraction['error']
                )

            # Skip LLM if insufficient text (likely scanned/image PDF)
            if extraction['text_length'] < self.min_text_for_analysis:
                result = PDFAnalysisResult(
                    url=url,
                    is_useful=False,
                    document_type='scanned_or_image',
                    title=extraction['title'],
                    description="PDF with insufficient extractable text (possibly scanned or image-based)",
                    page_count=extraction['page_count'],
                    text_length=extraction['text_length'],
                    confidence=0.3,
                    error=None
                )
                self._cache[cache_key] = result
                return result

            # Tier 2: LLM content analysis
            analysis = await self._analyze_content_with_llm(
                extraction['text'],
                extraction['title'],
                url
            )

            result = PDFAnalysisResult(
                url=url,
                is_useful=analysis['is_useful'],
                document_type=analysis['document_type'],
                title=extraction['title'] or analysis.get('title'),
                description=analysis['description'],
                page_count=extraction['page_count'],
                text_length=extraction['text_length'],
                confidence=analysis['confidence'],
                error=None
            )

            self._cache[cache_key] = result
            return result

        except Exception as e:
            logger.error(f"❌ PDF analysis failed for {url}: {e}")
            return PDFAnalysisResult(
                url=url,
                is_useful=False,
                document_type='error',
                title=None,
                description=None,
                page_count=0,
                text_length=0,
                confidence=0.0,
                error=str(e)
            )

    def _extract_pdf_content(self, pdf_content: bytes, url: str) -> Dict[str, Any]:
        """
        Tier 1: Extract text and metadata from PDF.
        No LLM usage - pure extraction.
        """
        result = {
            'text': '',
            'title': None,
            'author': None,
            'page_count': 0,
            'text_length': 0,
            'error': None
        }

        try:
            pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_content))
            result['page_count'] = len(pdf_reader.pages)

            # Extract metadata
            metadata = pdf_reader.metadata
            if metadata:
                result['title'] = metadata.get('/Title', '').strip() or None
                result['author'] = metadata.get('/Author', '').strip() or None

            # Extract text from first N pages
            text_parts = []
            for i in range(min(self.max_pages_to_extract, result['page_count'])):
                try:
                    page_text = pdf_reader.pages[i].extract_text()
                    if page_text:
                        text_parts.append(page_text)
                except Exception as e:
                    logger.debug(f"Failed to extract page {i}: {e}")

            result['text'] = '\n\n'.join(text_parts)
            result['text_length'] = len(result['text'])

            # Try to extract title from first page if not in metadata
            if not result['title'] and result['text']:
                result['title'] = self._extract_title_from_text(result['text'])

            # Infer title from URL if still missing
            if not result['title']:
                result['title'] = self._extract_title_from_url(url)

        except Exception as e:
            result['error'] = f"PDF extraction failed: {str(e)}"
            logger.warning(f"⚠️ PDF extraction error for {url}: {e}")

        return result

    def _extract_title_from_text(self, text: str) -> Optional[str]:
        """Try to extract title from first lines of PDF text"""
        lines = text.strip().split('\n')

        for line in lines[:10]:  # Check first 10 lines
            line = line.strip()
            # Title heuristics: reasonably long, not too long, starts with capital
            if 10 < len(line) < 200 and line[0].isupper():
                # Skip common non-title patterns
                skip_patterns = [
                    r'^(abstract|introduction|table of contents|page \d)',
                    r'^\d+[\.\)]\s',  # Numbered items
                    r'^(http|www\.)',  # URLs
                    r'^\w+@\w+',  # Emails
                ]
                if not any(re.match(p, line, re.IGNORECASE) for p in skip_patterns):
                    return line

        return None

    def _extract_title_from_url(self, url: str) -> Optional[str]:
        """Extract potential title from URL path"""
        parsed = urlparse(url)
        path = parsed.path

        # Get filename without extension
        filename = path.split('/')[-1]
        if filename.lower().endswith('.pdf'):
            filename = filename[:-4]

        # Clean up filename
        if filename and len(filename) > 3:
            # Replace common separators with spaces
            title = re.sub(r'[-_]+', ' ', filename)
            # Remove version numbers, dates
            title = re.sub(r'\b(v?\d+\.?\d*|20\d{2})\b', '', title)
            title = title.strip()
            if len(title) > 5:
                return title.title()

        return None

    async def _analyze_content_with_llm(
        self,
        text: str,
        title: Optional[str],
        url: str
    ) -> Dict[str, Any]:
        """
        Tier 2: Analyze content with LLM to determine usefulness and generate description.
        """
        # Truncate text for LLM
        truncated_text = text[:self.max_text_for_llm]
        if len(text) > self.max_text_for_llm:
            truncated_text += "\n[... text truncated ...]"

        # Determine domain context
        domain = urlparse(url).netloc.lower()
        domain_context = ""
        if 'arxiv' in domain:
            domain_context = "This is from arXiv (academic preprint server)."
        elif 'ieee' in domain or 'acm' in domain:
            domain_context = "This is from a major academic publisher."
        elif 'github' in domain:
            domain_context = "This is documentation from GitHub."

        prompt = f"""Analyze this PDF document and determine if it's a useful resource.

{domain_context}

Title: {title or 'Unknown'}
URL: {url}

Text content (first {self.max_text_for_llm} chars):
---
{truncated_text}
---

Respond in this exact format:
USEFUL: [yes/no]
TYPE: [research_paper/tutorial/whitepaper/slides/report/documentation/other]
CONFIDENCE: [0.0-1.0]
DESCRIPTION: [50-70 word description of what this document covers and its value]

Guidelines:
- USEFUL=yes if it contains educational, technical, or research content
- USEFUL=no if it's spam, marketing fluff, or low-quality content
- Be specific in the description about the actual content, not generic"""

        try:
            if self.use_openai:
                response = await self._call_openai(prompt)
            else:
                response = await self._call_local_llm(prompt)

            return self._parse_llm_response(response)

        except Exception as e:
            logger.warning(f"⚠️ LLM analysis failed: {e}")
            # Fallback: assume useful if we got this far
            return {
                'is_useful': True,
                'document_type': 'unknown',
                'description': f"PDF document with {len(text)} characters of content.",
                'confidence': 0.5
            }

    async def _call_openai(self, prompt: str) -> str:
        """Call OpenAI API"""
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not set")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        # Reasoning models (gpt-5, o1) need more tokens for their thinking process
        is_reasoning_model = 'gpt-5' in self.model or 'o1' in self.model
        max_tokens = 1000 if is_reasoning_model else 300

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_completion_tokens": max_tokens
        }

        # Only add temperature for models that support it
        # gpt-5-mini and similar reasoning models don't support temperature
        if not is_reasoning_model:
            payload["temperature"] = 0.3

        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status != 200:
                    error = await response.text()
                    raise Exception(f"OpenAI API error {response.status}: {error}")

                data = await response.json()
                return data['choices'][0]['message']['content']

    async def _call_local_llm(self, prompt: str) -> str:
        """Call local Ollama LLM"""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.3,
                "num_predict": 300
            }
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.llm_endpoint,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=60)
            ) as response:
                if response.status != 200:
                    raise Exception(f"Local LLM error {response.status}")

                data = await response.json()
                return data.get('response', '')

    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse structured LLM response"""
        result = {
            'is_useful': True,
            'document_type': 'unknown',
            'description': '',
            'confidence': 0.5
        }

        lines = response.strip().split('\n')

        for line in lines:
            line_clean = line.strip().replace('**', '')

            if line_clean.upper().startswith('USEFUL:'):
                value = line_clean.split(':', 1)[1].strip().lower()
                result['is_useful'] = value in ('yes', 'true', '1')

            elif line_clean.upper().startswith('TYPE:'):
                value = line_clean.split(':', 1)[1].strip().lower()
                valid_types = ['research_paper', 'tutorial', 'whitepaper', 'slides', 'report', 'documentation', 'other']
                result['document_type'] = value if value in valid_types else 'other'

            elif line_clean.upper().startswith('CONFIDENCE:'):
                try:
                    value = float(line_clean.split(':', 1)[1].strip())
                    result['confidence'] = max(0.0, min(1.0, value))
                except ValueError:
                    pass

            elif line_clean.upper().startswith('DESCRIPTION:'):
                result['description'] = line_clean.split(':', 1)[1].strip()

        # If description not found in structured format, use whole response
        if not result['description'] and len(response) > 20:
            result['description'] = response[:200].strip()

        return result

    def clear_cache(self):
        """Clear the analysis cache"""
        self._cache.clear()
        logger.info("📄 PDF analysis cache cleared")
