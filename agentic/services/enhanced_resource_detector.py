"""
Enhanced Resource Detection System

Advanced resource detection with semantic understanding, quality scoring,
and comprehensive resource type identification for AI community content.
"""

import re
import json
import logging
import hashlib
import asyncio
import aiohttp
from typing import Dict, List, Any, Optional, Tuple, Set
from urllib.parse import urlparse, urljoin
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import mimetypes
import PyPDF2
import docx
from io import BytesIO
import base64

from openai import OpenAI
from ..cache.smart_cache import SmartCache
from ..utils.error_handling import safe_async, retry_on_error

logger = logging.getLogger(__name__)


class ResourceType(Enum):
    """Enhanced resource type classifications"""
    # Academic & Research
    RESEARCH_PAPER = "research_paper"
    DATASET = "dataset"
    ACADEMIC_ARTICLE = "academic_article"
    PREPRINT = "preprint"
    
    # Code & Development
    CODE_REPOSITORY = "code_repository"
    CODE_SNIPPET = "code_snippet"
    API_DOCUMENTATION = "api_documentation"
    SOFTWARE_TOOL = "software_tool"
    LIBRARY_FRAMEWORK = "library_framework"
    
    # Learning & Education
    TUTORIAL = "tutorial"
    COURSE = "course"
    DOCUMENTATION = "documentation"
    BLOG_POST = "blog_post"
    VIDEO_TUTORIAL = "video_tutorial"
    WEBINAR = "webinar"
    
    # Business & Industry
    WHITEPAPER = "whitepaper"
    CASE_STUDY = "case_study"
    INDUSTRY_REPORT = "industry_report"
    PRESENTATION = "presentation"
    
    # AI/ML Specific
    MODEL = "model"
    MODEL_CARD = "model_card"
    BENCHMARK = "benchmark"
    EVALUATION = "evaluation"
    
    # Community & Social
    FORUM_POST = "forum_post"
    SOCIAL_MEDIA = "social_media"
    NEWS_ARTICLE = "news_article"
    PODCAST = "podcast"
    
    # Files & Media
    PDF_DOCUMENT = "pdf_document"
    SPREADSHEET = "spreadsheet"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    
    # Other
    UNKNOWN = "unknown"
    LOW_QUALITY = "low_quality"


class ResourceQuality(Enum):
    """Resource quality levels"""
    EXCELLENT = "excellent"  # 0.9-1.0
    HIGH = "high"           # 0.7-0.9
    GOOD = "good"           # 0.5-0.7
    FAIR = "fair"           # 0.3-0.5
    POOR = "poor"           # 0.1-0.3
    SPAM = "spam"           # 0.0-0.1


@dataclass
class ResourceMetadata:
    """Enhanced resource metadata"""
    # Basic info
    id: str
    type: ResourceType
    url: Optional[str] = None
    domain: Optional[str] = None
    title: Optional[str] = None
    description: Optional[str] = None
    
    # Quality metrics
    quality_score: float = 0.0
    quality_level: ResourceQuality = ResourceQuality.FAIR
    relevance_score: float = 0.0
    
    # Content analysis
    content_length: int = 0
    language: Optional[str] = None
    topics: List[str] = None
    keywords: List[str] = None
    
    # Technical details
    content_type: Optional[str] = None
    file_size: Optional[int] = None
    accessibility: bool = True
    
    # Temporal info
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    last_checked: Optional[datetime] = None
    
    # Source context
    message_id: Optional[str] = None
    channel_id: Optional[str] = None
    author: Optional[str] = None
    
    # Validation status
    is_valid: bool = True
    validation_errors: List[str] = None
    
    def __post_init__(self):
        if self.topics is None:
            self.topics = []
        if self.keywords is None:
            self.keywords = []
        if self.validation_errors is None:
            self.validation_errors = []


class EnhancedResourceDetector:
    """
    Advanced resource detection system with semantic understanding
    """
    
    def __init__(self, openai_client: OpenAI, config: Dict[str, Any] = None):
        self.openai_client = openai_client
        self.config = config or {}
        self.cache = SmartCache(self.config.get("cache", {}))
        
        # Advanced patterns for resource detection
        self._initialize_patterns()
        
        # Domain classifications
        self._initialize_domain_mappings()
        
        # HTTP session for validation
        self.session = None
        
        logger.info("🔍 Enhanced resource detector initialized")
    
    def _initialize_patterns(self):
        """Initialize advanced detection patterns"""
        # URL patterns - IMPROVED to capture more URL variations
        # Handles: special chars (+, ~, !, @), Unicode, encoded chars, parentheses (Wikipedia), etc.
        self.url_pattern = re.compile(
            r'https?://(?:[-\w.])+(?::\d+)?(?:/(?:[-\w.~!$&\'()*+,;=:@%]|(?:%[0-9A-Fa-f]{2}))*)*'
            r'(?:\?(?:[-\w.~!$&\'()*+,;=:@/?%]|(?:%[0-9A-Fa-f]{2}))*)?'
            r'(?:#(?:[-\w.~!$&\'()*+,;=:@/?%]|(?:%[0-9A-Fa-f]{2}))*)?'
        )
        
        # Code patterns - more comprehensive
        self.code_patterns = {
            'fenced_code': re.compile(r'```[\s\S]*?```'),
            'inline_code': re.compile(r'`[^`\n]+`'),
            'github_gist': re.compile(r'gist\.github\.com/[\w/]+'),
            'code_block': re.compile(r'(?:^|\n)(?:    |\t)[^\n]+(?:\n(?:    |\t)[^\n]+)*'),
            'function_def': re.compile(r'(?:def|function|class|interface)\s+\w+', re.IGNORECASE),
            'import_statement': re.compile(r'(?:import|from|#include|require)\s+[\w.]+', re.IGNORECASE)
        }
        
        # Academic patterns
        self.academic_patterns = {
            'doi': re.compile(r'10\.\d{4,}/[^\s]+'),
            'arxiv': re.compile(r'arxiv\.org/abs/[\d.]+'),
            'paper_title': re.compile(r'(?:paper|article|study|research):\s*([^.!?]+)', re.IGNORECASE),
            'citation': re.compile(r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+\(\d{4}\)', re.IGNORECASE)
        }
        
        # AI/ML specific patterns
        self.ai_ml_patterns = {
            'model_name': re.compile(r'\b(?:GPT|BERT|T5|LSTM|CNN|RNN|Transformer|CLIP|DALL-E|LLaMA|Claude|Gemini|Mixtral)\b', re.IGNORECASE),
            'dataset_name': re.compile(r'\b(?:ImageNet|COCO|MNIST|CIFAR|Wikipedia|Common Crawl|OpenWebText|BookCorpus)\b', re.IGNORECASE),
            'ml_framework': re.compile(r'\b(?:PyTorch|TensorFlow|JAX|Transformers|Hugging Face|scikit-learn|OpenAI|Anthropic)\b', re.IGNORECASE),
            'eval_metric': re.compile(r'\b(?:accuracy|precision|recall|F1|BLEU|ROUGE|perplexity|loss|AUC|mAP)\b', re.IGNORECASE)
        }
        
        # File extension patterns
        self.file_patterns = {
            'documents': re.compile(r'\.(?:pdf|doc|docx|ppt|pptx|txt|md|tex)$', re.IGNORECASE),
            'code': re.compile(r'\.(?:py|js|ts|java|cpp|c|h|r|sql|json|yaml|yml|xml|html|css)$', re.IGNORECASE),
            'data': re.compile(r'\.(?:csv|json|jsonl|parquet|h5|hdf5|npz|pkl|tsv)$', re.IGNORECASE),
            'images': re.compile(r'\.(?:png|jpg|jpeg|gif|svg|webp|bmp)$', re.IGNORECASE),
            'media': re.compile(r'\.(?:mp4|avi|mov|wmv|mp3|wav|flac|aac)$', re.IGNORECASE)
        }
    
    def _initialize_domain_mappings(self):
        """Initialize domain-to-resource-type mappings"""
        self.domain_mappings = {
            # Academic & Research
            'arxiv.org': ResourceType.PREPRINT,
            'scholar.google.com': ResourceType.ACADEMIC_ARTICLE,
            'pubmed.ncbi.nlm.nih.gov': ResourceType.RESEARCH_PAPER,
            'papers.nips.cc': ResourceType.RESEARCH_PAPER,
            'openreview.net': ResourceType.RESEARCH_PAPER,
            'ieee.org': ResourceType.RESEARCH_PAPER,
            'acm.org': ResourceType.RESEARCH_PAPER,
            'nature.com': ResourceType.RESEARCH_PAPER,
            'science.org': ResourceType.RESEARCH_PAPER,
            
            # Code & Development
            'github.com': ResourceType.CODE_REPOSITORY,
            'gitlab.com': ResourceType.CODE_REPOSITORY,
            'bitbucket.org': ResourceType.CODE_REPOSITORY,
            'stackoverflow.com': ResourceType.FORUM_POST,
            'docs.python.org': ResourceType.API_DOCUMENTATION,
            'pytorch.org': ResourceType.DOCUMENTATION,
            'tensorflow.org': ResourceType.DOCUMENTATION,
            
            # AI/ML Platforms (EXPANDED)
            'huggingface.co': ResourceType.MODEL,
            'openai.com': ResourceType.API_DOCUMENTATION,
            'anthropic.com': ResourceType.DOCUMENTATION,
            'kaggle.com': ResourceType.DATASET,
            'paperswithcode.com': ResourceType.BENCHMARK,
            'weights.biases.com': ResourceType.EVALUATION,
            'wandb.ai': ResourceType.EVALUATION,
            'replicate.com': ResourceType.MODEL,
            'lightning.ai': ResourceType.LIBRARY_FRAMEWORK,
            'mlflow.org': ResourceType.SOFTWARE_TOOL,
            'langchain.com': ResourceType.LIBRARY_FRAMEWORK,
            'llamaindex.ai': ResourceType.LIBRARY_FRAMEWORK,
            'together.ai': ResourceType.MODEL,
            'groq.com': ResourceType.SOFTWARE_TOOL,
            'mistral.ai': ResourceType.MODEL,
            'deepmind.com': ResourceType.RESEARCH_PAPER,
            'deepmind.google': ResourceType.RESEARCH_PAPER,
            'ai.meta.com': ResourceType.RESEARCH_PAPER,
            'stability.ai': ResourceType.MODEL,
            'colab.research.google.com': ResourceType.CODE_REPOSITORY,
            'colab.google': ResourceType.CODE_REPOSITORY,
            
            # Learning & Education
            'youtube.com': ResourceType.VIDEO_TUTORIAL,
            'youtu.be': ResourceType.VIDEO_TUTORIAL,
            'coursera.org': ResourceType.COURSE,
            'edx.org': ResourceType.COURSE,
            'udacity.com': ResourceType.COURSE,
            'medium.com': ResourceType.BLOG_POST,
            'towards.datascience.com': ResourceType.BLOG_POST,
            'deeplearning.ai': ResourceType.COURSE,
            
            # Business & Industry
            'mckinsey.com': ResourceType.INDUSTRY_REPORT,
            'deloitte.com': ResourceType.INDUSTRY_REPORT,
            'pwc.com': ResourceType.INDUSTRY_REPORT,
            'bcg.com': ResourceType.INDUSTRY_REPORT,
            'weforum.org': ResourceType.INDUSTRY_REPORT,
            
            # News & Media
            'techcrunch.com': ResourceType.NEWS_ARTICLE,
            'wired.com': ResourceType.NEWS_ARTICLE,
            'theverge.com': ResourceType.NEWS_ARTICLE,
            'arstechnica.com': ResourceType.NEWS_ARTICLE,
            'venturebeat.com': ResourceType.NEWS_ARTICLE,
            
            # Social & Community
            'reddit.com': ResourceType.FORUM_POST,
            'discord.com': ResourceType.SOCIAL_MEDIA,
            'twitter.com': ResourceType.SOCIAL_MEDIA,
            'linkedin.com': ResourceType.SOCIAL_MEDIA,
            'hackernews.com': ResourceType.FORUM_POST,
            
            # Low quality domains
            'tenor.com': ResourceType.LOW_QUALITY,
            'giphy.com': ResourceType.LOW_QUALITY,
            'imgur.com': ResourceType.LOW_QUALITY,
            'tiktok.com': ResourceType.LOW_QUALITY,
            'instagram.com': ResourceType.LOW_QUALITY,
        }
        
        # Quality domains (EXPANDED)
        self.high_quality_domains = {
            # Research
            'arxiv.org', 'scholar.google.com', 'nature.com', 'science.org',
            'semanticscholar.org', 'researchgate.net', 'openreview.net', 'aclanthology.org',
            # Code
            'github.com', 'gitlab.com', 'colab.research.google.com', 'colab.google',
            # AI/ML
            'huggingface.co', 'openai.com', 'anthropic.com', 'deepmind.com', 'deepmind.google',
            'ai.meta.com', 'mistral.ai', 'replicate.com', 'wandb.ai', 'lightning.ai',
            'langchain.com', 'llamaindex.ai', 'together.ai', 'stability.ai',
            # Education
            'deeplearning.ai', 'pytorch.org', 'tensorflow.org', 'kaggle.com',
            # Documentation
            'readthedocs.io', 'readthedocs.org', 'learn.microsoft.com', 'docs.python.org'
        }
        
        self.low_quality_domains = {
            'tenor.com', 'giphy.com', 'imgur.com', 'tiktok.com',
            'instagram.com', 'facebook.com', 'snapchat.com'
        }
    
    async def detect_resources(self, message: Dict[str, Any]) -> List[ResourceMetadata]:
        """
        Comprehensive resource detection from message content
        """
        resources = []
        content = message.get('content', '')
        attachments = message.get('attachments', [])
        
        # URLs
        url_resources = await self._detect_urls(content, message)
        resources.extend(url_resources)
        
        # Code snippets
        code_resources = await self._detect_code(content, message)
        resources.extend(code_resources)
        
        # Attachments
        attachment_resources = await self._detect_attachments(attachments, message)
        resources.extend(attachment_resources)
        
        # AI/ML entities
        ai_resources = await self._detect_ai_entities(content, message)
        resources.extend(ai_resources)
        
        # Academic references
        academic_resources = await self._detect_academic_references(content, message)
        resources.extend(academic_resources)
        
        # Apply quality scoring
        for resource in resources:
            resource.quality_score = await self._calculate_quality_score(resource, message)
            resource.quality_level = self._get_quality_level(resource.quality_score)
        
        # Filter out low quality resources
        high_quality_resources = [r for r in resources if r.quality_score >= self.config.get('min_quality_score', 0.3)]
        
        # Deduplicate
        deduplicated = await self._deduplicate_resources(high_quality_resources)
        
        return deduplicated
    
    async def _detect_urls(self, content: str, message: Dict[str, Any]) -> List[ResourceMetadata]:
        """Enhanced URL detection with comprehensive analysis"""
        resources = []
        urls = self.url_pattern.findall(content)
        
        for url in urls:
            try:
                parsed = urlparse(url)
                domain = parsed.netloc.lower()
                
                # Skip obviously low-quality domains
                if domain in self.low_quality_domains:
                    continue
                
                # Determine resource type
                resource_type = self._classify_url_type(url, domain)
                
                # Create metadata
                resource = ResourceMetadata(
                    id=hashlib.md5(url.encode()).hexdigest(),
                    type=resource_type,
                    url=url,
                    domain=domain,
                    message_id=message.get('message_id'),
                    channel_id=message.get('channel_id'),
                    author=message.get('author', {}).get('username'),
                    created_at=datetime.now(),
                    last_checked=datetime.now()
                )
                
                # Enhanced analysis
                await self._analyze_url_content(resource, content)
                
                resources.append(resource)
                
            except Exception as e:
                logger.debug(f"Error processing URL {url}: {e}")
                continue
        
        return resources
    
    async def _detect_code(self, content: str, message: Dict[str, Any]) -> List[ResourceMetadata]:
        """Enhanced code detection with quality analysis"""
        resources = []
        
        for pattern_name, pattern in self.code_patterns.items():
            matches = pattern.findall(content)
            
            for match in matches:
                # Filter out very short code snippets
                if len(match.strip()) < 20:
                    continue
                
                # Analyze code quality
                language = self._detect_code_language(match)
                quality_score = self._assess_code_quality(match, language)
                
                # Skip low-quality code
                if quality_score < 0.3:
                    continue
                
                resource = ResourceMetadata(
                    id=hashlib.md5(match.encode()).hexdigest(),
                    type=ResourceType.CODE_SNIPPET,
                    description=match[:100] + "..." if len(match) > 100 else match,
                    content_length=len(match),
                    language=language,
                    quality_score=quality_score,
                    message_id=message.get('message_id'),
                    channel_id=message.get('channel_id'),
                    author=message.get('author', {}).get('username'),
                    created_at=datetime.now()
                )
                
                resources.append(resource)
        
        return resources
    
    async def _detect_attachments(self, attachments: List[Dict], message: Dict[str, Any]) -> List[ResourceMetadata]:
        """Enhanced attachment detection with content analysis"""
        resources = []
        
        for attachment in attachments:
            filename = attachment.get('filename', '')
            content_type = attachment.get('content_type', '')
            file_size = attachment.get('size', 0)
            
            # Classify attachment type
            resource_type = self._classify_attachment_type(filename, content_type)
            
            # Skip very small files (likely not valuable)
            if file_size < 1024:  # 1KB
                continue
            
            resource = ResourceMetadata(
                id=hashlib.md5(filename.encode()).hexdigest(),
                type=resource_type,
                url=attachment.get('url'),
                title=filename,
                content_type=content_type,
                file_size=file_size,
                message_id=message.get('message_id'),
                channel_id=message.get('channel_id'),
                author=message.get('author', {}).get('username'),
                created_at=datetime.now()
            )
            
            # Extract content for certain file types
            if resource_type in [ResourceType.PDF_DOCUMENT, ResourceType.PRESENTATION]:
                await self._extract_file_content(resource, attachment)
            
            resources.append(resource)
        
        return resources
    
    async def _detect_ai_entities(self, content: str, message: Dict[str, Any]) -> List[ResourceMetadata]:
        """Detect AI/ML models, datasets, and tools mentioned"""
        resources = []
        
        # Model mentions
        for match in self.ai_ml_patterns['model_name'].finditer(content):
            model_name = match.group()
            
            resource = ResourceMetadata(
                id=hashlib.md5(f"model_{model_name}".encode()).hexdigest(),
                type=ResourceType.MODEL,
                title=model_name,
                description=f"AI model: {model_name}",
                message_id=message.get('message_id'),
                channel_id=message.get('channel_id'),
                author=message.get('author', {}).get('username'),
                created_at=datetime.now()
            )
            
            resources.append(resource)
        
        # Dataset mentions
        for match in self.ai_ml_patterns['dataset_name'].finditer(content):
            dataset_name = match.group()
            
            resource = ResourceMetadata(
                id=hashlib.md5(f"dataset_{dataset_name}".encode()).hexdigest(),
                type=ResourceType.DATASET,
                title=dataset_name,
                description=f"Dataset: {dataset_name}",
                message_id=message.get('message_id'),
                channel_id=message.get('channel_id'),
                author=message.get('author', {}).get('username'),
                created_at=datetime.now()
            )
            
            resources.append(resource)
        
        return resources
    
    async def _detect_academic_references(self, content: str, message: Dict[str, Any]) -> List[ResourceMetadata]:
        """Detect academic papers and references"""
        resources = []
        
        # DOI detection
        for match in self.academic_patterns['doi'].finditer(content):
            doi = match.group()
            
            resource = ResourceMetadata(
                id=hashlib.md5(f"doi_{doi}".encode()).hexdigest(),
                type=ResourceType.RESEARCH_PAPER,
                url=f"https://doi.org/{doi}",
                title=f"DOI: {doi}",
                description=f"Academic paper with DOI: {doi}",
                message_id=message.get('message_id'),
                channel_id=message.get('channel_id'),
                author=message.get('author', {}).get('username'),
                created_at=datetime.now()
            )
            
            resources.append(resource)
        
        # ArXiv detection
        for match in self.academic_patterns['arxiv'].finditer(content):
            arxiv_url = match.group()
            arxiv_id = arxiv_url.split('/')[-1]
            
            resource = ResourceMetadata(
                id=hashlib.md5(f"arxiv_{arxiv_id}".encode()).hexdigest(),
                type=ResourceType.PREPRINT,
                url=f"https://{arxiv_url}",
                title=f"ArXiv: {arxiv_id}",
                description=f"ArXiv preprint: {arxiv_id}",
                message_id=message.get('message_id'),
                channel_id=message.get('channel_id'),
                author=message.get('author', {}).get('username'),
                created_at=datetime.now()
            )
            
            resources.append(resource)
        
        return resources
    
    async def _calculate_quality_score(self, resource: ResourceMetadata, message: Dict[str, Any]) -> float:
        """Advanced quality scoring with multiple factors"""
        score = 0.0
        
        # Domain quality
        if resource.domain:
            if resource.domain in self.high_quality_domains:
                score += 0.4
            elif resource.domain in self.low_quality_domains:
                score -= 0.3
            else:
                score += 0.1
        
        # Resource type quality
        type_scores = {
            ResourceType.RESEARCH_PAPER: 0.4,
            ResourceType.DATASET: 0.35,
            ResourceType.MODEL: 0.35,
            ResourceType.CODE_REPOSITORY: 0.3,
            ResourceType.TUTORIAL: 0.3,
            ResourceType.DOCUMENTATION: 0.25,
            ResourceType.BLOG_POST: 0.2,
            ResourceType.NEWS_ARTICLE: 0.15,
            ResourceType.SOCIAL_MEDIA: 0.05,
            ResourceType.LOW_QUALITY: -0.5
        }
        
        score += type_scores.get(resource.type, 0.1)
        
        # Content length (for text resources)
        if resource.content_length > 0:
            if resource.content_length > 1000:
                score += 0.2
            elif resource.content_length > 500:
                score += 0.1
            elif resource.content_length < 50:
                score -= 0.1
        
        # File size (for attachments)
        if resource.file_size:
            if resource.file_size > 100000:  # 100KB
                score += 0.15
            elif resource.file_size > 10000:  # 10KB
                score += 0.1
            elif resource.file_size < 1000:  # 1KB
                score -= 0.2
        
        # Message context quality
        message_content = message.get('content', '')
        if len(message_content) > 200:  # Substantial context
            score += 0.1
        
        # Author reputation (could be enhanced with historical data)
        author_username = message.get('author', {}).get('username', '')
        if author_username:
            score += 0.05  # Small bonus for having an author
        
        # Time-based factors
        if resource.created_at and resource.created_at > datetime.now() - timedelta(days=30):
            score += 0.1  # Bonus for recent content
        
        return min(1.0, max(0.0, score))
    
    def _get_quality_level(self, score: float) -> ResourceQuality:
        """Convert quality score to quality level"""
        if score >= 0.9:
            return ResourceQuality.EXCELLENT
        elif score >= 0.7:
            return ResourceQuality.HIGH
        elif score >= 0.5:
            return ResourceQuality.GOOD
        elif score >= 0.3:
            return ResourceQuality.FAIR
        elif score >= 0.1:
            return ResourceQuality.POOR
        else:
            return ResourceQuality.SPAM
    
    def _classify_url_type(self, url: str, domain: str) -> ResourceType:
        """Classify URL type based on domain and path"""
        # Check domain mapping first
        if domain in self.domain_mappings:
            return self.domain_mappings[domain]
        
        # Check path patterns
        url_lower = url.lower()
        
        # GitHub-specific patterns
        if 'github.com' in domain:
            if '/blob/' in url_lower or '/tree/' in url_lower:
                return ResourceType.CODE_REPOSITORY
            elif '/releases/' in url_lower:
                return ResourceType.SOFTWARE_TOOL
            elif '/wiki/' in url_lower:
                return ResourceType.DOCUMENTATION
        
        # ArXiv patterns
        if 'arxiv.org' in domain and '/abs/' in url_lower:
            return ResourceType.PREPRINT
        
        # HuggingFace patterns
        if 'huggingface.co' in domain:
            if '/models/' in url_lower:
                return ResourceType.MODEL
            elif '/datasets/' in url_lower:
                return ResourceType.DATASET
            elif '/spaces/' in url_lower:
                return ResourceType.SOFTWARE_TOOL
        
        # File extension patterns
        for file_type, pattern in self.file_patterns.items():
            if pattern.search(url):
                if file_type == 'documents':
                    return ResourceType.PDF_DOCUMENT
                elif file_type == 'code':
                    return ResourceType.CODE_SNIPPET
                elif file_type == 'data':
                    return ResourceType.DATASET
                elif file_type == 'images':
                    return ResourceType.IMAGE
                elif file_type == 'media':
                    return ResourceType.VIDEO
        
        # Default classification
        return ResourceType.UNKNOWN
    
    def _classify_attachment_type(self, filename: str, content_type: str) -> ResourceType:
        """Classify attachment type"""
        filename_lower = filename.lower()
        
        # PDF documents
        if filename_lower.endswith('.pdf') or 'pdf' in content_type:
            return ResourceType.PDF_DOCUMENT
        
        # Office documents
        if any(filename_lower.endswith(ext) for ext in ['.doc', '.docx', '.ppt', '.pptx']):
            return ResourceType.PRESENTATION
        
        # Spreadsheets
        if any(filename_lower.endswith(ext) for ext in ['.xls', '.xlsx', '.csv']):
            return ResourceType.SPREADSHEET
        
        # Images
        if any(filename_lower.endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.gif', '.svg']):
            return ResourceType.IMAGE
        
        # Code files
        if any(filename_lower.endswith(ext) for ext in ['.py', '.js', '.ts', '.java', '.cpp', '.c', '.h']):
            return ResourceType.CODE_SNIPPET
        
        # Data files
        if any(filename_lower.endswith(ext) for ext in ['.json', '.csv', '.jsonl', '.parquet']):
            return ResourceType.DATASET
        
        return ResourceType.UNKNOWN
    
    def _detect_code_language(self, code: str) -> str:
        """Detect programming language from code snippet"""
        code_lower = code.lower()
        
        # Check for explicit language markers
        if code.startswith('```'):
            first_line = code.split('\n')[0]
            if len(first_line) > 3:
                return first_line[3:].strip()
        
        # Language-specific patterns
        if any(keyword in code_lower for keyword in ['def ', 'import ', 'print(', 'if __name__']):
            return 'python'
        elif any(keyword in code_lower for keyword in ['function', 'const ', 'let ', 'var ']):
            return 'javascript'
        elif any(keyword in code_lower for keyword in ['public class', 'private ', 'public static void']):
            return 'java'
        elif any(keyword in code_lower for keyword in ['#include', 'int main', 'printf']):
            return 'c'
        elif any(keyword in code_lower for keyword in ['SELECT ', 'FROM ', 'WHERE ']):
            return 'sql'
        
        return 'unknown'
    
    def _assess_code_quality(self, code: str, language: str) -> float:
        """Assess code quality based on various factors"""
        score = 0.5  # Base score
        
        # Length factor
        if len(code) > 100:
            score += 0.2
        elif len(code) > 50:
            score += 0.1
        
        # Complexity indicators
        if any(keyword in code.lower() for keyword in ['class', 'function', 'def']):
            score += 0.2
        
        # Comments
        if any(indicator in code for indicator in ['#', '//', '/*', '"""']):
            score += 0.1
        
        # Good practices
        if language == 'python':
            if 'import' in code.lower():
                score += 0.1
            if any(keyword in code for keyword in ['try:', 'except:', 'finally:']):
                score += 0.1
        
        return min(1.0, max(0.0, score))
    
    async def _analyze_url_content(self, resource: ResourceMetadata, context: str):
        """Analyze URL content for enhanced metadata"""
        # Extract title from context
        url_pos = context.find(resource.url)
        if url_pos > 0:
            # Look for title in preceding text
            preceding_text = context[:url_pos].strip()
            if preceding_text:
                # Simple title extraction heuristic
                lines = preceding_text.split('\n')
                for line in reversed(lines):
                    if len(line.strip()) > 10 and not line.strip().startswith('http'):
                        resource.title = line.strip()[:100]
                        break
        
        # Set relevance based on context
        ai_keywords = ['ai', 'machine learning', 'neural', 'model', 'dataset', 'algorithm']
        if any(keyword in context.lower() for keyword in ai_keywords):
            resource.relevance_score = 0.8
        else:
            resource.relevance_score = 0.5
    
    async def _extract_file_content(self, resource: ResourceMetadata, attachment: Dict):
        """Extract content from file attachments"""
        # This would require downloading and parsing files
        # For now, just set some metadata
        resource.accessibility = True
        resource.description = f"File: {attachment.get('filename', '')}"
    
    async def _deduplicate_resources(self, resources: List[ResourceMetadata]) -> List[ResourceMetadata]:
        """Remove duplicate resources"""
        seen_urls = set()
        seen_titles = set()
        unique_resources = []
        
        for resource in resources:
            # Check for URL duplicates
            if resource.url and resource.url in seen_urls:
                continue
            
            # Check for title duplicates
            if resource.title and resource.title in seen_titles:
                continue
            
            unique_resources.append(resource)
            
            if resource.url:
                seen_urls.add(resource.url)
            if resource.title:
                seen_titles.add(resource.title)
        
        return unique_resources
    
    async def validate_resources(self, resources: List[ResourceMetadata]) -> List[ResourceMetadata]:
        """Validate resources for accessibility and quality"""
        validated_resources = []
        
        for resource in resources:
            try:
                # URL validation
                if resource.url:
                    is_valid = await self._validate_url(resource.url)
                    resource.is_valid = is_valid
                    if not is_valid:
                        resource.validation_errors.append("URL not accessible")
                
                # Only include valid resources
                if resource.is_valid:
                    validated_resources.append(resource)
                    
            except Exception as e:
                logger.debug(f"Error validating resource {resource.id}: {e}")
                resource.validation_errors.append(str(e))
        
        return validated_resources
    
    @retry_on_error(max_retries=3, delay=1.0)
    async def _validate_url(self, url: str) -> bool:
        """Validate URL accessibility"""
        try:
            if not self.session:
                self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10))
            
            async with self.session.head(url) as response:
                return response.status < 400
                
        except Exception as e:
            logger.debug(f"URL validation failed for {url}: {e}")
            return False
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.session:
            await self.session.close() 