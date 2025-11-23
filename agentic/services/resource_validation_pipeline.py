"""
Resource Validation Pipeline

Comprehensive resource validation system for checking URL health,
content quality, accessibility, and freshness monitoring.
"""

import asyncio
import aiohttp
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import re
from urllib.parse import urlparse
import ssl
from bs4 import BeautifulSoup
import feedparser
import PyPDF2
from io import BytesIO

from .enhanced_resource_detector import ResourceMetadata, ResourceType, ResourceQuality
from .pdf_analyzer import PDFAnalyzer, PDFAnalysisResult
from ..utils.error_handling import retry_on_error, safe_async
from ..cache.smart_cache import SmartCache

logger = logging.getLogger(__name__)


class ValidationStatus(Enum):
    """Resource validation status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    BROKEN = "broken"
    INACCESSIBLE = "inaccessible"
    OUTDATED = "outdated"
    SPAM = "spam"


@dataclass
class ValidationResult:
    """Resource validation result"""
    resource_id: str
    status: ValidationStatus
    response_time: float
    http_status: Optional[int] = None
    content_length: Optional[int] = None
    content_type: Optional[str] = None
    last_modified: Optional[datetime] = None
    title: Optional[str] = None
    description: Optional[str] = None
    error_message: Optional[str] = None
    validation_timestamp: datetime = None
    
    def __post_init__(self):
        if self.validation_timestamp is None:
            self.validation_timestamp = datetime.now()


class ResourceValidationPipeline:
    """
    Comprehensive resource validation pipeline
    """

    def __init__(self, config: Dict[str, Any] = None, use_openai: bool = False):
        self.config = config or {}
        self.cache = SmartCache(self.config.get("cache", {}))
        self.session = None

        # Validation settings
        self.timeout = self.config.get("validation_timeout", 10)
        self.max_concurrent = self.config.get("max_concurrent_validations", 20)
        self.cache_ttl = self.config.get("validation_cache_ttl", 3600)  # 1 hour

        # Quality thresholds
        self.response_time_threshold = self.config.get("response_time_threshold", 5.0)
        self.content_length_threshold = self.config.get("content_length_threshold", 1000)

        # PDF Analyzer with LLM support (two-tier analysis)
        self.pdf_analyzer = PDFAnalyzer(use_openai=use_openai)
        self.use_openai = use_openai
        
        # Enhanced spam detection system (LOW-PRIORITY IMPROVEMENT)
        # Categorized spam patterns with weights
        self.spam_patterns = {
            # High confidence spam (weight: 3)
            'scam': [
                r'congratulations\s+you\s+(have\s+)?won',
                r'claim\s+your\s+(free\s+)?(prize|reward|gift)',
                r'you\s+have\s+been\s+selected',
                r'nigerian?\s+prince',
                r'transfer\s+\$?\d+\s*(million|thousand)',
                r'inheritance\s+fund',
            ],
            # Medium confidence spam (weight: 2)
            'marketing_spam': [
                r'click\s+here\s+to\s+(win|claim|get)',
                r'limited\s+time\s+offer',
                r'act\s+now\s+before',
                r'exclusive\s+deal\s+for\s+you',
                r'100%\s+(free|guaranteed)',
                r'no\s+risk',
                r'risk\s+free',
                r'double\s+your\s+(money|income)',
            ],
            # Low confidence spam (weight: 1) - could be legitimate
            'suspicious': [
                r'make\s+\$?\d+\s*(per\s+)?(hour|day|week)',
                r'work\s+from\s+home\s+and\s+earn',
                r'passive\s+income',
                r'get\s+rich\s+(quick|fast)',
                r'free\s+money',
                r'urgent\s+action\s+required',
                r'your\s+account\s+(has\s+been|will\s+be)\s+(suspended|closed)',
                r'verify\s+your\s+(account|identity)\s+immediately',
            ],
            # Crypto/investment scams (weight: 2)
            'crypto_scam': [
                r'guaranteed\s+(returns?|profits?|gains?)',
                r'invest\s+now\s+and\s+(double|triple)',
                r'crypto\s+(giveaway|airdrop)',
                r'send\s+\d+\s*(btc|eth|crypto)',
                r'elon\s+musk\s+(giveaway|giving)',
            ],
        }

        # Spam pattern weights
        self.spam_weights = {
            'scam': 3,
            'marketing_spam': 2,
            'suspicious': 1,
            'crypto_scam': 2,
        }

        # Spam score threshold (0-10 scale, higher = more likely spam)
        self.spam_threshold = 4

        # Domain reputation system (replaces simple blacklist)
        # Score: -1 (blocked), 0 (neutral), 1 (trusted)
        self.domain_reputation = {
            # Blocked domains (-1)
            'adfly.com': -1, 'adf.ly': -1, 'linkbucks.com': -1,
            'shorte.st': -1, 'bc.vc': -1, 'ouo.io': -1,
            'exe.io': -1, 'fc.lc': -1, 'gestyy.com': -1,

            # Suspicious but allowed (0) - URL shorteners from legitimate services
            'bit.ly': 0, 'tinyurl.com': 0, 'goo.gl': 0, 't.co': 0,
            'ow.ly': 0, 'buff.ly': 0, 'is.gd': 0, 'v.gd': 0,
            'rb.gy': 0, 'cutt.ly': 0, 'shorturl.at': 0,

            # Trusted domains (1) - major platforms
            'github.com': 1, 'arxiv.org': 1, 'huggingface.co': 1,
            'youtube.com': 1, 'youtu.be': 1, 'docs.google.com': 1,
            'drive.google.com': 1, 'colab.research.google.com': 1,
            'openai.com': 1, 'anthropic.com': 1, 'deepmind.com': 1,
            'pytorch.org': 1, 'tensorflow.org': 1, 'kaggle.com': 1,
        }

        # Content-based spam indicators
        self.spam_content_indicators = {
            'excessive_caps': (r'[A-Z]{10,}', 1),  # 10+ consecutive caps
            'excessive_exclamation': (r'!{3,}', 1),  # 3+ exclamation marks
            'excessive_emoji': (r'[\U0001F600-\U0001F64F]{5,}', 1),  # 5+ emojis in a row
            'hidden_text': (r'font-size:\s*0|display:\s*none|visibility:\s*hidden', 2),
            'base64_data': (r'data:.*base64', 1),
            'suspicious_form': (r'<form[^>]*action=["\'][^"\']*\.(php|cgi)', 2),
            'password_field': (r'type=["\']password["\']', 1),
            'external_scripts': (r'<script[^>]*src=["\']https?://(?!.*(?:google|facebook|twitter|cloudflare))', 1),
        }
        
        logger.info("🔍 Resource validation pipeline initialized")
    
    async def validate_resources(self, resources: List[ResourceMetadata]) -> List[Tuple[ResourceMetadata, ValidationResult]]:
        """
        Validate a list of resources
        """
        if not self.session:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout),
                connector=aiohttp.TCPConnector(limit=self.max_concurrent)
            )
        
        results = []
        
        # Process resources in batches to avoid overwhelming servers
        batch_size = self.max_concurrent
        for i in range(0, len(resources), batch_size):
            batch = resources[i:i + batch_size]
            
            tasks = [self._validate_single_resource(resource) for resource in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for resource, result in zip(batch, batch_results):
                if isinstance(result, Exception):
                    logger.error(f"Validation failed for resource {resource.id}: {result}")
                    result = ValidationResult(
                        resource_id=resource.id,
                        status=ValidationStatus.BROKEN,
                        response_time=0.0,
                        error_message=str(result)
                    )
                
                results.append((resource, result))
            
            # Small delay between batches
            await asyncio.sleep(0.1)
        
        return results
    
    async def _validate_single_resource(self, resource: ResourceMetadata) -> ValidationResult:
        """
        Validate a single resource
        """
        # Check cache first
        cache_key = f"validation:{resource.id}"
        cached_result = await self.cache.get(cache_key)
        if cached_result:
            return ValidationResult(**cached_result)
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            # URL-based validation
            if resource.url:
                result = await self._validate_url_resource(resource)
            else:
                # Non-URL resource (like code snippets, entities)
                result = await self._validate_content_resource(resource)
            
            # Calculate response time
            result.response_time = asyncio.get_event_loop().time() - start_time
            
            # Cache the result
            await self.cache.set(cache_key, result.__dict__, ttl=self.cache_ttl)
            
            return result
            
        except Exception as e:
            logger.error(f"Validation error for resource {resource.id}: {e}")
            return ValidationResult(
                resource_id=resource.id,
                status=ValidationStatus.BROKEN,
                response_time=asyncio.get_event_loop().time() - start_time,
                error_message=str(e)
            )
    
    async def _validate_url_resource(self, resource: ResourceMetadata) -> ValidationResult:
        """
        Validate URL-based resource with enhanced spam detection
        """
        url = resource.url
        parsed = urlparse(url)
        domain = parsed.netloc.lower()

        # Remove www. prefix for domain matching
        if domain.startswith('www.'):
            domain = domain[4:]

        # Check domain reputation
        domain_score = self._get_domain_reputation(domain)
        if domain_score == -1:
            return ValidationResult(
                resource_id=resource.id,
                status=ValidationStatus.SPAM,
                response_time=0.0,
                error_message=f"Blocked domain: {domain}"
            )

        # Check for suspicious URL patterns (adds to spam score)
        url_spam_score = self._calculate_url_spam_score(url)

        # If URL is very suspicious, reject early
        if url_spam_score >= self.spam_threshold:
            return ValidationResult(
                resource_id=resource.id,
                status=ValidationStatus.SPAM,
                response_time=0.0,
                error_message=f"Suspicious URL (spam score: {url_spam_score})"
            )
        
        # HTTP validation
        try:
            async with self.session.head(url, allow_redirects=True) as response:
                status = response.status
                content_type = response.headers.get('content-type', '')
                content_length = response.headers.get('content-length')
                last_modified = response.headers.get('last-modified')
                
                # Parse last modified
                last_modified_dt = None
                if last_modified:
                    try:
                        from email.utils import parsedate_to_datetime
                        last_modified_dt = parsedate_to_datetime(last_modified)
                    except:
                        pass
                
                # Determine validation status
                if status == 200:
                    validation_status = ValidationStatus.HEALTHY
                elif status < 400:
                    validation_status = ValidationStatus.HEALTHY
                elif status == 404:
                    validation_status = ValidationStatus.BROKEN
                elif status >= 500:
                    validation_status = ValidationStatus.DEGRADED
                else:
                    validation_status = ValidationStatus.INACCESSIBLE
                
                # For important resources, do a full content check
                if resource.type in [ResourceType.RESEARCH_PAPER, ResourceType.DATASET, 
                                   ResourceType.MODEL, ResourceType.TUTORIAL] and status == 200:
                    content_result = await self._validate_content_quality(url)
                    if content_result:
                        validation_status = content_result.status
                
                return ValidationResult(
                    resource_id=resource.id,
                    status=validation_status,
                    response_time=0.0,  # Will be set by caller
                    http_status=status,
                    content_length=int(content_length) if content_length else None,
                    content_type=content_type,
                    last_modified=last_modified_dt
                )
                
        except asyncio.TimeoutError:
            return ValidationResult(
                resource_id=resource.id,
                status=ValidationStatus.DEGRADED,
                response_time=0.0,
                error_message="Request timeout"
            )
        except Exception as e:
            return ValidationResult(
                resource_id=resource.id,
                status=ValidationStatus.INACCESSIBLE,
                response_time=0.0,
                error_message=str(e)
            )
    
    async def _validate_content_resource(self, resource: ResourceMetadata) -> ValidationResult:
        """
        Validate non-URL resource (code snippets, entities, etc.)
        """
        # For code snippets, validate syntax and quality
        if resource.type == ResourceType.CODE_SNIPPET:
            return self._validate_code_snippet(resource)
        
        # For other resources, basic validation
        return ValidationResult(
            resource_id=resource.id,
            status=ValidationStatus.HEALTHY,
            response_time=0.0
        )
    
    def _validate_code_snippet(self, resource: ResourceMetadata) -> ValidationResult:
        """
        Validate code snippet quality
        """
        # Basic code quality checks
        if resource.content_length < 20:
            return ValidationResult(
                resource_id=resource.id,
                status=ValidationStatus.BROKEN,
                response_time=0.0,
                error_message="Code snippet too short"
            )
        
        if resource.quality_score < 0.3:
            return ValidationResult(
                resource_id=resource.id,
                status=ValidationStatus.DEGRADED,
                response_time=0.0,
                error_message="Low quality code"
            )
        
        return ValidationResult(
            resource_id=resource.id,
            status=ValidationStatus.HEALTHY,
            response_time=0.0
        )
    
    async def _validate_content_quality(self, url: str) -> Optional[ValidationResult]:
        """
        Validate content quality by fetching and analyzing
        """
        try:
            async with self.session.get(url) as response:
                if response.status != 200:
                    return None
                
                content_type = response.headers.get('content-type', '')
                content = await response.text()
                
                # HTML content validation
                if 'text/html' in content_type:
                    return self._validate_html_content(content, url)
                
                # PDF content validation
                elif 'application/pdf' in content_type:
                    return self._validate_pdf_content(await response.read(), url)
                
                # JSON content validation
                elif 'application/json' in content_type:
                    return self._validate_json_content(content, url)
                
                return None
                
        except Exception as e:
            logger.debug(f"Content validation failed for {url}: {e}")
            return None
    
    def _validate_html_content(self, content: str, url: str) -> ValidationResult:
        """
        Validate HTML content quality with enhanced spam detection
        """
        try:
            soup = BeautifulSoup(content, 'html.parser')

            # Extract title and description
            title = soup.find('title')
            title_text = title.get_text().strip() if title else ""

            description = soup.find('meta', attrs={'name': 'description'})
            description_text = description.get('content', '').strip() if description else ""

            # Get text content for spam analysis
            text_content = soup.get_text()

            # Calculate spam score using the enhanced system
            content_spam_score, matched_patterns = self._calculate_content_spam_score(text_content)

            # Also check raw HTML for hidden spam indicators
            html_spam_score, html_patterns = self._calculate_content_spam_score(content)
            matched_patterns.extend(html_patterns)

            # Combine scores (content matters more than HTML structure)
            total_spam_score = content_spam_score + (html_spam_score // 2)

            # Get domain reputation for context
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            if domain.startswith('www.'):
                domain = domain[4:]
            domain_reputation = self._get_domain_reputation(domain)

            # Trusted domains get a spam score reduction
            if domain_reputation == 1:
                total_spam_score = max(0, total_spam_score - 2)

            # Determine status based on spam score and content quality
            if total_spam_score >= self.spam_threshold:
                status = ValidationStatus.SPAM
                error_msg = f"Spam detected (score: {total_spam_score}): {', '.join(matched_patterns[:3])}"
            elif len(text_content.strip()) < 200:
                status = ValidationStatus.DEGRADED
                error_msg = "Insufficient content"
            else:
                status = ValidationStatus.HEALTHY
                error_msg = None

            return ValidationResult(
                resource_id=hashlib.md5(url.encode()).hexdigest(),
                status=status,
                response_time=0.0,
                content_length=len(content),
                title=title_text[:100] if title_text else None,
                description=description_text[:200] if description_text else None,
                error_message=error_msg
            )

        except Exception as e:
            logger.debug(f"HTML validation failed for {url}: {e}")
            return ValidationResult(
                resource_id=hashlib.md5(url.encode()).hexdigest(),
                status=ValidationStatus.DEGRADED,
                response_time=0.0,
                error_message=str(e)
            )
    
    def _validate_pdf_content(self, content: bytes, url: str) -> ValidationResult:
        """
        Validate PDF content quality using two-tier PDF analysis.

        Tier 1: Text extraction and metadata (no LLM)
        Tier 2: LLM-based content analysis for usefulness and description
        """
        try:
            # Use the enhanced PDF analyzer with LLM support
            # Note: This is async, but we need sync here - use asyncio
            import asyncio

            # Get or create event loop
            try:
                loop = asyncio.get_running_loop()
                # If we're in an async context, create a new task
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self.pdf_analyzer.analyze_pdf(content, url))
                    pdf_result = future.result(timeout=60)
            except RuntimeError:
                # No running event loop, safe to use asyncio.run
                pdf_result = asyncio.run(self.pdf_analyzer.analyze_pdf(content, url))

            # Convert PDFAnalysisResult to ValidationResult
            if pdf_result.error:
                return ValidationResult(
                    resource_id=hashlib.md5(url.encode()).hexdigest(),
                    status=ValidationStatus.DEGRADED,
                    response_time=0.0,
                    content_length=len(content),
                    error_message=pdf_result.error
                )

            # Determine validation status based on PDF analysis
            if not pdf_result.is_useful:
                if pdf_result.document_type == 'scanned_or_image':
                    status = ValidationStatus.DEGRADED
                    error_msg = "PDF with insufficient extractable text (scanned/image-based)"
                else:
                    status = ValidationStatus.DEGRADED
                    error_msg = f"PDF not useful: {pdf_result.document_type}"
            elif pdf_result.confidence < 0.5:
                status = ValidationStatus.DEGRADED
                error_msg = f"Low confidence PDF analysis ({pdf_result.confidence:.2f})"
            else:
                status = ValidationStatus.HEALTHY
                error_msg = None

            # Build description from PDF analysis
            description_parts = []
            if pdf_result.title:
                description_parts.append(f"Title: {pdf_result.title}")
            description_parts.append(f"Type: {pdf_result.document_type}")
            description_parts.append(f"Pages: {pdf_result.page_count}")
            if pdf_result.description:
                description_parts.append(pdf_result.description)

            description = " | ".join(description_parts)

            return ValidationResult(
                resource_id=hashlib.md5(url.encode()).hexdigest(),
                status=status,
                response_time=0.0,
                content_length=len(content),
                title=pdf_result.title,
                description=description[:500] if description else f"PDF with {pdf_result.page_count} pages",
                error_message=error_msg
            )

        except Exception as e:
            logger.debug(f"PDF validation failed for {url}: {e}")
            return ValidationResult(
                resource_id=hashlib.md5(url.encode()).hexdigest(),
                status=ValidationStatus.DEGRADED,
                response_time=0.0,
                error_message=str(e)
            )
    
    def _validate_json_content(self, content: str, url: str) -> ValidationResult:
        """
        Validate JSON content quality
        """
        try:
            data = json.loads(content)
            
            # Basic JSON quality checks
            if isinstance(data, dict) and len(data) > 0:
                status = ValidationStatus.HEALTHY
            elif isinstance(data, list) and len(data) > 0:
                status = ValidationStatus.HEALTHY
            else:
                status = ValidationStatus.DEGRADED
            
            return ValidationResult(
                resource_id=hashlib.md5(url.encode()).hexdigest(),
                status=status,
                response_time=0.0,
                content_length=len(content),
                description=f"JSON with {len(data)} items" if isinstance(data, (dict, list)) else "JSON content"
            )
            
        except json.JSONDecodeError:
            return ValidationResult(
                resource_id=hashlib.md5(url.encode()).hexdigest(),
                status=ValidationStatus.BROKEN,
                response_time=0.0,
                error_message="Invalid JSON"
            )
    
    def _get_domain_reputation(self, domain: str) -> int:
        """
        Get domain reputation score.
        Returns: -1 (blocked), 0 (neutral/unknown), 1 (trusted)
        """
        # Check exact match first
        if domain in self.domain_reputation:
            return self.domain_reputation[domain]

        # Check if it's a subdomain of a known domain
        for known_domain, score in self.domain_reputation.items():
            if domain.endswith('.' + known_domain):
                return score

        # Unknown domain - neutral
        return 0

    def _calculate_url_spam_score(self, url: str) -> int:
        """
        Calculate spam score for a URL based on various heuristics.
        Returns score 0-10 (higher = more likely spam)
        """
        score = 0
        url_lower = url.lower()

        # Excessive URL parameters (often tracking/spam)
        if url_lower.count('?') > 2:
            score += 1
        if url_lower.count('&') > 8:
            score += 1
        if url_lower.count('&') > 15:
            score += 2

        # Suspicious PHP redirects
        php_patterns = [
            r'click\.php', r'redirect\.php', r'go\.php', r'out\.php',
            r'link\.php', r'url\.php', r'tracking\.php', r'track\.php',
            r'jump\.php', r'redir\.php'
        ]
        for pattern in php_patterns:
            if re.search(pattern, url_lower):
                score += 2
                break

        # Affiliate/tracking indicators (lower weight - can be legitimate)
        affiliate_patterns = [r'/aff/', r'affiliate', r'/ref/', r'referral=']
        for pattern in affiliate_patterns:
            if re.search(pattern, url_lower):
                score += 1
                break

        # Suspicious TLDs
        suspicious_tlds = ['.xyz', '.top', '.loan', '.win', '.bid', '.click', '.link']
        parsed = urlparse(url)
        if any(parsed.netloc.endswith(tld) for tld in suspicious_tlds):
            score += 1

        # Very long URLs (often obfuscated spam)
        if len(url) > 500:
            score += 1
        if len(url) > 1000:
            score += 2

        # Multiple redirects in URL
        if url_lower.count('http') > 1:
            score += 2

        # IP address instead of domain (suspicious)
        if re.match(r'https?://\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', url_lower):
            score += 2

        return min(score, 10)  # Cap at 10

    def _calculate_content_spam_score(self, content: str) -> Tuple[int, List[str]]:
        """
        Calculate spam score based on content analysis.
        Returns: (score, list of matched patterns)
        """
        score = 0
        matched_patterns = []
        content_lower = content.lower()

        # Check categorized spam patterns
        for category, patterns in self.spam_patterns.items():
            weight = self.spam_weights.get(category, 1)
            for pattern in patterns:
                if re.search(pattern, content_lower, re.IGNORECASE):
                    score += weight
                    matched_patterns.append(f"{category}: {pattern}")

        # Check content indicators
        for indicator_name, (pattern, weight) in self.spam_content_indicators.items():
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                score += weight * min(len(matches), 3)  # Cap repeated matches
                matched_patterns.append(f"{indicator_name}: {len(matches)} matches")

        return score, matched_patterns

    def _is_suspicious_url(self, url: str) -> bool:
        """
        Check if URL appears suspicious (legacy method for compatibility)
        """
        return self._calculate_url_spam_score(url) >= self.spam_threshold
    
    async def monitor_resource_health(self, resources: List[ResourceMetadata], 
                                    interval_hours: int = 24) -> Dict[str, List[ValidationResult]]:
        """
        Monitor resource health over time
        """
        health_report = {}
        
        # Group resources by type for monitoring
        resource_groups = {}
        for resource in resources:
            resource_type = resource.type.value
            if resource_type not in resource_groups:
                resource_groups[resource_type] = []
            resource_groups[resource_type].append(resource)
        
        # Validate each group
        for resource_type, group_resources in resource_groups.items():
            validation_results = await self.validate_resources(group_resources)
            health_report[resource_type] = [result for _, result in validation_results]
        
        return health_report
    
    async def generate_health_report(self, validation_results: List[Tuple[ResourceMetadata, ValidationResult]]) -> Dict[str, Any]:
        """
        Generate comprehensive health report
        """
        total_resources = len(validation_results)
        status_counts = {}
        avg_response_time = 0
        broken_resources = []
        excellent_resources = []
        
        for resource, result in validation_results:
            # Count statuses
            status = result.status.value
            status_counts[status] = status_counts.get(status, 0) + 1
            
            # Calculate average response time
            avg_response_time += result.response_time
            
            # Collect broken resources
            if result.status == ValidationStatus.BROKEN:
                broken_resources.append({
                    'resource_id': resource.id,
                    'url': resource.url,
                    'error': result.error_message
                })
            
            # Collect excellent resources
            if result.status == ValidationStatus.HEALTHY and result.response_time < 2.0:
                excellent_resources.append({
                    'resource_id': resource.id,
                    'url': resource.url,
                    'type': resource.type.value
                })
        
        avg_response_time = avg_response_time / total_resources if total_resources > 0 else 0
        
        # Calculate health score
        healthy_count = status_counts.get('healthy', 0)
        health_score = (healthy_count / total_resources) * 100 if total_resources > 0 else 0
        
        return {
            'timestamp': datetime.now().isoformat(),
            'total_resources': total_resources,
            'health_score': health_score,
            'status_distribution': status_counts,
            'average_response_time': avg_response_time,
            'broken_resources': broken_resources[:10],  # Top 10 broken
            'excellent_resources': excellent_resources[:10],  # Top 10 excellent
            'recommendations': self._generate_health_recommendations(status_counts, avg_response_time)
        }
    
    def _generate_health_recommendations(self, status_counts: Dict[str, int], avg_response_time: float) -> List[str]:
        """
        Generate health recommendations based on metrics
        """
        recommendations = []
        
        total = sum(status_counts.values())
        broken_pct = (status_counts.get('broken', 0) / total) * 100 if total > 0 else 0
        degraded_pct = (status_counts.get('degraded', 0) / total) * 100 if total > 0 else 0
        
        if broken_pct > 10:
            recommendations.append("High percentage of broken resources detected - consider cleanup")
        
        if degraded_pct > 20:
            recommendations.append("Many resources showing degraded performance - investigate server issues")
        
        if avg_response_time > 5.0:
            recommendations.append("Average response time is high - consider caching frequently accessed resources")
        
        if status_counts.get('spam', 0) > 5:
            recommendations.append("Spam content detected - review detection algorithms")
        
        if not recommendations:
            recommendations.append("Resource health is good - continue monitoring")
        
        return recommendations
    
    async def cleanup(self):
        """
        Cleanup resources
        """
        if self.session:
            await self.session.close() 