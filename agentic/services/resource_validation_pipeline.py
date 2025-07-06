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
    
    def __init__(self, config: Dict[str, Any] = None):
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
        
        # Spam detection patterns
        self.spam_patterns = [
            r'click\s+here\s+to\s+win',
            r'free\s+money',
            r'urgent\s+action\s+required',
            r'limited\s+time\s+offer',
            r'congratulations\s+you\s+have\s+won',
            r'increase\s+your\s+.+\s+fast',
            r'\\$\\d+\s+per\s+hour',
            r'work\s+from\s+home',
            r'make\s+\\$\\d+\s+daily'
        ]
        
        # Blacklisted domains
        self.blacklisted_domains = {
            'bit.ly', 'tinyurl.com', 'goo.gl', 't.co',  # URL shorteners (often spam)
            'adfly.com', 'adf.ly', 'linkbucks.com',      # Ad-based redirects
            'malware.com', 'phishing.com', 'spam.com'    # Known bad domains
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
        Validate URL-based resource
        """
        url = resource.url
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        
        # Check blacklisted domains
        if domain in self.blacklisted_domains:
            return ValidationResult(
                resource_id=resource.id,
                status=ValidationStatus.SPAM,
                response_time=0.0,
                error_message="Blacklisted domain"
            )
        
        # Check for suspicious URL patterns
        if self._is_suspicious_url(url):
            return ValidationResult(
                resource_id=resource.id,
                status=ValidationStatus.SPAM,
                response_time=0.0,
                error_message="Suspicious URL pattern"
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
        Validate HTML content quality
        """
        try:
            soup = BeautifulSoup(content, 'html.parser')
            
            # Extract title and description
            title = soup.find('title')
            title_text = title.get_text().strip() if title else ""
            
            description = soup.find('meta', attrs={'name': 'description'})
            description_text = description.get('content', '').strip() if description else ""
            
            # Check for spam indicators
            text_content = soup.get_text().lower()
            spam_score = sum(1 for pattern in self.spam_patterns if re.search(pattern, text_content))
            
            # Determine quality
            if spam_score > 2:
                status = ValidationStatus.SPAM
            elif len(text_content) < 200:
                status = ValidationStatus.DEGRADED
            else:
                status = ValidationStatus.HEALTHY
            
            return ValidationResult(
                resource_id=hashlib.md5(url.encode()).hexdigest(),
                status=status,
                response_time=0.0,
                content_length=len(content),
                title=title_text[:100] if title_text else None,
                description=description_text[:200] if description_text else None
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
        Validate PDF content quality
        """
        try:
            pdf_reader = PyPDF2.PdfReader(BytesIO(content))
            num_pages = len(pdf_reader.pages)
            
            # Extract text from first few pages
            text_content = ""
            for i in range(min(3, num_pages)):
                text_content += pdf_reader.pages[i].extract_text()
            
            # Basic quality checks
            if num_pages < 2:
                status = ValidationStatus.DEGRADED
            elif len(text_content) < 500:
                status = ValidationStatus.DEGRADED
            else:
                status = ValidationStatus.HEALTHY
            
            return ValidationResult(
                resource_id=hashlib.md5(url.encode()).hexdigest(),
                status=status,
                response_time=0.0,
                content_length=len(content),
                description=f"PDF with {num_pages} pages"
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
    
    def _is_suspicious_url(self, url: str) -> bool:
        """
        Check if URL appears suspicious
        """
        url_lower = url.lower()
        
        # Check for excessive URL parameters (often spam)
        if url_lower.count('?') > 2 or url_lower.count('&') > 10:
            return True
        
        # Check for suspicious patterns
        suspicious_patterns = [
            r'click\.php',
            r'redirect\.php',
            r'go\.php',
            r'out\.php',
            r'link\.php',
            r'url\.php',
            r'tracking\.php',
            r'affiliate',
            r'referral',
            r'promo',
            r'campaign'
        ]
        
        return any(re.search(pattern, url_lower) for pattern in suspicious_patterns)
    
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