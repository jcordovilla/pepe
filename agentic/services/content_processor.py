"""
Enhanced Content Processing Service
Modernized content classification and processing with AI integration
"""

import re
import json
import logging
import hashlib
from typing import Dict, List, Any, Optional
from urllib.parse import urlparse
from datetime import datetime

from openai import OpenAI
from ..cache.smart_cache import SmartCache

logger = logging.getLogger(__name__)

class ContentProcessingService:
    """
    Modern content processing with legacy-proven classification patterns
    
    Preserves battle-tested:
    - URL analysis and filtering
    - Code detection patterns
    - Resource classification rules
    - Attachment processing logic
    """
    
    def __init__(self, openai_client: OpenAI, cache_config: Optional[Dict[str, Any]] = None):
        self.openai_client = openai_client
        self.cache = SmartCache(cache_config or {})
        self.classification_cache_ttl = int((cache_config or {}).get("classification_cache_ttl", 86400))
        
        # Legacy-proven patterns
        self.url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )
        
        self.code_pattern = re.compile(r'```[\s\S]*?```|`[^`]*`')
        
        # Legacy noise filtering
        self.noise_domains = {
            'tenor.com', 'giphy.com', 'discord.com', 'cdn.discordapp.com',
            'imgur.com', 'zoom.us', 'meet.google.com', 'teams.microsoft.com'
        }
        
        logger.info("🔍 Content processor initialized with legacy patterns")
    
    async def analyze_message_content(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Comprehensive message content analysis
        Enhanced from legacy detection patterns
        """
        content = message.get('content', '')
        attachments = message.get('attachments', [])
        
        analysis = {
            "message_id": message.get('message_id'),
            "content_types": [],
            "resources": [],
            "code_snippets": [],
            "urls": [],
            "attachments_processed": [],
            "classifications": []
        }
        
        if not content and not attachments:
            return analysis
        
        # Legacy URL extraction and analysis
        urls = self.url_pattern.findall(content)
        for url in urls:
            url_analysis = await self._analyze_url(url, message)
            if url_analysis:
                analysis["urls"].append(url_analysis)
        
        # Legacy code detection
        code_blocks = self.code_pattern.findall(content)
        for code in code_blocks:
            code_analysis = await self._analyze_code_snippet(code, message)
            if code_analysis:
                analysis["code_snippets"].append(code_analysis)
        
        # Legacy attachment processing
        for attachment in attachments:
            attachment_analysis = await self._analyze_attachment(attachment, message)
            if attachment_analysis:
                analysis["attachments_processed"].append(attachment_analysis)
        
        # Enhanced AI-powered classification (modern addition)
        if content:
            classifications = await self._classify_content_ai(content)
            analysis["classifications"] = classifications
        
        return analysis
    
    async def _analyze_url(self, url: str, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        URL analysis with legacy filtering patterns
        """
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            
            # Legacy noise filtering
            if domain in self.noise_domains:
                return None
            
            # Legacy-style resource classification
            resource_type = "unknown"
            if any(ext in url.lower() for ext in ['.pdf', '.doc', '.docx']):
                resource_type = "document"
            elif any(ext in url.lower() for ext in ['.py', '.js', '.html', '.css']):
                resource_type = "code"
            elif any(domain_part in domain for domain_part in ['github.com', 'stackoverflow.com']):
                resource_type = "development"
            elif any(domain_part in domain for domain_part in ['youtube.com', 'vimeo.com']):
                resource_type = "video"
            
            return {
                "url": url,
                "domain": domain,
                "type": resource_type,
                "message_id": message.get('message_id'),
                "timestamp": message.get('timestamp'),
                "author": message.get('author', {}).get('username')
            }
            
        except Exception as e:
            logger.debug(f"Error analyzing URL {url}: {e}")
            return None
    
    async def _analyze_code_snippet(self, code: str, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Code snippet analysis with legacy patterns
        """
        # Legacy language detection
        language = "unknown"
        if code.startswith("```"):
            first_line = code.split('\n')[0]
            if len(first_line) > 3:
                language = first_line[3:].strip()
        
        return {
            "code": code,
            "language": language,
            "length": len(code),
            "message_id": message.get('message_id'),
            "channel_name": message.get('channel_name'),
            "author": message.get('author', {}).get('username'),
            "timestamp": message.get('timestamp')
        }
    
    async def _analyze_attachment(self, attachment: Dict[str, Any], message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Attachment analysis with legacy patterns
        """
        return {
            "filename": attachment.get('filename'),
            "size": attachment.get('size'),
            "url": attachment.get('url'),
            "content_type": attachment.get('content_type'),
            "message_id": message.get('message_id'),
            "channel_name": message.get('channel_name'),
            "author": message.get('author', {}).get('username'),
            "timestamp": message.get('timestamp')
        }
    
    async def _classify_content_ai(self, content: str) -> List[str]:
        """
        Completely overhauled AI-powered content classification with expanded categories and aggressive detection
        """
        try:
            cache_key = f"content_class:{hashlib.sha256(content.encode()).hexdigest()}"
            cached = await self.cache.get(cache_key)
            if cached:
                return cached

            system_prompt = (
                "You curate AI community content. Categories: question, code_help, "
                "resource_sharing, technical, educational, project, discussion, "
                "documentation, research, tutorial, meme, casual, general, low_quality. "
                "Choose up to 3 and reply with a JSON array."
            )

            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user", 
                        "content": f"Classify this content:\n\n{content[:2000]}"  # Increased context window
                    }
                ],
                max_tokens=100,
                temperature=0.0  # Deterministic for better consistency
            )
            
            result = response.choices[0].message.content
            if result:
                # Robust JSON parsing
                result = result.strip()
                
                # Remove any markdown formatting
                if result.startswith('```json'):
                    result = result[7:]
                elif result.startswith('```'):
                    result = result[3:]
                if result.endswith('```'):
                    result = result[:-3]
                result = result.strip()
                
                # Try to parse JSON
                parsed = json.loads(result)
                
                # Validate that all categories are valid
                valid_categories = {
                    "question", "code_help", "resource_sharing", "technical", 
                    "discussion", "educational", "project", "documentation", 
                    "research", "tutorial", "meme", "casual",
                    "general", "low_quality"
                }
                
                if isinstance(parsed, list):
                    validated = [cat for cat in parsed if cat in valid_categories]
                    if validated:
                        await self.cache.set(
                            cache_key, validated[:3], ttl=self.classification_cache_ttl
                        )
                        return validated[:3]  # Max 3 categories
                    else:
                        logger.warning(f"AI returned invalid categories: {parsed}")
                        heuristic = self._classify_content_heuristic(content)
                        await self.cache.set(
                            cache_key, heuristic, ttl=self.classification_cache_ttl
                        )
                        return heuristic
                else:
                    logger.warning(f"AI returned non-list result: {parsed}")
                    heuristic = self._classify_content_heuristic(content)
                    await self.cache.set(
                        cache_key, heuristic, ttl=self.classification_cache_ttl
                    )
                    return heuristic
            else:
                heuristic = self._classify_content_heuristic(content)
                await self.cache.set(cache_key, heuristic, ttl=self.classification_cache_ttl)
                return heuristic
            
        except json.JSONDecodeError as e:
            logger.warning(f"AI classification JSON parse error for content '{content[:50]}...': {e}")
            heuristic = self._classify_content_heuristic(content)
            await self.cache.set(cache_key, heuristic, ttl=self.classification_cache_ttl)
            return heuristic
        except Exception as e:
            logger.warning(f"AI classification failed for content '{content[:50]}...': {e}")
            heuristic = self._classify_content_heuristic(content)
            await self.cache.set(cache_key, heuristic, ttl=self.classification_cache_ttl)
            return heuristic

    def _classify_content_heuristic(self, content: str) -> List[str]:
        """Enhanced fallback heuristic classification with improved accuracy and quality focus"""
        classifications = []
        content_lower = content.lower()
        content_length = len(content.strip())
        
        # Early quality filtering - prioritize substantial content
        if content_length < 10:
            return ["low_quality"]
            
        # Enhanced low quality detection (prioritize quality filtering)
        low_quality_patterns = [
            'lol', 'lmao', 'haha', 'lmfao', 'rofl', 'meme', 'funny', 'hilarious',
            'omg', 'wtf', 'bruh', 'sus', 'cringe', 'based', 'ratio',
            r'\b(good morning|good night|gm|gn)\b', r'^[^a-zA-Z]*$',  # Only emojis/symbols
            r'^.{1,15}$'  # Very short posts
        ]
        
        # Check for low quality patterns
        is_low_quality = False
        for pattern in low_quality_patterns:
            if isinstance(pattern, str):
                if pattern in content_lower:
                    is_low_quality = True
                    break
            else:  # regex pattern
                if re.search(pattern, content_lower):
                    is_low_quality = True
                    break
        
        if is_low_quality and content_length < 50:  # Short + low quality indicators
            return ["low_quality"]
        
        # Question detection - improved patterns
        question_patterns = [
            r'\?', r'\bhow (to|do|can|should)\b', r'\bwhat (is|are|does|should)\b',
            r'\bwhy (does|is|should)\b', r'\bwhen (should|can|do)\b', 
            r'\bwhere (can|should|do)\b', r'\bwhich (is|are|should)\b',
            r'\bhelp\b', r'\bissue\b', r'\bproblem\b', r'\berror\b', 
            r'\bstuck\b', r'\bconfused\b', r'\bneed advice\b', r'\bany ideas\b',
            r'\bcan (someone|anyone)\b', r'\bdoes (anyone|someone)\b'
        ]
        
        is_question = any(re.search(pattern, content_lower) for pattern in question_patterns)
        if is_question:
            classifications.append("question")
        
        # Code help detection - comprehensive patterns
        code_patterns = [
            r'```', r'\bcode\b', r'\bfunction\b', r'\bdef \w+\(', r'\bclass \w+',
            r'\bimport \w+', r'\bfrom \w+ import\b', r'\bnpm install\b', r'\bpip install\b',
            r'\bgit (clone|push|pull|commit)\b', r'\brepository\b', r'\brepo\b',
            r'\bsyntax error\b', r'\btraceback\b', r'\bexception\b', r'\bdebugging\b',
            r'\bbug\b', r'\bstack trace\b', r'\bgithub\.com/\w+/\w+\b'
        ]
        
        has_code = any(re.search(pattern, content_lower) for pattern in code_patterns)
        if has_code:
            classifications.append("code_help")
        
        # Resource sharing detection - enhanced patterns
        resource_patterns = [
            r'https?://', r'arxiv\.org', r'github\.com', r'huggingface\.co', 
            r'kaggle\.com', r'scholar\.google', r'tensorflow\.org', r'pytorch\.org',
            r'\b(paper|article|research|study|dataset|model)\b',
            r'\b(tutorial|guide|documentation|docs)\b',
            r'\b(read this|check out|take a look|interesting|useful)\b',
            r'\b(resource|link|source|reference)\b'
        ]
        
        has_resources = any(re.search(pattern, content_lower) for pattern in resource_patterns)
        if has_resources and content_length > 30:  # Substantial resource sharing
            classifications.append("resource_sharing")
        
        # Research detection - specific patterns for academic content
        research_patterns = [
            r'\barxiv\.org\b', r'\b(paper|research|study|findings)\b',
            r'\bmethodology\b', r'\bresults\b', r'\bwhitepaper\b',
            r'\bacademic\b', r'\bscientific\b', r'\bpublication\b'
        ]
        
        has_research = any(re.search(pattern, content_lower) for pattern in research_patterns)
        if has_research and content_length > 40:
            classifications.append("research")
        
        # Tutorial detection - instructional content
        tutorial_patterns = [
            r'\btutorial\b', r'\bstep by step\b', r'\bwalkthrough\b',
            r'\bguide\b', r'\bhow to\b', r'\bbeginners?\b',
            r'\blearn\b', r'\binstruction\b'
        ]
        
        has_tutorial = any(re.search(pattern, content_lower) for pattern in tutorial_patterns)
        if has_tutorial and content_length > 50:
            classifications.append("tutorial")
        
        # Documentation detection
        documentation_patterns = [
            r'\bdocumentation\b', r'\bdocs\b', r'\bmanual\b',
            r'\breference\b', r'\bspecification\b', r'\bapi\b',
            r'\bwhitepaper\b', r'\bguide\b'
        ]
        
        has_documentation = any(re.search(pattern, content_lower) for pattern in documentation_patterns)
        if has_documentation and content_length > 40:
            classifications.append("documentation")
        
        # Meme detection
        meme_patterns = [
            r'\bmeme\b', r'\bfunny\b', r'\bhilarious\b', r'\blol\b',
            r'\blmao\b', r'\bhaha\b', r'\bjoke\b', r'\bcomedy\b'
        ]
        
        has_meme = any(re.search(pattern, content_lower) for pattern in meme_patterns)
        if has_meme:
            classifications.append("meme")
        
        # Casual detection  
        casual_patterns = [
            r'\b(good morning|good night|gm|gn|hello|hi|hey)\b',
            r'\b(how are you|what\'s up|wassup)\b',
            r'\b(have a (great|good|nice) day)\b'
        ]
        
        has_casual = any(re.search(pattern, content_lower) for pattern in casual_patterns)
        if has_casual and content_length < 100:  # Keep casual short
            classifications.append("casual")
        
        # Technical discussion detection - AI/ML focused
        technical_patterns = [
            r'\b(ai|artificial intelligence|machine learning|ml|deep learning|dl)\b',
            r'\b(neural network|transformer|attention|bert|gpt|llm)\b',
            r'\b(algorithm|model|training|inference|optimization)\b',
            r'\b(pytorch|tensorflow|scikit-learn|numpy|pandas)\b',
            r'\b(gradient|loss|accuracy|precision|recall|f1)\b',
            r'\b(classification|regression|clustering|supervised|unsupervised)\b',
            r'\b(nlp|computer vision|cv|reinforcement learning|rl)\b',
            r'\b(architecture|framework|implementation|performance)\b'
        ]
        
        is_technical = any(re.search(pattern, content_lower) for pattern in technical_patterns)
        if is_technical and content_length > 40:  # Substantial technical content
            classifications.append("technical")
        
        # Educational content detection
        educational_patterns = [
            r'\b(tutorial|learn|guide|explanation|example|walkthrough)\b',
            r'\b(step by step|beginner|introduction|basics|fundamentals)\b',
            r'\b(concept|theory|practice|exercise|lesson)\b',
            r'\b(understand|explain|demonstrate|show how)\b'
        ]
        
        has_educational = any(re.search(pattern, content_lower) for pattern in educational_patterns)
        if has_educational and content_length > 50:  # Substantial educational content
            classifications.append("educational")
        
        # Project/showcase detection
        project_patterns = [
            r'\b(project|built|created|developed|working on)\b',
            r'\b(demo|showcase|portfolio|collaboration|open source)\b',
            r'\b(feedback|review|thoughts|opinions)\b',
            r'\b(deployed|launched|released|published)\b'
        ]
        
        has_project = any(re.search(pattern, content_lower) for pattern in project_patterns)
        if has_project and content_length > 40:
            classifications.append("project")
        
        # Discussion detection for longer, thoughtful content
        if (content_length > 200 and 
            not any(cls in classifications for cls in ['question', 'code_help', 'low_quality']) and
            not is_low_quality):
            classifications.append("discussion")
        
        # General category for short but acceptable content
        if (not classifications and 
            content_length > 20 and 
            content_length < 100 and 
            not is_low_quality):
            classifications.append("general")
        
        # Default handling
        if not classifications:
            if is_low_quality or content_length < 20:
                return ["low_quality"]
            else:
                return ["general"]
            
        return classifications[:3]  # Limit to max 3 categories
    
    def calculate_quality_score(self, analysis: Dict[str, Any]) -> float:
        """
        Enhanced quality scoring algorithm with focus on substantial, educational content
        
        Prioritizes quality over quantity by rewarding:
        - Educational and technical value
        - Substantial content length
        - High-quality resources and links
        - Meaningful discussions
        
        Returns: Quality score between 0.0 and 1.0
        """
        score = 0.0
        classifications = analysis.get("classifications", [])
        urls = analysis.get("urls", [])
        code_snippets = analysis.get("code_snippets", [])
        attachments = analysis.get("attachments_processed", [])
        content = analysis.get("content", "")
        content_length = len(content.strip())
        
        # Immediate penalty for low quality content
        if "low_quality" in classifications:
            return max(0.05, score - 0.8)  # Heavy penalty, but not zero
        
        # Base score from content length and substance
        if content_length > 500:  # Very substantial content
            score += 0.4
        elif content_length > 200:  # Good content length
            score += 0.3
        elif content_length > 100:  # Moderate content
            score += 0.2
        elif content_length > 50:   # Minimal acceptable content
            score += 0.1
        else:  # Very short content
            score -= 0.1
        
        # Classification-based scoring with quality focus
        classification_weights = {
            "resource_sharing": 0.35,   # High value for sharing resources
            "educational": 0.35,        # High value for educational content
            "technical": 0.30,          # High value for technical discussions
            "research": 0.40,           # Very high value for research content
            "tutorial": 0.35,           # High value for tutorials
            "documentation": 0.30,      # High value for documentation
            "project": 0.25,            # Good value for projects
            "code_help": 0.20,          # Moderate value for code help
            "discussion": 0.15,         # Basic value for discussions
            "question": 0.10,           # Lower value for simple questions
            "general": 0.05,            # Low value for general content
            "casual": 0.02,             # Very low value for casual content
            "meme": 0.01,               # Minimal value for memes
            "low_quality": -0.50        # Penalty for low quality content
        }
        
        # Apply classification bonuses
        for classification in classifications:
            if classification in classification_weights:
                score += classification_weights[classification]
        
        # High-quality URL assessment with stricter criteria
        high_quality_domains = [
            '.edu', '.org', 'arxiv.org', 'github.com', 'huggingface.co', 
            'kaggle.com', 'scholar.google.com', 'tensorflow.org', 'pytorch.org',
            'scikit-learn.org', 'numpy.org', 'pandas.pydata.org', 'jupyter.org',
            'openai.com', 'deepmind.com', 'ai.google', 'research.microsoft.com',
            'papers.nips.cc', 'proceedings.mlr.press', 'jmlr.org'
        ]
        
        medium_quality_domains = [
            'medium.com', 'towardsdatascience.com', 'analyticsvidhya.com',
            'kdnuggets.com', 'machinelearningmastery.com', 'distill.pub',
            'blog.openai.com', 'ai.googleblog.com'
        ]
        
        low_quality_domains = [
            'tenor.com', 'giphy.com', 'imgur.com', 'reddit.com/r/memes',
            'twitter.com', 'tiktok.com', 'instagram.com', 'facebook.com'
        ]
        
        for url_data in urls:
            domain = url_data.get("domain", "").lower()
            url_type = url_data.get("type", "")
            
            # High-quality domains get significant bonus
            if any(domain.endswith(d) for d in high_quality_domains):
                score += 0.4
            elif any(domain.endswith(d) for d in medium_quality_domains):
                score += 0.2
            elif any(domain.endswith(d) for d in low_quality_domains):
                score -= 0.3  # Penalty for low-quality domains
            elif url_type in ['research', 'documentation', 'tutorial']:
                score += 0.3
            else:
                score += 0.1  # Small bonus for any URL
        
        # Code snippet quality assessment - favor substantial code
        total_code_length = sum(len(code.get("code", "")) for code in code_snippets)
        if total_code_length > 500:  # Substantial code
            score += 0.3
        elif total_code_length > 200:  # Moderate code
            score += 0.2
        elif total_code_length > 50:   # Some code
            score += 0.1
        
        # Attachment quality assessment
        for attachment in attachments:
            filename = attachment.get("filename", "").lower()
            content_type = attachment.get("content_type", "").lower()
            
            # High-value file types
            if any(ext in filename for ext in ['.pdf', '.docx', '.pptx', '.ipynb', '.py', '.js', '.cpp']):
                score += 0.3
            elif any(ext in filename for ext in ['.png', '.jpg', '.jpeg', '.svg']) and 'screenshot' in filename:
                score += 0.15  # Screenshots can be valuable for code help
            elif content_type.startswith('image/'):
                score += 0.05  # Basic image content
            else:
                score += 0.1   # Other attachments
        
        # Bonus for multi-modal content (text + resources + code)
        has_text = content_length > 50
        has_urls = len(urls) > 0
        has_code = len(code_snippets) > 0
        has_attachments = len(attachments) > 0
        
        content_types = sum([has_text, has_urls, has_code, has_attachments])
        if content_types >= 3:  # Rich, multi-modal content
            score += 0.2
        elif content_types >= 2:  # Good content variety
            score += 0.1
        
        # Special bonuses for high-value combinations
        if ("resource_sharing" in classifications and 
            ("technical" in classifications or "educational" in classifications) and
            has_urls and content_length > 100):
            score += 0.25  # Bonus for well-explained resource sharing
        
        if ("code_help" in classifications and 
            has_code and content_length > 100):
            score += 0.2   # Bonus for substantial code help with explanation
        
        # Ensure score is within bounds
        return min(1.0, max(0.0, score))
    
    def should_include_resource(self, analysis: Dict[str, Any], quality_threshold: float = 0.5) -> bool:
        """
        Determine if a resource should be included with stricter quality filtering
        
        Implements "quality over quantity" philosophy by:
        - Raising default quality threshold from 0.3 to 0.5
        - Requiring substantial content for most categories
        - Prioritizing educational and technical value
        - Strict filtering of low-quality content
        
        Args:
            analysis: Content analysis result
            quality_threshold: Minimum quality score for inclusion (default 0.5 - stricter)
            
        Returns: True if resource meets quality criteria
        """
        quality_score = self.calculate_quality_score(analysis)
        classifications = analysis.get("classifications", [])
        content = analysis.get("content", "")
        content_length = len(content.strip())
        
        # Immediate exclusion criteria - prioritize quality
        if "low_quality" in classifications:
            return False  # Zero tolerance for low quality content
        
        # Content length requirements - ensure substantial content
        if content_length < 30:  # Very short content rarely has value
            return False
        
        # Stricter quality threshold enforcement
        if quality_score >= quality_threshold:
            return True
        
        # Special high-value categories with relaxed but still meaningful thresholds
        exceptional_value_types = {
            "resource_sharing": 0.4,   # Must be substantial resource sharing
            "educational": 0.4,        # Must be meaningful educational content
            "technical": 0.35,         # Technical discussions need depth
            "project": 0.3             # Project showcases can be slightly lower
        }
        
        for category, min_score in exceptional_value_types.items():
            if (category in classifications and 
                quality_score >= min_score and
                content_length > 80):  # Require substantial content even for exceptions
                return True
        
        # Code help with actual code gets special consideration
        if ("code_help" in classifications and 
            len(analysis.get("code_snippets", [])) > 0 and
            quality_score >= 0.3 and
            content_length > 50):
            return True
        
        # Multi-modal content (text + URLs/attachments + substantial length)
        has_urls = len(analysis.get("urls", [])) > 0
        has_attachments = len(analysis.get("attachments_processed", [])) > 0
        
        if (has_urls and content_length > 100 and quality_score >= 0.35):
            return True
        
        if (has_attachments and content_length > 80 and quality_score >= 0.3):
            return True
        
        # Default: strict quality filtering
        return False
