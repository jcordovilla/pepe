"""
Validation System

AI-powered validation system for query-answer pairs with quality assessment,
relevance scoring, and improvement suggestions.
"""

import asyncio
import json
import re
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ValidationMethod(Enum):
    """Validation method types"""
    HEURISTIC = "heuristic"
    AI_POWERED = "ai_powered"
    HUMAN = "human"
    HYBRID = "hybrid"


class QualityDimension(Enum):
    """Quality assessment dimensions"""
    RELEVANCE = "relevance"
    COMPLETENESS = "completeness"
    ACCURACY = "accuracy"
    CLARITY = "clarity"
    HELPFULNESS = "helpfulness"


@dataclass
class ValidationMetrics:
    """Validation metrics for a query-answer pair"""
    relevance_score: float
    completeness_score: float
    accuracy_score: float
    clarity_score: float
    helpfulness_score: float
    overall_score: float
    confidence: float


@dataclass
class QualityIssue:
    """Identified quality issue"""
    dimension: QualityDimension
    severity: str  # "low", "medium", "high"
    description: str
    suggestion: str


@dataclass
class ValidationReport:
    """Comprehensive validation report"""
    query_id: int
    validation_method: ValidationMethod
    metrics: ValidationMetrics
    issues: List[QualityIssue]
    improvements: List[str]
    validated_at: datetime
    validator_info: Dict[str, Any]


class ValidationSystem:
    """
    Comprehensive validation system for query-answer quality assessment.
    
    Features:
    - Multiple validation methods (heuristic, AI-powered, human)
    - Detailed quality scoring across multiple dimensions
    - Issue identification and improvement suggestions
    - Continuous learning from feedback
    - Integration with query repository
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.validation_enabled = config.get("validation_enabled", True)
        self.default_method = ValidationMethod(config.get("default_method", "heuristic"))
        self.enable_ai_validation = config.get("enable_ai_validation", False)
        
        # Quality thresholds
        self.quality_thresholds = config.get("quality_thresholds", {
            "excellent": 4.5,
            "good": 3.5,
            "average": 2.5,
            "poor": 1.5
        })
        
        # Validation rules and patterns
        self.validation_rules = self._load_validation_rules()
        
        # Learning data for continuous improvement
        self.feedback_data = []
        
        logger.info("Validation System initialized with local Llama model")
    
    def _load_validation_rules(self) -> Dict[str, Any]:
        """Load validation rules and patterns"""
        return {
            "min_answer_length": 10,
            "max_answer_length": 5000,
            "required_elements": ["substantive_content"],
            "quality_indicators": {
                "positive": [
                    r"\b(detailed|comprehensive|specific|examples?|steps?|guide)\b",
                    r"\b(explanation|because|therefore|however)\b",
                    r"\b(first|second|finally|additionally)\b",
                    r"[0-9]+\.",  # Numbered lists
                    r"•",  # Bullet points
                ],
                "negative": [
                    r"\b(sorry|can'?t|unable|don'?t know|not sure)\b",
                    r"\b(maybe|perhaps|might|could be)\b",
                    r"^.{0,20}$",  # Very short answers
                    r"\b(error|failed|problem)\b"
                ]
            },
            "relevance_keywords": {
                "high_relevance": [
                    r"\b(specific|exactly|precisely|directly)\b",
                    r"\b(answer|solution|method|approach)\b"
                ],
                "low_relevance": [
                    r"\b(general|vague|unclear|confusing)\b",
                    r"\b(off.?topic|unrelated|irrelevant)\b"
                ]
            }
        }
    
    async def validate_query_answer(
        self,
        query_id: int,
        query_text: str,
        answer_text: str,
        context: Optional[Dict[str, Any]] = None,
        method: Optional[ValidationMethod] = None
    ) -> ValidationReport:
        """
        Validate a query-answer pair using specified method.
        
        Args:
            query_id: ID of the query-answer pair
            query_text: Original user query
            answer_text: System response
            context: Additional context information
            method: Validation method to use
            
        Returns:
            Comprehensive validation report
        """
        if not self.validation_enabled:
            return self._create_default_report(query_id)
        
        validation_method = method or self.default_method
        
        try:
            if validation_method == ValidationMethod.HEURISTIC:
                return await self._heuristic_validation(
                    query_id, query_text, answer_text, context
                )
            elif validation_method == ValidationMethod.AI_POWERED:
                return await self._ai_powered_validation(
                    query_id, query_text, answer_text, context
                )
            elif validation_method == ValidationMethod.HYBRID:
                return await self._hybrid_validation(
                    query_id, query_text, answer_text, context
                )
            else:
                return await self._heuristic_validation(
                    query_id, query_text, answer_text, context
                )
                
        except Exception as e:
            logger.error(f"Validation failed for query {query_id}: {e}")
            return self._create_error_report(query_id, str(e))
    
    async def _heuristic_validation(
        self,
        query_id: int,
        query_text: str,
        answer_text: str,
        context: Optional[Dict[str, Any]] = None
    ) -> ValidationReport:
        """Perform heuristic-based validation"""
        metrics = self._calculate_heuristic_metrics(query_text, answer_text)
        issues = self._identify_heuristic_issues(query_text, answer_text, metrics)
        improvements = self._generate_heuristic_improvements(issues, metrics)
        
        return ValidationReport(
            query_id=query_id,
            validation_method=ValidationMethod.HEURISTIC,
            metrics=metrics,
            issues=issues,
            improvements=improvements,
            validated_at=datetime.utcnow(),
            validator_info={"method": "heuristic", "version": "1.0"}
        )
    
    def _calculate_heuristic_metrics(
        self,
        query_text: str,
        answer_text: str
    ) -> ValidationMetrics:
        """Calculate quality metrics using heuristic methods"""
        
        # Relevance score based on keyword overlap and semantic similarity
        relevance_score = self._calculate_relevance_score(query_text, answer_text)
        
        # Completeness based on answer length and structure
        completeness_score = self._calculate_completeness_score(answer_text)
        
        # Accuracy based on confidence indicators and error markers
        accuracy_score = self._calculate_accuracy_score(answer_text)
        
        # Clarity based on readability and structure
        clarity_score = self._calculate_clarity_score(answer_text)
        
        # Helpfulness based on actionable content and specificity
        helpfulness_score = self._calculate_helpfulness_score(query_text, answer_text)
        
        # Overall score (weighted average)
        overall_score = (
            relevance_score * 0.25 +
            completeness_score * 0.20 +
            accuracy_score * 0.20 +
            clarity_score * 0.15 +
            helpfulness_score * 0.20
        )
        
        # Confidence based on multiple factors
        confidence = min(0.8, (completeness_score + clarity_score) / 2)
        
        return ValidationMetrics(
            relevance_score=relevance_score,
            completeness_score=completeness_score,
            accuracy_score=accuracy_score,
            clarity_score=clarity_score,
            helpfulness_score=helpfulness_score,
            overall_score=overall_score,
            confidence=confidence
        )
    
    def _calculate_relevance_score(self, query_text: str, answer_text: str) -> float:
        """Calculate relevance score using keyword overlap and patterns"""
        # Simple keyword overlap
        query_words = set(re.findall(r'\w+', query_text.lower()))
        answer_words = set(re.findall(r'\w+', answer_text.lower()))
        
        overlap = len(query_words.intersection(answer_words))
        max_possible = len(query_words)
        
        if max_possible == 0:
            base_score = 0.5
        else:
            base_score = min(1.0, overlap / max_possible)
        
        # Check for high relevance indicators
        high_rel_count = sum(
            1 for pattern in self.validation_rules["relevance_keywords"]["high_relevance"]
            if re.search(pattern, answer_text, re.IGNORECASE)
        )
        
        # Check for low relevance indicators
        low_rel_count = sum(
            1 for pattern in self.validation_rules["relevance_keywords"]["low_relevance"]
            if re.search(pattern, answer_text, re.IGNORECASE)
        )
        
        # Adjust score based on indicators
        adjustment = (high_rel_count * 0.1) - (low_rel_count * 0.2)
        final_score = max(0.0, min(1.0, base_score + adjustment))
        
        return final_score * 5.0  # Scale to 0-5
    
    def _calculate_completeness_score(self, answer_text: str) -> float:
        """Calculate completeness score based on length and structure"""
        length = len(answer_text)
        
        # Base score from length
        if length < 20:
            length_score = 0.1
        elif length < 50:
            length_score = 0.3
        elif length < 100:
            length_score = 0.6
        elif length < 300:
            length_score = 0.8
        else:
            length_score = 1.0
        
        # Structure indicators
        structure_score = 0.0
        
        # Lists and organization
        if re.search(r'[0-9]+\.|\*|\-|•', answer_text):
            structure_score += 0.2
        
        # Multiple sentences
        sentences = len(re.findall(r'[.!?]+', answer_text))
        if sentences > 1:
            structure_score += min(0.3, sentences * 0.1)
        
        # Examples or explanations
        if re.search(r'\b(example|for instance|such as|specifically)\b', answer_text, re.IGNORECASE):
            structure_score += 0.2
        
        final_score = min(1.0, length_score + structure_score)
        return final_score * 5.0  # Scale to 0-5
    
    def _calculate_accuracy_score(self, answer_text: str) -> float:
        """Calculate accuracy score based on confidence indicators"""
        # Start with neutral score
        base_score = 0.7
        
        # Positive indicators
        positive_count = sum(
            1 for pattern in self.validation_rules["quality_indicators"]["positive"]
            if re.search(pattern, answer_text, re.IGNORECASE)
        )
        
        # Negative indicators
        negative_count = sum(
            1 for pattern in self.validation_rules["quality_indicators"]["negative"]
            if re.search(pattern, answer_text, re.IGNORECASE)
        )
        
        # Adjust score
        adjustment = (positive_count * 0.05) - (negative_count * 0.15)
        final_score = max(0.2, min(1.0, base_score + adjustment))
        
        return final_score * 5.0  # Scale to 0-5
    
    def _calculate_clarity_score(self, answer_text: str) -> float:
        """Calculate clarity score based on readability"""
        # Simple readability metrics
        sentences = len(re.findall(r'[.!?]+', answer_text))
        words = len(re.findall(r'\w+', answer_text))
        
        if sentences == 0:
            return 1.0  # Single fragment
        
        avg_words_per_sentence = words / sentences
        
        # Optimal range is 10-20 words per sentence
        if 10 <= avg_words_per_sentence <= 20:
            readability_score = 1.0
        elif avg_words_per_sentence < 5:
            readability_score = 0.6  # Too choppy
        elif avg_words_per_sentence > 30:
            readability_score = 0.4  # Too complex
        else:
            readability_score = 0.8
        
        # Check for clear structure
        structure_bonus = 0.0
        if re.search(r'\b(first|second|third|finally|in conclusion)\b', answer_text, re.IGNORECASE):
            structure_bonus += 0.1
        
        if re.search(r'\b(however|therefore|additionally|furthermore)\b', answer_text, re.IGNORECASE):
            structure_bonus += 0.1
        
        final_score = min(1.0, readability_score + structure_bonus)
        return final_score * 5.0  # Scale to 0-5
    
    def _calculate_helpfulness_score(self, query_text: str, answer_text: str) -> float:
        """Calculate helpfulness score based on actionability and specificity"""
        # Actionable content indicators
        actionable_patterns = [
            r'\b(step|steps|method|approach|way|how to)\b',
            r'\b(click|select|choose|enter|type)\b',
            r'\b(first|then|next|after|finally)\b',
            r'\b(should|must|need to|have to)\b'
        ]
        
        actionable_count = sum(
            1 for pattern in actionable_patterns
            if re.search(pattern, answer_text, re.IGNORECASE)
        )
        
        # Specificity indicators
        specific_patterns = [
            r'\b[A-Z][a-z]+\s[A-Z][a-z]+\b',  # Proper nouns
            r'\b\d+\b',  # Numbers
            r'\bhttps?://\S+\b',  # URLs
            r'\b[A-Z]{2,}\b'  # Acronyms
        ]
        
        specific_count = sum(
            1 for pattern in specific_patterns
            if re.search(pattern, answer_text)
        )
        
        # Calculate base helpfulness
        helpfulness = min(1.0, (actionable_count * 0.15) + (specific_count * 0.1) + 0.5)
        
        # Check if answer directly addresses query type
        query_lower = query_text.lower()
        if any(word in query_lower for word in ['how', 'what', 'why', 'where', 'when']):
            if any(word in answer_text.lower() for word in ['because', 'by', 'through', 'using']):
                helpfulness += 0.2
        
        return min(1.0, helpfulness) * 5.0  # Scale to 0-5
    
    def _identify_heuristic_issues(
        self,
        query_text: str,
        answer_text: str,
        metrics: ValidationMetrics
    ) -> List[QualityIssue]:
        """Identify quality issues using heuristic analysis"""
        issues = []
        
        # Length issues
        if len(answer_text) < 20:
            issues.append(QualityIssue(
                dimension=QualityDimension.COMPLETENESS,
                severity="high",
                description="Answer is too short to be comprehensive",
                suggestion="Provide more detailed explanation with examples"
            ))
        
        if len(answer_text) > 2000:
            issues.append(QualityIssue(
                dimension=QualityDimension.CLARITY,
                severity="medium",
                description="Answer is very long and may be overwhelming",
                suggestion="Consider summarizing key points or breaking into sections"
            ))
        
        # Relevance issues
        if metrics.relevance_score < 2.0:
            issues.append(QualityIssue(
                dimension=QualityDimension.RELEVANCE,
                severity="high",
                description="Answer appears to be off-topic or not directly relevant",
                suggestion="Focus more directly on the specific question asked"
            ))
        
        # Accuracy issues
        if re.search(r'\b(sorry|can\'?t|unable|don\'?t know)\b', answer_text, re.IGNORECASE):
            issues.append(QualityIssue(
                dimension=QualityDimension.ACCURACY,
                severity="medium",
                description="Answer indicates uncertainty or inability to help",
                suggestion="Provide partial information or guide user to resources"
            ))
        
        # Clarity issues
        sentences = len(re.findall(r'[.!?]+', answer_text))
        words = len(re.findall(r'\w+', answer_text))
        if sentences > 0 and words / sentences > 25:
            issues.append(QualityIssue(
                dimension=QualityDimension.CLARITY,
                severity="medium",
                description="Sentences are too long and may be hard to follow",
                suggestion="Break down complex sentences into shorter, clearer ones"
            ))
        
        # Helpfulness issues
        if metrics.helpfulness_score < 2.0:
            issues.append(QualityIssue(
                dimension=QualityDimension.HELPFULNESS,
                severity="medium",
                description="Answer lacks actionable advice or specific information",
                suggestion="Include specific steps, examples, or practical guidance"
            ))
        
        return issues
    
    def _generate_heuristic_improvements(
        self,
        issues: List[QualityIssue],
        metrics: ValidationMetrics
    ) -> List[str]:
        """Generate improvement suggestions based on identified issues"""
        improvements = []
        
        # Add issue-specific suggestions
        for issue in issues:
            improvements.append(issue.suggestion)
        
        # General improvements based on scores
        if metrics.completeness_score < 3.0:
            improvements.append("Add more detailed explanations and examples")
        
        if metrics.clarity_score < 3.0:
            improvements.append("Improve readability with better structure and clearer language")
        
        if metrics.helpfulness_score < 3.0:
            improvements.append("Include more practical, actionable advice")
        
        if metrics.relevance_score < 3.0:
            improvements.append("Focus more directly on the specific question asked")
        
        # Remove duplicates while preserving order
        seen = set()
        unique_improvements = []
        for improvement in improvements:
            if improvement not in seen:
                seen.add(improvement)
                unique_improvements.append(improvement)
        
        return unique_improvements
    
    async def _ai_powered_validation(
        self,
        query_id: int,
        query_text: str,
        answer_text: str,
        context: Optional[Dict[str, Any]] = None
    ) -> ValidationReport:
        """Perform AI-powered validation (placeholder for future implementation)"""
        # For now, fall back to heuristic validation
        # In a full implementation, this would use an AI model to assess quality
        
        logger.warning("AI-powered validation not yet implemented, falling back to heuristic")
        return await self._heuristic_validation(query_id, query_text, answer_text, context)
    
    async def _hybrid_validation(
        self,
        query_id: int,
        query_text: str,
        answer_text: str,
        context: Optional[Dict[str, Any]] = None
    ) -> ValidationReport:
        """Perform hybrid validation combining multiple methods"""
        # Start with heuristic validation
        heuristic_report = await self._heuristic_validation(
            query_id, query_text, answer_text, context
        )
        
        # In a full implementation, this would combine with AI validation
        # For now, just enhance the heuristic report
        
        # Add hybrid-specific improvements
        additional_improvements = [
            "Consider multiple validation perspectives",
            "Cross-reference with similar successful answers"
        ]
        
        heuristic_report.improvements.extend(additional_improvements)
        heuristic_report.validation_method = ValidationMethod.HYBRID
        heuristic_report.validator_info["method"] = "hybrid"
        
        return heuristic_report
    
    def _create_default_report(self, query_id: int) -> ValidationReport:
        """Create default validation report when validation is disabled"""
        return ValidationReport(
            query_id=query_id,
            validation_method=ValidationMethod.HEURISTIC,
            metrics=ValidationMetrics(
                relevance_score=3.0,
                completeness_score=3.0,
                accuracy_score=3.0,
                clarity_score=3.0,
                helpfulness_score=3.0,
                overall_score=3.0,
                confidence=0.5
            ),
            issues=[],
            improvements=[],
            validated_at=datetime.utcnow(),
            validator_info={"method": "disabled", "note": "Validation disabled"}
        )
    
    def _create_error_report(self, query_id: int, error_message: str) -> ValidationReport:
        """Create error validation report"""
        return ValidationReport(
            query_id=query_id,
            validation_method=ValidationMethod.HEURISTIC,
            metrics=ValidationMetrics(
                relevance_score=0.0,
                completeness_score=0.0,
                accuracy_score=0.0,
                clarity_score=0.0,
                helpfulness_score=0.0,
                overall_score=0.0,
                confidence=0.0
            ),
            issues=[QualityIssue(
                dimension=QualityDimension.ACCURACY,
                severity="high",
                description=f"Validation error: {error_message}",
                suggestion="Review system configuration and try again"
            )],
            improvements=["Fix validation system error"],
            validated_at=datetime.utcnow(),
            validator_info={"method": "error", "error": error_message}
        )
    
    async def record_feedback(
        self,
        query_id: int,
        user_feedback: Dict[str, Any],
        validation_report: Optional[ValidationReport] = None
    ):
        """Record user feedback for continuous learning"""
        feedback_entry = {
            "query_id": query_id,
            "user_feedback": user_feedback,
            "validation_report": validation_report,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        self.feedback_data.append(feedback_entry)
        
        # Trim feedback data if too large
        if len(self.feedback_data) > 1000:
            self.feedback_data = self.feedback_data[-1000:]
        
        logger.info(f"Recorded feedback for query {query_id}")
    
    def get_validation_statistics(self) -> Dict[str, Any]:
        """Get validation system statistics"""
        return {
            "validation_enabled": self.validation_enabled,
            "default_method": self.default_method.value,
            "total_feedback_entries": len(self.feedback_data),
            "quality_thresholds": self.quality_thresholds,
            "ai_validation_enabled": self.enable_ai_validation,
            "timestamp": datetime.utcnow().isoformat()
        }
