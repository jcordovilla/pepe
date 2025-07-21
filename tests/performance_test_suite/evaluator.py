"""
Response Evaluator

Evaluates bot responses against expected dummy answers using multiple metrics.
Provides comprehensive assessment of response quality, relevance, and format accuracy.
Now includes Llama model integration for semantic evaluation.
"""

import json
import logging
import re
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Represents evaluation results for a single response."""
    query_id: int
    query: str
    expected_structure: Dict[str, Any]
    actual_response: str
    overall_score: float
    metrics: Dict[str, float]
    detailed_analysis: Dict[str, Any]
    recommendations: List[str]


class LlamaEvaluator:
    """
    Llama-based evaluator for semantic understanding and quality assessment.
    
    Uses Llama model to evaluate:
    - Semantic relevance
    - Response quality
    - Coherence and logical flow
    - Context appropriateness
    """
    
    def __init__(self, model_name: str = "llama3.1:8b"):
        self.model_name = model_name
        self.model = None
        self._initialize_model()
        
        logger.info(f"LlamaEvaluator initialized with model: {model_name}")
    
    def _initialize_model(self):
        """Initialize the Llama model."""
        try:
            import ollama
            
            # Test if model is available
            try:
                ollama.show(self.model_name)
                self.model = ollama
                logger.info(f"Llama model {self.model_name} loaded successfully")
            except Exception as e:
                logger.warning(f"Llama model {self.model_name} not available: {e}")
                logger.info("Falling back to rule-based evaluation only")
                self.model = None
                
        except ImportError:
            logger.warning("Ollama not available. Install with: pip install ollama")
            self.model = None
    
    def evaluate_semantic_relevance(self, query: str, response: str) -> float:
        """
        Evaluate semantic relevance using Llama model.
        
        Args:
            query: The original query
            response: The bot's response
            
        Returns:
            Relevance score between 0.0 and 1.0
        """
        if not self.model:
            return self._fallback_semantic_relevance(query, response)
        
        try:
            prompt = f"""Rate how well this response answers the query on a scale of 0.0 to 1.0.

Query: {query}
Response: {response}

Consider:
- Does the response directly address the question asked?
- Is the information relevant to what was requested?
- Does it provide useful information related to the query?
- Is it appropriate for the context?

Respond with only a number between 0.0 and 1.0, where:
0.0 = Completely irrelevant or doesn't answer the question
0.5 = Partially relevant or somewhat addresses the question
1.0 = Completely relevant and fully answers the question

Score:"""
            
            result = self.model.generate(
                model=self.model_name,
                prompt=prompt,
                options={"temperature": 0.1}
            )
            
            # Extract numeric score from response
            score_text = result['response'].strip()
            score = self._extract_score_from_text(score_text)
            
            logger.debug(f"Llama semantic relevance score: {score} for query: {query[:50]}...")
            return score
            
        except Exception as e:
            logger.error(f"Error in Llama semantic evaluation: {e}")
            return self._fallback_semantic_relevance(query, response)
    
    def evaluate_response_quality(self, response: str, query: str, context: str = "Discord bot response") -> float:
        """
        Evaluate response quality using Llama model.
        
        Args:
            response: The bot's response
            query: The original query
            context: Additional context
            
        Returns:
            Quality score between 0.0 and 1.0
        """
        if not self.model:
            return self._fallback_quality_evaluation(response)
        
        try:
            prompt = f"""Rate the quality of this Discord bot response on a scale of 0.0 to 1.0.

Context: {context}
Query: {query}
Response: {response}

Consider:
- Is the information accurate and correct?
- Is the response helpful and useful?
- Is it appropriate for a Discord community?
- Is the tone and style suitable?
- Does it provide value to the user?

Respond with only a number between 0.0 and 1.0, where:
0.0 = Poor quality, inaccurate, unhelpful
0.5 = Average quality, partially helpful
1.0 = Excellent quality, accurate, very helpful

Score:"""
            
            result = self.model.generate(
                model=self.model_name,
                prompt=prompt,
                options={"temperature": 0.1}
            )
            
            score_text = result['response'].strip()
            score = self._extract_score_from_text(score_text)
            
            logger.debug(f"Llama quality score: {score}")
            return score
            
        except Exception as e:
            logger.error(f"Error in Llama quality evaluation: {e}")
            return self._fallback_quality_evaluation(response)
    
    def evaluate_coherence(self, response: str) -> float:
        """
        Evaluate response coherence using Llama model.
        
        Args:
            response: The bot's response
            
        Returns:
            Coherence score between 0.0 and 1.0
        """
        if not self.model:
            return self._fallback_coherence_evaluation(response)
        
        try:
            prompt = f"""Rate the coherence and logical flow of this response on a scale of 0.0 to 1.0.

Response: {response}

Consider:
- Does the response flow logically from one point to the next?
- Are the ideas well-connected and organized?
- Is the reasoning clear and understandable?
- Does it make sense as a complete response?

Respond with only a number between 0.0 and 1.0, where:
0.0 = Incoherent, confusing, poorly organized
0.5 = Somewhat coherent, partially organized
1.0 = Very coherent, well-organized, logical flow

Score:"""
            
            result = self.model.generate(
                model=self.model_name,
                prompt=prompt,
                options={"temperature": 0.1}
            )
            
            score_text = result['response'].strip()
            score = self._extract_score_from_text(score_text)
            
            logger.debug(f"Llama coherence score: {score}")
            return score
            
        except Exception as e:
            logger.error(f"Error in Llama coherence evaluation: {e}")
            return self._fallback_coherence_evaluation(response)
    
    def _extract_score_from_text(self, text: str) -> float:
        """Extract numeric score from Llama response text."""
        try:
            # Look for numbers in the text
            import re
            numbers = re.findall(r'\d+\.?\d*', text)
            if numbers:
                score = float(numbers[0])
                # Ensure score is between 0 and 1
                return max(0.0, min(1.0, score))
            else:
                # Fallback: look for words that indicate score
                text_lower = text.lower()
                if any(word in text_lower for word in ['excellent', 'perfect', '1.0', '1']):
                    return 1.0
                elif any(word in text_lower for word in ['good', '0.8', '0.9']):
                    return 0.8
                elif any(word in text_lower for word in ['average', '0.5', '0.6']):
                    return 0.5
                elif any(word in text_lower for word in ['poor', 'bad', '0.0', '0.1']):
                    return 0.2
                else:
                    return 0.5  # Default middle score
        except Exception as e:
            logger.error(f"Error extracting score from text '{text}': {e}")
            return 0.5
    
    def _fallback_semantic_relevance(self, query: str, response: str) -> float:
        """Fallback semantic relevance evaluation using rule-based approach."""
        # Extract key terms from query
        key_terms = self._extract_key_terms(query)
        response_lower = response.lower()
        
        # Check term coverage
        term_coverage = sum(1 for term in key_terms if term.lower() in response_lower) / len(key_terms) if key_terms else 0
        
        # Check for question-answer patterns
        question_words = ['what', 'how', 'why', 'when', 'where', 'who', 'which']
        has_question = any(word in query.lower() for word in question_words)
        
        if has_question:
            # Look for answer indicators
            answer_indicators = ['is', 'are', 'was', 'were', 'will', 'can', 'should', 'because', 'due to', 'since']
            has_answer = any(indicator in response_lower for indicator in answer_indicators)
            if has_answer:
                term_coverage = min(1.0, term_coverage + 0.3)
        
        return min(term_coverage, 1.0)
    
    def _fallback_quality_evaluation(self, response: str) -> float:
        """Fallback quality evaluation using rule-based approach."""
        quality_score = 0.5  # Base score
        
        # Length quality
        if 50 <= len(response) <= 500:
            quality_score += 0.2
        elif len(response) > 500:
            quality_score += 0.1
        
        # Structure quality
        if '\n' in response or '•' in response or '-' in response:
            quality_score += 0.1
        
        # Technical content quality
        technical_indicators = ['because', 'therefore', 'however', 'additionally', 'furthermore']
        if any(indicator in response.lower() for indicator in technical_indicators):
            quality_score += 0.1
        
        # Code block quality
        if '```' in response:
            quality_score += 0.1
        
        return min(quality_score, 1.0)
    
    def _fallback_coherence_evaluation(self, response: str) -> float:
        """Fallback coherence evaluation using rule-based approach."""
        coherence_score = 0.5  # Base score
        
        # Sentence structure
        sentences = re.split(r'[.!?]+', response)
        valid_sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        if len(valid_sentences) > 1:
            coherence_score += 0.2
        
        # Logical connectors
        connectors = ['however', 'therefore', 'furthermore', 'additionally', 'in addition', 'also', 'but', 'and']
        connector_count = sum(1 for connector in connectors if connector in response.lower())
        
        if connector_count > 0:
            coherence_score += 0.2
        
        # Paragraph structure
        paragraphs = response.split('\n\n')
        if len(paragraphs) > 1:
            coherence_score += 0.1
        
        return min(coherence_score, 1.0)
    
    def _extract_key_terms(self, query: str) -> List[str]:
        """Extract key terms from a query."""
        # Remove common words and extract meaningful terms
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'what', 'when', 'where', 'why', 'how', 'show', 'me', 'give', 'tell', 'find', 'get', 'create', 'generate', 'analyze', 'summarize'}
        
        words = re.findall(r'\b\w+\b', query.lower())
        key_terms = [word for word in words if word not in common_words and len(word) > 2]
        
        return key_terms


class ResponseEvaluator:
    """
    Evaluates bot responses against expected dummy answers.
    
    Evaluation criteria:
    - Response relevance and completeness (Llama + rule-based)
    - Format accuracy and structure (rule-based)
    - Content quality and coherence (Llama + rule-based)
    - Semantic similarity (Llama)
    - Response time performance (rule-based)
    """
    
    def __init__(self):
        """Initialize the response evaluator with hybrid evaluation."""
        self.llama_evaluator = LlamaEvaluator()
        
        # Initialize hybrid evaluator for better evaluation methodology
        self.hybrid_evaluator = HybridEvaluator({
            "llama_model": "llama3.1:8b",
            "weights": {
                "objective": 0.6,
                "ai": 0.4
            }
        })
        
        logger.info("ResponseEvaluator initialized with hybrid evaluation")
    
    def evaluate_responses(self, queries: List[Any], responses: List[Any]) -> List[EvaluationResult]:
        """
        Evaluate all bot responses against expected dummy answers.
        
        Args:
            queries: List of TestQuery objects with expected structures
            responses: List of BotResponse objects with actual responses
            
        Returns:
            List of EvaluationResult objects with comprehensive evaluation
        """
        logger.info(f"Starting evaluation of {len(responses)} responses with Llama integration...")
        
        self.evaluation_results = []
        
        # Create a mapping of query_id to query for easy lookup
        query_map = {query.id: query for query in queries}
        
        for response in responses:
            try:
                if not response or not hasattr(response, 'query_id') or response.query_id not in query_map:
                    logger.warning(f"Skipping malformed or unmatched response: {response}")
                    continue
                query = query_map[response.query_id]
                evaluation = self._evaluate_single_response(query, response)
                self.evaluation_results.append(evaluation)
            except Exception as e:
                logger.error(f"Error evaluating response {getattr(response, 'query_id', '?')}: {e}")
        
        logger.info(f"Completed evaluation of {len(self.evaluation_results)} responses")
        return self.evaluation_results
    
    def _evaluate_single_response(self, query: Any, response: Any) -> EvaluationResult:
        """Evaluate a single response using hybrid evaluation methodology."""
        try:
            if not response.success:
                return self._evaluate_failed_response(query, response)
            
            # Use hybrid evaluator for comprehensive evaluation
            evaluation_result = self.hybrid_evaluator.evaluate_response(
                query=query.query,
                response=response.response,
                expected_structure=query.expected_structure
            )
            
            # Extract scores from hybrid evaluation
            overall_score = evaluation_result["overall_score"]
            objective_score = evaluation_result["objective_score"]["overall"]
            ai_score = evaluation_result["ai_score"]["overall"]
            
            # Create metrics dictionary
            metrics = {
                "overall": overall_score,
                "objective": objective_score,
                "ai": ai_score,
                "format_accuracy": evaluation_result["objective_score"]["metrics"].get("format_accuracy", 0.5),
                "completeness": evaluation_result["objective_score"]["metrics"].get("completeness", 0.5),
                "performance": evaluation_result["objective_score"]["metrics"].get("performance", 0.5),
                "content_matching": evaluation_result["objective_score"]["metrics"].get("content_matching", 0.5),
                "semantic_similarity": evaluation_result["objective_score"]["metrics"].get("semantic_similarity", 0.5)
            }
            
            # Add AI metrics if available
            if evaluation_result["ai_score"].get("ai_available", False):
                ai_metrics = evaluation_result["ai_score"]["metrics"]
                metrics.update({
                    "relevance": ai_metrics.get("relevance", 0.5),
                    "quality": ai_metrics.get("quality", 0.5),
                    "coherence": ai_metrics.get("coherence", 0.5)
                })
            
            # Generate detailed analysis
            analysis = evaluation_result["analysis"]
            
            # Generate recommendations
            recommendations = analysis.get("recommendations", [])
            
            return EvaluationResult(
                query_id=query.id,
                query=query.query,
                expected_structure=query.expected_structure,
                actual_response=response.response,
                overall_score=overall_score,
                metrics=metrics,
                detailed_analysis=analysis,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Error evaluating response for query {query.id}: {e}")
            return self._evaluate_failed_response(query, response)
    
    def _evaluate_failed_response(self, query: Any, response: Any) -> EvaluationResult:
        """Evaluate a failed response."""
        return EvaluationResult(
            query_id=query.id,
            query=query.query,
            expected_structure=query.expected_response_structure,
            actual_response="",
            overall_score=0.0,
            metrics={
                'relevance': 0.0,
                'quality': 0.0,
                'format': 0.0,
                'completeness': 0.0,
                'coherence': 0.0,
                'semantic_similarity': 0.0,
                'performance': 0.0
            },
            detailed_analysis={
                'failure_reason': response.error_message,
                'error_type': self._classify_error(response.error_message)
            },
            recommendations=[f"Fix error: {response.error_message}"]
        )
    
    def _calculate_relevance_score(self, query: Any, response: Any) -> float:
        """
        Calculate relevance score using Llama model.
        
        Returns:
            Score between 0.0 and 1.0
        """
        return self.llama_evaluator.evaluate_semantic_relevance(query.query, response.response)
    
    def _has_unknown_or_redundant_content(self, response_text: str) -> bool:
        # Check for unknown/default values
        unknown_patterns = ["Unknown Channel", "Unknown time", "Unknown", "No messages found"]
        for pat in unknown_patterns:
            if pat.lower() in response_text.lower():
                return True
        # Check for repeated lines/blocks
        lines = [l.strip() for l in response_text.split("\n") if l.strip()]
        seen = set()
        for line in lines:
            if line in seen:
                return True
            seen.add(line)
        return False

    def _calculate_format_score(self, query: Any, response: Any) -> float:
        """
        Calculate format accuracy score (rule-based).
        
        Returns:
            Score between 0.0 and 1.0
        """
        if not response.response:
            return 0.0
        
        expected_structure = query.expected_response_structure
        response_text = response.response
        
        format_scores = []
        
        # Penalize unknown/redundant content
        if self._has_unknown_or_redundant_content(response_text):
            format_scores.append(0.0)
        
        # Check for expected structure elements
        if 'type' in expected_structure:
            expected_type = expected_structure['type']
            if expected_type in response_text.lower():
                format_scores.append(1.0)
            else:
                format_scores.append(0.0)
        
        # Check for list formatting
        if any(key in expected_structure for key in ['channels', 'users', 'topics', 'highlights']):
            list_items = re.findall(r'[-•*]\s+\w+', response_text)
            if list_items:
                format_scores.append(1.0)
            else:
                format_scores.append(0.5)
        
        # Check for numerical data
        if any(key in expected_structure for key in ['metrics', 'statistics', 'count']):
            numbers = re.findall(r'\d+', response_text)
            if numbers:
                format_scores.append(1.0)
            else:
                format_scores.append(0.0)
        
        # Check for structured sections
        sections = re.findall(r'^[A-Z][^:]*:', response_text, re.MULTILINE)
        if sections:
            format_scores.append(0.8)
        else:
            format_scores.append(0.4)
        
        return sum(format_scores) / len(format_scores) if format_scores else 0.5
    
    def _calculate_completeness_score(self, query: Any, response: Any) -> float:
        """
        Calculate completeness score based on expected response structure.
        
        Returns:
            Score between 0.0 and 1.0
        """
        if not response.response:
            return 0.0
        
        expected_structure = query.expected_response_structure
        response_text = response.response
        
        completeness_scores = []
        
        # Penalize unknown/redundant content
        if self._has_unknown_or_redundant_content(response_text):
            completeness_scores.append(0.0)
        
        # Check for required fields in expected structure
        required_fields = self._get_required_fields(expected_structure)
        for field in required_fields:
            if field.lower() in response_text.lower():
                completeness_scores.append(1.0)
            else:
                completeness_scores.append(0.0)
        
        # Check response length adequacy
        expected_length = self._estimate_expected_length(expected_structure)
        actual_length = len(response_text)
        
        if expected_length > 0:
            length_ratio = min(actual_length / expected_length, 2.0)  # Cap at 2x
            completeness_scores.append(length_ratio)
        
        # Check for multiple data points (for statistical responses)
        if 'statistics' in expected_structure or 'metrics' in expected_structure:
            data_points = re.findall(r'\d+', response_text)
            if len(data_points) >= 3:
                completeness_scores.append(1.0)
            elif len(data_points) >= 1:
                completeness_scores.append(0.5)
            else:
                completeness_scores.append(0.0)
        
        return sum(completeness_scores) / len(completeness_scores) if completeness_scores else 0.5
    
    def _calculate_coherence_score(self, response: Any) -> float:
        """
        Calculate response coherence using Llama model.
        
        Returns:
            Score between 0.0 and 1.0
        """
        return self.llama_evaluator.evaluate_coherence(response.response)
    
    def _calculate_semantic_similarity(self, query: Any, response: Any) -> float:
        """
        Calculate semantic similarity between expected and actual response.
        
        Returns:
            Score between 0.0 and 1.0
        """
        if not response.response:
            return 0.0
        
        # For summary/topic queries, prioritize semantic relevance
        if query.category in ['feedback_summary', 'trending_topics', 'digests']:
            return self.llama_evaluator.evaluate_semantic_relevance(query.query, response.response)
        
        # For specific queries, prioritize content matching
        else:
            return self._calculate_content_matching(query, response)
    
    def _calculate_content_matching(self, query: Any, response: Any) -> float:
        """Calculate content matching for specific queries."""
        # Use sequence matcher for text similarity
        expected_text = str(query.expected_response_structure)
        actual_text = response.response
        
        similarity = SequenceMatcher(None, expected_text.lower(), actual_text.lower()).ratio()
        
        return similarity
    
    def _calculate_performance_score(self, response: Any) -> float:
        """
        Calculate performance score based on response time and length.
        
        Returns:
            Score between 0.0 and 1.0
        """
        if not response.success:
            return 0.0
        
        performance_scores = []
        
        # Response time scoring (lower is better, up to 10 seconds is good)
        if response.response_time <= 5:
            performance_scores.append(1.0)
        elif response.response_time <= 10:
            performance_scores.append(0.8)
        elif response.response_time <= 20:
            performance_scores.append(0.6)
        elif response.response_time <= 30:
            performance_scores.append(0.4)
        else:
            performance_scores.append(0.2)
        
        # Response length scoring (adequate length is good)
        if response.metadata and 'response_length' in response.metadata:
            length = response.metadata['response_length']
            if 50 <= length <= 500:
                performance_scores.append(1.0)
            elif 20 <= length < 50 or 500 < length <= 1000:
                performance_scores.append(0.8)
            elif length < 20:
                performance_scores.append(0.4)
            else:
                performance_scores.append(0.6)
        
        return sum(performance_scores) / len(performance_scores) if performance_scores else 0.5
    
    def _get_required_fields(self, expected_structure: Dict[str, Any]) -> List[str]:
        """Get required fields from expected structure."""
        required = []
        
        for key, value in expected_structure.items():
            if isinstance(value, dict):
                required.extend(self._get_required_fields(value))
            elif isinstance(value, list):
                required.append(key)
            else:
                required.append(key)
        
        return required
    
    def _estimate_expected_length(self, expected_structure: Dict[str, Any]) -> int:
        """Estimate expected response length based on structure."""
        base_length = 100
        
        # Add length for each field
        for key, value in expected_structure.items():
            if isinstance(value, dict):
                base_length += 50
            elif isinstance(value, list):
                base_length += len(value) * 30
            else:
                base_length += 20
        
        return base_length
    
    def _classify_error(self, error_message: str) -> str:
        """Classify error type from error message."""
        error_lower = error_message.lower()
        
        if 'timeout' in error_lower:
            return 'Timeout'
        elif 'api' in error_lower:
            return 'API Error'
        elif 'connection' in error_lower:
            return 'Connection Error'
        elif 'rate limit' in error_lower:
            return 'Rate Limit'
        elif 'not found' in error_lower:
            return 'Not Found'
        else:
            return 'Unknown Error'
    
    def _generate_detailed_analysis(self, query: Any, response: Any, metrics: Dict[str, float]) -> Dict[str, Any]:
        """Generate detailed analysis of the response."""
        return {
            'query_analysis': {
                'category': query.category,
                'complexity': query.complexity,
                'edge_case': query.edge_case,
                'time_bound': query.time_bound
            },
            'response_analysis': {
                'length': len(response.response),
                'word_count': len(response.response.split()),
                'response_time': response.response_time,
                'success': response.success
            },
            'metric_breakdown': metrics,
            'llama_evaluation': {
                'model_used': self.llama_evaluator.model_name if self.llama_evaluator.model else 'None',
                'semantic_relevance': metrics['relevance'],
                'quality_assessment': metrics['quality'],
                'coherence_evaluation': metrics['coherence']
            },
            'strengths': self._identify_strengths(metrics),
            'weaknesses': self._identify_weaknesses(metrics)
        }
    
    def _generate_recommendations(self, query: Any, response: Any, metrics: Dict[str, float]) -> List[str]:
        """Generate improvement recommendations."""
        recommendations = []
        
        if metrics['relevance'] < 0.7:
            recommendations.append("Improve response relevance to query topic using better query understanding")
        
        if metrics['quality'] < 0.7:
            recommendations.append("Enhance response quality and accuracy")
        
        if metrics['format'] < 0.7:
            recommendations.append("Standardize response formatting and structure")
        
        if metrics['completeness'] < 0.7:
            recommendations.append("Provide more comprehensive information")
        
        if metrics['coherence'] < 0.7:
            recommendations.append("Improve response coherence and logical flow")
        
        if metrics['performance'] < 0.7:
            recommendations.append("Optimize response time and efficiency")
        
        if not recommendations:
            recommendations.append("Response meets quality standards")
        
        return recommendations
    
    def _identify_strengths(self, metrics: Dict[str, float]) -> List[str]:
        """Identify response strengths."""
        strengths = []
        
        if metrics['relevance'] > 0.8:
            strengths.append("High semantic relevance to query")
        
        if metrics['quality'] > 0.8:
            strengths.append("Excellent response quality")
        
        if metrics['format'] > 0.8:
            strengths.append("Good formatting and structure")
        
        if metrics['completeness'] > 0.8:
            strengths.append("Comprehensive response")
        
        if metrics['coherence'] > 0.8:
            strengths.append("Clear and coherent")
        
        if metrics['performance'] > 0.8:
            strengths.append("Good performance")
        
        return strengths
    
    def _identify_weaknesses(self, metrics: Dict[str, float]) -> List[str]:
        """Identify response weaknesses."""
        weaknesses = []
        
        if metrics['relevance'] < 0.6:
            weaknesses.append("Low semantic relevance to query")
        
        if metrics['quality'] < 0.6:
            weaknesses.append("Poor response quality")
        
        if metrics['format'] < 0.6:
            weaknesses.append("Poor formatting")
        
        if metrics['completeness'] < 0.6:
            weaknesses.append("Incomplete response")
        
        if metrics['coherence'] < 0.6:
            weaknesses.append("Unclear or incoherent")
        
        if metrics['performance'] < 0.6:
            weaknesses.append("Performance issues")
        
        return weaknesses
    
    def get_evaluation_summary(self) -> Dict[str, Any]:
        """Get a comprehensive summary of all evaluations."""
        if not self.evaluation_results:
            return {}
        
        # Calculate overall statistics
        total_evaluations = len(self.evaluation_results)
        successful_evaluations = [e for e in self.evaluation_results if e.overall_score > 0]
        
        if successful_evaluations:
            overall_scores = [e.overall_score for e in successful_evaluations]
            avg_overall_score = sum(overall_scores) / len(overall_scores)
            
            # Calculate average scores for each metric
            metric_averages = {}
            for metric in ['relevance', 'quality', 'format', 'completeness', 'coherence', 'semantic_similarity', 'performance']:
                scores = [e.metrics[metric] for e in successful_evaluations]
                metric_averages[metric] = sum(scores) / len(scores)
            
            # Category performance
            category_performance = {}
            for evaluation in successful_evaluations:
                category = evaluation.detailed_analysis['query_analysis']['category']
                if category not in category_performance:
                    category_performance[category] = []
                category_performance[category].append(evaluation.overall_score)
            
            for category in category_performance:
                category_performance[category] = sum(category_performance[category]) / len(category_performance[category])
            
            # Complexity performance
            complexity_performance = {}
            for evaluation in successful_evaluations:
                complexity = evaluation.detailed_analysis['query_analysis']['complexity']
                if complexity not in complexity_performance:
                    complexity_performance[complexity] = []
                complexity_performance[complexity].append(evaluation.overall_score)
            
            for complexity in complexity_performance:
                complexity_performance[complexity] = sum(complexity_performance[complexity]) / len(complexity_performance[complexity])
            
            summary = {
                "total_evaluations": total_evaluations,
                "successful_evaluations": len(successful_evaluations),
                "success_rate": (len(successful_evaluations) / total_evaluations) * 100,
                "overall_performance": {
                    "average_score": avg_overall_score,
                    "min_score": min(overall_scores),
                    "max_score": max(overall_scores)
                },
                "metric_performance": metric_averages,
                "category_performance": category_performance,
                "complexity_performance": complexity_performance,
                "quality_distribution": {
                    "excellent": len([s for s in overall_scores if s >= 0.9]),
                    "good": len([s for s in overall_scores if 0.7 <= s < 0.9]),
                    "fair": len([s for s in overall_scores if 0.5 <= s < 0.7]),
                    "poor": len([s for s in overall_scores if s < 0.5])
                },
                "llama_integration": {
                    "model_used": self.llama_evaluator.model_name if self.llama_evaluator.model else "None",
                    "semantic_evaluation": metric_averages.get('relevance', 0),
                    "quality_evaluation": metric_averages.get('quality', 0),
                    "coherence_evaluation": metric_averages.get('coherence', 0)
                }
            }
        else:
            summary = {
                "total_evaluations": total_evaluations,
                "successful_evaluations": 0,
                "success_rate": 0.0,
                "overall_performance": {
                    "average_score": 0.0,
                    "min_score": 0.0,
                    "max_score": 0.0
                }
            }
        
        return summary
    
    def save_evaluation_results(self, filename: Optional[str] = None) -> str:
        """Save evaluation results to a JSON file."""
        if not filename:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filename = f"tests/performance_test_suite/data/evaluation_results_{timestamp}.json"
        
        # Ensure directory exists
        import os
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Convert results to serializable format
        results_data = []
        for result in self.evaluation_results:
            result_dict = asdict(result)
            results_data.append(result_dict)
        
        data = {
            "metadata": {
                "generated_at": datetime.utcnow().isoformat(),
                "total_evaluations": len(self.evaluation_results),
                "evaluation_summary": self.get_evaluation_summary(),
                "llama_integration": {
                    "model_used": self.llama_evaluator.model_name,
                    "model_available": self.llama_evaluator.model is not None
                }
            },
            "evaluations": results_data
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"Evaluation results saved to: {filename}")
        return filename 


class HybridEvaluator:
    """
    Hybrid evaluator that combines objective metrics with AI evaluation.
    
    Provides:
    - Objective metrics for reliability and consistency
    - AI evaluation for semantic understanding
    - Balanced scoring methodology
    - Fallback mechanisms when AI is unavailable
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Evaluation weights
        self.weights = {
            "objective": 0.6,  # 60% weight for objective metrics
            "ai": 0.4         # 40% weight for AI evaluation
        }
        
        # Objective metric weights
        self.objective_weights = {
            "format_accuracy": 0.25,
            "completeness": 0.25,
            "performance": 0.20,
            "content_matching": 0.15,
            "semantic_similarity": 0.15
        }
        
        # AI evaluation weights
        self.ai_weights = {
            "relevance": 0.40,
            "quality": 0.35,
            "coherence": 0.25
        }
        
        # Initialize AI evaluator
        self.ai_evaluator = LlamaEvaluator(
            self.config.get("llama_model", "llama3.1:8b")
        )
        
        logger.info("HybridEvaluator initialized")
    
    def evaluate_response(self, query: str, response: str, expected_structure: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Evaluate response using hybrid methodology.
        
        Args:
            query: Original query
            response: Bot response
            expected_structure: Expected response structure
            
        Returns:
            Comprehensive evaluation results
        """
        try:
            # Calculate objective metrics
            objective_score = self._calculate_objective_metrics(query, response, expected_structure)
            
            # Calculate AI evaluation (if available)
            ai_score = self._calculate_ai_evaluation(query, response)
            
            # Combine scores using weights
            final_score = self._combine_scores(objective_score, ai_score)
            
            # Generate detailed analysis
            analysis = self._generate_detailed_analysis(
                query, response, objective_score, ai_score, final_score
            )
            
            return {
                "overall_score": final_score,
                "objective_score": objective_score,
                "ai_score": ai_score,
                "weights_used": self.weights,
                "analysis": analysis,
                "evaluation_method": "hybrid"
            }
            
        except Exception as e:
            logger.error(f"Error in hybrid evaluation: {e}")
            return self._fallback_evaluation(query, response)
    
    def _calculate_objective_metrics(self, query: str, response: str, expected_structure: Dict[str, Any] = None) -> float:
        """Calculate objective metrics score."""
        try:
            metrics = {}
            
            # Format accuracy
            metrics["format_accuracy"] = self._calculate_format_accuracy(response, expected_structure)
            
            # Completeness
            metrics["completeness"] = self._calculate_completeness(response, expected_structure)
            
            # Performance (response length appropriateness)
            metrics["performance"] = self._calculate_performance_metric(response, query)
            
            # Content matching
            metrics["content_matching"] = self._calculate_content_matching(query, response)
            
            # Semantic similarity
            metrics["semantic_similarity"] = self._calculate_semantic_similarity(query, response)
            
            # Calculate weighted score
            weighted_score = sum(
                metrics[metric] * self.objective_weights[metric]
                for metric in self.objective_weights.keys()
            )
            
            return {
                "overall": weighted_score,
                "metrics": metrics,
                "weights": self.objective_weights
            }
            
        except Exception as e:
            logger.error(f"Error calculating objective metrics: {e}")
            return {"overall": 0.5, "metrics": {}, "weights": self.objective_weights}
    
    def _calculate_ai_evaluation(self, query: str, response: str) -> Dict[str, Any]:
        """Calculate AI evaluation score."""
        try:
            if not self.ai_evaluator.model:
                # Fallback to basic metrics if AI is unavailable
                return {
                    "overall": 0.5,
                    "metrics": {
                        "relevance": 0.5,
                        "quality": 0.5,
                        "coherence": 0.5
                    },
                    "weights": self.ai_weights,
                    "ai_available": False
                }
            
            # Get AI evaluations
            relevance = self.ai_evaluator.evaluate_semantic_relevance(query, response)
            quality = self.ai_evaluator.evaluate_response_quality(response, query)
            coherence = self.ai_evaluator.evaluate_coherence(response)
            
            metrics = {
                "relevance": relevance,
                "quality": quality,
                "coherence": coherence
            }
            
            # Calculate weighted score
            weighted_score = sum(
                metrics[metric] * self.ai_weights[metric]
                for metric in self.ai_weights.keys()
            )
            
            return {
                "overall": weighted_score,
                "metrics": metrics,
                "weights": self.ai_weights,
                "ai_available": True
            }
            
        except Exception as e:
            logger.error(f"Error calculating AI evaluation: {e}")
            return {
                "overall": 0.5,
                "metrics": {"relevance": 0.5, "quality": 0.5, "coherence": 0.5},
                "weights": self.ai_weights,
                "ai_available": False
            }
    
    def _combine_scores(self, objective_score: Dict[str, Any], ai_score: Dict[str, Any]) -> float:
        """Combine objective and AI scores using weights."""
        try:
            objective_weight = self.weights["objective"]
            ai_weight = self.weights["ai"]
            
            # If AI is not available, adjust weights
            if not ai_score.get("ai_available", False):
                objective_weight = 1.0
                ai_weight = 0.0
            
            combined_score = (
                objective_score["overall"] * objective_weight +
                ai_score["overall"] * ai_weight
            )
            
            return combined_score
            
        except Exception as e:
            logger.error(f"Error combining scores: {e}")
            return 0.5
    
    def _calculate_format_accuracy(self, response: str, expected_structure: Dict[str, Any] = None) -> float:
        """Calculate format accuracy score."""
        try:
            if not expected_structure:
                return 0.5  # Neutral score if no expected structure
            
            # Check for required fields
            required_fields = expected_structure.get("required_fields", [])
            if not required_fields:
                return 0.5
            
            # Simple format checking
            score = 0.0
            for field in required_fields:
                if field.lower() in response.lower():
                    score += 1.0
            
            return score / len(required_fields) if required_fields else 0.5
            
        except Exception as e:
            logger.error(f"Error calculating format accuracy: {e}")
            return 0.5
    
    def _calculate_completeness(self, response: str, expected_structure: Dict[str, Any] = None) -> float:
        """Calculate completeness score."""
        try:
            if not expected_structure:
                return 0.5
            
            # Check expected length
            expected_length = expected_structure.get("expected_length", 100)
            actual_length = len(response)
            
            if actual_length == 0:
                return 0.0
            
            # Score based on length appropriateness
            length_ratio = min(actual_length / expected_length, 2.0)  # Cap at 2x
            if length_ratio >= 0.8 and length_ratio <= 1.2:
                return 1.0
            elif length_ratio >= 0.5 and length_ratio <= 1.5:
                return 0.8
            else:
                return 0.5
            
        except Exception as e:
            logger.error(f"Error calculating completeness: {e}")
            return 0.5
    
    def _calculate_performance_metric(self, response: str, query: str) -> float:
        """Calculate performance metric."""
        try:
            # Simple performance metric based on response appropriateness
            query_length = len(query)
            response_length = len(response)
            
            if response_length == 0:
                return 0.0
            
            # Ideal response should be proportional to query
            ideal_ratio = 3.0  # Response should be ~3x query length
            actual_ratio = response_length / query_length if query_length > 0 else 1.0
            
            if actual_ratio >= 0.5 and actual_ratio <= 5.0:
                return 1.0
            elif actual_ratio >= 0.2 and actual_ratio <= 10.0:
                return 0.7
            else:
                return 0.3
            
        except Exception as e:
            logger.error(f"Error calculating performance metric: {e}")
            return 0.5
    
    def _calculate_content_matching(self, query: str, response: str) -> float:
        """Calculate content matching score."""
        try:
            # Extract key terms from query
            query_terms = set(word.lower() for word in query.split() if len(word) > 3)
            response_terms = set(word.lower() for word in response.split() if len(word) > 3)
            
            if not query_terms:
                return 0.5
            
            # Calculate overlap
            overlap = len(query_terms.intersection(response_terms))
            match_ratio = overlap / len(query_terms)
            
            return min(match_ratio * 2, 1.0)  # Scale up but cap at 1.0
            
        except Exception as e:
            logger.error(f"Error calculating content matching: {e}")
            return 0.5
    
    def _calculate_semantic_similarity(self, query: str, response: str) -> float:
        """Calculate semantic similarity score."""
        try:
            # Simple semantic similarity using word overlap
            query_words = set(query.lower().split())
            response_words = set(response.lower().split())
            
            if not query_words:
                return 0.5
            
            # Calculate Jaccard similarity
            intersection = len(query_words.intersection(response_words))
            union = len(query_words.union(response_words))
            
            similarity = intersection / union if union > 0 else 0.0
            return similarity
            
        except Exception as e:
            logger.error(f"Error calculating semantic similarity: {e}")
            return 0.5
    
    def _generate_detailed_analysis(self, query: str, response: str, objective_score: Dict[str, Any], 
                                  ai_score: Dict[str, Any], final_score: float) -> Dict[str, Any]:
        """Generate detailed analysis of evaluation results."""
        try:
            analysis = {
                "overall_assessment": self._get_overall_assessment(final_score),
                "strengths": self._identify_strengths(objective_score, ai_score),
                "weaknesses": self._identify_weaknesses(objective_score, ai_score),
                "recommendations": self._generate_recommendations(objective_score, ai_score),
                "score_breakdown": {
                    "objective": objective_score,
                    "ai": ai_score,
                    "final": final_score
                }
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error generating detailed analysis: {e}")
            return {"overall_assessment": "neutral", "error": str(e)}
    
    def _get_overall_assessment(self, score: float) -> str:
        """Get overall assessment based on score."""
        if score >= 0.9:
            return "excellent"
        elif score >= 0.7:
            return "good"
        elif score >= 0.5:
            return "fair"
        else:
            return "poor"
    
    def _identify_strengths(self, objective_score: Dict[str, Any], ai_score: Dict[str, Any]) -> List[str]:
        """Identify strengths in the response."""
        strengths = []
        
        # Check objective metrics
        for metric, score in objective_score.get("metrics", {}).items():
            if score >= 0.8:
                strengths.append(f"Strong {metric.replace('_', ' ')}")
        
        # Check AI metrics
        if ai_score.get("ai_available", False):
            for metric, score in ai_score.get("metrics", {}).items():
                if score >= 0.8:
                    strengths.append(f"High {metric} quality")
        
        return strengths
    
    def _identify_weaknesses(self, objective_score: Dict[str, Any], ai_score: Dict[str, Any]) -> List[str]:
        """Identify weaknesses in the response."""
        weaknesses = []
        
        # Check objective metrics
        for metric, score in objective_score.get("metrics", {}).items():
            if score < 0.5:
                weaknesses.append(f"Poor {metric.replace('_', ' ')}")
        
        # Check AI metrics
        if ai_score.get("ai_available", False):
            for metric, score in ai_score.get("metrics", {}).items():
                if score < 0.5:
                    weaknesses.append(f"Low {metric} quality")
        
        return weaknesses
    
    def _generate_recommendations(self, objective_score: Dict[str, Any], ai_score: Dict[str, Any]) -> List[str]:
        """Generate improvement recommendations."""
        recommendations = []
        
        # Check for specific issues
        metrics = objective_score.get("metrics", {})
        if metrics.get("format_accuracy", 1.0) < 0.7:
            recommendations.append("Improve response format and structure")
        
        if metrics.get("completeness", 1.0) < 0.7:
            recommendations.append("Provide more comprehensive information")
        
        if metrics.get("performance", 1.0) < 0.7:
            recommendations.append("Adjust response length for better balance")
        
        return recommendations
    
    def _fallback_evaluation(self, query: str, response: str) -> Dict[str, Any]:
        """Fallback evaluation when hybrid evaluation fails."""
        return {
            "overall_score": 0.5,
            "objective_score": {"overall": 0.5, "metrics": {}},
            "ai_score": {"overall": 0.5, "metrics": {}, "ai_available": False},
            "weights_used": self.weights,
            "analysis": {
                "overall_assessment": "neutral",
                "strengths": [],
                "weaknesses": ["Evaluation failed"],
                "recommendations": ["Check system configuration"]
            },
            "evaluation_method": "fallback"
        } 