"""
Response Evaluator

Evaluates bot responses against expected dummy answers using multiple metrics.
Provides comprehensive assessment of response quality, relevance, and format accuracy.
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


class ResponseEvaluator:
    """
    Evaluates bot responses against expected dummy answers.
    
    Evaluation criteria:
    - Response relevance and completeness
    - Format accuracy and structure
    - Content quality and coherence
    - Semantic similarity
    - Response time performance
    """
    
    def __init__(self):
        self.evaluation_results = []
        
        logger.info("ResponseEvaluator initialized")
    
    def evaluate_responses(self, queries: List[Any], responses: List[Any]) -> List[EvaluationResult]:
        """
        Evaluate all bot responses against expected dummy answers.
        
        Args:
            queries: List of TestQuery objects with expected structures
            responses: List of BotResponse objects with actual responses
            
        Returns:
            List of EvaluationResult objects with comprehensive evaluation
        """
        logger.info(f"Starting evaluation of {len(responses)} responses...")
        
        self.evaluation_results = []
        
        # Create a mapping of query_id to query for easy lookup
        query_map = {query.id: query for query in queries}
        
        for response in responses:
            if response.query_id in query_map:
                query = query_map[response.query_id]
                evaluation = self._evaluate_single_response(query, response)
                self.evaluation_results.append(evaluation)
            else:
                logger.warning(f"No matching query found for response {response.query_id}")
        
        logger.info(f"Completed evaluation of {len(self.evaluation_results)} responses")
        return self.evaluation_results
    
    def _evaluate_single_response(self, query: Any, response: Any) -> EvaluationResult:
        """
        Evaluate a single response against its expected structure.
        
        Args:
            query: TestQuery object with expected structure
            response: BotResponse object with actual response
            
        Returns:
            EvaluationResult with comprehensive evaluation
        """
        if not response.success:
            return self._evaluate_failed_response(query, response)
        
        # Calculate individual metrics
        relevance_score = self._calculate_relevance_score(query, response)
        format_score = self._calculate_format_score(query, response)
        completeness_score = self._calculate_completeness_score(query, response)
        coherence_score = self._calculate_coherence_score(response)
        semantic_score = self._calculate_semantic_similarity(query, response)
        performance_score = self._calculate_performance_score(response)
        
        # Calculate overall score (weighted average)
        weights = {
            'relevance': 0.25,
            'format': 0.20,
            'completeness': 0.20,
            'coherence': 0.15,
            'semantic': 0.15,
            'performance': 0.05
        }
        
        overall_score = (
            relevance_score * weights['relevance'] +
            format_score * weights['format'] +
            completeness_score * weights['completeness'] +
            coherence_score * weights['coherence'] +
            semantic_score * weights['semantic'] +
            performance_score * weights['performance']
        )
        
        metrics = {
            'relevance': relevance_score,
            'format': format_score,
            'completeness': completeness_score,
            'coherence': coherence_score,
            'semantic_similarity': semantic_score,
            'performance': performance_score
        }
        
        detailed_analysis = self._generate_detailed_analysis(query, response, metrics)
        recommendations = self._generate_recommendations(query, response, metrics)
        
        return EvaluationResult(
            query_id=query.id,
            query=query.query,
            expected_structure=query.expected_response_structure,
            actual_response=response.response,
            overall_score=overall_score,
            metrics=metrics,
            detailed_analysis=detailed_analysis,
            recommendations=recommendations
        )
    
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
        Calculate relevance score based on query category and response content.
        
        Returns:
            Score between 0.0 and 1.0
        """
        if not response.response:
            return 0.0
        
        response_lower = response.response.lower()
        query_lower = query.query.lower()
        
        # Extract key terms from query
        key_terms = self._extract_key_terms(query.query)
        
        # Check if response addresses the query category
        category_relevance = self._check_category_relevance(query.category, response.response)
        
        # Check if key terms are mentioned
        term_coverage = sum(1 for term in key_terms if term.lower() in response_lower) / len(key_terms) if key_terms else 0
        
        # Check if response structure matches expected type
        structure_match = self._check_structure_match(query.expected_response_structure, response.response)
        
        # Weighted average
        relevance_score = (
            category_relevance * 0.4 +
            term_coverage * 0.4 +
            structure_match * 0.2
        )
        
        return min(relevance_score, 1.0)
    
    def _calculate_format_score(self, query: Any, response: Any) -> float:
        """
        Calculate format accuracy score.
        
        Returns:
            Score between 0.0 and 1.0
        """
        if not response.response:
            return 0.0
        
        expected_structure = query.expected_response_structure
        response_text = response.response
        
        format_scores = []
        
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
        Calculate response coherence and readability score.
        
        Returns:
            Score between 0.0 and 1.0
        """
        if not response.response:
            return 0.0
        
        response_text = response.response
        
        coherence_scores = []
        
        # Check sentence structure
        sentences = re.split(r'[.!?]+', response_text)
        valid_sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        if valid_sentences:
            coherence_scores.append(0.8)
        else:
            coherence_scores.append(0.3)
        
        # Check for logical connectors
        connectors = ['however', 'therefore', 'furthermore', 'additionally', 'in addition', 'also']
        connector_count = sum(1 for connector in connectors if connector in response_text.lower())
        if connector_count > 0:
            coherence_scores.append(0.9)
        else:
            coherence_scores.append(0.6)
        
        # Check for paragraph structure
        paragraphs = response_text.split('\n\n')
        if len(paragraphs) > 1:
            coherence_scores.append(0.8)
        else:
            coherence_scores.append(0.5)
        
        # Check for repetition (negative indicator)
        words = response_text.lower().split()
        unique_words = set(words)
        repetition_ratio = len(unique_words) / len(words) if words else 0
        
        if repetition_ratio > 0.7:
            coherence_scores.append(0.9)
        elif repetition_ratio > 0.5:
            coherence_scores.append(0.7)
        else:
            coherence_scores.append(0.4)
        
        return sum(coherence_scores) / len(coherence_scores)
    
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
            return self._calculate_semantic_relevance(query, response)
        
        # For specific queries, prioritize content matching
        else:
            return self._calculate_content_matching(query, response)
    
    def _calculate_semantic_relevance(self, query: Any, response: Any) -> float:
        """Calculate semantic relevance for summary/topic queries."""
        # Extract key concepts from query
        query_concepts = self._extract_key_terms(query.query)
        
        # Check if response addresses these concepts
        response_lower = response.response.lower()
        concept_coverage = 0
        
        for concept in query_concepts:
            if concept.lower() in response_lower:
                concept_coverage += 1
        
        return concept_coverage / len(query_concepts) if query_concepts else 0.5
    
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
    
    def _extract_key_terms(self, query: str) -> List[str]:
        """Extract key terms from a query."""
        # Remove common words and extract meaningful terms
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'what', 'when', 'where', 'why', 'how', 'show', 'me', 'give', 'tell', 'find', 'get', 'create', 'generate', 'analyze', 'summarize'}
        
        words = re.findall(r'\b\w+\b', query.lower())
        key_terms = [word for word in words if word not in common_words and len(word) > 2]
        
        return key_terms
    
    def _check_category_relevance(self, category: str, response: str) -> float:
        """Check if response is relevant to the query category."""
        category_keywords = {
            'server_analysis': ['channel', 'user', 'activity', 'message', 'server'],
            'feedback_summary': ['feedback', 'suggestion', 'complaint', 'improvement'],
            'trending_topics': ['trend', 'popular', 'discussion', 'topic'],
            'qa_concepts': ['question', 'answer', 'help', 'support'],
            'statistics': ['statistic', 'number', 'count', 'percentage', 'metric'],
            'structure_analysis': ['structure', 'organization', 'channel', 'flow'],
            'digests': ['summary', 'digest', 'overview', 'highlight'],
            'conversational_leadership': ['discussion', 'engagement', 'community', 'leadership']
        }
        
        if category in category_keywords:
            keywords = category_keywords[category]
            response_lower = response.lower()
            matches = sum(1 for keyword in keywords if keyword in response_lower)
            return min(matches / len(keywords), 1.0)
        
        return 0.5
    
    def _check_structure_match(self, expected_structure: Dict[str, Any], response: str) -> float:
        """Check if response structure matches expected structure."""
        response_lower = response.lower()
        
        # Check for expected structure type
        if 'type' in expected_structure:
            expected_type = expected_structure['type']
            if expected_type in response_lower:
                return 1.0
        
        # Check for expected fields
        field_matches = 0
        total_fields = 0
        
        for key in expected_structure.keys():
            if key != 'type':
                total_fields += 1
                if key in response_lower:
                    field_matches += 1
        
        return field_matches / total_fields if total_fields > 0 else 0.5
    
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
            'strengths': self._identify_strengths(metrics),
            'weaknesses': self._identify_weaknesses(metrics)
        }
    
    def _generate_recommendations(self, query: Any, response: Any, metrics: Dict[str, float]) -> List[str]:
        """Generate improvement recommendations."""
        recommendations = []
        
        if metrics['relevance'] < 0.7:
            recommendations.append("Improve response relevance to query topic")
        
        if metrics['format'] < 0.7:
            recommendations.append("Enhance response formatting and structure")
        
        if metrics['completeness'] < 0.7:
            recommendations.append("Provide more comprehensive information")
        
        if metrics['coherence'] < 0.7:
            recommendations.append("Improve response coherence and readability")
        
        if metrics['semantic_similarity'] < 0.7:
            recommendations.append("Better align response with expected content")
        
        if metrics['performance'] < 0.7:
            recommendations.append("Optimize response time and efficiency")
        
        if not recommendations:
            recommendations.append("Response meets quality standards")
        
        return recommendations
    
    def _identify_strengths(self, metrics: Dict[str, float]) -> List[str]:
        """Identify response strengths."""
        strengths = []
        
        if metrics['relevance'] > 0.8:
            strengths.append("High relevance to query")
        
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
            weaknesses.append("Low relevance to query")
        
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
            for metric in ['relevance', 'format', 'completeness', 'coherence', 'semantic_similarity', 'performance']:
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
                "evaluation_summary": self.get_evaluation_summary()
            },
            "evaluations": results_data
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"Evaluation results saved to: {filename}")
        return filename 