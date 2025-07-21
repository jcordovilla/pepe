"""
Report Generator

Generates comprehensive performance test reports with conclusions and recommendations.
Provides actionable insights for improving the agentic architecture and workflows.
Now includes Llama model integration for sophisticated analysis and architectural insights.
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class PerformanceRecommendation:
    """Represents a specific performance improvement recommendation."""
    category: str
    priority: str  # high, medium, low
    title: str
    description: str
    impact: str
    effort: str  # low, medium, high
    implementation_notes: str
    expected_improvement: str


@dataclass
class PerformanceReport:
    """Represents a comprehensive performance test report."""
    test_summary: Dict[str, Any]
    content_analysis_summary: Dict[str, Any]
    query_coverage_summary: Dict[str, Any]
    execution_summary: Dict[str, Any]
    evaluation_summary: Dict[str, Any]
    key_findings: List[str]
    recommendations: List[PerformanceRecommendation]
    architecture_insights: Dict[str, Any]
    next_steps: List[str]


class LlamaReportAnalyzer:
    """
    Llama-powered analyzer for sophisticated report generation and architectural insights.
    
    Uses Llama model to:
    - Analyze evaluation patterns and trends
    - Generate architectural recommendations
    - Identify root causes of performance issues
    - Suggest specific improvements to agentic components
    """
    
    def __init__(self, model_name: str = "llama3.1:8b"):
        self.model_name = model_name
        self.model = None
        self._initialize_model()
        
        logger.info(f"LlamaReportAnalyzer initialized with model: {model_name}")
    
    def _initialize_model(self):
        """Initialize the Llama model."""
        try:
            import ollama
            
            # Test if model is available
            try:
                ollama.show(self.model_name)
                self.model = ollama
                logger.info(f"Llama model {self.model_name} loaded successfully for report analysis")
            except Exception as e:
                logger.warning(f"Llama model {self.model_name} not available for report analysis: {e}")
                logger.info("Falling back to rule-based report generation only")
                self.model = None
                
        except ImportError:
            logger.warning("Ollama not available for report analysis. Install with: pip install ollama")
            self.model = None
    
    def analyze_evaluation_patterns(self, evaluation_results: List[Any]) -> Dict[str, Any]:
        """
        Analyze evaluation results to identify patterns and trends using Llama.
        
        Args:
            evaluation_results: List of evaluation results
            
        Returns:
            Dictionary with pattern analysis
        """
        if not self.model:
            return self._fallback_pattern_analysis(evaluation_results)
        
        try:
            # Prepare evaluation data for analysis
            analysis_data = self._prepare_evaluation_data(evaluation_results)
            
            prompt = f"""Analyze these Discord bot performance evaluation results and identify key patterns, trends, and insights.

Evaluation Data:
{json.dumps(analysis_data, indent=2)}

Please provide analysis in the following JSON format:
{{
    "performance_patterns": [
        {{
            "pattern": "description of pattern",
            "frequency": "how often it occurs",
            "impact": "positive/negative/neutral",
            "root_cause": "likely cause of pattern"
        }}
    ],
    "category_performance": {{
        "best_performing": "category name",
        "worst_performing": "category name",
        "category_insights": [
            {{
                "category": "category name",
                "strength": "main strength",
                "weakness": "main weakness",
                "improvement_area": "specific area to improve"
            }}
        ]
    }},
    "quality_trends": [
        {{
            "trend": "description of trend",
            "significance": "high/medium/low",
            "implication": "what this means for the system"
        }}
    ],
    "architectural_implications": [
        {{
            "component": "agentic component name",
            "issue": "specific issue identified",
            "recommendation": "specific improvement suggestion"
        }}
    ]
}}

Focus on identifying:
1. Performance patterns across different query types
2. Quality trends and their implications
3. Specific architectural components that need attention
4. Root causes of performance issues
5. Actionable improvement recommendations

Analysis:"""
            
            result = self.model.generate(
                model=self.model_name,
                prompt=prompt,
                options={"temperature": 0.1}
            )
            
            # Parse the JSON response
            analysis_text = result['response'].strip()
            analysis = self._extract_json_from_response(analysis_text)
            
            logger.debug(f"Llama pattern analysis completed")
            return analysis
            
        except Exception as e:
            logger.error(f"Error in Llama pattern analysis: {e}")
            return self._fallback_pattern_analysis(evaluation_results)
    
    def generate_architectural_recommendations(self, evaluation_summary: Dict[str, Any], evaluation_results: List[Any]) -> List[Dict[str, Any]]:
        """
        Generate specific architectural recommendations using Llama.
        
        Args:
            evaluation_summary: Summary of evaluation results
            evaluation_results: Detailed evaluation results
            
        Returns:
            List of architectural recommendations
        """
        if not self.model:
            return self._fallback_architectural_recommendations(evaluation_summary, evaluation_results)
        
        try:
            # Prepare data for architectural analysis
            arch_data = {
                "evaluation_summary": evaluation_summary,
                "sample_evaluations": [self._extract_architectural_data(result) for result in evaluation_results[:10]]
            }
            
            prompt = f"""Analyze this Discord bot performance data and generate specific architectural recommendations for the agentic system.

Architectural Data:
{json.dumps(arch_data, indent=2)}

Consider the following agentic architecture components:
- Query Understanding & Processing
- Content Retrieval & Vector Search
- Response Generation & Formatting
- Memory & Context Management
- Error Handling & Recovery
- Performance Optimization
- Database & Storage Systems

Generate recommendations in this JSON format:
{{
    "architectural_recommendations": [
        {{
            "component": "specific component name",
            "issue": "detailed description of the issue",
            "recommendation": "specific improvement suggestion",
            "priority": "high/medium/low",
            "effort": "low/medium/high",
            "expected_impact": "description of expected improvement",
            "implementation_notes": "specific implementation guidance"
        }}
    ],
    "system_optimization": [
        {{
            "area": "system area to optimize",
            "current_state": "description of current state",
            "optimization": "specific optimization strategy",
            "benefit": "expected benefit"
        }}
    ],
    "component_interactions": [
        {{
            "interaction": "description of component interaction",
            "issue": "specific interaction issue",
            "solution": "proposed solution"
        }}
    ]
}}

Focus on:
1. Specific agentic components that need improvement
2. System-level optimizations
3. Component interaction issues
4. Practical implementation guidance
5. Expected performance improvements

Recommendations:"""
            
            result = self.model.generate(
                model=self.model_name,
                prompt=prompt,
                options={"temperature": 0.1}
            )
            
            recommendations_text = result['response'].strip()
            recommendations = self._extract_json_from_response(recommendations_text)
            
            logger.debug(f"Llama architectural recommendations generated")
            return recommendations
            
        except Exception as e:
            logger.error(f"Error in Llama architectural recommendations: {e}")
            return self._fallback_architectural_recommendations(evaluation_summary, evaluation_results)
    
    def analyze_root_causes(self, evaluation_results: List[Any], execution_summary: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze root causes of performance issues using Llama.
        
        Args:
            evaluation_results: List of evaluation results
            execution_summary: Summary of execution results
            
        Returns:
            Dictionary with root cause analysis
        """
        if not self.model:
            return self._fallback_root_cause_analysis(evaluation_results, execution_summary)
        
        try:
            # Prepare data for root cause analysis
            rca_data = {
                "execution_summary": execution_summary,
                "failed_evaluations": [result for result in evaluation_results if result.overall_score < 0.5],
                "performance_issues": [result for result in evaluation_results if result.metrics.get('performance', 1.0) < 0.7]
            }
            
            prompt = f"""Analyze these Discord bot performance issues and identify root causes.

Root Cause Analysis Data:
{json.dumps(rca_data, indent=2)}

Provide analysis in this JSON format:
{{
    "root_causes": [
        {{
            "issue": "description of the issue",
            "symptoms": ["list of symptoms"],
            "root_cause": "identified root cause",
            "component": "affected component",
            "severity": "high/medium/low",
            "fix_strategy": "specific fix strategy"
        }}
    ],
    "system_bottlenecks": [
        {{
            "bottleneck": "description of bottleneck",
            "impact": "performance impact",
            "location": "where in the system",
            "solution": "proposed solution"
        }}
    ],
    "failure_patterns": [
        {{
            "pattern": "description of failure pattern",
            "frequency": "how often it occurs",
            "trigger": "what triggers it",
            "prevention": "how to prevent it"
        }}
    ]
}}

Focus on:
1. Identifying true root causes vs. symptoms
2. System-level bottlenecks
3. Failure patterns and their triggers
4. Specific fix strategies
5. Prevention measures

Root Cause Analysis:"""
            
            result = self.model.generate(
                model=self.model_name,
                prompt=prompt,
                options={"temperature": 0.1}
            )
            
            rca_text = result['response'].strip()
            rca_analysis = self._extract_json_from_response(rca_text)
            
            logger.debug(f"Llama root cause analysis completed")
            return rca_analysis
            
        except Exception as e:
            logger.error(f"Error in Llama root cause analysis: {e}")
            return self._fallback_root_cause_analysis(evaluation_results, execution_summary)
    
    def generate_improvement_strategy(self, evaluation_summary: Dict[str, Any], recommendations: List[Any]) -> Dict[str, Any]:
        """
        Generate comprehensive improvement strategy using Llama.
        
        Args:
            evaluation_summary: Summary of evaluation results
            recommendations: List of recommendations
            
        Returns:
            Dictionary with improvement strategy
        """
        if not self.model:
            return self._fallback_improvement_strategy(evaluation_summary, recommendations)
        
        try:
            strategy_data = {
                "evaluation_summary": evaluation_summary,
                "recommendations": [asdict(rec) for rec in recommendations]
            }
            
            prompt = f"""Based on these Discord bot performance evaluation results and recommendations, generate a comprehensive improvement strategy.

Strategy Data:
{json.dumps(strategy_data, indent=2)}

Generate strategy in this JSON format:
{{
    "immediate_actions": [
        {{
            "action": "specific action to take",
            "priority": "high/medium/low",
            "timeline": "when to implement",
            "resources": "resources needed",
            "success_metrics": "how to measure success"
        }}
    ],
    "short_term_goals": [
        {{
            "goal": "specific goal",
            "timeline": "timeframe",
            "dependencies": ["list of dependencies"],
            "milestones": ["list of milestones"]
        }}
    ],
    "long_term_vision": [
        {{
            "vision": "long-term vision",
            "components": ["components involved"],
            "benefits": ["expected benefits"],
            "risks": ["potential risks"]
        }}
    ],
    "implementation_phases": [
        {{
            "phase": "phase name",
            "duration": "estimated duration",
            "focus": "what to focus on",
            "deliverables": ["list of deliverables"]
        }}
    ]
}}

Focus on:
1. Prioritized action plan
2. Realistic timelines
3. Resource requirements
4. Success metrics
5. Risk mitigation

Improvement Strategy:"""
            
            result = self.model.generate(
                model=self.model_name,
                prompt=prompt,
                options={"temperature": 0.1}
            )
            
            strategy_text = result['response'].strip()
            strategy = self._extract_json_from_response(strategy_text)
            
            logger.debug(f"Llama improvement strategy generated")
            return strategy
            
        except Exception as e:
            logger.error(f"Error in Llama improvement strategy: {e}")
            return self._fallback_improvement_strategy(evaluation_summary, recommendations)
    
    def _extract_json_from_response(self, text: str) -> Dict[str, Any]:
        """Extract JSON from Llama response text."""
        try:
            # Look for JSON content between curly braces
            import re
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                return json.loads(json_str)
            else:
                logger.warning("No JSON found in Llama response")
                return {}
        except Exception as e:
            logger.error(f"Error parsing JSON from Llama response: {e}")
            return {}
    
    def _prepare_evaluation_data(self, evaluation_results: List[Any]) -> Dict[str, Any]:
        """Prepare evaluation data for Llama analysis."""
        return {
            "total_evaluations": len(evaluation_results),
            "score_distribution": {
                "excellent": len([r for r in evaluation_results if r.overall_score >= 0.9]),
                "good": len([r for r in evaluation_results if 0.7 <= r.overall_score < 0.9]),
                "fair": len([r for r in evaluation_results if 0.5 <= r.overall_score < 0.7]),
                "poor": len([r for r in evaluation_results if r.overall_score < 0.5])
            },
            "category_performance": self._aggregate_category_performance(evaluation_results),
            "metric_averages": self._calculate_metric_averages(evaluation_results),
            "sample_evaluations": [
                {
                    "query_id": r.query_id,
                    "category": r.detailed_analysis.get('query_analysis', {}).get('category', 'unknown'),
                    "overall_score": r.overall_score,
                    "metrics": r.metrics,
                    "strengths": r.detailed_analysis.get('strengths', []),
                    "weaknesses": r.detailed_analysis.get('weaknesses', [])
                }
                for r in evaluation_results[:5]  # Sample first 5
            ]
        }
    
    def _extract_architectural_data(self, evaluation_result: Any) -> Dict[str, Any]:
        """Extract architectural-relevant data from evaluation result."""
        return {
            "query_category": evaluation_result.detailed_analysis.get('query_analysis', {}).get('category', 'unknown'),
            "performance_metrics": evaluation_result.metrics,
            "response_time": evaluation_result.detailed_analysis.get('response_analysis', {}).get('response_time', 0),
            "success": evaluation_result.detailed_analysis.get('response_analysis', {}).get('success', True),
            "strengths": evaluation_result.detailed_analysis.get('strengths', []),
            "weaknesses": evaluation_result.detailed_analysis.get('weaknesses', [])
        }
    
    def _aggregate_category_performance(self, evaluation_results: List[Any]) -> Dict[str, Any]:
        """Aggregate performance by category."""
        category_scores = {}
        for result in evaluation_results:
            category = result.detailed_analysis.get('query_analysis', {}).get('category', 'unknown')
            if category not in category_scores:
                category_scores[category] = []
            category_scores[category].append(result.overall_score)
        
        return {
            category: {
                "average_score": sum(scores) / len(scores),
                "count": len(scores),
                "min_score": min(scores),
                "max_score": max(scores)
            }
            for category, scores in category_scores.items()
        }
    
    def _calculate_metric_averages(self, evaluation_results: List[Any]) -> Dict[str, float]:
        """Calculate average scores for each metric."""
        metric_sums = {}
        metric_counts = {}
        
        for result in evaluation_results:
            for metric, score in result.metrics.items():
                if metric not in metric_sums:
                    metric_sums[metric] = 0
                    metric_counts[metric] = 0
                metric_sums[metric] += score
                metric_counts[metric] += 1
        
        return {
            metric: metric_sums[metric] / metric_counts[metric]
            for metric in metric_sums.keys()
        }
    
    def _fallback_pattern_analysis(self, evaluation_results: List[Any]) -> Dict[str, Any]:
        """Fallback pattern analysis using rule-based approach."""
        return {
            "performance_patterns": [
                {
                    "pattern": "Score distribution analysis",
                    "frequency": "All evaluations",
                    "impact": "neutral",
                    "root_cause": "Overall system performance"
                }
            ],
            "category_performance": {
                "best_performing": "unknown",
                "worst_performing": "unknown",
                "category_insights": []
            },
            "quality_trends": [],
            "architectural_implications": []
        }
    
    def _fallback_architectural_recommendations(self, evaluation_summary: Dict[str, Any], evaluation_results: List[Any]) -> List[Dict[str, Any]]:
        """Fallback architectural recommendations using rule-based approach."""
        return {
            "architectural_recommendations": [
                {
                    "component": "General System",
                    "issue": "Performance varies across query types",
                    "recommendation": "Implement targeted optimizations",
                    "priority": "medium",
                    "effort": "medium",
                    "expected_impact": "Improved overall performance",
                    "implementation_notes": "Focus on weakest performing areas"
                }
            ],
            "system_optimization": [],
            "component_interactions": []
        }
    
    def _fallback_root_cause_analysis(self, evaluation_results: List[Any], execution_summary: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback root cause analysis using rule-based approach."""
        return {
            "root_causes": [
                {
                    "issue": "Performance variation",
                    "symptoms": ["Mixed evaluation scores"],
                    "root_cause": "System optimization needed",
                    "component": "General",
                    "severity": "medium",
                    "fix_strategy": "Targeted improvements"
                }
            ],
            "system_bottlenecks": [],
            "failure_patterns": []
        }
    
    def _fallback_improvement_strategy(self, evaluation_summary: Dict[str, Any], recommendations: List[Any]) -> Dict[str, Any]:
        """Fallback improvement strategy using rule-based approach."""
        return {
            "immediate_actions": [
                {
                    "action": "Review evaluation results",
                    "priority": "high",
                    "timeline": "Immediate",
                    "resources": "Development team",
                    "success_metrics": "Improved scores"
                }
            ],
            "short_term_goals": [],
            "long_term_vision": [],
            "implementation_phases": []
        }


class ReportGenerator:
    """
    Generates comprehensive performance test reports.
    
    Features:
    - Detailed analysis of test results
    - Actionable recommendations
    - Architecture insights
    - Performance improvement strategies
    - Llama-powered sophisticated analysis
    """
    
    def __init__(self):
        self.report = None
        self.llama_analyzer = LlamaReportAnalyzer()
        
        logger.info("ReportGenerator initialized with Llama integration")
    
    def generate_comprehensive_report(
        self,
        content_analysis: Dict[str, Any],
        query_summary: Dict[str, Any],
        execution_summary: Dict[str, Any],
        evaluation_summary: Dict[str, Any],
        evaluation_results: List[Any]
    ) -> PerformanceReport:
        """
        Generate a comprehensive performance test report with Llama-powered analysis.
        
        Args:
            content_analysis: Results from content analysis
            query_summary: Summary of generated queries
            execution_summary: Summary of bot execution
            evaluation_summary: Summary of response evaluation
            evaluation_results: Detailed evaluation results
            
        Returns:
            Comprehensive PerformanceReport object
        """
        logger.info("Generating comprehensive performance test report with Llama analysis...")
        
        # Generate test summary
        test_summary = self._generate_test_summary(
            content_analysis, query_summary, execution_summary, evaluation_summary
        )
        
        # Generate key findings with Llama analysis
        key_findings = self._generate_key_findings_with_llama(
            content_analysis, execution_summary, evaluation_summary, evaluation_results
        )
        
        # Generate recommendations with Llama insights
        recommendations = self._generate_recommendations_with_llama(
            evaluation_summary, evaluation_results, execution_summary
        )
        
        # Generate architecture insights with Llama analysis
        architecture_insights = self._generate_architecture_insights_with_llama(
            content_analysis, evaluation_summary, evaluation_results
        )
        
        # Generate next steps with Llama strategy
        next_steps = self._generate_next_steps_with_llama(
            recommendations, architecture_insights, evaluation_summary
        )
        
        # Create comprehensive report
        self.report = PerformanceReport(
            test_summary=test_summary,
            content_analysis_summary=content_analysis,
            query_coverage_summary=query_summary,
            execution_summary=execution_summary,
            evaluation_summary=evaluation_summary,
            key_findings=key_findings,
            recommendations=recommendations,
            architecture_insights=architecture_insights,
            next_steps=next_steps
        )
        
        logger.info("Comprehensive performance test report generated with Llama analysis")
        return self.report
    
    def _generate_test_summary(
        self,
        content_analysis: Dict[str, Any],
        query_summary: Dict[str, Any],
        execution_summary: Dict[str, Any],
        evaluation_summary: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate overall test summary."""
        return {
            "test_execution": {
                "timestamp": datetime.utcnow().isoformat(),
                "duration": "Comprehensive multi-phase test",
                "phases": ["Content Analysis", "Query Generation", "Bot Execution", "Response Evaluation"]
            },
            "scope": {
                "messages_analyzed": content_analysis.get("server_overview", {}).get("total_messages", 0),
                "queries_generated": query_summary.get("total_queries", 0),
                "queries_executed": execution_summary.get("total_queries", 0),
                "responses_evaluated": evaluation_summary.get("total_evaluations", 0)
            },
            "overall_performance": {
                "success_rate": execution_summary.get("success_rate", 0),
                "average_quality_score": evaluation_summary.get("overall_performance", {}).get("average_score", 0),
                "system_reliability": self._calculate_reliability_score(execution_summary)
            }
        }
    
    def _generate_key_findings(
        self,
        content_analysis: Dict[str, Any],
        execution_summary: Dict[str, Any],
        evaluation_summary: Dict[str, Any],
        evaluation_results: List[Any]
    ) -> List[str]:
        """Generate key findings from the test results."""
        findings = []
        
        # Performance findings
        success_rate = execution_summary.get("success_rate", 0)
        if success_rate < 80:
            findings.append(f"System reliability needs improvement: {success_rate:.1f}% success rate")
        elif success_rate >= 95:
            findings.append(f"Excellent system reliability: {success_rate:.1f}% success rate")
        else:
            findings.append(f"Good system reliability: {success_rate:.1f}% success rate")
        
        # Quality findings
        avg_score = evaluation_summary.get("overall_performance", {}).get("average_score", 0)
        if avg_score < 0.6:
            findings.append(f"Response quality needs significant improvement: {avg_score:.2f} average score")
        elif avg_score < 0.8:
            findings.append(f"Response quality has room for improvement: {avg_score:.2f} average score")
        else:
            findings.append(f"Good response quality: {avg_score:.2f} average score")
        
        # Category performance findings
        category_performance = evaluation_summary.get("category_performance", {})
        if category_performance:
            best_category = max(category_performance.items(), key=lambda x: x[1])
            worst_category = min(category_performance.items(), key=lambda x: x[1])
            findings.append(f"Best performing category: {best_category[0]} ({best_category[1]:.2f} score)")
            findings.append(f"Category needing improvement: {worst_category[0]} ({worst_category[1]:.2f} score)")
        
        # Complexity findings
        complexity_performance = evaluation_summary.get("complexity_performance", {})
        if complexity_performance:
            for complexity, score in complexity_performance.items():
                if score < 0.6:
                    findings.append(f"{complexity.capitalize()} complexity queries need improvement: {score:.2f} score")
        
        # Error analysis findings
        error_summary = execution_summary.get("error_summary", {})
        if error_summary.get("total_errors", 0) > 0:
            common_errors = error_summary.get("common_errors", {})
            if common_errors:
                most_common_error = max(common_errors.items(), key=lambda x: x[1])
                findings.append(f"Most common error: {most_common_error[0]} ({most_common_error[1]} occurrences)")
        
        # Content insights
        content_insights = content_analysis.get("key_insights", {})
        if content_insights:
            if content_insights.get("top_topics"):
                findings.append(f"Server focuses on: {', '.join(content_insights['top_topics'][:3])}")
        
        return findings
    
    def _generate_recommendations(
        self,
        evaluation_summary: Dict[str, Any],
        evaluation_results: List[Any],
        execution_summary: Dict[str, Any]
    ) -> List[PerformanceRecommendation]:
        """Generate actionable performance improvement recommendations."""
        recommendations = []
        
        # System reliability recommendations
        success_rate = execution_summary.get("success_rate", 0)
        if success_rate < 90:
            recommendations.append(PerformanceRecommendation(
                category="System Reliability",
                priority="high",
                title="Improve Error Handling and Recovery",
                description="Implement robust error handling mechanisms to increase success rate",
                impact="High - Will improve overall system reliability",
                effort="medium",
                implementation_notes="Add retry logic, better error classification, and fallback mechanisms",
                expected_improvement="Increase success rate to 95%+"
            ))
        
        # Response quality recommendations
        metric_performance = evaluation_summary.get("metric_performance", {})
        if metric_performance:
            # Relevance improvements
            if metric_performance.get("relevance", 1.0) < 0.8:
                recommendations.append(PerformanceRecommendation(
                    category="Response Quality",
                    priority="high",
                    title="Enhance Query Understanding and Relevance",
                    description="Improve query interpretation and response relevance",
                    impact="High - Will significantly improve user satisfaction",
                    effort="high",
                    implementation_notes="Refine system prompts, improve query classification, enhance context understanding",
                    expected_improvement="Increase relevance score to 0.85+"
                ))
            
            # Format improvements
            if metric_performance.get("format", 1.0) < 0.8:
                recommendations.append(PerformanceRecommendation(
                    category="Response Quality",
                    priority="medium",
                    title="Standardize Response Formatting",
                    description="Implement consistent response formatting across all query types",
                    impact="Medium - Will improve readability and user experience",
                    effort="medium",
                    implementation_notes="Create response templates, implement format validation, add structure guidelines",
                    expected_improvement="Increase format score to 0.85+"
                ))
            
            # Completeness improvements
            if metric_performance.get("completeness", 1.0) < 0.8:
                recommendations.append(PerformanceRecommendation(
                    category="Response Quality",
                    priority="medium",
                    title="Enhance Response Completeness",
                    description="Ensure responses provide comprehensive information",
                    impact="Medium - Will reduce follow-up questions",
                    effort="medium",
                    implementation_notes="Implement completeness checks, add missing information detection",
                    expected_improvement="Increase completeness score to 0.85+"
                ))
        
        # Performance recommendations
        performance_metrics = evaluation_summary.get("metric_performance", {})
        if performance_metrics.get("performance", 1.0) < 0.8:
            recommendations.append(PerformanceRecommendation(
                category="Performance",
                priority="medium",
                title="Optimize Response Time",
                description="Reduce average response time for better user experience",
                impact="Medium - Will improve user satisfaction",
                effort="medium",
                implementation_notes="Implement caching, optimize database queries, parallel processing",
                expected_improvement="Reduce average response time to <5 seconds"
            ))
        
        # Category-specific recommendations
        category_performance = evaluation_summary.get("category_performance", {})
        for category, score in category_performance.items():
            if score < 0.7:
                recommendations.append(PerformanceRecommendation(
                    category=f"Category: {category}",
                    priority="medium",
                    title=f"Improve {category.replace('_', ' ').title()} Performance",
                    description=f"Enhance bot performance for {category} queries",
                    impact="Medium - Will improve specific use case performance",
                    effort="medium",
                    implementation_notes=f"Analyze {category} query patterns, optimize specific workflows",
                    expected_improvement=f"Increase {category} score to 0.75+"
                ))
        
        # Architecture recommendations
        recommendations.append(PerformanceRecommendation(
            category="Architecture",
            priority="high",
            title="Implement Advanced Error Recovery",
            description="Add sophisticated error recovery mechanisms to the agentic system",
            impact="High - Will improve system robustness",
            effort="high",
            implementation_notes="Implement retry strategies, alternative approaches, graceful degradation",
            expected_improvement="Reduce system failures by 80%"
        ))
        
        recommendations.append(PerformanceRecommendation(
            category="Architecture",
            priority="medium",
            title="Optimize Resource Management",
            description="Improve resource allocation and usage efficiency",
            impact="Medium - Will reduce costs and improve performance",
            effort="medium",
            implementation_notes="Implement connection pooling, caching strategies, resource monitoring",
            expected_improvement="Reduce resource usage by 30%"
        ))
        
        return recommendations
    
    def _generate_architecture_insights(
        self,
        content_analysis: Dict[str, Any],
        evaluation_summary: Dict[str, Any],
        evaluation_results: List[Any]
    ) -> Dict[str, Any]:
        """Generate insights about the agentic architecture."""
        insights = {
            "current_state": {
                "overall_health": self._assess_architecture_health(evaluation_summary),
                "strengths": self._identify_architecture_strengths(evaluation_summary),
                "weaknesses": self._identify_architecture_weaknesses(evaluation_summary)
            },
            "performance_patterns": {
                "query_type_performance": evaluation_summary.get("category_performance", {}),
                "complexity_performance": evaluation_summary.get("complexity_performance", {}),
                "quality_distribution": evaluation_summary.get("quality_distribution", {})
            },
            "scalability_assessment": {
                "current_capacity": self._assess_current_capacity(content_analysis),
                "bottlenecks": self._identify_bottlenecks(evaluation_results),
                "scalability_concerns": self._identify_scalability_concerns(evaluation_summary)
            },
            "improvement_opportunities": {
                "high_impact": self._identify_high_impact_improvements(evaluation_summary),
                "low_effort": self._identify_low_effort_improvements(evaluation_summary),
                "architectural": self._identify_architectural_improvements(evaluation_summary)
            }
        }
        
        return insights
    
    def _generate_next_steps(
        self,
        recommendations: List[PerformanceRecommendation],
        architecture_insights: Dict[str, Any]
    ) -> List[str]:
        """Generate actionable next steps."""
        next_steps = []
        
        # Prioritize recommendations
        high_priority = [r for r in recommendations if r.priority == "high"]
        medium_priority = [r for r in recommendations if r.priority == "medium"]
        
        next_steps.append("Immediate Actions (High Priority):")
        for rec in high_priority[:3]:  # Top 3 high priority
            next_steps.append(f"  - {rec.title}")
        
        next_steps.append("Short-term Improvements (Medium Priority):")
        for rec in medium_priority[:3]:  # Top 3 medium priority
            next_steps.append(f"  - {rec.title}")
        
        next_steps.append("Architecture Enhancements:")
        insights = architecture_insights.get("improvement_opportunities", {})
        if insights.get("high_impact"):
            next_steps.append(f"  - Focus on: {insights['high_impact'][0] if insights['high_impact'] else 'System reliability'}")
        
        next_steps.append("Monitoring and Validation:")
        next_steps.append("  - Implement continuous performance monitoring")
        next_steps.append("  - Set up automated quality checks")
        next_steps.append("  - Schedule regular performance reviews")
        
        return next_steps
    
    def _calculate_reliability_score(self, execution_summary: Dict[str, Any]) -> float:
        """Calculate system reliability score."""
        success_rate = execution_summary.get("success_rate", 0)
        reliability = success_rate / 100
        error_summary = execution_summary.get("error_summary", {})
        total_errors = error_summary.get("total_errors", 0)
        total_queries = execution_summary.get("total_queries", 1)
        if total_queries == 0:
            error_rate = 1.0
            reliability = 0.0
        else:
            error_rate = total_errors / total_queries
            reliability = reliability * (1 - error_rate)
        return min(reliability, 1.0)
    
    def _assess_architecture_health(self, evaluation_summary: Dict[str, Any]) -> str:
        """Assess overall architecture health."""
        avg_score = evaluation_summary.get("overall_performance", {}).get("average_score", 0)
        success_rate = evaluation_summary.get("success_rate", 0)
        
        if avg_score >= 0.8 and success_rate >= 95:
            return "Excellent"
        elif avg_score >= 0.7 and success_rate >= 90:
            return "Good"
        elif avg_score >= 0.6 and success_rate >= 80:
            return "Fair"
        else:
            return "Needs Improvement"
    
    def _identify_architecture_strengths(self, evaluation_summary: Dict[str, Any]) -> List[str]:
        """Identify architecture strengths."""
        strengths = []
        
        metric_performance = evaluation_summary.get("metric_performance", {})
        if metric_performance.get("coherence", 0) > 0.8:
            strengths.append("Good response coherence and readability")
        
        if metric_performance.get("format", 0) > 0.8:
            strengths.append("Consistent response formatting")
        
        success_rate = evaluation_summary.get("success_rate", 0)
        if success_rate > 90:
            strengths.append("High system reliability")
        
        return strengths
    
    def _identify_architecture_weaknesses(self, evaluation_summary: Dict[str, Any]) -> List[str]:
        """Identify architecture weaknesses."""
        weaknesses = []
        
        metric_performance = evaluation_summary.get("metric_performance", {})
        if metric_performance.get("relevance", 1.0) < 0.7:
            weaknesses.append("Low response relevance to queries")
        
        if metric_performance.get("completeness", 1.0) < 0.7:
            weaknesses.append("Incomplete responses")
        
        if metric_performance.get("performance", 1.0) < 0.7:
            weaknesses.append("Slow response times")
        
        success_rate = evaluation_summary.get("success_rate", 0)
        if success_rate < 85:
            weaknesses.append("System reliability issues")
        
        return weaknesses
    
    def _assess_current_capacity(self, content_analysis: Dict[str, Any]) -> str:
        """Assess current system capacity."""
        total_messages = content_analysis.get("server_overview", {}).get("total_messages", 0)
        
        if total_messages > 50000:
            return "High capacity needed"
        elif total_messages > 10000:
            return "Medium capacity"
        else:
            return "Low capacity"
    
    def _identify_bottlenecks(self, evaluation_results: List[Any]) -> List[str]:
        """Identify potential bottlenecks."""
        bottlenecks = []
        
        # Analyze response times
        response_times = [r.response_time for r in evaluation_results if hasattr(r, 'response_time')]
        if response_times:
            avg_time = sum(response_times) / len(response_times)
            if avg_time > 10:
                bottlenecks.append("Slow response times")
        
        # Analyze error patterns
        errors = [r for r in evaluation_results if not r.success]
        if len(errors) > len(evaluation_results) * 0.1:  # More than 10% errors
            bottlenecks.append("High error rate")
        
        return bottlenecks
    
    def _identify_scalability_concerns(self, evaluation_summary: Dict[str, Any]) -> List[str]:
        """Identify scalability concerns."""
        concerns = []
        
        performance_metrics = evaluation_summary.get("metric_performance", {})
        if performance_metrics.get("performance", 1.0) < 0.7:
            concerns.append("Performance degradation under load")
        
        success_rate = evaluation_summary.get("success_rate", 0)
        if success_rate < 90:
            concerns.append("Reliability issues at scale")
        
        return concerns
    
    def _identify_high_impact_improvements(self, evaluation_summary: Dict[str, Any]) -> List[str]:
        """Identify high-impact improvement opportunities."""
        improvements = []
        
        metric_performance = evaluation_summary.get("metric_performance", {})
        if metric_performance.get("relevance", 1.0) < 0.7:
            improvements.append("Query understanding and relevance")
        
        success_rate = evaluation_summary.get("success_rate", 0)
        if success_rate < 90:
            improvements.append("System reliability and error handling")
        
        return improvements
    
    def _identify_low_effort_improvements(self, evaluation_summary: Dict[str, Any]) -> List[str]:
        """Identify low-effort improvement opportunities."""
        improvements = []
        
        metric_performance = evaluation_summary.get("metric_performance", {})
        if metric_performance.get("format", 1.0) < 0.8:
            improvements.append("Response formatting standardization")
        
        if metric_performance.get("coherence", 1.0) < 0.8:
            improvements.append("Response coherence improvements")
        
        return improvements
    
    def _identify_architectural_improvements(self, evaluation_summary: Dict[str, Any]) -> List[str]:
        """Identify architectural improvement opportunities."""
        improvements = []
        
        improvements.append("Enhanced error recovery mechanisms")
        improvements.append("Optimized resource management")
        improvements.append("Improved query processing pipeline")
        improvements.append("Better state management")
        
        return improvements
    
    def _generate_key_findings_with_llama(
        self,
        content_analysis: Dict[str, Any],
        execution_summary: Dict[str, Any],
        evaluation_summary: Dict[str, Any],
        evaluation_results: List[Any]
    ) -> List[str]:
        """Generate key findings with Llama-powered pattern analysis."""
        # Get Llama pattern analysis
        pattern_analysis = self.llama_analyzer.analyze_evaluation_patterns(evaluation_results)
        
        # Combine with rule-based findings
        rule_based_findings = self._generate_key_findings(
            content_analysis, execution_summary, evaluation_summary, evaluation_results
        )
        
        # Add Llama insights
        llama_findings = []
        
        if pattern_analysis.get('performance_patterns'):
            for pattern in pattern_analysis['performance_patterns']:
                llama_findings.append(
                    f"Pattern identified: {pattern['pattern']} (Impact: {pattern['impact']}, Root cause: {pattern['root_cause']})"
                )
        
        if pattern_analysis.get('quality_trends'):
            for trend in pattern_analysis['quality_trends']:
                llama_findings.append(
                    f"Quality trend: {trend['trend']} (Significance: {trend['significance']}, Implication: {trend['implication']})"
                )
        
        if pattern_analysis.get('architectural_implications'):
            for implication in pattern_analysis['architectural_implications']:
                llama_findings.append(
                    f"Architectural implication: {implication['component']} - {implication['issue']} (Recommendation: {implication['recommendation']})"
                )
        
        return rule_based_findings + llama_findings
    
    def _generate_recommendations_with_llama(
        self,
        evaluation_summary: Dict[str, Any],
        evaluation_results: List[Any],
        execution_summary: Dict[str, Any]
    ) -> List[PerformanceRecommendation]:
        """Generate recommendations with Llama-powered architectural analysis."""
        # Get Llama architectural recommendations
        llama_recommendations = self.llama_analyzer.generate_architectural_recommendations(
            evaluation_summary, evaluation_results
        )
        
        # Get rule-based recommendations
        rule_based_recommendations = self._generate_recommendations(
            evaluation_summary, evaluation_results, execution_summary
        )
        
        # Convert Llama recommendations to PerformanceRecommendation objects
        llama_perf_recommendations = []
        
        if llama_recommendations.get('architectural_recommendations'):
            for rec in llama_recommendations['architectural_recommendations']:
                llama_perf_recommendations.append(
                    PerformanceRecommendation(
                        category=rec.get('component', 'Architecture'),
                        priority=rec.get('priority', 'medium'),
                        title=rec.get('issue', 'Architectural improvement'),
                        description=rec.get('recommendation', ''),
                        impact=rec.get('expected_impact', 'Improved performance'),
                        effort=rec.get('effort', 'medium'),
                        implementation_notes=rec.get('implementation_notes', ''),
                        expected_improvement=rec.get('expected_impact', 'Better performance')
                    )
                )
        
        return rule_based_recommendations + llama_perf_recommendations
    
    def _generate_architecture_insights_with_llama(
        self,
        content_analysis: Dict[str, Any],
        evaluation_summary: Dict[str, Any],
        evaluation_results: List[Any]
    ) -> Dict[str, Any]:
        """Generate architecture insights with Llama-powered analysis."""
        # Get rule-based insights
        rule_based_insights = self._generate_architecture_insights(
            content_analysis, evaluation_summary, evaluation_results
        )
        
        # Get Llama root cause analysis
        root_cause_analysis = self.llama_analyzer.analyze_root_causes(
            evaluation_results, {"evaluation_summary": evaluation_summary}
        )
        
        # Get Llama improvement strategy
        improvement_strategy = self.llama_analyzer.generate_improvement_strategy(
            evaluation_summary, []
        )
        
        # Combine insights
        enhanced_insights = rule_based_insights.copy()
        enhanced_insights.update({
            'llama_analysis': {
                'pattern_analysis': self.llama_analyzer.analyze_evaluation_patterns(evaluation_results),
                'root_cause_analysis': root_cause_analysis,
                'improvement_strategy': improvement_strategy,
                'model_used': self.llama_analyzer.model_name if self.llama_analyzer.model else 'None'
            }
        })
        
        return enhanced_insights
    
    def _generate_next_steps_with_llama(
        self,
        recommendations: List[PerformanceRecommendation],
        architecture_insights: Dict[str, Any],
        evaluation_summary: Dict[str, Any]
    ) -> List[str]:
        """Generate next steps with Llama-powered strategy."""
        # Get rule-based next steps
        rule_based_steps = self._generate_next_steps(recommendations, architecture_insights)
        
        # Get Llama improvement strategy
        improvement_strategy = self.llama_analyzer.generate_improvement_strategy(
            evaluation_summary, recommendations
        )
        
        # Add Llama strategy steps
        llama_steps = []
        
        if improvement_strategy.get('immediate_actions'):
            for action in improvement_strategy['immediate_actions']:
                llama_steps.append(
                    f"Immediate: {action['action']} (Priority: {action['priority']}, Timeline: {action['timeline']})"
                )
        
        if improvement_strategy.get('short_term_goals'):
            for goal in improvement_strategy['short_term_goals']:
                llama_steps.append(
                    f"Short-term: {goal['goal']} (Timeline: {goal['timeline']})"
                )
        
        if improvement_strategy.get('implementation_phases'):
            for phase in improvement_strategy['implementation_phases']:
                llama_steps.append(
                    f"Phase: {phase['phase']} - {phase['focus']} (Duration: {phase['duration']})"
                )
        
        return rule_based_steps + llama_steps
    
    def save_report(self, filename: Optional[str] = None) -> str:
        """Save the comprehensive report to a JSON file."""
        if not self.report:
            raise ValueError("No report generated. Call generate_comprehensive_report first.")
        
        if not filename:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filename = f"tests/performance_test_suite/data/comprehensive_report_{timestamp}.json"
        
        # Ensure directory exists
        import os
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Convert report to serializable format
        report_dict = asdict(self.report)
        
        with open(filename, 'w') as f:
            json.dump(report_dict, f, indent=2, default=str)
        
        logger.info(f"Comprehensive report saved to: {filename}")
        return filename
    
    def generate_executive_summary(self) -> str:
        """Generate an executive summary of the report."""
        if not self.report:
            return "No report available."
        
        summary = f"""
# Performance Test Report - Executive Summary

## Test Overview
- **Total Queries Tested**: {self.report.test_summary['scope']['queries_executed']}
- **Success Rate**: {self.report.execution_summary.get('success_rate', 0):.1f}%
- **Average Quality Score**: {self.report.evaluation_summary.get('overall_performance', {}).get('average_score', 0):.2f}
- **System Reliability**: {self.report.test_summary['overall_performance']['system_reliability']:.2f}

## Key Findings
{chr(10).join(f"- {finding}" for finding in self.report.key_findings[:5])}

## Top Recommendations
{chr(10).join(f"- **{rec.priority.upper()}**: {rec.title}" for rec in self.report.recommendations[:3])}

## Next Steps
{chr(10).join(f"- {step}" for step in self.report.next_steps[:5])}

## Architecture Health: {self.report.architecture_insights['current_state']['overall_health']}
"""
        
        return summary 