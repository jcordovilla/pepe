"""
Report Generator

Generates comprehensive performance test reports with conclusions and recommendations.
Provides actionable insights for improving the agentic architecture and workflows.
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


class ReportGenerator:
    """
    Generates comprehensive performance test reports.
    
    Features:
    - Detailed analysis of test results
    - Actionable recommendations
    - Architecture insights
    - Performance improvement strategies
    """
    
    def __init__(self):
        self.report = None
        
        logger.info("ReportGenerator initialized")
    
    def generate_comprehensive_report(
        self,
        content_analysis: Dict[str, Any],
        query_summary: Dict[str, Any],
        execution_summary: Dict[str, Any],
        evaluation_summary: Dict[str, Any],
        evaluation_results: List[Any]
    ) -> PerformanceReport:
        """
        Generate a comprehensive performance test report.
        
        Args:
            content_analysis: Results from content analysis
            query_summary: Summary of generated queries
            execution_summary: Summary of bot execution
            evaluation_summary: Summary of response evaluation
            evaluation_results: Detailed evaluation results
            
        Returns:
            Comprehensive PerformanceReport object
        """
        logger.info("Generating comprehensive performance test report...")
        
        # Generate test summary
        test_summary = self._generate_test_summary(
            content_analysis, query_summary, execution_summary, evaluation_summary
        )
        
        # Generate key findings
        key_findings = self._generate_key_findings(
            content_analysis, execution_summary, evaluation_summary, evaluation_results
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            evaluation_summary, evaluation_results, execution_summary
        )
        
        # Generate architecture insights
        architecture_insights = self._generate_architecture_insights(
            content_analysis, evaluation_summary, evaluation_results
        )
        
        # Generate next steps
        next_steps = self._generate_next_steps(recommendations, architecture_insights)
        
        # Create comprehensive report
        self.report = PerformanceReport(
            test_summary=test_summary,
            content_analysis_summary=content_analysis.get("server_overview", {}),
            query_coverage_summary=query_summary,
            execution_summary=execution_summary,
            evaluation_summary=evaluation_summary,
            key_findings=key_findings,
            recommendations=recommendations,
            architecture_insights=architecture_insights,
            next_steps=next_steps
        )
        
        logger.info("Comprehensive performance test report generated")
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
        
        # Convert percentage to 0-1 scale
        reliability = success_rate / 100
        
        # Consider error patterns
        error_summary = execution_summary.get("error_summary", {})
        total_errors = error_summary.get("total_errors", 0)
        total_queries = execution_summary.get("total_queries", 1)
        
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