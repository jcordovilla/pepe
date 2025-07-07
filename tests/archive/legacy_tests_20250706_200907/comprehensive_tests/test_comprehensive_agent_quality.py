"""
Comprehensive Agent Response Quality Evaluation System

This advanced test system evaluates agent behavior across multiple dimensions:
- Quantitative metrics (response time, accuracy scores, success rates)
- Qualitative analysis (strengths, weaknesses, user experience)
- Performance insights (bottlenecks, improvement areas, optimization opportunities)
- Actionable recommendations for system enhancement
"""

import asyncio
import json
import os
import sys
import time
import statistics
import re
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import logging

import pytest
from openai import OpenAI

# Add project root to path
sys.path.insert(0, os.path.abspath('.'))

from agentic.agents.orchestrator import AgentOrchestrator
# Import agents to ensure they're registered
from agentic.agents import SearchAgent, AnalysisAgent, PlanningAgent

logger = logging.getLogger(__name__)


class QueryComplexity(Enum):
    """Query complexity levels"""
    SIMPLE = "simple"           # Basic keyword search
    MODERATE = "moderate"       # Multi-parameter queries
    COMPLEX = "complex"         # Multi-step reasoning
    VERY_COMPLEX = "very_complex"  # Multi-agent coordination


class ResponseQuality(Enum):
    """Response quality levels"""
    EXCELLENT = "excellent"     # 9-10 score
    GOOD = "good"              # 7-8 score  
    ACCEPTABLE = "acceptable"   # 5-6 score
    POOR = "poor"              # 3-4 score
    FAILING = "failing"        # 0-2 score


@dataclass
class ComprehensiveEvaluation:
    """Comprehensive evaluation structure with quantitative and qualitative metrics"""
    # Basic Info
    query: str
    response: str
    category: str
    complexity: QueryComplexity
    
    # Quantitative Metrics (0-10 scale)
    accuracy_score: float
    relevance_score: float
    completeness_score: float
    clarity_score: float
    usefulness_score: float
    factual_correctness: float
    context_understanding: float
    technical_precision: float
    user_experience_score: float
    overall_score: float
    
    # Performance Metrics
    response_time: float
    processing_steps: int
    agents_involved: List[str]
    memory_usage: Optional[float]
    api_calls_made: int
    cache_hit_rate: float
    
    # Qualitative Analysis
    strengths: List[str]
    weaknesses: List[str]
    specific_issues: List[str]
    improvement_suggestions: List[str]
    user_experience_notes: List[str]
    
    # Technical Assessment
    query_understanding_quality: str
    task_decomposition_quality: str
    agent_coordination_quality: str
    response_synthesis_quality: str
    error_handling_quality: str
    
    # Confidence and Reliability
    confidence_level: float
    consistency_score: float
    robustness_indicator: str


@dataclass
class PerformanceInsights:
    """Actionable performance insights"""
    category: str
    current_performance: float
    performance_trend: str
    bottlenecks: List[str]
    optimization_opportunities: List[str]
    priority_level: str
    estimated_impact: str
    implementation_complexity: str
    recommended_actions: List[str]


@dataclass
class SystemHealthReport:
    """Comprehensive system health and performance report"""
    timestamp: str
    overall_health_score: float
    performance_by_category: Dict[str, float]
    performance_by_complexity: Dict[str, float]
    response_time_analysis: Dict[str, float]
    quality_distribution: Dict[ResponseQuality, int]
    agent_utilization: Dict[str, float]
    system_bottlenecks: List[str]
    critical_issues: List[str]
    improvement_priorities: List[PerformanceInsights]
    technical_recommendations: List[str]
    user_experience_recommendations: List[str]


class AdvancedResponseEvaluator:
    """
    Advanced AI-powered evaluator with comprehensive metrics
    """
    
    def __init__(self, openai_client: OpenAI):
        self.client = openai_client
        self.evaluation_prompt = """
You are an expert AI system evaluator specializing in conversational AI and agentic architectures. 

Evaluate this Discord bot response comprehensively across these dimensions:

**QUANTITATIVE SCORES (0-10 scale):**
1. **Accuracy**: Factual correctness and truthfulness
2. **Relevance**: How well it addresses the specific query
3. **Completeness**: Comprehensive coverage without verbosity
4. **Clarity**: Structure, readability, and understandability
5. **Usefulness**: Actionable value and practical utility
6. **Factual Correctness**: Accuracy of data and claims
7. **Context Understanding**: Grasp of user intent and context
8. **Technical Precision**: Accuracy of technical details
9. **User Experience**: Overall user satisfaction potential

**QUALITATIVE ANALYSIS:**
- **Strengths**: What the response does well
- **Weaknesses**: Areas needing improvement
- **Specific Issues**: Concrete problems identified
- **User Experience Notes**: UX observations

**TECHNICAL ASSESSMENT:**
- **Query Understanding**: How well the system understood the request
- **Response Synthesis**: Quality of information compilation
- **Error Handling**: Graceful handling of issues

**Query**: {query}
**Response**: {response}
**Context**: Discord bot for message search, analysis, and digest generation

Respond in JSON format:
{{
    "quantitative_scores": {{
        "accuracy": <0-10>,
        "relevance": <0-10>,
        "completeness": <0-10>,
        "clarity": <0-10>,
        "usefulness": <0-10>,
        "factual_correctness": <0-10>,
        "context_understanding": <0-10>,
        "technical_precision": <0-10>,
        "user_experience": <0-10>,
        "overall": <average>
    }},
    "qualitative_analysis": {{
        "strengths": ["strength1", "strength2"],
        "weaknesses": ["weakness1", "weakness2"],
        "specific_issues": ["issue1", "issue2"],
        "user_experience_notes": ["note1", "note2"]
    }},
    "technical_assessment": {{
        "query_understanding": "excellent|good|fair|poor",
        "response_synthesis": "excellent|good|fair|poor",
        "error_handling": "excellent|good|fair|poor"
    }},
    "confidence_level": <0-10>,
    "improvement_suggestions": ["suggestion1", "suggestion2"]
}}

Be thorough, objective, and provide actionable insights.
"""

    async def evaluate_response(self, query: str, response: str, 
                              processing_details: Dict[str, Any]) -> ComprehensiveEvaluation:
        """Comprehensive evaluation of agent response"""
        try:
            # Get AI evaluation
            ai_evaluation = await self._get_ai_evaluation(query, response)
            
            # Extract quantitative metrics
            scores = ai_evaluation.get("quantitative_scores", {})
            qualitative = ai_evaluation.get("qualitative_analysis", {})
            technical = ai_evaluation.get("technical_assessment", {})
            
            # Determine complexity
            complexity = self._assess_query_complexity(query)
            
            # Calculate performance metrics
            response_time = processing_details.get("response_time", 0)
            agents_involved = processing_details.get("agents_involved", [])
            
            return ComprehensiveEvaluation(
                query=query,
                response=response,
                category=processing_details.get("category", "unknown"),
                complexity=complexity,
                
                # Quantitative scores
                accuracy_score=scores.get("accuracy", 0),
                relevance_score=scores.get("relevance", 0),
                completeness_score=scores.get("completeness", 0),
                clarity_score=scores.get("clarity", 0),
                usefulness_score=scores.get("usefulness", 0),
                factual_correctness=scores.get("factual_correctness", 0),
                context_understanding=scores.get("context_understanding", 0),
                technical_precision=scores.get("technical_precision", 0),
                user_experience_score=scores.get("user_experience", 0),
                overall_score=scores.get("overall", 0),
                
                # Performance metrics
                response_time=response_time,
                processing_steps=processing_details.get("processing_steps", 0),
                agents_involved=agents_involved,
                memory_usage=processing_details.get("memory_usage"),
                api_calls_made=processing_details.get("api_calls", 0),
                cache_hit_rate=processing_details.get("cache_hit_rate", 0),
                
                # Qualitative analysis
                strengths=qualitative.get("strengths", []),
                weaknesses=qualitative.get("weaknesses", []),
                specific_issues=qualitative.get("specific_issues", []),
                improvement_suggestions=ai_evaluation.get("improvement_suggestions", []),
                user_experience_notes=qualitative.get("user_experience_notes", []),
                
                # Technical assessment
                query_understanding_quality=technical.get("query_understanding", "unknown"),
                task_decomposition_quality=technical.get("task_decomposition", "unknown"),
                agent_coordination_quality=technical.get("agent_coordination", "unknown"),
                response_synthesis_quality=technical.get("response_synthesis", "unknown"),
                error_handling_quality=technical.get("error_handling", "unknown"),
                
                # Confidence metrics
                confidence_level=ai_evaluation.get("confidence_level", 0),
                consistency_score=self._calculate_consistency_score(scores),
                robustness_indicator=self._assess_robustness(scores, response_time)
            )
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return self._create_error_evaluation(query, response, str(e))
    
    async def _get_ai_evaluation(self, query: str, response: str) -> Dict[str, Any]:
        """Get AI evaluation from OpenAI"""
        try:
            prompt = self.evaluation_prompt.format(query=query, response=response)
            
            completion = await asyncio.to_thread(
                self.client.chat.completions.create,
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert AI system evaluator."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=1500
            )
            
            response_text = completion.choices[0].message.content
            if response_text:
                return json.loads(response_text)
            else:
                return self._get_fallback_evaluation()
                
        except Exception as e:
            logger.error(f"AI evaluation failed: {e}")
            return self._get_fallback_evaluation()
    
    def _assess_query_complexity(self, query: str) -> QueryComplexity:
        """Assess query complexity based on content"""
        query_lower = query.lower()
        
        # Count complexity indicators
        complexity_indicators = [
            len(query.split()) > 15,  # Long query
            "and" in query_lower and "or" in query_lower,  # Boolean logic
            len(re.findall(r'#\w+|@\w+|\w+day|\w+week|\w+month', query)) > 2,  # Multiple entities
            any(word in query_lower for word in ["analyze", "compare", "summarize", "digest"]),  # Analysis tasks
            query_lower.count("?") > 1,  # Multiple questions
        ]
        
        complexity_score = sum(complexity_indicators)
        
        if complexity_score >= 4:
            return QueryComplexity.VERY_COMPLEX
        elif complexity_score >= 3:
            return QueryComplexity.COMPLEX
        elif complexity_score >= 2:
            return QueryComplexity.MODERATE
        else:
            return QueryComplexity.SIMPLE
    
    def _calculate_consistency_score(self, scores: Dict[str, float]) -> float:
        """Calculate consistency across different score dimensions"""
        if not scores:
            return 0.0
        
        score_values = [v for v in scores.values() if isinstance(v, (int, float))]
        if len(score_values) < 2:
            return 10.0
        
        std_dev = statistics.stdev(score_values)
        # Lower standard deviation = higher consistency
        consistency = max(0, 10 - std_dev)
        return round(consistency, 2)
    
    def _assess_robustness(self, scores: Dict[str, float], response_time: float) -> str:
        """Assess system robustness based on scores and performance"""
        avg_score = statistics.mean([v for v in scores.values() if isinstance(v, (int, float))])
        
        if avg_score >= 8 and response_time <= 3:
            return "high"
        elif avg_score >= 6 and response_time <= 7:
            return "medium"
        else:
            return "low"
    
    def _get_fallback_evaluation(self) -> Dict[str, Any]:
        """Fallback evaluation when AI evaluation fails"""
        return {
            "quantitative_scores": {
                "accuracy": 5,
                "relevance": 5,
                "completeness": 5,
                "clarity": 5,
                "usefulness": 5,
                "factual_correctness": 5,
                "context_understanding": 5,
                "technical_precision": 5,
                "user_experience": 5,
                "overall": 5
            },
            "qualitative_analysis": {
                "strengths": ["Response generated"],
                "weaknesses": ["Could not evaluate thoroughly"],
                "specific_issues": ["Evaluation system failure"],
                "user_experience_notes": ["Unable to assess UX"]
            },
            "technical_assessment": {
                "query_understanding": "unknown",
                "response_synthesis": "unknown",
                "error_handling": "unknown"
            },
            "confidence_level": 3,
            "improvement_suggestions": ["Fix evaluation system", "Ensure AI evaluation availability"]
        }
    
    def _create_error_evaluation(self, query: str, response: str, error: str) -> ComprehensiveEvaluation:
        """Create evaluation for error cases"""
        return ComprehensiveEvaluation(
            query=query,
            response=response or "No response generated",
            category="error",
            complexity=QueryComplexity.SIMPLE,
            
            # All scores zero for errors
            accuracy_score=0,
            relevance_score=0,
            completeness_score=0,
            clarity_score=0,
            usefulness_score=0,
            factual_correctness=0,
            context_understanding=0,
            technical_precision=0,
            user_experience_score=0,
            overall_score=0,
            
            # Performance metrics
            response_time=0,
            processing_steps=0,
            agents_involved=[],
            memory_usage=None,
            api_calls_made=0,
            cache_hit_rate=0,
            
            # Error details
            strengths=[],
            weaknesses=[f"System error: {error}"],
            specific_issues=[error],
            improvement_suggestions=["Fix system error", "Improve error handling"],
            user_experience_notes=["System failure affects user experience"],
            
            # Technical assessment
            query_understanding_quality="poor",
            task_decomposition_quality="poor",
            agent_coordination_quality="poor",
            response_synthesis_quality="poor",
            error_handling_quality="poor",
            
            # Low confidence
            confidence_level=0,
            consistency_score=0,
            robustness_indicator="low"
        )


class ComprehensiveAgentQualityTest:
    """
    Comprehensive test suite that measures all aspects of agent behavior
    and provides actionable insights for improvement
    """
    
    def __init__(self):
        # Initialize orchestrator with basic config
        config = {
            "agents": {},
            "memory": {"db_path": "data/conversation_memory.db"},
            "task_planner": {}
        }
        
        # Ensure agents are registered by importing and creating them
        from agentic.agents.base_agent import agent_registry
        logger.info("Initializing agents for testing...")
        
        # Import agent classes to trigger registration
        try:
            search_agent = SearchAgent(config.get("search_agent", {}))
            analysis_agent = AnalysisAgent(config.get("analysis_agent", {}))
            planning_agent = PlanningAgent(config.get("planning_agent", {}))
            logger.info(f"Agents registered: {list(agent_registry._agents.keys())}")
        except Exception as e:
            logger.warning(f"Could not initialize all agents: {e}")
        
        self.orchestrator = AgentOrchestrator(config)
        self.evaluator = AdvancedResponseEvaluator(
            OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        )
        
        # Comprehensive test scenarios
        self.test_scenarios = [
            # Simple queries
            {
                "query": "Find messages about Python",
                "category": "basic_search",
                "complexity": QueryComplexity.SIMPLE,
                "expected_agents": ["search"],
                "success_criteria": {"min_score": 7.0, "max_time": 3.0}
            },
            
            # Temporal queries
            {
                "query": "Show me discussions from last week about machine learning",
                "category": "temporal_search",
                "complexity": QueryComplexity.MODERATE,
                "expected_agents": ["search", "analysis"],
                "success_criteria": {"min_score": 7.5, "max_time": 5.0}
            },
            
            # Digest generation
            {
                "query": "Create a weekly digest for #ai-research channel",
                "category": "digest_generation",
                "complexity": QueryComplexity.COMPLEX,
                "expected_agents": ["search", "digest", "analysis"],
                "success_criteria": {"min_score": 8.0, "max_time": 8.0}
            },
            
            # Complex analysis
            {
                "query": "Analyze engagement patterns and identify trending topics from the past month",
                "category": "complex_analysis",
                "complexity": QueryComplexity.VERY_COMPLEX,
                "expected_agents": ["search", "analysis", "digest"],
                "success_criteria": {"min_score": 7.0, "max_time": 10.0}
            },
            
            # Multi-intent queries
            {
                "query": "Find Python discussions from yesterday, summarize key points, and identify who was most active",
                "category": "multi_intent",
                "complexity": QueryComplexity.VERY_COMPLEX,
                "expected_agents": ["search", "analysis", "digest"],
                "success_criteria": {"min_score": 7.5, "max_time": 10.0}
            },
            
            # Error handling
            {
                "query": "Show me messages from #nonexistent-channel-12345",
                "category": "error_handling",
                "complexity": QueryComplexity.SIMPLE,
                "expected_agents": ["search"],
                "success_criteria": {"min_score": 6.0, "max_time": 3.0}
            },
            
            # Ambiguous queries
            {
                "query": "What's been happening?",
                "category": "ambiguous_query",
                "complexity": QueryComplexity.MODERATE,
                "expected_agents": ["search", "analysis"],
                "success_criteria": {"min_score": 6.0, "max_time": 5.0}
            },
            
            # User-specific queries
            {
                "query": "What has @john_doe been working on recently?",
                "category": "user_specific",
                "complexity": QueryComplexity.MODERATE,
                "expected_agents": ["search"],
                "success_criteria": {"min_score": 7.0, "max_time": 4.0}
            },
            
            # Performance queries
            {
                "query": "How active has our server been this month compared to last month?",
                "category": "performance_analysis",
                "complexity": QueryComplexity.COMPLEX,
                "expected_agents": ["search", "analysis"],
                "success_criteria": {"min_score": 7.5, "max_time": 8.0}
            },
            
            # Channel-specific analysis
            {
                "query": "Analyze conversation patterns in #general and identify peak activity times",
                "category": "channel_analysis",
                "complexity": QueryComplexity.COMPLEX,
                "expected_agents": ["search", "analysis"],
                "success_criteria": {"min_score": 7.0, "max_time": 7.0}
            }
        ]
    
    async def run_comprehensive_evaluation(self) -> SystemHealthReport:
        """Run comprehensive evaluation and generate actionable insights"""
        logger.info("Starting comprehensive agent quality evaluation")
        
        # First, perform system health checks
        system_issues = await self._perform_system_health_checks()
        
        evaluations = []
        start_time = time.time()
        
        # Run all test scenarios
        for scenario in self.test_scenarios:
            evaluation = await self._run_single_scenario(scenario)
            evaluations.append(evaluation)
            
            # Brief pause to avoid overwhelming the system
            await asyncio.sleep(0.5)
        
        total_time = time.time() - start_time
        
        # Generate comprehensive report with system issues
        report = self._generate_system_health_report(evaluations, total_time, system_issues)
        
        # Save detailed results
        await self._save_comprehensive_report(report, evaluations)
        
        return report
    
    async def _perform_system_health_checks(self) -> List[str]:
        """Perform basic system health checks before running tests"""
        issues = []
        
        # Check agent registration
        from agentic.agents.base_agent import agent_registry
        if not agent_registry._agents:
            issues.append("CRITICAL: No agents registered in the system")
        else:
            logger.info(f"Agents available: {list(agent_registry._agents.keys())}")
        
        # Test basic orchestrator functionality
        try:
            test_response = await self.orchestrator.process_query(
                "test", "test_user", {"test": True}
            )
            if isinstance(test_response, dict):
                if not test_response.get('results') and "didn't find specific results" in str(test_response):
                    issues.append("WARNING: Orchestrator returns generic responses, no real data processing")
        except Exception as e:
            issues.append(f"CRITICAL: Orchestrator basic functionality failed: {str(e)}")
        
        # Check if we have required environment variables
        required_vars = ['OPENAI_API_KEY', 'DISCORD_TOKEN', 'GUILD_ID']
        for var in required_vars:
            if not os.getenv(var):
                issues.append(f"CRITICAL: Missing required environment variable: {var}")
        
        return issues
    
    async def _run_single_scenario(self, scenario: Dict[str, Any]) -> ComprehensiveEvaluation:
        """Run a single test scenario"""
        query = scenario["query"]
        category = scenario["category"]
        
        logger.info(f"Testing scenario: {category} - {query[:50]}...")
        
        start_time = time.time()
        
        try:
            # Execute query
            response = await self.orchestrator.process_query(
                query=query,
                user_id="test_user",
                context={"test_scenario": True, "category": category}
            )
            
            response_time = time.time() - start_time
            
            # Convert response to string
            if isinstance(response, dict):
                response_text = json.dumps(response, indent=2)
            else:
                response_text = str(response)
            
            # Create processing details
            processing_details = {
                "category": category,
                "response_time": response_time,
                "agents_involved": scenario.get("expected_agents", []),
                "processing_steps": len(scenario.get("expected_agents", [])),
                "api_calls": 1,  # Simplified
                "cache_hit_rate": 0.0,  # Would need to be tracked in real system
            }
            
            # Comprehensive evaluation
            evaluation = await self.evaluator.evaluate_response(
                query, response_text, processing_details
            )
            
            return evaluation
            
        except Exception as e:
            logger.error(f"Scenario failed: {category} - {e}")
            return self.evaluator._create_error_evaluation(query, "", str(e))
    
    def _generate_system_health_report(self, evaluations: List[ComprehensiveEvaluation], 
                                     total_time: float, system_issues: Optional[List[str]] = None) -> SystemHealthReport:
        """Generate comprehensive system health report"""
        
        # Calculate overall metrics
        overall_scores = [e.overall_score for e in evaluations]
        overall_health = statistics.mean(overall_scores) if overall_scores else 0
        
        # Performance by category
        performance_by_category = {}
        for category in set(e.category for e in evaluations):
            category_scores = [e.overall_score for e in evaluations if e.category == category]
            performance_by_category[category] = statistics.mean(category_scores) if category_scores else 0
        
        # Performance by complexity
        performance_by_complexity = {}
        for complexity in QueryComplexity:
            complexity_scores = [e.overall_score for e in evaluations if e.complexity == complexity]
            if complexity_scores:
                performance_by_complexity[complexity.value] = statistics.mean(complexity_scores)
        
        # Response time analysis
        response_times = [e.response_time for e in evaluations if e.response_time > 0]
        response_time_analysis = {
            "average": statistics.mean(response_times) if response_times else 0,
            "median": statistics.median(response_times) if response_times else 0,
            "p95": self._percentile(response_times, 95) if response_times else 0,
            "max": max(response_times) if response_times else 0
        }
        
        # Quality distribution
        quality_distribution = {}
        for evaluation in evaluations:
            quality = self._get_quality_level(evaluation.overall_score)
            quality_distribution[quality] = quality_distribution.get(quality, 0) + 1
        
        # Identify bottlenecks and issues
        bottlenecks = self._identify_bottlenecks(evaluations)
        critical_issues = self._identify_critical_issues(evaluations, system_issues or [])
        
        # Generate improvement priorities
        improvement_priorities = self._generate_improvement_priorities(evaluations)
        
        # Technical and UX recommendations
        technical_recommendations = self._generate_technical_recommendations(evaluations)
        ux_recommendations = self._generate_ux_recommendations(evaluations)
        
        return SystemHealthReport(
            timestamp=datetime.now().isoformat(),
            overall_health_score=overall_health,
            performance_by_category=performance_by_category,
            performance_by_complexity=performance_by_complexity,
            response_time_analysis=response_time_analysis,
            quality_distribution=quality_distribution,
            agent_utilization={},  # Would need agent-specific tracking
            system_bottlenecks=bottlenecks,
            critical_issues=critical_issues,
            improvement_priorities=improvement_priorities,
            technical_recommendations=technical_recommendations,
            user_experience_recommendations=ux_recommendations
        )
    
    def _get_quality_level(self, score: float) -> ResponseQuality:
        """Get quality level from score"""
        if score >= 9:
            return ResponseQuality.EXCELLENT
        elif score >= 7:
            return ResponseQuality.GOOD
        elif score >= 5:
            return ResponseQuality.ACCEPTABLE
        elif score >= 3:
            return ResponseQuality.POOR
        else:
            return ResponseQuality.FAILING
    
    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile"""
        if not data:
            return 0
        sorted_data = sorted(data)
        index = (percentile / 100) * (len(sorted_data) - 1)
        if index.is_integer():
            return sorted_data[int(index)]
        else:
            lower = sorted_data[int(index)]
            upper = sorted_data[int(index) + 1]
            return lower + (upper - lower) * (index - int(index))
    
    def _identify_bottlenecks(self, evaluations: List[ComprehensiveEvaluation]) -> List[str]:
        """Identify system bottlenecks"""
        bottlenecks = []
        
        # Response time bottlenecks
        slow_responses = [e for e in evaluations if e.response_time > 7]
        if len(slow_responses) > len(evaluations) * 0.3:
            bottlenecks.append("Response time consistently above 7 seconds")
        
        # Quality bottlenecks
        low_quality = [e for e in evaluations if e.overall_score < 6]
        if len(low_quality) > len(evaluations) * 0.4:
            bottlenecks.append("Overall quality scores consistently below acceptable threshold")
        
        # Specific dimension bottlenecks
        accuracy_issues = [e for e in evaluations if e.accuracy_score < 6]
        if len(accuracy_issues) > len(evaluations) * 0.3:
            bottlenecks.append("Accuracy scores indicate factual correctness issues")
        
        clarity_issues = [e for e in evaluations if e.clarity_score < 6]
        if len(clarity_issues) > len(evaluations) * 0.3:
            bottlenecks.append("Clarity scores indicate response structure and readability issues")
        
        return bottlenecks
    
    def _identify_critical_issues(self, evaluations: List[ComprehensiveEvaluation], 
                                system_issues: Optional[List[str]] = None) -> List[str]:
        """Identify critical issues requiring immediate attention"""
        critical_issues = []
        
        # Add system-level issues first
        if system_issues:
            critical_issues.extend(system_issues)
        
        # System failures
        failures = [e for e in evaluations if e.overall_score == 0]
        if failures:
            critical_issues.append(f"CRITICAL: {len(failures)} complete system failures detected")
        
        # Very low overall performance
        very_poor = [e for e in evaluations if e.overall_score < 3]
        if len(very_poor) > len(evaluations) * 0.3:
            critical_issues.append("CRITICAL: >30% of responses have failing quality scores (<3/10)")
        
        # Agent coordination failures
        agent_failures = [e for e in evaluations if "No agent found" in str(e.specific_issues)]
        if len(agent_failures) > len(evaluations) * 0.5:
            critical_issues.append("CRITICAL: Agent coordination failing in >50% of queries")
        
        # Poor error handling
        poor_error_handling = [e for e in evaluations if e.error_handling_quality == "poor"]
        if len(poor_error_handling) > 2:
            critical_issues.append("CRITICAL: Poor error handling across multiple scenarios")
        
        # Consistently low user experience
        poor_ux = [e for e in evaluations if e.user_experience_score < 4]
        if len(poor_ux) > len(evaluations) * 0.25:
            critical_issues.append("CRITICAL: User experience scores indicate significant usability problems")
        
        # Data processing failures
        no_results = [e for e in evaluations if "didn't find specific results" in e.response or "failed to provide specific results" in str(e.weaknesses)]
        if len(no_results) > len(evaluations) * 0.6:
            critical_issues.append("CRITICAL: System failing to provide meaningful results in >60% of queries")
        
        return critical_issues
    
    def _generate_improvement_priorities(self, evaluations: List[ComprehensiveEvaluation]) -> List[PerformanceInsights]:
        """Generate prioritized improvement recommendations"""
        priorities = []
        
        # Analyze each category
        categories = set(e.category for e in evaluations)
        for category in categories:
            category_evals = [e for e in evaluations if e.category == category]
            avg_score = statistics.mean([e.overall_score for e in category_evals])
            
            if avg_score < 7:
                # Identify specific issues
                common_weaknesses = []
                for eval in category_evals:
                    common_weaknesses.extend(eval.weaknesses)
                
                weakness_counts = {}
                for weakness in common_weaknesses:
                    weakness_counts[weakness] = weakness_counts.get(weakness, 0) + 1
                
                top_weaknesses = sorted(weakness_counts.items(), key=lambda x: x[1], reverse=True)[:3]
                
                priority = PerformanceInsights(
                    category=category,
                    current_performance=avg_score,
                    performance_trend="declining" if avg_score < 6 else "stable",
                    bottlenecks=[w[0] for w in top_weaknesses],
                    optimization_opportunities=[
                        f"Address {w[0]}" for w in top_weaknesses
                    ],
                    priority_level="high" if avg_score < 6 else "medium",
                    estimated_impact="significant" if avg_score < 6 else "moderate",
                    implementation_complexity="medium",
                    recommended_actions=[
                        f"Focus on improving {category} responses",
                        "Analyze failure patterns in this category",
                        "Implement targeted optimizations"
                    ]
                )
                priorities.append(priority)
        
        # Sort by priority and performance
        priorities.sort(key=lambda x: (x.priority_level == "high", -x.current_performance))
        
        return priorities[:5]  # Top 5 priorities
    
    def _generate_technical_recommendations(self, evaluations: List[ComprehensiveEvaluation]) -> List[str]:
        """Generate technical improvement recommendations"""
        recommendations = []
        
        # Response time recommendations
        slow_responses = [e for e in evaluations if e.response_time > 5]
        if len(slow_responses) > len(evaluations) * 0.3:
            recommendations.extend([
                "Implement response caching for common queries",
                "Optimize database query performance",
                "Consider parallel processing for complex queries"
            ])
        
        # Accuracy recommendations
        accuracy_issues = [e for e in evaluations if e.accuracy_score < 7]
        if len(accuracy_issues) > len(evaluations) * 0.2:
            recommendations.extend([
                "Improve fact-checking mechanisms",
                "Enhance data validation processes",
                "Implement confidence scoring for responses"
            ])
        
        # Agent coordination recommendations
        coordination_issues = [e for e in evaluations if e.agent_coordination_quality in ["poor", "fair"]]
        if coordination_issues:
            recommendations.extend([
                "Improve agent communication protocols",
                "Implement better task handoff mechanisms",
                "Add agent performance monitoring"
            ])
        
        return recommendations
    
    def _generate_ux_recommendations(self, evaluations: List[ComprehensiveEvaluation]) -> List[str]:
        """Generate user experience improvement recommendations"""
        recommendations = []
        
        # Clarity recommendations
        clarity_issues = [e for e in evaluations if e.clarity_score < 7]
        if clarity_issues:
            recommendations.extend([
                "Improve response formatting and structure",
                "Add clear section headers and bullet points",
                "Implement progressive disclosure for complex information"
            ])
        
        # Usefulness recommendations
        usefulness_issues = [e for e in evaluations if e.usefulness_score < 7]
        if usefulness_issues:
            recommendations.extend([
                "Include more actionable insights in responses",
                "Add follow-up suggestions and related queries",
                "Provide context and explanation for results"
            ])
        
        # Error handling UX
        error_handling_issues = [e for e in evaluations if e.error_handling_quality == "poor"]
        if error_handling_issues:
            recommendations.extend([
                "Improve error message clarity and helpfulness",
                "Provide specific guidance when queries fail",
                "Implement graceful degradation for partial failures"
            ])
        
        return recommendations
    
    async def _save_comprehensive_report(self, report: SystemHealthReport, 
                                       evaluations: List[ComprehensiveEvaluation]):
        """Save comprehensive report and raw data"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save summary report
        report_path = f"tests/reports/comprehensive_report_{timestamp}.json"
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        
        # Convert report to JSON-serializable format
        report_dict = asdict(report)
        # Convert ResponseQuality enum keys to strings
        if 'quality_distribution' in report_dict:
            report_dict['quality_distribution'] = {
                str(k): v for k, v in report_dict['quality_distribution'].items()
            }
        
        with open(report_path, 'w') as f:
            json.dump(report_dict, f, indent=2)
        
        # Save detailed evaluations
        details_path = f"tests/reports/detailed_evaluations_{timestamp}.json"
        
        # Convert evaluations to JSON-serializable format
        serializable_evaluations = []
        for evaluation in evaluations:
            eval_dict = asdict(evaluation)
            # Convert enum to string
            eval_dict['complexity'] = evaluation.complexity.value
            serializable_evaluations.append(eval_dict)
        
        with open(details_path, 'w') as f:
            json.dump(serializable_evaluations, f, indent=2)
        
        logger.info(f"Comprehensive report saved to: {report_path}")
        logger.info(f"Detailed evaluations saved to: {details_path}")


# Pytest integration
@pytest.mark.comprehensive
@pytest.mark.quality
@pytest.mark.asyncio
async def test_comprehensive_agent_quality():
    """Comprehensive agent quality test with actionable insights"""
    
    # Skip if no OpenAI API key
    if not os.getenv('OPENAI_API_KEY'):
        pytest.skip("OPENAI_API_KEY not set")
    
    # Run comprehensive test
    test_runner = ComprehensiveAgentQualityTest()
    report = await test_runner.run_comprehensive_evaluation()
    
    # Assert minimum thresholds - these should be realistic for a working system
    MIN_HEALTH_SCORE = 7.0  # A working system should score at least 7/10
    MAX_CRITICAL_ISSUES = 1  # At most 1 critical issue acceptable
    MIN_ACCEPTABLE_RATE = 80.0  # 80% of responses should be acceptable (score >= 5)
    
    assert report.overall_health_score >= MIN_HEALTH_SCORE, (
        f"SYSTEM FAILURE: Health score {report.overall_health_score:.1f} below minimum {MIN_HEALTH_SCORE}. "
        f"Critical issues: {report.critical_issues}"
    )
    
    assert len(report.critical_issues) <= MAX_CRITICAL_ISSUES, (
        f"SYSTEM FAILURE: {len(report.critical_issues)} critical issues detected (max {MAX_CRITICAL_ISSUES}): "
        f"{report.critical_issues}"
    )
    
    # Calculate acceptable response rate
    total_responses = sum(report.quality_distribution.values())
    acceptable_responses = sum(
        count for quality, count in report.quality_distribution.items()
        if quality in [ResponseQuality.EXCELLENT, ResponseQuality.GOOD, ResponseQuality.ACCEPTABLE]
    )
    acceptable_rate = (acceptable_responses / total_responses * 100) if total_responses > 0 else 0
    
    assert acceptable_rate >= MIN_ACCEPTABLE_RATE, (
        f"Acceptable response rate {acceptable_rate:.1f}% below threshold {MIN_ACCEPTABLE_RATE}%"
    )
    
    # Print comprehensive summary
    print(f"\n{'='*80}")
    print("🔍 COMPREHENSIVE AGENT QUALITY EVALUATION RESULTS")
    print(f"{'='*80}")
    print(f"🎯 Overall Health Score: {report.overall_health_score:.1f}/10")
    print(f"📊 Acceptable Response Rate: {acceptable_rate:.1f}%")
    print(f"⚡ Average Response Time: {report.response_time_analysis['average']:.1f}s")
    print(f"🔧 Critical Issues: {len(report.critical_issues)}")
    print(f"📈 Improvement Priorities: {len(report.improvement_priorities)}")
    
    # Category performance
    print(f"\n📋 Performance by Category:")
    for category, score in sorted(report.performance_by_category.items(), key=lambda x: x[1], reverse=True):
        status = "✅" if score >= 7 else "⚠️" if score >= 5 else "❌"
        print(f"  {status} {category.replace('_', ' ').title()}: {score:.1f}/10")
    
    # Top improvement priorities
    if report.improvement_priorities:
        print(f"\n🎯 Top Improvement Priorities:")
        for i, priority in enumerate(report.improvement_priorities[:3], 1):
            print(f"  {i}. {priority.category} (Score: {priority.current_performance:.1f})")
            print(f"     Priority: {priority.priority_level.upper()}")
            print(f"     Actions: {priority.recommended_actions[0] if priority.recommended_actions else 'No specific actions'}")
    
    print(f"{'='*80}")


# Standalone execution
if __name__ == "__main__":
    async def main():
        test_runner = ComprehensiveAgentQualityTest()
        report = await test_runner.run_comprehensive_evaluation()
        
        print("\n" + "="*80)
        print("🔍 COMPREHENSIVE AGENT EVALUATION COMPLETED")
        print("="*80)
        print(f"🎯 System Health Score: {report.overall_health_score:.1f}/10")
        print(f"⏱️  Average Response Time: {report.response_time_analysis['average']:.1f}s")
        print(f"🔧 Critical Issues: {len(report.critical_issues)}")
        
        if report.critical_issues:
            print(f"\n❌ Critical Issues:")
            for issue in report.critical_issues:
                print(f"  • {issue}")
        
        if report.improvement_priorities:
            print(f"\n📈 Priority Improvements:")
            for priority in report.improvement_priorities[:3]:
                print(f"  🎯 {priority.category}: {priority.current_performance:.1f}/10 ({priority.priority_level} priority)")
        
        print(f"\n📁 Detailed reports saved to tests/reports/")
        print("="*80)
        
        return report
    
    asyncio.run(main())
