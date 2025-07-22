"""
Synthetic Data Generator

Generates synthetic test data for edge cases, stress scenarios, and load testing.
Creates realistic but challenging scenarios to test system robustness.
"""

import random
import string
import json
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import asyncio
from pathlib import Path
from dataclasses import asdict

from tests.performance_test_suite.query_generator import TestQuery

logger = logging.getLogger(__name__)

@dataclass
class SyntheticScenario:
    """Represents a synthetic test scenario."""
    name: str
    description: str
    queries: List[TestQuery]
    expected_challenges: List[str]
    stress_level: str  # low, medium, high, extreme

class SyntheticDataGenerator:
    """
    Generates synthetic test data for comprehensive testing scenarios.
    """
    
    def __init__(self):
        self.scenarios = []
        self.edge_case_patterns = self._load_edge_case_patterns()
        self.stress_patterns = self._load_stress_patterns()
        
        logger.info("SyntheticDataGenerator initialized")

    def _load_edge_case_patterns(self) -> Dict[str, List[str]]:
        """Load patterns for edge case generation."""
        return {
            "very_long_queries": [
                "This is an extremely long query that contains many words and goes on for a very long time " * 50,
                "A query with " + "repetitive " * 100 + "words",
                "Query with " + "nested " * 20 + "parentheses " * 20 + "and " * 20 + "brackets " * 20
            ],
            "malformed_input": [
                "",  # Empty query
                "   ",  # Whitespace only
                "SELECT * FROM users; DROP TABLE messages;",  # SQL injection
                "<script>alert('xss')</script>",  # XSS attempt
                "query" + "\x00" * 100,  # Null bytes
                "query" + "\xff" * 100,  # Invalid UTF-8
            ],
            "unicode_edge_cases": [
                "Query with emojis 🚀🎉💻🔥",
                "Query with special chars: áéíóú ñ ç ß",
                "Query with " + "🚀" * 50,  # Many emojis
                "Query with " + "ñ" * 100,  # Many special chars
            ],
            "concurrent_scenarios": [
                "What are the most active channels?",
                "Show me recent activity",
                "Generate a weekly digest",
                "Find messages about AI",
                "Analyze user engagement"
            ]
        }

    def _load_stress_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Load patterns for stress testing."""
        return {
            "high_frequency": {
                "queries_per_second": 20,
                "duration_seconds": 60,
                "query_types": ["simple_search", "complex_analysis", "digest_generation"]
            },
            "memory_intensive": {
                "large_context": True,
                "complex_queries": True,
                "batch_size": 1000
            },
            "cpu_intensive": {
                "complex_analytics": True,
                "pattern_matching": True,
                "statistical_analysis": True
            },
            "mixed_load": {
                "concurrent_users": 10,
                "query_diversity": "high",
                "duration_minutes": 5
            }
        }

    def generate_edge_case_scenarios(self) -> List[SyntheticScenario]:
        """Generate edge case test scenarios."""
        scenarios = []
        
        # Very long queries
        long_query_scenario = self._create_long_query_scenario()
        scenarios.append(long_query_scenario)
        
        # Malformed input
        malformed_scenario = self._create_malformed_input_scenario()
        scenarios.append(malformed_scenario)
        
        # Unicode edge cases
        unicode_scenario = self._create_unicode_edge_scenario()
        scenarios.append(unicode_scenario)
        
        # Empty and whitespace queries
        empty_scenario = self._create_empty_query_scenario()
        scenarios.append(empty_scenario)
        
        # SQL injection attempts
        injection_scenario = self._create_injection_scenario()
        scenarios.append(injection_scenario)
        
        logger.info(f"Generated {len(scenarios)} edge case scenarios")
        return scenarios

    def generate_stress_scenarios(self) -> List[SyntheticScenario]:
        """Generate stress test scenarios."""
        scenarios = []
        
        # High frequency queries
        high_freq_scenario = self._create_high_frequency_scenario()
        scenarios.append(high_freq_scenario)
        
        # Memory intensive
        memory_scenario = self._create_memory_intensive_scenario()
        scenarios.append(memory_scenario)
        
        # CPU intensive
        cpu_scenario = self._create_cpu_intensive_scenario()
        scenarios.append(cpu_scenario)
        
        # Mixed load
        mixed_scenario = self._create_mixed_load_scenario()
        scenarios.append(mixed_scenario)
        
        # Concurrent users
        concurrent_scenario = self._create_concurrent_user_scenario()
        scenarios.append(concurrent_scenario)
        
        logger.info(f"Generated {len(scenarios)} stress scenarios")
        return scenarios

    def generate_load_test_scenarios(self) -> List[SyntheticScenario]:
        """Generate load testing scenarios."""
        scenarios = []
        
        # Gradual load increase
        gradual_scenario = self._create_gradual_load_scenario()
        scenarios.append(gradual_scenario)
        
        # Spike load
        spike_scenario = self._create_spike_load_scenario()
        scenarios.append(spike_scenario)
        
        # Sustained load
        sustained_scenario = self._create_sustained_load_scenario()
        scenarios.append(sustained_scenario)
        
        # Burst load
        burst_scenario = self._create_burst_load_scenario()
        scenarios.append(burst_scenario)
        
        logger.info(f"Generated {len(scenarios)} load test scenarios")
        return scenarios

    def _create_long_query_scenario(self) -> SyntheticScenario:
        """Create scenario with very long queries."""
        queries = []
        
        for i, pattern in enumerate(self.edge_case_patterns["very_long_queries"]):
            queries.append(TestQuery(
                id=f"long_query_{i+1}",
                query=pattern,
                category="edge_case",
                subcategory="very_long_query",
                complexity="extreme",
                expected_response_structure={
                    "error_handling": "graceful_degradation",
                    "response_time_limit": 30.0,
                    "should_truncate": True
                },
                edge_case=True
            ))
        
        return SyntheticScenario(
            name="very_long_queries",
            description="Test system handling of extremely long queries",
            queries=queries,
            expected_challenges=[
                "Memory usage with long text",
                "Processing time for large inputs",
                "Response truncation handling"
            ],
            stress_level="high"
        )

    def _create_malformed_input_scenario(self) -> SyntheticScenario:
        """Create scenario with malformed input."""
        queries = []
        
        for i, pattern in enumerate(self.edge_case_patterns["malformed_input"]):
            queries.append(TestQuery(
                id=f"malformed_{i+1}",
                query=pattern,
                category="edge_case",
                subcategory="malformed_input",
                complexity="extreme",
                expected_response_structure={
                    "error_handling": "robust_validation",
                    "security_check": True,
                    "graceful_fallback": True
                },
                edge_case=True
            ))
        
        return SyntheticScenario(
            name="malformed_input",
            description="Test system handling of malformed and malicious input",
            queries=queries,
            expected_challenges=[
                "Input validation",
                "Security vulnerability prevention",
                "Error message appropriateness"
            ],
            stress_level="medium"
        )

    def _create_unicode_edge_scenario(self) -> SyntheticScenario:
        """Create scenario with Unicode edge cases."""
        queries = []
        
        for i, pattern in enumerate(self.edge_case_patterns["unicode_edge_cases"]):
            queries.append(TestQuery(
                id=f"unicode_{i+1}",
                query=pattern,
                category="edge_case",
                subcategory="unicode_handling",
                complexity="moderate",
                expected_response_structure={
                    "unicode_support": True,
                    "encoding_handling": "utf8",
                    "emoji_processing": True
                },
                edge_case=True
            ))
        
        return SyntheticScenario(
            name="unicode_edge_cases",
            description="Test system handling of Unicode and special characters",
            queries=queries,
            expected_challenges=[
                "UTF-8 encoding handling",
                "Emoji processing",
                "Special character support"
            ],
            stress_level="low"
        )

    def _create_empty_query_scenario(self) -> SyntheticScenario:
        """Create scenario with empty and whitespace queries."""
        empty_queries = ["", "   ", "\n", "\t", "  \n  "]
        queries = []
        
        for i, query in enumerate(empty_queries):
            queries.append(TestQuery(
                id=f"empty_{i+1}",
                query=query,
                category="edge_case",
                subcategory="empty_query",
                complexity="simple",
                expected_response_structure={
                    "input_validation": True,
                    "helpful_error_message": True,
                    "graceful_handling": True
                },
                edge_case=True
            ))
        
        return SyntheticScenario(
            name="empty_queries",
            description="Test system handling of empty and whitespace-only queries",
            queries=queries,
            expected_challenges=[
                "Input validation",
                "User-friendly error messages",
                "Graceful degradation"
            ],
            stress_level="low"
        )

    def _create_injection_scenario(self) -> SyntheticScenario:
        """Create scenario with injection attempts."""
        injection_queries = [
            "SELECT * FROM users; DROP TABLE messages;",
            "<script>alert('xss')</script>",
            "'; DROP TABLE users; --",
            "1' OR '1'='1",
            "admin'--",
            "'; INSERT INTO users VALUES ('hacker', 'password'); --"
        ]
        
        queries = []
        for i, query in enumerate(injection_queries):
            queries.append(TestQuery(
                id=f"injection_{i+1}",
                query=query,
                category="edge_case",
                subcategory="injection_attempt",
                complexity="extreme",
                expected_response_structure={
                    "security_validation": True,
                    "sanitization": True,
                    "safe_response": True
                },
                edge_case=True
            ))
        
        return SyntheticScenario(
            name="injection_attempts",
            description="Test system security against injection attacks",
            queries=queries,
            expected_challenges=[
                "SQL injection prevention",
                "XSS prevention",
                "Input sanitization"
            ],
            stress_level="high"
        )

    def _create_high_frequency_scenario(self) -> SyntheticScenario:
        """Create high frequency query scenario."""
        base_queries = [
            "What are the most active channels?",
            "Show me recent activity",
            "Find messages about AI",
            "Generate a weekly digest",
            "Analyze user engagement"
        ]
        
        queries = []
        for i in range(100):  # 100 rapid queries
            base_query = random.choice(base_queries)
            queries.append(TestQuery(
                id=f"high_freq_{i+1}",
                query=base_query,
                category="stress_test",
                subcategory="high_frequency",
                complexity="simple",
                expected_response_structure={
                    "rate_limiting": True,
                    "concurrent_handling": True,
                    "response_time": "< 5.0"
                },
                edge_case=False
            ))
        
        return SyntheticScenario(
            name="high_frequency_queries",
            description="Test system under high query frequency",
            queries=queries,
            expected_challenges=[
                "Rate limiting",
                "Concurrent request handling",
                "Resource management"
            ],
            stress_level="high"
        )

    def _create_memory_intensive_scenario(self) -> SyntheticScenario:
        """Create memory intensive scenario."""
        queries = []
        
        # Large context queries
        large_context_queries = [
            "Analyze all messages from the last 6 months and provide detailed insights about user behavior patterns, engagement trends, and content analysis with comprehensive statistics and visualizations",
            "Generate a complete server analysis including all channels, all users, all messages, all reactions, all threads, all attachments, and all metadata with detailed breakdowns and cross-references",
            "Create a comprehensive digest of all server activity including detailed user profiles, message content analysis, engagement metrics, trending topics, and predictive analytics"
        ]
        
        for i, query in enumerate(large_context_queries):
            queries.append(TestQuery(
                id=f"memory_{i+1}",
                query=query,
                category="stress_test",
                subcategory="memory_intensive",
                complexity="complex",
                expected_response_structure={
                    "memory_management": True,
                    "large_data_handling": True,
                    "streaming_processing": True
                },
                edge_case=False
            ))
        
        return SyntheticScenario(
            name="memory_intensive",
            description="Test system memory handling with large data processing",
            queries=queries,
            expected_challenges=[
                "Memory usage optimization",
                "Large dataset processing",
                "Garbage collection efficiency"
            ],
            stress_level="high"
        )

    def _create_cpu_intensive_scenario(self) -> SyntheticScenario:
        """Create CPU intensive scenario."""
        queries = []
        
        # Complex analytical queries
        complex_queries = [
            "Perform advanced statistical analysis on all server messages including sentiment analysis, topic modeling, user clustering, engagement prediction, and trend forecasting with machine learning algorithms",
            "Generate comprehensive analytics with complex aggregations, time series analysis, correlation studies, and predictive modeling for all server metrics and user interactions",
            "Create detailed user behavior analysis with pattern recognition, anomaly detection, clustering algorithms, and predictive analytics for engagement optimization"
        ]
        
        for i, query in enumerate(complex_queries):
            queries.append(TestQuery(
                id=f"cpu_{i+1}",
                query=query,
                category="stress_test",
                subcategory="cpu_intensive",
                complexity="complex",
                expected_response_structure={
                    "cpu_optimization": True,
                    "algorithm_efficiency": True,
                    "processing_timeout": 60.0
                },
                edge_case=False
            ))
        
        return SyntheticScenario(
            name="cpu_intensive",
            description="Test system CPU handling with complex analytical queries",
            queries=queries,
            expected_challenges=[
                "CPU utilization optimization",
                "Algorithm efficiency",
                "Processing time management"
            ],
            stress_level="high"
        )

    def _create_mixed_load_scenario(self) -> SyntheticScenario:
        """Create mixed load scenario."""
        queries = []
        
        # Mix of different query types
        query_types = [
            ("simple", "What are the most active channels?"),
            ("complex", "Generate a comprehensive weekly digest with detailed analytics"),
            ("moderate", "Find messages about machine learning from the last month"),
            ("simple", "Show me recent activity"),
            ("complex", "Analyze user engagement patterns and predict future trends"),
            ("moderate", "What did @username say about AI?"),
            ("simple", "Find shared resources"),
            ("complex", "Create detailed server health report with recommendations")
        ]
        
        for i in range(50):  # 50 mixed queries
            complexity, query = random.choice(query_types)
            queries.append(TestQuery(
                id=f"mixed_{i+1}",
                query=query,
                category="stress_test",
                subcategory="mixed_load",
                complexity=complexity,
                expected_response_structure={
                    "load_balancing": True,
                    "priority_handling": True,
                    "resource_allocation": True
                },
                edge_case=False
            ))
        
        return SyntheticScenario(
            name="mixed_load",
            description="Test system under mixed query load",
            queries=queries,
            expected_challenges=[
                "Load balancing",
                "Resource allocation",
                "Priority handling"
            ],
            stress_level="medium"
        )

    def _create_concurrent_user_scenario(self) -> SyntheticScenario:
        """Create concurrent user scenario."""
        queries = []
        
        # Simulate multiple users with different query patterns
        user_patterns = [
            ("user1", "What are the most active channels?"),
            ("user2", "Generate a weekly digest"),
            ("user3", "Find messages about AI"),
            ("user4", "Analyze my activity"),
            ("user5", "Show me trending topics")
        ]
        
        for i in range(100):  # 100 queries from different users
            user, base_query = random.choice(user_patterns)
            queries.append(TestQuery(
                id=f"concurrent_{i+1}",
                query=base_query,
                category="stress_test",
                subcategory="concurrent_users",
                complexity="moderate",
                expected_response_structure={
                    "user_isolation": True,
                    "concurrent_processing": True,
                    "session_management": True
                },
                edge_case=False
            ))
        
        return SyntheticScenario(
            name="concurrent_users",
            description="Test system with multiple concurrent users",
            queries=queries,
            expected_challenges=[
                "User session isolation",
                "Concurrent processing",
                "Resource sharing"
            ],
            stress_level="high"
        )

    def _create_gradual_load_scenario(self) -> SyntheticScenario:
        """Create gradual load increase scenario."""
        queries = []
        
        # Start with few queries, gradually increase
        base_query = "What are the most active channels?"
        
        for i in range(50):
            # Gradually increase complexity
            if i < 10:
                complexity = "simple"
                query = base_query
            elif i < 30:
                complexity = "moderate"
                query = "Find messages about AI from the last week"
            else:
                complexity = "complex"
                query = "Generate a comprehensive weekly digest with detailed analytics"
            
            queries.append(TestQuery(
                id=f"gradual_{i+1}",
                query=query,
                category="load_test",
                subcategory="gradual_increase",
                complexity=complexity,
                expected_response_structure={
                    "scalability": True,
                    "resource_adaptation": True,
                    "performance_consistency": True
                },
                edge_case=False
            ))
        
        return SyntheticScenario(
            name="gradual_load_increase",
            description="Test system scalability with gradual load increase",
            queries=queries,
            expected_challenges=[
                "Scalability",
                "Resource adaptation",
                "Performance consistency"
            ],
            stress_level="medium"
        )

    def _create_spike_load_scenario(self) -> SyntheticScenario:
        """Create spike load scenario."""
        queries = []
        
        # Sudden spike of simple queries
        for i in range(50):
            queries.append(TestQuery(
                id=f"spike_{i+1}",
                query="What are the most active channels?",
                category="load_test",
                subcategory="spike_load",
                complexity="simple",
                expected_response_structure={
                    "spike_handling": True,
                    "queue_management": True,
                    "recovery_time": "< 30.0"
                },
                edge_case=False
            ))
        
        return SyntheticScenario(
            name="spike_load",
            description="Test system handling of sudden load spikes",
            queries=queries,
            expected_challenges=[
                "Spike handling",
                "Queue management",
                "Recovery time"
            ],
            stress_level="high"
        )

    def _create_sustained_load_scenario(self) -> SyntheticScenario:
        """Create sustained load scenario."""
        queries = []
        
        # Sustained moderate load
        for i in range(100):
            query = random.choice([
                "What are the most active channels?",
                "Show me recent activity",
                "Find messages about AI",
                "Generate a weekly digest"
            ])
            
            queries.append(TestQuery(
                id=f"sustained_{i+1}",
                query=query,
                category="load_test",
                subcategory="sustained_load",
                complexity="moderate",
                expected_response_structure={
                    "sustained_performance": True,
                    "resource_stability": True,
                    "memory_management": True
                },
                edge_case=False
            ))
        
        return SyntheticScenario(
            name="sustained_load",
            description="Test system under sustained moderate load",
            queries=queries,
            expected_challenges=[
                "Sustained performance",
                "Resource stability",
                "Memory management"
            ],
            stress_level="medium"
        )

    def _create_burst_load_scenario(self) -> SyntheticScenario:
        """Create burst load scenario."""
        queries = []
        
        # Burst of complex queries
        complex_queries = [
            "Generate a comprehensive weekly digest with detailed analytics",
            "Analyze all user engagement patterns and predict future trends",
            "Create detailed server health report with recommendations",
            "Perform advanced statistical analysis on all server messages"
        ]
        
        for i in range(20):  # 20 complex queries in burst
            query = random.choice(complex_queries)
            queries.append(TestQuery(
                id=f"burst_{i+1}",
                query=query,
                category="load_test",
                subcategory="burst_load",
                complexity="complex",
                expected_response_structure={
                    "burst_handling": True,
                    "priority_queue": True,
                    "resource_allocation": True
                },
                edge_case=False
            ))
        
        return SyntheticScenario(
            name="burst_load",
            description="Test system handling of burst of complex queries",
            queries=queries,
            expected_challenges=[
                "Burst handling",
                "Priority queue management",
                "Resource allocation"
            ],
            stress_level="extreme"
        )

    def save_scenarios(self, scenarios: List[SyntheticScenario], filename: Optional[str] = None) -> str:
        """Save scenarios to JSON file."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"synthetic_scenarios_{timestamp}.json"
        
        # Convert scenarios to serializable format
        serializable_scenarios = []
        for scenario in scenarios:
            scenario_dict = {
                "name": scenario.name,
                "description": scenario.description,
                "expected_challenges": scenario.expected_challenges,
                "stress_level": scenario.stress_level,
                "queries": [asdict(query) for query in scenario.queries]
            }
            serializable_scenarios.append(scenario_dict)
        
        # Save to file
        scenarios_dir = Path("tests/performance_test_suite/data")
        scenarios_dir.mkdir(exist_ok=True)
        
        file_path = scenarios_dir / filename
        with open(file_path, 'w') as f:
            json.dump(serializable_scenarios, f, indent=2, default=str)
        
        logger.info(f"Synthetic scenarios saved to: {file_path}")
        return str(file_path)

    def get_scenario_summary(self, scenarios: List[SyntheticScenario]) -> Dict[str, Any]:
        """Get summary of generated scenarios."""
        total_queries = sum(len(scenario.queries) for scenario in scenarios)
        
        stress_levels = {}
        categories = {}
        
        for scenario in scenarios:
            stress_levels[scenario.stress_level] = stress_levels.get(scenario.stress_level, 0) + 1
            
            for query in scenario.queries:
                categories[query.category] = categories.get(query.category, 0) + 1
        
        return {
            "total_scenarios": len(scenarios),
            "total_queries": total_queries,
            "stress_level_distribution": stress_levels,
            "category_distribution": categories,
            "scenario_names": [s.name for s in scenarios]
        } 