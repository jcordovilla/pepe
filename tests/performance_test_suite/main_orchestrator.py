"""
Main Performance Test Orchestrator

Coordinates the execution of the complete performance test suite.
Orchestrates content analysis, query generation, bot execution, evaluation, and reporting.
Updated for v2 agentic architecture with comprehensive progress reporting.
"""

import asyncio
import json
import logging
import sys
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
from tqdm import tqdm

# Import test suite components
from tests.performance_test_suite.content_analyzer import ContentAnalyzer
from tests.performance_test_suite.query_generator import QueryGenerator
from tests.performance_test_suite.bot_runner import BotRunner
from tests.performance_test_suite.evaluator import ResponseEvaluator
from tests.performance_test_suite.report_generator import ReportGenerator

logger = logging.getLogger(__name__)

# Add color utility
class Color:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_phase(msg):
    print(f"{Color.HEADER}{Color.BOLD}\n=== {msg} ==={Color.ENDC}")
    sys.stdout.flush()

def print_success(msg):
    print(f"{Color.OKGREEN}{msg}{Color.ENDC}")
    sys.stdout.flush()

def print_warning(msg):
    print(f"{Color.WARNING}{msg}{Color.ENDC}")
    sys.stdout.flush()

def print_error(msg):
    print(f"{Color.FAIL}{msg}{Color.ENDC}")
    sys.stdout.flush()

def print_info(msg):
    print(f"{Color.OKBLUE}{msg}{Color.ENDC}")
    sys.stdout.flush()

def print_progress(phase: str, current: int, total: int, description: str = ""):
    """Print progress with timestamp and percentage."""
    percentage = (current / total) * 100 if total > 0 else 0
    timestamp = datetime.now().strftime("%H:%M:%S")
    progress_bar = "█" * int(percentage / 5) + "░" * (20 - int(percentage / 5))
    print(f"{Color.OKCYAN}[{timestamp}] {phase}: [{progress_bar}] {current}/{total} ({percentage:.1f}%) {description}{Color.ENDC}")
    sys.stdout.flush()


class PerformanceTestOrchestrator:
    """
    Main orchestrator for the comprehensive performance test suite.
    
    Coordinates:
    1. Content analysis of Discord server
    2. Generation of diverse test queries for v2 agents
    3. Execution of queries against the bot
    4. Evaluation of responses against expected answers
    5. Generation of comprehensive report with recommendations
    
    Updated for v2 agentic architecture with enhanced progress tracking.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        self.results = {}
        self.start_time = None
        self.phase_times = {}
        
        # Initialize components
        self.content_analyzer = ContentAnalyzer(self.config.get("database_path"))
        self.content_analyzer.preflight_schema_check()
        self.query_generator = QueryGenerator()
        self.bot_runner = BotRunner(self.config.get("bot_api_endpoint"))
        self.evaluator = ResponseEvaluator()
        self.report_generator = ReportGenerator()
        
        logger.info("PerformanceTestOrchestrator initialized for v2 agentic architecture")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for the test suite."""
        return {
            "database_path": "data/discord_messages.db",
            "bot_api_endpoint": "http://localhost:8000",
            "output_directory": "tests/performance_test_suite/data",
            "sample_percentage": 0.15,
            "query_count": 35,  # Updated for v2 agents (5 per agent type)
            "enable_error_scenarios": True,
            "save_intermediate_results": True,
            "use_production_config": True,
            "real_data_mode": True,
            "v2_agents": {
                "router": True,
                "qa": True,
                "stats": True,
                "digest": True,
                "trend": True,
                "structure": True,
                "selfcheck": True
            }
        }
    
    async def run_performance_tests(self) -> Dict[str, Any]:
        """
        Run the complete performance test suite with comprehensive progress reporting.
        
        Returns:
            Complete test results with timing and performance metrics
        """
        self.start_time = time.time()
        print_phase("🚀 Starting Performance Test Suite - V2 Agentic Architecture")
        print_info(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print_info(f"Configuration: {self.config['query_count']} queries, {self.config['sample_percentage']*100}% sample")
        
        try:
            # Phase 1: Content Analysis
            phase_start = time.time()
            print_phase("📊 Phase 1: Content Analysis")
            print_info("Analyzing Discord server content and structure...")
            
            content_analysis = await self._run_content_analysis()
            self.phase_times['content_analysis'] = time.time() - phase_start
            
            print_success(f"✅ Content analysis completed in {self.phase_times['content_analysis']:.2f}s")
            print_info(f"   - Analyzed {content_analysis.get('total_messages', 0):,} messages")
            print_info(f"   - Found {content_analysis.get('unique_channels', 0)} channels")
            print_info(f"   - Identified {content_analysis.get('unique_users', 0)} users")
            
            # Phase 2: Query Generation
            phase_start = time.time()
            print_phase("🎯 Phase 2: Query Generation")
            print_info("Generating test queries for v2 agents...")
            
            queries = await self._run_query_generation(content_analysis)
            self.phase_times['query_generation'] = time.time() - phase_start
            
            print_success(f"✅ Query generation completed in {self.phase_times['query_generation']:.2f}s")
            print_info(f"   - Generated {len(queries)} queries")
            
            # Show query distribution by agent type
            agent_distribution = self._analyze_query_distribution(queries)
            for agent, count in agent_distribution.items():
                print_info(f"   - {agent.upper()}: {count} queries")
            
            # Phase 3: Bot Execution
            phase_start = time.time()
            print_phase("🤖 Phase 3: Bot Execution")
            print_info("Executing queries against v2 agentic system...")
            
            responses = await self._run_bot_execution(queries)
            self.phase_times['bot_execution'] = time.time() - phase_start
            
            print_success(f"✅ Bot execution completed in {self.phase_times['bot_execution']:.2f}s")
            print_info(f"   - Executed {len(responses)} queries")
            print_info(f"   - Success rate: {self._calculate_success_rate(responses):.1f}%")
            print_info(f"   - Average response time: {self._calculate_avg_response_time(responses):.2f}s")
            
            # Phase 4: Response Evaluation
            phase_start = time.time()
            print_phase("📋 Phase 4: Response Evaluation")
            print_info("Evaluating responses against expected outcomes...")
            
            evaluations = await self._run_response_evaluation(queries, responses)
            self.phase_times['response_evaluation'] = time.time() - phase_start
            
            print_success(f"✅ Response evaluation completed in {self.phase_times['response_evaluation']:.2f}s")
            print_info(f"   - Evaluated {len(evaluations)} responses")
            
            # Phase 5: Report Generation
            phase_start = time.time()
            print_phase("📈 Phase 5: Report Generation")
            print_info("Generating comprehensive performance report...")
            
            report = await self._run_report_generation(queries, responses, evaluations)
            self.phase_times['report_generation'] = time.time() - phase_start
            
            print_success(f"✅ Report generation completed in {self.phase_times['report_generation']:.2f}s")
            
            # Compile final results
            total_time = time.time() - self.start_time
            self.results = {
                "test_summary": {
                    "total_duration": total_time,
                    "phase_times": self.phase_times,
                    "queries_executed": len(queries),
                    "responses_received": len(responses),
                    "success_rate": self._calculate_success_rate(responses),
                    "average_response_time": self._calculate_avg_response_time(responses)
                },
                "content_analysis": content_analysis,
                "queries": queries,
                "responses": responses,
                "evaluations": evaluations,
                "report": report,
                "agent_performance": self._analyze_agent_performance(responses)
            }
            
            # Final summary
            print_phase("🎉 Test Suite Complete")
            print_success(f"Total execution time: {total_time:.2f}s")
            print_success(f"Results saved to: {self.config['output_directory']}")
            
            return self.results
            
        except Exception as e:
            print_error(f"❌ Test suite failed: {str(e)}")
            logger.error(f"Test suite execution failed: {e}", exc_info=True)
            raise
    
    async def _run_content_analysis(self) -> Dict[str, Any]:
        """Run content analysis with progress reporting."""
        print_info("Starting content analysis...")
        
        try:
            # Initialize content analyzer
            await self._initialize_components()
            
            # Run analysis with progress updates
            analysis_result = await self.content_analyzer.analyze_content_with_progress()
            
            # Save intermediate results
            if self.config.get("save_intermediate_results"):
                output_dir = Path(self.config["output_directory"])
                output_dir.mkdir(parents=True, exist_ok=True)
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                analysis_file = output_dir / f"content_analysis_{timestamp}.json"
                
                with open(analysis_file, 'w') as f:
                    json.dump(analysis_result, f, indent=2, default=str)
                
                print_info(f"Content analysis saved to: {analysis_file}")
            
            return analysis_result
            
        except Exception as e:
            print_error(f"Content analysis failed: {str(e)}")
            logger.error(f"Content analysis error: {e}", exc_info=True)
            raise
    
    async def _run_query_generation(self, content_analysis: Dict[str, Any]) -> List[Any]:
        """Run query generation with progress reporting."""
        print_info("Starting query generation for v2 agents...")
        
        try:
            # Update query generator with content analysis
            self.query_generator.content_analysis = content_analysis
            
            # Generate queries with progress tracking
            queries = self.query_generator.generate_test_queries()
            
            # Save intermediate results
            if self.config.get("save_intermediate_results"):
                output_dir = Path(self.config["output_directory"])
                output_dir.mkdir(parents=True, exist_ok=True)
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                queries_file = output_dir / f"test_queries_{timestamp}.json"
                
                # Convert queries to serializable format
                serializable_queries = []
                for query in queries:
                    query_dict = query.__dict__.copy()
                    query_dict['expected_response_structure'] = query.expected_response_structure
                    serializable_queries.append(query_dict)
                
                with open(queries_file, 'w') as f:
                    json.dump(serializable_queries, f, indent=2, default=str)
                
                print_info(f"Test queries saved to: {queries_file}")
            
            return queries
            
        except Exception as e:
            print_error(f"Query generation failed: {str(e)}")
            logger.error(f"Query generation error: {e}", exc_info=True)
            raise
    
    async def _run_bot_execution(self, queries: List[Any]) -> List[Any]:
        """Run bot execution with comprehensive progress reporting."""
        print_info("Starting bot execution with v2 agents...")
        
        try:
            # Execute queries with progress tracking
            responses = await self.bot_runner.run_queries_with_progress(queries)
            
            # Save intermediate results
            if self.config.get("save_intermediate_results"):
                output_dir = Path(self.config["output_directory"])
                output_dir.mkdir(parents=True, exist_ok=True)
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                responses_file = output_dir / f"bot_responses_{timestamp}.json"
                
                # Convert responses to serializable format
                serializable_responses = []
                for response in responses:
                    response_dict = response.__dict__.copy()
                    serializable_responses.append(response_dict)
                
                with open(responses_file, 'w') as f:
                    json.dump(serializable_responses, f, indent=2, default=str)
                
                print_info(f"Bot responses saved to: {responses_file}")
            
            return responses
            
        except Exception as e:
            print_error(f"Bot execution failed: {str(e)}")
            logger.error(f"Bot execution error: {e}", exc_info=True)
            raise
    
    async def _run_response_evaluation(self, queries: List[Any], responses: List[Any]) -> List[Any]:
        """Run response evaluation with progress reporting."""
        print_info("Starting response evaluation...")
        
        try:
            # Evaluate responses with progress tracking
            evaluations = await self.evaluator.evaluate_responses_with_progress(queries, responses)
            
            # Save intermediate results
            if self.config.get("save_intermediate_results"):
                output_dir = Path(self.config["output_directory"])
                output_dir.mkdir(parents=True, exist_ok=True)
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                evaluations_file = output_dir / f"evaluations_{timestamp}.json"
                
                with open(evaluations_file, 'w') as f:
                    json.dump(evaluations, f, indent=2, default=str)
                
                print_info(f"Evaluations saved to: {evaluations_file}")
            
            return evaluations
            
        except Exception as e:
            print_error(f"Response evaluation failed: {str(e)}")
            logger.error(f"Response evaluation error: {e}", exc_info=True)
            raise
    
    async def _run_report_generation(self, queries: List[Any], responses: List[Any], evaluations: List[Any]) -> Dict[str, Any]:
        """Run report generation with progress reporting."""
        print_info("Starting report generation...")
        
        try:
            # Generate comprehensive report
            report = await self.report_generator.generate_comprehensive_report(
                queries, responses, evaluations, self.phase_times
            )
            
            # Save final report
            output_dir = Path(self.config["output_directory"])
            output_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = output_dir / f"performance_report_{timestamp}.json"
            
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            print_info(f"Final report saved to: {report_file}")
            
            return report
            
        except Exception as e:
            print_error(f"Report generation failed: {str(e)}")
            logger.error(f"Report generation error: {e}", exc_info=True)
            raise
    
    def _analyze_query_distribution(self, queries: List[Any]) -> Dict[str, int]:
        """Analyze distribution of queries by agent type."""
        distribution = {}
        for query in queries:
            # Extract agent type from query category or subcategory
            agent_type = self._extract_agent_type_from_query(query)
            distribution[agent_type] = distribution.get(agent_type, 0) + 1
        return distribution
    
    def _extract_agent_type_from_query(self, query: Any) -> str:
        """Extract the target agent type from a query."""
        category = query.category.lower()
        subcategory = query.subcategory.lower()
        
        # Map query categories to v2 agent types
        if any(word in category for word in ['qa', 'question', 'answer']):
            return 'qa'
        elif any(word in category for word in ['stats', 'statistics', 'metrics']):
            return 'stats'
        elif any(word in category for word in ['digest', 'summary', 'summarize']):
            return 'digest'
        elif any(word in category for word in ['trend', 'trending', 'topic']):
            return 'trend'
        elif any(word in category for word in ['structure', 'channel', 'organization']):
            return 'structure'
        else:
            return 'router'  # Default to router for general queries
    
    def _calculate_success_rate(self, responses: List[Any]) -> float:
        """Calculate success rate from responses."""
        if not responses:
            return 0.0
        successful = sum(1 for r in responses if r.success)
        return (successful / len(responses)) * 100
    
    def _calculate_avg_response_time(self, responses: List[Any]) -> float:
        """Calculate average response time."""
        if not responses:
            return 0.0
        total_time = sum(r.response_time for r in responses)
        return total_time / len(responses)
    
    def _analyze_agent_performance(self, responses: List[Any]) -> Dict[str, Any]:
        """Analyze performance by agent type."""
        agent_stats = {}
        
        for response in responses:
            # Extract agent type from metadata
            agent_type = response.metadata.get('agent_used', 'unknown')
            
            if agent_type not in agent_stats:
                agent_stats[agent_type] = {
                    'count': 0,
                    'success_count': 0,
                    'total_time': 0.0,
                    'avg_time': 0.0
                }
            
            agent_stats[agent_type]['count'] += 1
            agent_stats[agent_type]['total_time'] += response.response_time
            
            if response.success:
                agent_stats[agent_type]['success_count'] += 1
        
        # Calculate averages
        for agent_type, stats in agent_stats.items():
            if stats['count'] > 0:
                stats['avg_time'] = stats['total_time'] / stats['count']
                stats['success_rate'] = (stats['success_count'] / stats['count']) * 100
        
        return agent_stats
    
    async def _initialize_components(self):
        """Initialize all test components."""
        print_info("Initializing test components...")
        
        # Initialize components that need async setup
        await self.content_analyzer.initialize()
        await self.bot_runner.initialize()
        await self.evaluator.initialize()
        
        print_success("✅ All components initialized")
    
    async def cleanup(self):
        """Clean up resources."""
        print_info("Cleaning up test resources...")
        
        await self.content_analyzer.cleanup()
        await self.bot_runner.cleanup()
        await self.evaluator.cleanup()
        
        print_success("✅ Cleanup completed")
    
    def get_test_summary(self) -> Dict[str, Any]:
        """Get a summary of the test execution."""
        if not self.results:
            return {"status": "No tests executed"}
        
        return {
            "status": "completed",
            "total_duration": self.results["test_summary"]["total_duration"],
            "queries_executed": self.results["test_summary"]["queries_executed"],
            "success_rate": self.results["test_summary"]["success_rate"],
            "average_response_time": self.results["test_summary"]["average_response_time"],
            "phase_times": self.phase_times,
            "agent_performance": self.results.get("agent_performance", {})
        }


async def main():
    """Main entry point for the performance test suite."""
    try:
        # Parse command line arguments
        import argparse
        parser = argparse.ArgumentParser(description="Performance Test Suite for V2 Agentic Architecture")
        parser.add_argument("--config", type=str, help="Configuration file path")
        parser.add_argument("--output-dir", type=str, help="Output directory")
        parser.add_argument("--query-count", type=int, help="Number of queries to generate")
        
        args = parser.parse_args()
        
        # Load configuration
        config = {}
        if args.config:
            with open(args.config, 'r') as f:
                config = json.load(f)
        
        # Override with command line arguments
        if args.output_dir:
            config["output_directory"] = args.output_dir
        if args.query_count:
            config["query_count"] = args.query_count
        
        # Run tests
        orchestrator = PerformanceTestOrchestrator(config)
        results = await orchestrator.run_performance_tests()
        
        # Print summary
        summary = orchestrator.get_test_summary()
        print_phase("📊 Test Summary")
        print_info(f"Total Duration: {summary['total_duration']:.2f}s")
        print_info(f"Queries Executed: {summary['queries_executed']}")
        print_info(f"Success Rate: {summary['success_rate']:.1f}%")
        print_info(f"Average Response Time: {summary['average_response_time']:.2f}s")
        
        # Print agent performance
        print_phase("🤖 Agent Performance")
        for agent, stats in summary.get('agent_performance', {}).items():
            print_info(f"{agent.upper()}: {stats['count']} queries, {stats['success_rate']:.1f}% success, {stats['avg_time']:.2f}s avg")
        
        await orchestrator.cleanup()
        
    except KeyboardInterrupt:
        print_warning("\n⚠️ Test execution interrupted by user")
    except Exception as e:
        print_error(f"❌ Test execution failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main()) 