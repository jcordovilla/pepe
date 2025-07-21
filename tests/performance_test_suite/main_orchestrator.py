"""
Main Performance Test Orchestrator

Coordinates the execution of the complete performance test suite.
Orchestrates content analysis, query generation, bot execution, evaluation, and reporting.
"""

import asyncio
import json
import logging
import sys
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

# Import test suite components
from tests.performance_test_suite.content_analyzer import ContentAnalyzer
from tests.performance_test_suite.query_generator import QueryGenerator
from tests.performance_test_suite.bot_runner import BotRunner
from tests.performance_test_suite.evaluator import ResponseEvaluator
from tests.performance_test_suite.report_generator import ReportGenerator

logger = logging.getLogger(__name__)


class PerformanceTestOrchestrator:
    """
    Main orchestrator for the comprehensive performance test suite.
    
    Coordinates:
    1. Content analysis of Discord server
    2. Generation of 50 diverse test queries
    3. Execution of queries against the bot
    4. Evaluation of responses against expected answers
    5. Generation of comprehensive report with recommendations
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        self.results = {}
        
        # Initialize components
        self.content_analyzer = ContentAnalyzer(self.config.get("database_path"))
        self.content_analyzer.preflight_schema_check()
        self.query_generator = QueryGenerator()
        self.bot_runner = BotRunner(self.config.get("bot_api_endpoint"))
        self.evaluator = ResponseEvaluator()
        self.report_generator = ReportGenerator()
        
        logger.info("PerformanceTestOrchestrator initialized")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for the test suite."""
        return {
            "database_path": "data/discord_messages.db",  # Same as production
            "bot_api_endpoint": "http://localhost:8000",  # Not used - we use in-process AgentAPI
            "output_directory": "tests/performance_test_suite/data",
            "sample_percentage": 0.15,
            "query_count": 30,  # Updated to 30 queries
            "enable_error_scenarios": True,
            "save_intermediate_results": True,
            # Ensure we use the same configuration as production
            "use_production_config": True,
            "real_data_mode": True
        }
    
    async def run_performance_tests(self) -> Dict[str, Any]:
        """
        Run comprehensive performance tests with parallel execution and resource optimization.
        
        Returns:
            Complete test results with performance metrics
        """
        try:
            logger.info("Starting comprehensive performance tests...")
            
            # Initialize service container for resource optimization
            from agentic.services.service_container import initialize_global_services, cleanup_global_services
            await initialize_global_services(self.config)
            
            # Initialize components with shared services
            await self._initialize_components()
            
            # Run tests with parallel execution
            results = await self._run_tests_parallel()
            
            # Clean up resources
            await cleanup_global_services()
            
            logger.info("Performance tests completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Error in performance tests: {e}")
            # Ensure cleanup happens even on error
            try:
                from agentic.services.service_container import cleanup_global_services
                await cleanup_global_services()
            except:
                pass
            raise
    
    async def _initialize_components(self):
        """Initialize all test components with shared services."""
        try:
            logger.info("Initializing test components with shared services...")
            
            # Get service container
            from agentic.services.service_container import get_service_container
            service_container = get_service_container()
            
            # Initialize content analyzer
            self.content_analyzer = ContentAnalyzer(self.config.get("database_path"))
            self.content_analyzer.preflight_schema_check()
            
            # Initialize query generator with shared services
            self.query_generator = QueryGenerator(self.config)
            service_container.inject_services(self.query_generator)
            
            # Initialize bot runner with shared services
            self.bot_runner = BotRunner()
            service_container.inject_services(self.bot_runner)
            
            # Initialize evaluator with shared services
            self.evaluator = ResponseEvaluator()
            service_container.inject_services(self.evaluator)
            
            # Initialize report generator
            self.report_generator = ReportGenerator()
            
            logger.info("All components initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            raise
    
    async def _run_tests_parallel(self) -> Dict[str, Any]:
        """Run tests with parallel execution for better performance."""
        try:
            logger.info("Running tests with parallel execution...")
            
            # Step 1: Content Analysis (can run in parallel with other setup)
            content_analysis_task = asyncio.create_task(
                self.content_analyzer.analyze_server_content()
            )
            
            # Step 2: Generate queries (can run in parallel with content analysis)
            query_generation_task = asyncio.create_task(
                self.query_generator.generate_test_queries()
            )
            
            # Wait for both tasks to complete
            content_analysis, queries = await asyncio.gather(
                content_analysis_task, 
                query_generation_task
            )
            
            logger.info(f"Generated {len(queries)} test queries")
            
            # Step 3: Run queries in parallel
            responses = await self.bot_runner.run_queries_parallel(
                queries, 
                max_concurrent=5
            )
            
            logger.info(f"Executed {len(responses)} queries")
            
            # Step 4: Evaluate responses in parallel
            evaluation_results = await self._evaluate_responses_parallel(queries, responses)
            
            # Step 5: Generate comprehensive report
            report = await self.report_generator.generate_comprehensive_report(
                content_analysis=content_analysis,
                queries=queries,
                responses=responses,
                evaluation_results=evaluation_results,
                config=self.config
            )
            
            return report
            
        except Exception as e:
            logger.error(f"Error in parallel test execution: {e}")
            raise
    
    async def _evaluate_responses_parallel(self, queries: List[Any], responses: List[Any]) -> List[Any]:
        """Evaluate responses in parallel for better performance."""
        try:
            logger.info(f"Evaluating {len(responses)} responses in parallel...")
            
            # Create evaluation tasks
            evaluation_tasks = []
            for query, response in zip(queries, responses):
                task = asyncio.create_task(
                    self._evaluate_single_response_async(query, response)
                )
                evaluation_tasks.append(task)
            
            # Execute all evaluations in parallel
            evaluation_results = await asyncio.gather(*evaluation_tasks, return_exceptions=True)
            
            # Handle any exceptions
            final_results = []
            for i, result in enumerate(evaluation_results):
                if isinstance(result, Exception):
                    logger.error(f"Evaluation {i} failed: {result}")
                    # Create fallback result
                    final_results.append(self._create_fallback_evaluation_result(queries[i], responses[i]))
                else:
                    final_results.append(result)
            
            logger.info(f"Completed {len(final_results)} evaluations")
            return final_results
            
        except Exception as e:
            logger.error(f"Error in parallel response evaluation: {e}")
            raise
    
    async def _evaluate_single_response_async(self, query: Any, response: Any) -> Any:
        """Evaluate a single response asynchronously."""
        try:
            return self.evaluator._evaluate_single_response(query, response)
        except Exception as e:
            logger.error(f"Error in async evaluation: {e}")
            raise
    
    def _create_fallback_evaluation_result(self, query: Any, response: Any) -> Any:
        """Create a fallback evaluation result when evaluation fails."""
        from .evaluator import EvaluationResult
        
        return EvaluationResult(
            query_id=query.id,
            query=query.query,
            expected_structure=query.expected_structure,
            actual_response=response.response,
            overall_score=0.0,
            metrics={"overall": 0.0, "error": "Evaluation failed"},
            detailed_analysis={"error": "Evaluation failed"},
            recommendations=["Check evaluation system"]
        )
    
    async def _save_all_results(self):
        """Save all test results to a single file."""
        try:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.config['output_directory']}/complete_test_results_{timestamp}.json"
            Path(filename).parent.mkdir(parents=True, exist_ok=True)
            
            with open(filename, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
            
            logger.info(f"Complete test results saved to: {filename}")
            
        except Exception as e:
            logger.error(f"Error saving complete results: {e}")
    
    async def cleanup(self):
        """Clean up resources."""
        try:
            await self.bot_runner.cleanup()
            logger.info("Cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    def get_test_summary(self) -> Dict[str, Any]:
        """Get a summary of the test execution."""
        if not self.results:
            return {"status": "No test results available"}
        
        return {
            "status": "Test completed successfully",
            "timestamp": self.results["test_execution"]["timestamp"],
            "summary": self.results["summary"],
            "key_findings": self.results["comprehensive_report"].key_findings[:5],
            "top_recommendations": [
                rec.title for rec in self.results["comprehensive_report"].recommendations[:3]
            ]
        }


async def main():
    """Main entry point for the performance test suite."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('tests/performance_test_suite/performance_test.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger.info("Starting Performance Test Suite")
    
    # Configuration
    config = {
        "database_path": "data/discord_messages.db",
        "bot_api_endpoint": "http://localhost:8000",
        "output_directory": "tests/performance_test_suite/data",
        "sample_percentage": 0.15,
        "query_count": 50,
        "enable_error_scenarios": True,
        "save_intermediate_results": True
    }
    
    # Create orchestrator
    orchestrator = PerformanceTestOrchestrator(config)
    
    try:
        # Run complete test suite
        results = await orchestrator.run_performance_tests()
        
        # Print summary
        summary = orchestrator.get_test_summary()
        print("\n" + "="*60)
        print("PERFORMANCE TEST SUITE - EXECUTION SUMMARY")
        print("="*60)
        print(f"Status: {summary['status']}")
        print(f"Timestamp: {summary['timestamp']}")
        print(f"Total Queries: {summary['summary']['total_queries']}")
        print(f"Successful Responses: {summary['summary']['successful_responses']}")
        print(f"Average Quality Score: {summary['summary']['average_quality_score']:.2f}")
        print(f"System Reliability: {summary['summary']['system_reliability']:.2f}")
        
        print("\nKey Findings:")
        for finding in summary['key_findings']:
            print(f"  - {finding}")
        
        print("\nTop Recommendations:")
        for rec in summary['top_recommendations']:
            print(f"  - {rec}")
        
        print("\n" + "="*60)
        print("Test suite execution completed successfully!")
        print("Check the generated reports for detailed analysis and recommendations.")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Test suite execution failed: {e}")
        print(f"\nERROR: Test suite execution failed: {e}")
        sys.exit(1)
    
    finally:
        # Cleanup
        await orchestrator.cleanup()


if __name__ == "__main__":
    asyncio.run(main()) 